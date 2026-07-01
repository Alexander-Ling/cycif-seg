from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qtpy import QtCore, QtWidgets
from napari.qt.threading import thread_worker
from napari.settings import get_settings
from napari.utils.notifications import show_warning

from cycif_seg.io.ome_tiff import (
    LazyChannelImage,
    inspect_tiff_pyramid,
    inspect_tiff_yxc,
    is_tiff_loading_debug,
    load_channel_downsampled,
    load_channel_multiscale_lazy,
    load_single_channel_tiff_native,
)


class ViewerTileRuntime:
    """Small owner for viewer async/tile scheduling state.

    Napari 0.7 can slice layers off the main thread, but the setting is off by
    default.  Step 2a multiscale layers are the only app path that needs this
    aggressive viewer behavior, so the controller owns the opt-in and the
    generation counters used to ignore stale worker callbacks.
    """

    def __init__(self, *, max_workers: int | None = None):
        cpu = os.cpu_count() or 2
        self.max_workers = max(1, int(max_workers or min(8, max(2, cpu // 2))))
        self._pool: ThreadPoolExecutor | None = None
        self._load_generation = 0
        self._channel_generation = 0

    def configure_for_multiscale(self) -> None:
        try:
            get_settings().experimental.async_ = True
        except Exception:
            pass

        try:
            from cycif_seg.io import ome_tiff as _ome_tiff

            if _ome_tiff.dask is not None:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix="cycif-viewer-tile",
                    )
                _ome_tiff.dask.config.set(scheduler="threads", pool=self._pool)
        except Exception:
            pass

    def next_load_generation(self) -> int:
        self._load_generation += 1
        self._channel_generation += 1
        return self._load_generation

    def current_load_generation(self) -> int:
        return self._load_generation

    def is_current_load(self, generation: int) -> bool:
        return int(generation) == int(self._load_generation)

    def next_channel_generation(self) -> int:
        self._channel_generation += 1
        return self._channel_generation

    def is_current_channel(self, generation: int) -> bool:
        return int(generation) == int(self._channel_generation)

    def shutdown(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)


class ImageController:
    """Handles image loading and channel display management.

    This controller intentionally operates on the owning main widget (``w``)
    to minimize invasive state rewrites during refactors.
    """

    def __init__(self, w):
        self.w = w
        self.tile_runtime = ViewerTileRuntime()

    # --------------------------
    # Multiscale / contrast helpers
    # --------------------------

    @staticmethod
    def _contrast_limits_from_thumbnail(path: str, channel_index: int, dtype) -> tuple[float, float] | None:
        """Return (lo, hi) contrast limits from a tiny downsampled thumbnail.

        Uses the smallest available pyramid level (or a 16× block-average downsample
        for non-pyramidal files) to compute 1st/99th percentiles without loading the
        full-resolution channel.  Returns None if the thumbnail load fails so callers
        can fall back to dtype-range limits.
        """
        try:
            thumb = load_channel_downsampled(path, channel_index, factor=16)
            if thumb is not None and thumb.size > 0:
                lo = float(np.percentile(thumb, 1.0))
                hi = float(np.percentile(thumb, 99.0))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return (lo, hi)
        except Exception:
            pass
        return None

    @staticmethod
    def _dtype_contrast_limits(dtype) -> tuple[float, float]:
        """Fallback contrast limits from the dtype range."""
        try:
            if np.issubdtype(np.dtype(dtype), np.integer):
                info = np.iinfo(np.dtype(dtype))
                return (float(info.min), float(info.max))
        except Exception:
            pass
        return (0.0, 1.0)

    def _get_contrast_limits(self, path: str, channel_index: int, dtype) -> tuple[float, float]:
        lims = self._contrast_limits_from_thumbnail(path, channel_index, dtype)
        return lims if lims is not None else self._dtype_contrast_limits(dtype)

    def _want_multiscale(self) -> bool:
        w = self.w
        return bool(getattr(w, "chk_multiscale", None) and w.chk_multiscale.isChecked())

    def _load_multiscale_levels(self, path: str | None, channel_index: int) -> list | None:
        if not path or not self._want_multiscale():
            return None
        self.tile_runtime.configure_for_multiscale()
        return load_channel_multiscale_lazy(path, int(channel_index))

    @staticmethod
    def _dask_available() -> bool:
        try:
            from cycif_seg.io import ome_tiff as _ome_tiff

            return _ome_tiff.dask is not None and _ome_tiff.da is not None
        except Exception:
            return False

    @classmethod
    def _can_use_lazy_pyramid(cls, path: str, *, want_multiscale: bool) -> bool:
        if not want_multiscale or not cls._dask_available():
            return False
        try:
            return bool(inspect_tiff_pyramid(path).get("is_pyramidal"))
        except Exception:
            return False

    @classmethod
    def _load_initial_display_probe(cls, path: str, *, want_multiscale: bool) -> tuple[bool, np.ndarray | None]:
        if cls._can_use_lazy_pyramid(path, want_multiscale=want_multiscale):
            return True, None
        return False, np.asarray(load_single_channel_tiff_native(path, 0))

    def _add_or_update_channel_layer(
        self,
        channel_index: int,
        *,
        levels: list | None = None,
        plane: np.ndarray | None = None,
    ) -> bool:
        w = self.w
        if w.img is None or w.ch_names is None:
            return False

        i = int(channel_index)
        nm = w.ch_names[i]
        path = getattr(w.img, "path", None)
        if levels is None and plane is None:
            levels = self._load_multiscale_levels(path, i)

        if levels is not None:
            if is_tiff_loading_debug():
                print(
                    f"[viewer ch={i}] multiscale path: {len(levels)} level(s), "
                    f"shapes={[tuple(l.shape) for l in levels]}",
                    flush=True,
                )
            clims = self._get_contrast_limits(str(path), i, w.img.dtype)
            if is_tiff_loading_debug():
                print(f"[viewer ch={i}] contrast_limits={clims}", flush=True)
            if nm in w.viewer.layers:
                layer = w.viewer.layers[nm]
                layer.data = levels
                try:
                    layer.multiscale = len(levels) > 1
                    layer.contrast_limits = clims
                except Exception:
                    pass
                layer.visible = True
            else:
                w.viewer.add_image(
                    levels,
                    name=nm,
                    multiscale=(len(levels) > 1),
                    contrast_limits=clims,
                    blending="additive",
                    opacity=0.6,
                    gamma=0.3,
                    colormap=w._colormap_for_channel(i),
                    cache=True,
                )
            try:
                w._connect_layer_dirty(w.viewer.layers[nm])
            except Exception:
                pass
            return True

        if plane is None:
            if is_tiff_loading_debug():
                print(f"[viewer ch={i}] fallback to numpy (multiscale unavailable)", flush=True)
            plane = w.img[..., i]

        if nm in w.viewer.layers:
            layer = w.viewer.layers[nm]
            layer.data = plane
            layer.visible = True
        else:
            w.viewer.add_image(
                plane,
                name=nm,
                blending="additive",
                opacity=0.6,
                gamma=0.3,
                colormap=w._colormap_for_channel(i),
            )
            try:
                w._connect_layer_dirty(w.viewer.layers[nm])
            except Exception:
                pass
        return False

    # --------------------------
    # Channel selection utilities
    # --------------------------

    def set_all_channels(self, checked: bool) -> None:
        w = self.w
        for i in range(w.list_channels.count()):
            it = w.list_channels.item(i)
            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def get_selected_channels(self) -> list[int]:
        w = self.w
        idxs: list[int] = []
        for i in range(w.list_channels.count()):
            it = w.list_channels.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                idxs.append(i)
        return idxs

    def _update_apply_channels_button_state(self) -> None:
        w = self.w
        try:
            if not getattr(w, "btn_apply_channels", None):
                return
            sel = set(self.get_selected_channels())
            disp = set(getattr(w, "_displayed_channel_indices", []) or [])
            w.btn_apply_channels.setEnabled(sel != disp)
        except Exception:
            try:
                w.btn_apply_channels.setEnabled(False)
            except Exception:
                pass

    def on_channel_selection_changed(self) -> None:
        """Called when the channel checklist selection changes."""
        w = self.w
        self._update_apply_channels_button_state()
        try:
            if w.project is not None:
                w.project.mark_dirty()
                w._update_project_label()
        except Exception:
            pass

    # --------------------------
    # Channel display application
    # --------------------------

    def _remove_all_channels_layer(self) -> None:
        w = self.w
        base = getattr(w, "_all_channels_layer_name", "All Channels")
        to_remove = [
            lyr
            for lyr in list(w.viewer.layers)
            if isinstance(getattr(lyr, "name", None), str) and getattr(lyr, "name").startswith(base)
        ]
        for lyr in to_remove:
            try:
                w.viewer.layers.remove(lyr)
            except Exception:
                try:
                    w.viewer.layers.remove(getattr(lyr, "name", ""))
                except Exception:
                    pass

    def sync_displayed_channel_layers(self) -> None:
        w = self.w
        if w.img is None or w.ch_names is None:
            return

        selected = list(self.get_selected_channels())
        if not selected:
            selected = [0]
            try:
                w.list_channels.blockSignals(True)
                if w.list_channels.count() > 0:
                    w.list_channels.item(0).setCheckState(QtCore.Qt.Checked)
            finally:
                w.list_channels.blockSignals(False)

        self._apply_selected_channels_now(selected)

    def _apply_selected_channels_now(self, indices: list[int]) -> None:
        w = self.w
        if w.img is None or w.ch_names is None:
            return

        indices = sorted(set(int(i) for i in indices))
        should_display = set(indices)

        # Remove non-selected channel layers
        for i, nm in enumerate(w.ch_names):
            if nm in w.viewer.layers and i not in should_display:
                try:
                    w.viewer.layers.remove(nm)
                except Exception:
                    pass

        for i in indices:
            nm = w.ch_names[i]
            if nm not in w.viewer.layers:
                self._add_or_update_channel_layer(i)

        # Keep scribbles on top
        if w._scribbles_layer_name not in w.viewer.layers:
            if w.img is not None:
                w.layers.ensure_scribbles_layer(image_shape=w.img.shape[:2], name=w._scribbles_layer_name)
        else:
            lyr = w.viewer.layers[w._scribbles_layer_name]
            try:
                w.viewer.layers.remove(w._scribbles_layer_name)
                w.viewer.layers.append(lyr)
            except Exception:
                pass

        w._displayed_channel_indices = indices
        self._update_apply_channels_button_state()

    def _apply_selected_channels_incremental(self, indices: list[int], *, done_cb=None) -> None:
        w = self.w
        if w.img is None or w.ch_names is None:
            if done_cb:
                done_cb()
            return

        self._remove_all_channels_layer()

        indices = sorted(set(int(i) for i in indices))
        should_display = set(indices)

        to_remove = [i for i, nm in enumerate(list(w.ch_names)) if nm in w.viewer.layers and i not in should_display]
        to_add = [i for i in indices if w.ch_names[i] not in w.viewer.layers]
        total_ops = max(1, len(to_remove) + len(to_add))
        progress = {"n": 0}

        try:
            if getattr(w, "prog", None) is not None:
                w.prog.setVisible(True)
                w.prog.setRange(0, total_ops)
                w.prog.setValue(0)
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        def _bump_progress():
            progress["n"] += 1
            try:
                if getattr(w, "prog", None) is not None and w.prog.isVisible():
                    w.prog.setValue(min(progress["n"], total_ops))
                    QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        # Remove any per-channel layers no longer selected.
        for i in to_remove:
            nm = w.ch_names[i]
            try:
                if nm in w.viewer.layers:
                    w.viewer.layers.remove(nm)
            except Exception:
                try:
                    w.viewer.layers.remove(str(nm))
                except Exception:
                    pass
            _bump_progress()

        def _finish():
            # Keep scribbles on top
            if w._scribbles_layer_name not in w.viewer.layers:
                if w.img is not None:
                    w.layers.ensure_scribbles_layer(image_shape=w.img.shape[:2], name=w._scribbles_layer_name)
            else:
                try:
                    lyr = w.viewer.layers[w._scribbles_layer_name]
                    w.viewer.layers.remove(w._scribbles_layer_name)
                    w.viewer.layers.append(lyr)
                except Exception:
                    pass

            w._displayed_channel_indices = indices
            self._update_apply_channels_button_state()
            if done_cb:
                done_cb()

        if not to_add:
            _finish()
            return

        # Fast path for pyramidal files: dask array creation is near-instant so
        # we can add all new channels synchronously without a background worker.
        channel_generation = self.tile_runtime.next_channel_generation()
        _img_path = getattr(w.img, "path", None) if w.img else None
        _probe = self._load_multiscale_levels(_img_path, 0)
        _multiscale_ok = bool(_probe is not None)
        if is_tiff_loading_debug():
            print(f"[viewer incremental] want_multiscale={self._want_multiscale()} ok={_multiscale_ok}, adding channels={to_add}", flush=True)

        if _multiscale_ok and _img_path:
            for i in to_add:
                if not self.tile_runtime.is_current_channel(channel_generation):
                    return
                _levels = self._load_multiscale_levels(_img_path, i)
                if _levels is None:
                    _multiscale_ok = False
                    break
                try:
                    self._add_or_update_channel_layer(i, levels=_levels)
                except Exception:
                    _multiscale_ok = False
                    break
                _bump_progress()
            if _multiscale_ok:
                _finish()
                return
            # If something went wrong mid-loop, fall through to worker-based path

        @thread_worker
        def _load_planes_worker(path, load_indices):
            for idx in load_indices:
                plane = np.asarray(load_single_channel_tiff_native(path, int(idx)))
                yield (int(idx), plane)

        worker = _load_planes_worker(w.path, to_add)
        w._channel_update_worker = worker

        @worker.yielded.connect
        def _on_plane_loaded(result):
            if not self.tile_runtime.is_current_channel(channel_generation):
                return
            i, plane = result
            if i not in should_display:
                return
            try:
                self._add_or_update_channel_layer(i, plane=plane)
                _bump_progress()
            except Exception:
                raise

        @worker.returned.connect
        def _on_done(_):
            if not self.tile_runtime.is_current_channel(channel_generation):
                return
            _finish()

        @worker.errored.connect
        def _on_err(e):
            if not self.tile_runtime.is_current_channel(channel_generation):
                return
            show_warning(f"Update displayed channels failed: {e}")
            try:
                w.set_status(f"Update displayed channels failed: {e}")
            except Exception:
                pass
            _finish()

        worker.start()

    def on_apply_channel_selection(self) -> None:
        w = self.w
        if w.img is None or w.ch_names is None:
            return

        target = self.get_selected_channels()
        if not target:
            target = [0]

        w.set_status("Updating displayed channels…")
        w.prog.setVisible(True)
        w.prog.setRange(0, 0)  # indeterminate/busy
        w.prog.setValue(0)
        try:
            w.btn_apply_channels.setEnabled(False)
        except Exception:
            pass

        def _done():
            try:
                w._mark_project_dirty()
                w.set_status("Displayed channels updated.")
            finally:
                w.prog.setRange(0, 1)
                w.prog.setValue(1)
                try:
                    self._update_apply_channels_button_state()
                except Exception:
                    pass
                try:
                    w.btn_apply_channels.setEnabled(True)
                except Exception:
                    pass

        self._apply_selected_channels_incremental(target, done_cb=_done)

    # --------------------------
    # Image loading
    # --------------------------

    def _load_image_from_path(self, path: str, *, record_input: bool = True) -> None:
        w = self.w
        if not path:
            return

        load_generation = self.tile_runtime.next_load_generation()
        want_multiscale = self._want_multiscale()
        if want_multiscale:
            self.tile_runtime.configure_for_multiscale()

        w.path = path
        w.lbl_file.setText(path)

        if record_input and w.project is not None:
            try:
                w.project.add_input(Path(path))
                w._update_project_label()
            except Exception:
                pass

        w.set_status("Loading image…")
        w.prog.setVisible(True)
        w.prog.setRange(0, 3)
        w.prog.setValue(0)
        try:
            w.btn_load.setEnabled(False)
        except Exception:
            pass

        @thread_worker
        def _load_worker(p, use_multiscale):
            info = inspect_tiff_yxc(p)
            yield ("metadata", info)
            used_lazy_pyramid, first_plane = self._load_initial_display_probe(
                p,
                want_multiscale=bool(use_multiscale),
            )
            if used_lazy_pyramid:
                yield ("lazy_pyramid", None)
                return {"info": info, "used_lazy_pyramid": True, "first_plane": None}
            yield ("first_plane", first_plane)
            return {"info": info, "used_lazy_pyramid": False, "first_plane": first_plane}

        worker = _load_worker(path, want_multiscale)
        state = {"info": None, "first_plane": None, "used_lazy_pyramid": False}
        w._image_load_worker = worker

        @worker.yielded.connect
        def _on_progress(payload):
            if not self.tile_runtime.is_current_load(load_generation):
                return
            kind, value = payload
            if kind == "metadata":
                state["info"] = value
                try:
                    w.prog.setValue(1)
                except Exception:
                    pass
            elif kind == "first_plane":
                state["first_plane"] = value
                try:
                    w.prog.setValue(2)
                except Exception:
                    pass
            elif kind == "lazy_pyramid":
                state["used_lazy_pyramid"] = True
                try:
                    w.prog.setValue(2)
                except Exception:
                    pass

        @worker.returned.connect
        def _on_loaded(result):
            if not self.tile_runtime.is_current_load(load_generation):
                return
            info = dict((result or {}).get("info") or state.get("info") or {})
            used_lazy_pyramid = bool((result or {}).get("used_lazy_pyramid") or state.get("used_lazy_pyramid"))
            ch_names = list(info.get("channel_names") or [])
            if not ch_names:
                ch_names = ["Channel 0"]

            w.img = LazyChannelImage(path)
            if not used_lazy_pyramid:
                try:
                    first_plane = (result or {}).get("first_plane")
                    if first_plane is None:
                        first_plane = state.get("first_plane")
                    if first_plane is not None:
                        w.img._root._cache[0] = np.asarray(first_plane)
                except Exception:
                    pass
            w.ch_names = ch_names

            # Populate channel list
            w.list_channels.blockSignals(True)
            w.list_channels.clear()
            for i, nm in enumerate(w.ch_names):
                it = QtWidgets.QListWidgetItem(nm)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if i == 0 else QtCore.Qt.Unchecked)
                w.list_channels.addItem(it)
            w.list_channels.blockSignals(False)

            # Reset viewer layers for the new image
            w._prob_layers = {}
            try:
                w.viewer.layers.clear()
            except Exception:
                pass

            self.sync_displayed_channel_layers()

            # Apply pending restore if requested
            if getattr(w, "_pending_restore", None):
                try:
                    w._apply_project_restore(w._pending_restore)
                finally:
                    w._pending_restore = None

            shp = tuple(int(v) for v in getattr(w.img, 'shape', (0, 0, 0)))
            w.set_status(f"Loaded image {shp} with {len(w.ch_names)} channels.")

            w.prog.setValue(3)
            try:
                w.btn_load.setEnabled(True)
            except Exception:
                pass

        @worker.errored.connect
        def _on_load_err(e):
            if not self.tile_runtime.is_current_load(load_generation):
                return
            show_warning(f"Load failed: {e}")
            w.set_status(f"Load failed: {e}")
            w.prog.setRange(0, 1)
            w.prog.setValue(0)
            try:
                w.btn_load.setEnabled(True)
            except Exception:
                pass

        worker.start()

    def on_load(self) -> None:
        w = self.w
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            w,
            "Open multi-channel TIFF/OME-TIFF",
            w._dialog_start_dir(),
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self._load_image_from_path(path, record_input=True)
