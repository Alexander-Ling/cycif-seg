from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from qtpy import QtCore, QtWidgets
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_warning, show_info

from cycif_seg.io.ome_tiff import load_multichannel_tiff, load_multichannel_tiff_lazy


class ImageController:
    """Handles image loading and channel display management.

    This controller intentionally operates on the owning main widget (``w``)
    to minimize invasive state rewrites during refactors.
    """

    def __init__(self, w):
        self.w = w

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

        # Add missing selected layers
        for i in indices:
            nm = w.ch_names[i]
            if nm not in w.viewer.layers:
                w.viewer.add_image(
                    w.img[..., i],
                    name=nm,
                    blending="additive",
                    opacity=0.6,
                    colormap=w._colormap_for_channel(i),
                )
                try:
                    w._connect_layer_dirty(w.viewer.layers[nm])
                except Exception:
                    pass

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

        # Remove any per-channel layers no longer selected
        for i, nm in enumerate(list(w.ch_names)):
            if nm in w.viewer.layers and i not in should_display:
                try:
                    w.viewer.layers.remove(nm)
                except Exception:
                    try:
                        w.viewer.layers[nm].visible = False
                    except Exception:
                        pass

        # Toggle visibility for existing layers
        for i, nm in enumerate(list(w.ch_names)):
            if nm in w.viewer.layers:
                try:
                    w.viewer.layers[nm].visible = (i in should_display)
                except Exception:
                    pass

        to_add = [i for i in indices if w.ch_names[i] not in w.viewer.layers]

        # Configure determinate progress for layer creation only.
        try:
            if getattr(w, "prog", None) is not None:
                w.prog.setVisible(True)
                w.prog.setRange(0, max(1, len(to_add)))
                w.prog.setValue(0)
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass

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

        def _add_next(k: int):
            if k >= len(to_add):
                _finish()
                return

            i = to_add[k]
            nm = w.ch_names[i]

            # Update progress before add_image so the user sees movement.
            try:
                if getattr(w, "prog", None) is not None and w.prog.isVisible():
                    w.prog.setValue(min(k, w.prog.maximum()))
                    QtWidgets.QApplication.processEvents()
            except Exception:
                pass

            try:
                w.viewer.add_image(
                    w.img[..., i],
                    name=nm,
                    blending="additive",
                    opacity=0.6,
                    colormap=w._colormap_for_channel(i),
                )
                try:
                    w._connect_layer_dirty(w.viewer.layers[nm])
                except Exception:
                    pass
                try:
                    w.viewer.layers[nm].visible = True
                except Exception:
                    pass
            finally:
                try:
                    if getattr(w, "prog", None) is not None and w.prog.isVisible():
                        w.prog.setValue(min(k + 1, w.prog.maximum()))
                        QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
                QtCore.QTimer.singleShot(0, lambda: _add_next(k + 1))

        QtCore.QTimer.singleShot(0, lambda: _add_next(0))

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
        w.prog.setRange(0, 0)  # indeterminate/busy
        w.prog.setValue(0)
        try:
            w.btn_load.setEnabled(False)
        except Exception:
            pass

        @thread_worker
        def _load_worker(p):
            try:
                img_cyx, names = load_multichannel_tiff_lazy(p)
                try:
                    import dask.array as da  # type: ignore
                    img_yxc = da.moveaxis(img_cyx, 0, 2)
                except Exception:
                    img_yxc = img_cyx
                return img_yxc, names
            except Exception:
                return load_multichannel_tiff(p)

        worker = _load_worker(path)

        @worker.returned.connect
        def _on_loaded(result):
            w.img, w.ch_names = result

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

            w.set_status(f"Loaded image {w.img.shape} with {len(w.ch_names)} channels.")

            w.prog.setRange(0, 1)
            w.prog.setValue(1)
            try:
                w.btn_load.setEnabled(True)
            except Exception:
                pass

        @worker.errored.connect
        def _on_load_err(e):
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
