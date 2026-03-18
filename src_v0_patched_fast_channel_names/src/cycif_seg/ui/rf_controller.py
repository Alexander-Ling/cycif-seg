from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from qtpy import QtCore
from napari.utils.notifications import show_info, show_warning


class RFController:
    """
    Controller for Step 2a RF pixel model (model A) train+predict and probability layers.

    This extracts the large "train + tiled prediction + UI updates" logic out of main_widget.py.
    """

    def __init__(self, widget):
        # widget is CycIFMVPWidget (kept loose to avoid circular import)
        self.w = widget

        self._rf_worker = None
        self._rf_run_id = 0

        # Throttled refresh for large probability layers (avoid per-tile repaint cost)
        self._prob_dirty = False
        self._prob_refresh_timer = QtCore.QTimer(self.w)
        self._prob_refresh_timer.setInterval(100)  # ms (10 Hz)
        self._prob_refresh_timer.timeout.connect(self._refresh_prob_layers_if_dirty)

        # Keep these on the widget too for backward-compat with other code paths.
        if not hasattr(self.w, "_prob_layers"):
            self.w._prob_layers = {}
        if not hasattr(self.w, "_P"):
            self.w._P = None

    # ------------------------------------------------------------------
    # Public hooks from UI
    # ------------------------------------------------------------------

    def on_alpha_change(self, v: int):
        a = float(v) / 100.0
        for lyr in getattr(self.w, "_prob_layers", {}).values():
            try:
                lyr.opacity = a
            except Exception:
                pass

    def cancel_worker(self):
        w = getattr(self, "_rf_worker", None)
        if w is None:
            return
        try:
            if hasattr(w, "quit"):
                w.quit()
            if hasattr(w, "cancel"):
                w.cancel()
        except Exception:
            pass
        self._rf_worker = None
        try:
            self._prob_refresh_timer.stop()
            self.w.prog.setVisible(False)
        except Exception:
            pass

    def request_cancel(self):
        """Cooperatively cancel the current RF job.

        We bump the run id so any late yields are ignored, then ask the worker to stop.
        """
        try:
            self._rf_run_id += 1
        except Exception:
            pass
        self.cancel_worker()

    def on_train_predict(self):
        """
        Train the RF pixel classifier from scribbles and predict tiled probabilities.
        Populates/updates 3 napari Image layers:
          - P(nucleus)
          - P(nuc_boundary)
          - P(background)
        """
        self.cancel_worker()
        self._rf_run_id += 1
        my_run_id = self._rf_run_id

        def is_cancelled() -> bool:
            return my_run_id != self._rf_run_id

        if self.w.img is None:
            show_warning("Load an image first.")
            return

        use_ch = self.w.get_selected_channels()
        if len(use_ch) == 0:
            show_warning("Select at least one channel.")
            return

        img_for_rf = self.w.img.subset(use_ch) if hasattr(self.w.img, "subset") else self.w.img
        use_ch_local = list(range(len(use_ch)))

        scrib_layer = self.w.layers.ensure_scribbles_layer(
            image_shape=self.w.img.shape[:2],
            name=self.w._scribbles_layer_name,
        )
        S = np.asarray(scrib_layer.data)

        train_mask = S > 0
        n_train = int(train_mask.sum())
        if n_train < 2000:
            show_warning(f"Not enough labeled pixels ({n_train}). Paint more scribbles.")
            return

        H, W, _ = img_for_rf.shape

        # Allocate probability volume
        self.w._P = np.zeros((H, W, 3), dtype=np.float32)

        alpha = float(self.w.slider_alpha.value()) / 100.0

        # Create probability layers if missing
        viewer = self.w.viewer
        if "P(nucleus)" not in viewer.layers:
            self.w._prob_layers["P_nuc"] = viewer.add_image(
                self.w._P[..., 0],
                name="P(nucleus)",
                opacity=alpha,
                blending="additive",
                colormap="magenta",
            )
        if "P(nuc_boundary)" not in viewer.layers:
            self.w._prob_layers["P_nb"] = viewer.add_image(
                self.w._P[..., 1],
                name="P(nuc_boundary)",
                opacity=alpha,
                blending="additive",
                colormap="yellow",
            )
        if "P(background)" not in viewer.layers:
            self.w._prob_layers["P_bg"] = viewer.add_image(
                self.w._P[..., 2],
                name="P(background)",
                opacity=alpha,
                blending="additive",
                colormap="gray",
            )

        # Ensure layers point at the current _P buffer
        name_to_idx = {"P(nucleus)": 0, "P(nuc_boundary)": 1, "P(background)": 2}
        for nm, ii in name_to_idx.items():
            if nm in viewer.layers:
                try:
                    viewer.layers[nm].data = self.w._P[..., ii]
                    viewer.layers[nm].refresh()
                except Exception:
                    pass

        from cycif_seg.predict.workers import predict_rf_worker
        from cycif_seg.features.multiscale import build_features
        from cycif_seg.features.zarr_tile_cache import ZarrTileFeatureCache, FeatureCacheConfig

        self.w.set_status("Training + predicting (RF)…")
        self.w.prog.setVisible(True)
        self.w.prog.setRange(0, 0)  # indeterminate during training
        self.w.prog.setValue(0)

        # Choose a priority point near what the user is looking at
        try:
            cy, cx = viewer.camera.center[-2], viewer.camera.center[-1]
        except Exception:
            cy, cx = float(viewer.dims.point[0]), float(viewer.dims.point[1])

        center_yx = (float(cy), float(cx))

        # Optional on-disk feature cache (Zarr) to avoid recomputing features on every run.
        # This caches *tile-local* feature maps per channel and reuses them across retrains.
        get_tile_features = None
        get_point_features = None
        cache_for_stats = None
        try:
            prj = getattr(self.w, "project", None)
            if prj is not None and ZarrTileFeatureCache.available():
                # Cache path: <project>/features/step2a_rf/<image_fp>/<cfg_hash>/
                img_shape = tuple(int(x) for x in img_for_rf.shape)
                img_fp = ZarrTileFeatureCache.compute_image_fingerprint(getattr(self.w, "path", None), img_shape)
                cfg = FeatureCacheConfig()  # current hard-coded feature set
                # Normalize intensities to [0,1] before feature computation so features can be cached as float16 safely.
                cfg.norm_mode = "p0.5_p99.5"
                cache_dir = prj.root / "features" / "step2a_rf" / img_fp / cfg.hash()
                cache = ZarrTileFeatureCache(
                    cache_dir,
                    image_fingerprint=img_fp,
                    image_shape_yxc=img_shape,
                    tile_size=512,
                    cfg=cfg,
                )
                cache_for_stats = cache
                img_for_stats = img_for_rf

                # Pre-initialize per-channel stores serially to avoid concurrent
                # metadata writes from tile worker threads (important on Windows).
                cache.prepare_channels(use_ch_local)

                def _get_tile_features(y0, y1, x0, x1, use_channels, tile_img):
                    return cache.get_tile_features(y0, y1, x0, x1, list(use_channels), tile_img, img=img_for_stats)

                get_tile_features = _get_tile_features

                def _get_point_features(ys, xs, use_channels, img):
                    return cache.get_point_features(ys, xs, list(use_channels), img)

                get_point_features = _get_point_features

                # Record cache info in manifest for transparency (best-effort).
                try:
                    prj.manifest.setdefault("step2a", {})
                    prj.manifest["step2a"].setdefault("feature_cache", {})
                    prj.manifest["step2a"]["feature_cache"].update(
                        {
                            "kind": "zarr_tile_cache",
                            "image_fingerprint": img_fp,
                            "tile_size": 512,
                            "cfg": cfg.to_dict(),
                            "cfg_hash": cfg.hash(),
                            "root": prj.relpath(cache_dir),
                        }
                    )
                    prj.mark_dirty()
                    self.w._update_project_label()
                except Exception:
                    pass

        except Exception as e:
            try:
                print(f"[FeatureCache] WARNING: disabled (init failed): {e}")
            except Exception:
                pass
            get_tile_features = None
            get_point_features = None
            cache_for_stats = None


        worker = predict_rf_worker(
            img_for_rf,
            use_ch_local,
            S,
            build_features,
            tile_size=512,
            center_yx=center_yx,
            run_id=my_run_id,
            is_cancelled=is_cancelled,
            get_tile_features=get_tile_features,
            get_point_features=get_point_features,
            progress_every=2,
            batch_tiles=4,
            feature_workers=3,
            prefetch_tiles=16,
            rf_n_jobs=12,
        )
        self._rf_worker = worker

        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Prediction failed: {e}")
            self.w.status.setText(f"Prediction failed: {e}")
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            self.w.prog.setVisible(False)
            try:
                self.w.btn_train.setEnabled(True)
                if hasattr(self.w, "btn_stop_train"):
                    self.w.btn_stop_train.setEnabled(False)
            except Exception:
                pass

        @worker.yielded.connect
        def _on_tile(result):
            kind = result[0]

            if kind == "status":
                _, run_id, msg = result
                if run_id != self._rf_run_id:
                    return
                self.w.status.setText(str(msg))
                return

            if kind == "trained_model":
                _, run_id, model, meta = result
                if run_id != self._rf_run_id:
                    return
                meta = dict(meta or {})
                meta.setdefault("kind", "rf_pixel_model_A")
                meta["use_channels"] = list(use_ch)
                meta["use_channels_local"] = list(use_ch_local)
                self.w._pending_models.append({"model": model, "meta": meta, "saved_path": None})
                self.w._mark_project_dirty()
                return

            if kind == "progress_init":
                _, run_id, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                self.w.prog.setRange(0, int(n_tiles))
                self.w.prog.setValue(0)
                try:
                    self._prob_refresh_timer.start()
                except Exception:
                    pass
                return

            if kind == "progress":
                _, run_id, i, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                self.w.prog.setValue(int(i))
                return

            if kind == "tile":
                _, run_id, y0, y1, x0, x1, P_tile = result
                if run_id != self._rf_run_id:
                    return
                self.w._P[y0:y1, x0:x1, :] = P_tile
                self._prob_dirty = True
                return

            return

        @worker.finished.connect
        def _on_done():
            if my_run_id != self._rf_run_id:
                return
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            self._prob_dirty = True
            self._refresh_prob_layers_if_dirty()
            try:
                self.w.prog.setValue(self.w.prog.maximum())
            except Exception:
                pass
            self.w.prog.setVisible(False)
            self.w.set_status("Prediction complete.")
            try:
                if cache_for_stats is not None:
                    print(cache_for_stats.stats_summary())
            except Exception:
                pass
            try:
                self.w.btn_train.setEnabled(True)
                if hasattr(self.w, "btn_stop_train"):
                    self.w.btn_stop_train.setEnabled(False)
            except Exception:
                pass

        worker.start()

    def propagate_nuclei_edits(
        self,
        base_labels: np.ndarray,
        edited_labels: np.ndarray,
        use_channels: Sequence[int],
        *,
        tile_size: int,
        center_yx: tuple[float, float],
        batch_tiles: int = 4,
        feature_workers: int = 3,
        progress_every: int = 2,
    ) -> None:
        """
        Run RF propagation worker to update probabilities from edited nuclei labels.

        This is used by Step 2b "Propagate edits" (model B/C training later).
        """
        # Cancel any in-flight RF work
        self.cancel_worker()
        self._rf_run_id += 1
        run_id = self._rf_run_id

        if self.w.img is None:
            show_warning("Load an image first.")
            return

        H, W = self.w.img.shape[:2]
        if self.w._P is None or self.w._P.shape[:2] != (H, W) or self.w._P.shape[2] != 3:
            self.w._P = np.zeros((H, W, 3), dtype=np.float32)
        else:
            self.w._P[...] = 0.0

        # Ensure prob layers exist and point at current buffer
        viewer = self.w.viewer
        alpha = float(self.w.slider_alpha.value()) / 100.0
        if "P(nucleus)" not in viewer.layers:
            self.w._prob_layers["P_nuc"] = viewer.add_image(
                self.w._P[..., 0],
                name="P(nucleus)",
                opacity=alpha,
                blending="additive",
                colormap="magenta",
            )
        if "P(nuc_boundary)" not in viewer.layers:
            self.w._prob_layers["P_nb"] = viewer.add_image(
                self.w._P[..., 1],
                name="P(nuc_boundary)",
                opacity=alpha,
                blending="additive",
                colormap="yellow",
            )
        if "P(background)" not in viewer.layers:
            self.w._prob_layers["P_bg"] = viewer.add_image(
                self.w._P[..., 2],
                name="P(background)",
                opacity=alpha,
                blending="additive",
                colormap="gray",
            )
        name_to_idx = {"P(nucleus)": 0, "P(nuc_boundary)": 1, "P(background)": 2}
        for nm, ii in name_to_idx.items():
            if nm in viewer.layers:
                try:
                    viewer.layers[nm].data = self.w._P[..., ii]
                    viewer.layers[nm].refresh()
                except Exception:
                    pass

        from cycif_seg.predict.workers import propagate_nuclei_edits_worker
        from cycif_seg.features.multiscale import build_features

        self.w.set_status("Propagating nuclei edits (RF)…")
        self.w.prog.setVisible(True)
        self.w.prog.setRange(0, 0)
        self.w.prog.setValue(0)

        worker = propagate_nuclei_edits_worker(
            self.w.img,
            list(use_channels),
            base_labels,
            edited_labels,
            build_features,
            tile_size,
            center_yx,
            run_id,
            batch_tiles=batch_tiles,
            feature_workers=feature_workers,
            progress_every=progress_every,
        )
        self._rf_worker = worker

        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Edit propagation failed: {e}")
            self.w.status.setText(f"Edit propagation failed: {e}")
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            self.w.prog.setVisible(False)

        @worker.yielded.connect
        def _on_msg(msg):
            if not isinstance(msg, tuple) or len(msg) == 0:
                return
            kind = msg[0]

            if kind == "stage":
                _, rid, text, step, total = msg
                if rid != self._rf_run_id:
                    return
                self.w.status.setText(str(text))
                return

            if kind == "tile":
                _, rid, y0, y1, x0, x1, P_tile = msg
                if rid != self._rf_run_id:
                    return
                self.w._P[y0:y1, x0:x1, : P_tile.shape[2]] = P_tile
                self._prob_dirty = True
                return

            if kind == "progress":
                _, rid, done, total = msg
                if rid != self._rf_run_id:
                    return
                self.w.status.setText(f"Propagating edits… tiles {done}/{total}")
                return

            if kind == "trained_model":
                _, rid, model, meta = msg
                if rid != self._rf_run_id:
                    return
                meta = dict(meta or {})
                meta.setdefault("kind", "rf_from_nuclei_edits")
                self.w._pending_models.append({"model": model, "meta": meta, "saved_path": None})
                self.w._mark_project_dirty()
                return

        @worker.finished.connect
        def _on_done():
            if run_id != self._rf_run_id:
                return
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            self._prob_dirty = True
            self._refresh_prob_layers_if_dirty()
            self.w.prog.setRange(0, 1)
            self.w.prog.setValue(1)
            self.w.prog.setVisible(False)
            self.w.set_status("Edits propagated. You can now Generate nuclei.")

        try:
            self._prob_refresh_timer.start()
        except Exception:
            pass
        worker.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_prob_layers_if_dirty(self):
        if not self._prob_dirty:
            return
        self._prob_dirty = False
        try:
            for nm in ("P(nucleus)", "P(nuc_boundary)", "P(background)"):
                if nm in self.w.viewer.layers:
                    self.w.viewer.layers[nm].refresh()
        except Exception:
            pass