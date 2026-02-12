from __future__ import annotations

import os
import numpy as np
from qtpy import QtWidgets, QtCore

import napari
from napari.utils.notifications import show_info, show_warning

from cycif_seg.io.ome_tiff import load_multichannel_tiff
from cycif_seg.features.multiscale import build_features
from cycif_seg.model.rf_pixel import train_rf, predict_proba_tiled


class CycIFMVPWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.img = None            # (Y,X,C) float32
        self.ch_names = None
        self.path = None

        self._rf_worker = None
        self._rf_run_id = 0

        self._prob_layers = {}
        self._scribbles_layer_name = "Scribbles (0=unlabeled,1=nuc,2=cyto,3=bg)"

        layout = QtWidgets.QVBoxLayout(self)

        # File load
        file_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load TIFF/OME-TIFF…")
        self.lbl_file = QtWidgets.QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        file_row.addWidget(self.btn_load)
        file_row.addWidget(self.lbl_file, 1)
        layout.addLayout(file_row)

        # Channel checklist header + filter
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Channels used for RF training:"))
        self.chk_display_selected_only = QtWidgets.QCheckBox("Display selected channels only")
        header_row.addWidget(self.chk_display_selected_only)
        layout.addLayout(header_row)

        self.list_channels = QtWidgets.QListWidget()
        self.list_channels.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.list_channels, 1)

        ch_btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        ch_btn_row.addWidget(self.btn_all)
        ch_btn_row.addWidget(self.btn_none)
        layout.addLayout(ch_btn_row)

        layout.addWidget(QtWidgets.QLabel(
            "Scribbles: paint into the Labels layer using values:\n"
            "1 = nucleus, 2 = cytoplasm, 3 = background (0 = unlabeled)\n"
            "Tip: select the Scribbles layer and set the current label value in napari."
        ))

        # Training controls
        self.btn_train = QtWidgets.QPushButton("Train + Predict (RF)")
        layout.addWidget(self.btn_train)

        # --- Cell segmentation controls ---
        layout.addWidget(QtWidgets.QLabel("Cell segmentation (nucleus-seeded):"))

        grid = QtWidgets.QGridLayout()

        self.spin_nuc_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_nuc_thresh.setRange(0.0, 1.0)
        self.spin_nuc_thresh.setSingleStep(0.05)
        self.spin_nuc_thresh.setValue(0.35)

        self.spin_cyto_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_cyto_thresh.setRange(0.0, 1.0)
        self.spin_cyto_thresh.setSingleStep(0.05)
        self.spin_cyto_thresh.setValue(0.35)

        self.spin_min_nuc_area = QtWidgets.QSpinBox()
        self.spin_min_nuc_area.setRange(0, 10_000)
        self.spin_min_nuc_area.setValue(30)

        self.spin_min_cell_area = QtWidgets.QSpinBox()
        self.spin_min_cell_area.setRange(0, 100_000)
        self.spin_min_cell_area.setValue(200)

        grid.addWidget(QtWidgets.QLabel("Nuc thresh"), 0, 0)
        grid.addWidget(self.spin_nuc_thresh, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Cyto thresh"), 0, 2)
        grid.addWidget(self.spin_cyto_thresh, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Min nucleus area"), 1, 0)
        grid.addWidget(self.spin_min_nuc_area, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Min cell area"), 1, 2)
        grid.addWidget(self.spin_min_cell_area, 1, 3)

        layout.addLayout(grid)

        self.btn_cells = QtWidgets.QPushButton("Generate Cells (watershed)")
        layout.addWidget(self.btn_cells)

        self.btn_cells.clicked.connect(self.on_generate_cells)


        # Opacity
        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(QtWidgets.QLabel("Overlay opacity:"))
        self.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(100)
        self.slider_alpha.setValue(40)
        op_row.addWidget(self.slider_alpha, 1)
        layout.addLayout(op_row)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.prog = QtWidgets.QProgressBar()
        self.prog.setVisible(False)
        layout.addWidget(self.prog)

        # Signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_all.clicked.connect(lambda: self.set_all_channels(True))
        self.btn_none.clicked.connect(lambda: self.set_all_channels(False))
        self.btn_train.clicked.connect(self.on_train_predict)
        self.slider_alpha.valueChanged.connect(self.on_alpha_change)
        self.chk_display_selected_only.stateChanged.connect(self.sync_displayed_channel_layers)
        self.list_channels.itemChanged.connect(lambda _: self.on_channel_selection_changed())

    def _cancel_rf_worker(self):
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

    def set_status(self, msg: str):
        self.status.setText(msg)
        show_info(msg)

    def on_alpha_change(self, v):
        a = float(v) / 100.0
        for lyr in self._prob_layers.values():
            lyr.opacity = a

    def _channel_layer_names(self) -> set[str]:
        return set(self.ch_names or [])

    def sync_displayed_channel_layers(self):
        """
        If 'Display selected channels only' is checked, remove non-selected channel layers
        from the napari layer list. If unchecked, ensure all channel layers are present.
        Scribbles and probability layers are preserved.
        """
        if self.img is None or self.ch_names is None:
            return

        selected = set(self.get_selected_channels())
        display_selected_only = self.chk_display_selected_only.isChecked()

        # Determine which channel indices should be displayed
        if display_selected_only:
            should_display = {i for i in selected}
        else:
            should_display = set(range(len(self.ch_names)))

        # Remove channel layers that should NOT be displayed
        for i, nm in enumerate(self.ch_names):
            if nm in self.viewer.layers and i not in should_display:
                self.viewer.layers.remove(nm)

        # Add missing channel layers that SHOULD be displayed
        for i in sorted(should_display):
            nm = self.ch_names[i]
            if nm not in self.viewer.layers:
                self.viewer.add_image(
                    self.img[..., i],
                    name=nm,
                    blending="additive",
                    opacity=0.6,
                    colormap=self._colormap_for_channel(i),
                )

        # Keep scribbles on top (re-add if needed)
        if self._scribbles_layer_name not in self.viewer.layers:
            self.ensure_scribbles_layer()
        else:
            # Move to top for painting convenience
            lyr = self.viewer.layers[self._scribbles_layer_name]
            self.viewer.layers.remove(self._scribbles_layer_name)
            self.viewer.layers.append(lyr)

    def on_channel_selection_changed(self):
        """
        Called whenever an item in the channel checklist changes.
        If display-selected-only mode is enabled, update viewer layers immediately.
        """
        if getattr(self, "chk_display_selected_only", None) and self.chk_display_selected_only.isChecked():
            self.sync_displayed_channel_layers()


    def _colormap_for_channel(self, i: int) -> str:
        palette = ["blue", "green", "red", "magenta", "cyan", "yellow"]
        return palette[i % len(palette)]

    def set_all_channels(self, checked: bool):
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def get_selected_channels(self) -> list[int]:
        idxs = []
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                idxs.append(i)
        return idxs

    def ensure_scribbles_layer(self):
        if self._scribbles_layer_name in self.viewer.layers:
            return self.viewer.layers[self._scribbles_layer_name]

        if self.img is None:
            raise RuntimeError("Load an image first.")

        H, W, _ = self.img.shape
        scrib = np.zeros((H, W), dtype=np.uint8)
        layer = self.viewer.add_labels(
            scrib,
            name=self._scribbles_layer_name,
            opacity=0.6,
        )
        return layer

    def _add_channel_layers(self):
        """
        Add one napari Image layer per channel, named using ch_names.
        """
        assert self.img is not None and self.ch_names is not None

        # Clear viewer
        self.viewer.layers.clear()

        # Add per-channel layers
        for i, nm in enumerate(self.ch_names):
            self.viewer.add_image(
                self.img[..., i],
                name=nm,
                blending="additive",
                opacity=0.6,
                colormap=self._colormap_for_channel(i),
            )

        # Add scribbles last so it paints over everything
        self.ensure_scribbles_layer()

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open multi-channel TIFF/OME-TIFF",
            os.getcwd(),
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        self.path = path
        self.lbl_file.setText(path)

        self.img, self.ch_names = load_multichannel_tiff(path)

        # Populate channel list with correct names
        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        for nm in self.ch_names:
            it = QtWidgets.QListWidgetItem(nm)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)  # default select all
            self.list_channels.addItem(it)
        self.list_channels.blockSignals(False)

        # Add named layers
        self._prob_layers = {}
        self._add_channel_layers()

        self.sync_displayed_channel_layers()

        self.set_status(f"Loaded image {self.img.shape} with {len(self.ch_names)} channels.")

    def on_train_predict(self):

        self._cancel_rf_worker()
        self._rf_run_id += 1
        my_run_id = self._rf_run_id

        def is_cancelled():
            return my_run_id != self._rf_run_id

        if self.img is None:
            show_warning("Load an image first.")
            return

        use_ch = self.get_selected_channels()
        if len(use_ch) == 0:
            show_warning("Select at least one channel.")
            return

        scrib_layer = self.ensure_scribbles_layer()
        S = np.asarray(scrib_layer.data)

        train_mask = S > 0
        n_train = int(train_mask.sum())
        if n_train < 2000:
            show_warning(f"Not enough labeled pixels ({n_train}). Paint more scribbles.")
            return

        H, W, _ = self.img.shape

        # Allocate probability volume
        self._P = np.zeros((H, W, 3), dtype=np.float32)

        alpha = float(self.slider_alpha.value()) / 100.0

        # Create probability layers if missing
        if "P(nucleus)" not in self.viewer.layers:
            self._prob_layers["P_nuc"] = self.viewer.add_image(
                self._P[..., 0],
                name="P(nucleus)",
                opacity=alpha,
                blending="additive",
                colormap="magenta",
            )
        if "P(cytoplasm)" not in self.viewer.layers:
            self._prob_layers["P_cyto"] = self.viewer.add_image(
                self._P[..., 1],
                name="P(cytoplasm)",
                opacity=alpha,
                blending="additive",
                colormap="green",
            )

        from cycif_seg.predict.workers import predict_rf_worker
        from cycif_seg.features.multiscale import build_features

        self.set_status("Training + predicting (background)...")
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)  # indeterminate during training
        self.prog.setValue(0)

        # Choose a priority point near what the user is looking at
        try:
            # camera.center is usually (z, y, x) in 2D display; take last 2
            cy, cx = self.viewer.camera.center[-2], self.viewer.camera.center[-1]
        except Exception:
            # fall back to dims point (y, x)
            cy, cx = float(self.viewer.dims.point[0]), float(self.viewer.dims.point[1])

        center_yx = (float(cy), float(cx))

        worker = predict_rf_worker(
            self.img,
            use_ch,
            S,
            build_features,
            tile_size=512,
            center_yx=center_yx,
            run_id=my_run_id,
            is_cancelled=is_cancelled,
        )
        self._rf_worker = worker

        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Prediction failed: {e}")
            self.status.setText(f"Prediction failed: {e}")
            self.prog.setVisible(False)

        @worker.yielded.connect
        def _on_tile(result):
            kind = result[0]

            if kind == "status":
                _, run_id, msg = result
                if run_id != self._rf_run_id:
                    return
                self.status.setText(str(msg))
                return

            if kind == "progress_init":
                _, run_id, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                self.prog.setRange(0, int(n_tiles))
                self.prog.setValue(0)
                return

            if kind == "progress":
                _, run_id, i, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                # range already set; update value
                self.prog.setValue(int(i))
                return

            if kind == "tile":
                _, run_id, y0, y1, x0, x1, P_tile = result
                if run_id != self._rf_run_id:
                    return
                # write probabilities
                self._P[y0:y1, x0:x1, :] = P_tile
                # refresh views
                if "P(nucleus)" in self.viewer.layers:
                    self.viewer.layers["P(nucleus)"].refresh()
                if "P(cytoplasm)" in self.viewer.layers:
                    self.viewer.layers["P(cytoplasm)"].refresh()
                return
            # Unknown message type; ignore safely.
            return            

        @worker.finished.connect
        def _on_done():
            if my_run_id != self._rf_run_id:
                # stale/cancelled run
                return
            self.prog.setValue(self.prog.maximum())
            self.prog.setVisible(False)
            self.set_status("Prediction complete.")

        worker.start()

    def on_generate_cells(self):
        if "P(nucleus)" not in self.viewer.layers or "P(cytoplasm)" not in self.viewer.layers:
            show_warning("Run Train + Predict first to create P(nucleus) and P(cytoplasm).")
            return

        p_nuc = np.asarray(self.viewer.layers["P(nucleus)"].data).astype(np.float32, copy=False)
        p_cyto = np.asarray(self.viewer.layers["P(cytoplasm)"].data).astype(np.float32, copy=False)

        params = {
            "nuc_thresh": float(self.spin_nuc_thresh.value()),
            "cyto_thresh": float(self.spin_cyto_thresh.value()),
            "min_nucleus_area": int(self.spin_min_nuc_area.value()),
            "min_cell_area": int(self.spin_min_cell_area.value()),
            # conservative defaults; we can expose later if needed
            "peak_min_distance": 6,
            "peak_footprint": 9,
        }

        self.set_status("Generating cells (background)…")

        from cycif_seg.instance.workers import cells_from_probs_worker

        worker = cells_from_probs_worker(p_nuc, p_cyto, params)

        @worker.returned.connect
        def _on_returned(result):
            labels, debug = result

            # Add / update Cells layer
            if "Cells" in self.viewer.layers:
                self.viewer.layers["Cells"].data = labels
            else:
                self.viewer.add_labels(labels, name="Cells", opacity=0.6)

            self.set_status(f"Cells generated: N={int(labels.max())} (0=unassigned cytoplasm).")

            # (Optional) Keep debug around for next milestone (split/local resegment)
            self._cells_debug = debug

        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Cell generation failed: {e}")
            self.status.setText(f"Cell generation failed: {e}")

        worker.start()

