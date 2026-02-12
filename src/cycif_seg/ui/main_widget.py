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
        self.chk_show_selected_only = QtWidgets.QCheckBox("Show selected only")
        header_row.addWidget(self.chk_show_selected_only)
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

        # Signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_all.clicked.connect(lambda: self.set_all_channels(True))
        self.btn_none.clicked.connect(lambda: self.set_all_channels(False))
        self.btn_train.clicked.connect(self.on_train_predict)
        self.slider_alpha.valueChanged.connect(self.on_alpha_change)
        self.chk_show_selected_only.stateChanged.connect(self.apply_channel_filter)
        self.list_channels.itemChanged.connect(lambda _: self.apply_channel_filter())

    def set_status(self, msg: str):
        self.status.setText(msg)
        show_info(msg)

    def on_alpha_change(self, v):
        a = float(v) / 100.0
        for lyr in self._prob_layers.values():
            lyr.opacity = a

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

    def apply_channel_filter(self):
        show_only = self.chk_show_selected_only.isChecked()
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            if not show_only:
                it.setHidden(False)
            else:
                it.setHidden(it.checkState() != QtCore.Qt.Checked)

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
            self.viewer.add_image(self.img[..., i], name=nm)

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

        self.apply_channel_filter()

        # Add named layers
        self._prob_layers = {}
        self._add_channel_layers()

        self.set_status(f"Loaded image {self.img.shape} with {len(self.ch_names)} channels.")

    def on_train_predict(self):
        if self.img is None:
            show_warning("Load an image first.")
            return

        use_ch = self.get_selected_channels()
        if len(use_ch) == 0:
            show_warning("Select at least one channel for training.")
            return

        scrib_layer = self.ensure_scribbles_layer()
        S = np.asarray(scrib_layer.data)

        train_mask = S > 0
        n_train = int(train_mask.sum())
        if n_train < 2000:
            show_warning(f"Not enough labeled pixels ({n_train}). Paint more scribbles (aim for >2k).")
            return

        self.set_status(f"Building features from channels {use_ch} and training on {n_train} pixels…")

        X_full = build_features(self.img, use_ch)
        X_train = X_full[train_mask].reshape(-1, X_full.shape[-1])
        y_train = (S[train_mask] - 1).astype(np.uint8)  # 0..2

        rf = train_rf(X_train, y_train)

        self.set_status("Predicting probabilities (tiled)…")
        P = predict_proba_tiled(rf, X_full, tile=512)

        alpha = float(self.slider_alpha.value()) / 100.0
        P_nuc = P[..., 0]
        P_cyto = P[..., 1]

        # Add/update probability layers (named)
        if "P(nucleus)" in self.viewer.layers:
            self.viewer.layers["P(nucleus)"].data = P_nuc
        else:
            self._prob_layers["P_nuc"] = self.viewer.add_image(
                P_nuc, name="P(nucleus)", opacity=alpha, blending="additive", colormap="magenta"
            )

        if "P(cytoplasm)" in self.viewer.layers:
            self.viewer.layers["P(cytoplasm)"].data = P_cyto
        else:
            self._prob_layers["P_cyto"] = self.viewer.add_image(
                P_cyto, name="P(cytoplasm)", opacity=alpha, blending="additive", colormap="green"
            )

        self.set_status("Done. Inspect P(nucleus)/P(cytoplasm), add scribbles, retrain as needed.")
