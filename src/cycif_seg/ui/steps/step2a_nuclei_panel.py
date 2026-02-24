from __future__ import annotations

from qtpy import QtWidgets, QtCore


class Step2aNucleiPanel(QtWidgets.QWidget):
    """UI panel for Step 2a (initial nuclei segmentation).

    This panel only builds widgets/layout. All behavior and signal wiring remains
    in main_widget.py (for now) to keep the refactor safe and incremental.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Load image for nuclei segmentation
        file_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load TIFF/OME-TIFF…")
        self.lbl_file = QtWidgets.QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        file_row.addWidget(self.btn_load)
        file_row.addWidget(self.lbl_file, 1)
        layout.addLayout(file_row)

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Channels used for RF training:"))
        layout.addLayout(header_row)

        self.list_channels = QtWidgets.QListWidget()
        self.list_channels.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.list_channels, 1)

        ch_btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        self.btn_apply_channels = QtWidgets.QPushButton("Update displayed channels")
        self.btn_apply_channels.setEnabled(False)
        ch_btn_row.addWidget(self.btn_all)
        ch_btn_row.addWidget(self.btn_none)
        ch_btn_row.addWidget(self.btn_apply_channels)
        layout.addLayout(ch_btn_row)

        layout.addWidget(
            QtWidgets.QLabel(
                "Scribbles: paint into the Labels layer using values:\n"
                "1 = nucleus, 2 = nucleus boundary/overlap, 3 = background (0 = unlabeled)\n"
                "Tip: select the Scribbles layer and set the current label value in napari."
            )
        )

        self.btn_train = QtWidgets.QPushButton("Train + Predict (RF)")
        layout.addWidget(self.btn_train)

        grid = QtWidgets.QGridLayout()

        self.spin_nuc_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_nuc_thresh.setRange(0.0, 1.0)
        self.spin_nuc_thresh.setSingleStep(0.05)
        self.spin_nuc_thresh.setValue(0.35)

        self.spin_min_nuc_area = QtWidgets.QSpinBox()
        self.spin_min_nuc_area.setRange(0, 10_000_000)
        self.spin_min_nuc_area.setValue(30)

        grid.addWidget(QtWidgets.QLabel("Nucleus prob thresh"), 0, 0)
        grid.addWidget(self.spin_nuc_thresh, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min nucleus area (px)"), 1, 0)
        grid.addWidget(self.spin_min_nuc_area, 1, 1)
        layout.addLayout(grid)

        self.btn_nuclei = QtWidgets.QPushButton("Generate Nuclei (instances)")
        layout.addWidget(self.btn_nuclei)

        self.chk_show_nuc_markers = QtWidgets.QCheckBox("Show nucleus markers overlay")
        self.chk_show_nuc_markers.setChecked(False)
        layout.addWidget(self.chk_show_nuc_markers)

        # Opacity
        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(QtWidgets.QLabel("Overlay opacity:"))
        self.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(100)
        self.slider_alpha.setValue(40)
        op_row.addWidget(self.slider_alpha, 1)
        layout.addLayout(op_row)

        layout.addStretch(0)
