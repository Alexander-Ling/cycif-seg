from __future__ import annotations

from qtpy import QtWidgets


class Step0StitchPanel(QtWidgets.QWidget):
    """UI panel for Stage 0 (tile stitching)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.btn_batch_stitch = QtWidgets.QPushButton("Batch stitch cycles…")
        row.addWidget(self.btn_batch_stitch)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(
            QtWidgets.QLabel(
                "Stage 0 (Stitch): find area_{x}_{y}.ome.tiff tiles inside each cycle folder, "
                "estimate translations on a chosen stitch channel, and write a stitched "
                "pyramidal OME-TIFF back into each cycle folder."
            )
        )
        layout.addStretch(1)
