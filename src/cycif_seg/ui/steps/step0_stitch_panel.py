from __future__ import annotations

from qtpy import QtWidgets

from cycif_seg.ui.help_button import STITCH_HELP, make_help_button


class Step0StitchPanel(QtWidgets.QWidget):
    """UI panel for Step 0 (tile stitching)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.btn_batch_stitch = QtWidgets.QPushButton("Batch stitch cycles…")
        row.addWidget(self.btn_batch_stitch)
        row.addWidget(make_help_button(self, "Batch stitch", STITCH_HELP))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(
            QtWidgets.QLabel(
                "Step 0 (Stitch): find tile files each cycle folder, "
                "estimate translations on a chosen stitch channel, and write a stitched "
                "pyramidal OME-TIFF back into each cycle folder."
            )
        )
        layout.addStretch(1)
