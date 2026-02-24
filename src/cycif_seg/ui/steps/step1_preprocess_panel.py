from __future__ import annotations

from qtpy import QtWidgets


class Step1PreprocessPanel(QtWidgets.QWidget):
    """UI panel for Step 1 (Preprocess).

    Keep this panel intentionally small and UI-only. Any processing logic should live in
    controllers/services and be called from the main widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        file_row = QtWidgets.QHBoxLayout()
        self.btn_merge_cycles = QtWidgets.QPushButton("Merge/Register cycles (beta)…")
        file_row.addWidget(self.btn_merge_cycles)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        layout.addWidget(
            QtWidgets.QLabel(
                "Preprocessing (Step 1, beta): merge 2+ cycle OME-TIFFs into one co-registered OME-TIFF.\n"
                "Current MVP does translation-only registration using a chosen nuclear marker (default: DAPI)."
            )
        )

        layout.addStretch(1)
