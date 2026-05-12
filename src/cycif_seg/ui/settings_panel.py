from __future__ import annotations

from qtpy import QtWidgets


class SettingsPanel(QtWidgets.QWidget):
    """App-wide settings panel (Settings tab)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        debug_group = QtWidgets.QGroupBox("Debug / Verbose Output")
        debug_layout = QtWidgets.QVBoxLayout(debug_group)

        self.chk_debug_tiff_loading = QtWidgets.QCheckBox(
            "Print TIFF/OME-TIFF loading debug messages"
        )
        self.chk_debug_tiff_loading.setChecked(False)
        self.chk_debug_tiff_loading.setToolTip(
            "Print verbose console output when loading TIFF and OME-TIFF files,\n"
            "including pyramid level detection and multiscale channel loading."
        )
        debug_layout.addWidget(self.chk_debug_tiff_loading)

        self.chk_debug_preprocess = QtWidgets.QCheckBox(
            "Print batch preprocessing debug messages"
        )
        self.chk_debug_preprocess.setChecked(False)
        self.chk_debug_preprocess.setToolTip(
            "Print verbose console output during batch preprocessing (Step 1).\n"
            "Shows strip mode configuration, per-strip progress, RAM usage,\n"
            "registration channel sizes, and computed shifts per cycle.\n"
            "Useful for verifying that RAM-efficient strip processing is active."
        )
        debug_layout.addWidget(self.chk_debug_preprocess)

        layout.addWidget(debug_group)
        layout.addStretch(1)
