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
            "Useful for verifying that RAM-efficient strip processing is active.\n"
            "Also prints the full stack traceback to the console for any\n"
            "sample that fails, to help diagnose the underlying error."
        )
        debug_layout.addWidget(self.chk_debug_preprocess)

        self.chk_debug_elastic_touchup = QtWidgets.QCheckBox(
            "Save elastic touch-up debug images"
        )
        self.chk_debug_elastic_touchup.setChecked(False)
        self.chk_debug_elastic_touchup.setToolTip(
            "After each moving cycle, save the registered DAPI channel twice:\n"
            "  cycle_NNN_1_after_rigid.tiff  — after rigid island refinement only\n"
            "  cycle_NNN_2_after_elastic.tiff — after B-spline elastic correction\n"
            "Output folder: <output_stem>_debug/ next to the output OME-TIFF.\n"
            "Only has effect when elastic touch-up is enabled for the sample."
        )
        debug_layout.addWidget(self.chk_debug_elastic_touchup)

        self.chk_debug_elastic_field = QtWidgets.QCheckBox(
            "Print elastic touch-up field statistics"
        )
        self.chk_debug_elastic_field.setChecked(False)
        self.chk_debug_elastic_field.setToolTip(
            "Print per-island diagnostics during elastic touch-up:\n"
            "  - island size, bounding box, and post-rigid correlation score\n"
            "  - whether each island was skipped (corr > threshold) or processed\n"
            "  - whether SimpleITK/elastix is installed and working\n"
            "  - displacement field magnitude (max, mean, p95) when applied\n"
            "  - per-cycle summary of how many corrections were applied\n"
            "Only has effect when elastic touch-up is enabled for the sample."
        )
        debug_layout.addWidget(self.chk_debug_elastic_field)

        layout.addWidget(debug_group)
        layout.addStretch(1)
