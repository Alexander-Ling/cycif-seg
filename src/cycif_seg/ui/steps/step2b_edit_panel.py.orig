from __future__ import annotations

from qtpy import QtWidgets


class Step2bEditPanel(QtWidgets.QWidget):
    """Step 2b panel: nuclei touch-up UI.

    This widget contains only UI construction (buttons/inputs/layout). The actual
    tool logic lives in controllers (e.g., NucleiEditController).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(
            QtWidgets.QLabel(
                "Touch up nuclei by manually editing the instance labels and (optionally)\n"
                "training a model to propagate similar edits elsewhere."
            )
        )

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Nuclei edit tools:"))

        edit_row = QtWidgets.QHBoxLayout()
        self.btn_make_nuclei_edit = QtWidgets.QPushButton("Make/Select editable nuclei")
        self.btn_propagate_nuclei_edits = QtWidgets.QPushButton("Propagate edits (RF)")
        edit_row.addWidget(self.btn_make_nuclei_edit)
        edit_row.addWidget(self.btn_propagate_nuclei_edits)
        layout.addLayout(edit_row)

        tools_row = QtWidgets.QHBoxLayout()
        self.btn_split_cut = QtWidgets.QPushButton("Split (cut line)")
        self.btn_split_cut.setCheckable(True)
        self.btn_merge_lasso = QtWidgets.QPushButton("Merge (line)")
        self.btn_merge_lasso.setCheckable(True)
        self.btn_delete_nucleus = QtWidgets.QPushButton("Delete nucleus")
        self.btn_delete_nucleus.setCheckable(True)
        self.btn_draw_new_nucleus = QtWidgets.QPushButton("Draw new nucleus")
        self.btn_draw_new_nucleus.setCheckable(True)
        self.btn_erase_nucleus = QtWidgets.QPushButton("Erode (eraser)")
        self.btn_erase_nucleus.setCheckable(True)
        tools_row.addWidget(self.btn_split_cut)
        tools_row.addWidget(self.btn_merge_lasso)
        tools_row.addWidget(self.btn_delete_nucleus)
        tools_row.addWidget(self.btn_draw_new_nucleus)
        tools_row.addWidget(self.btn_erase_nucleus)
        layout.addLayout(tools_row)

        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(QtWidgets.QLabel("Erode iters"))
        self.spin_erode_iters = QtWidgets.QSpinBox()
        self.spin_erode_iters.setRange(1, 50)
        self.spin_erode_iters.setValue(2)
        params_row.addWidget(self.spin_erode_iters)
        params_row.addSpacing(12)
        params_row.addWidget(QtWidgets.QLabel("Brush"))
        self.spin_brush = QtWidgets.QSpinBox()
        self.spin_brush.setRange(1, 200)
        self.spin_brush.setValue(4)
        params_row.addWidget(self.spin_brush)
        params_row.addStretch(1)
        layout.addLayout(params_row)

        self.chk_auto_regen_nuclei = QtWidgets.QCheckBox(
            "Auto-regenerate nuclei after propagation"
        )
        self.chk_auto_regen_nuclei.setChecked(True)
        layout.addWidget(self.chk_auto_regen_nuclei)

        layout.addWidget(QtWidgets.QLabel("Nuclei instance segmentation (boundary-aware):"))
        layout.addStretch(0)
