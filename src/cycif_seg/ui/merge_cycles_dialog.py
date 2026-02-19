from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets

from cycif_seg.io.ome_tiff import load_multichannel_tiff


@dataclass
class CycleUIConfig:
    path: str
    cycle: int
    channel_names: list[str]
    marker_names: list[str]
    antibody_names: list[str]
    registration_channel: str


class MergeRegisterCyclesDialog(QtWidgets.QDialog):
    """Collect step (1) metadata + per-cycle registration channel in a visual way."""

    def __init__(self, parent=None, *, paths: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Merge/Register cycles")
        self.setModal(True)

        self._paths = list(paths)
        self._cycles: list[CycleUIConfig] = []

        root = QtWidgets.QVBoxLayout(self)

        # Global metadata
        form = QtWidgets.QFormLayout()
        self.txt_tissue = QtWidgets.QLineEdit()
        self.txt_species = QtWidgets.QLineEdit()
        form.addRow("Tissue type:", self.txt_tissue)
        form.addRow("Species:", self.txt_species)
        root.addLayout(form)

        root.addWidget(
            QtWidgets.QLabel(
                "For each cycle, choose a registration channel (default DAPI if present),\n"
                "and optionally edit marker/antibody names per channel.\n"
                "Output channels will be named as: <marker>_cy<cycle>."
            )
        )

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        inner = QtWidgets.QWidget()
        self.scroll.setWidget(inner)
        self.inner_layout = QtWidgets.QVBoxLayout(inner)

        self._build_cycle_panels()

        # Output path
        out_row = QtWidgets.QHBoxLayout()
        self.txt_output = QtWidgets.QLineEdit()
        self.btn_browse_out = QtWidgets.QPushButton("Browse…")
        out_row.addWidget(QtWidgets.QLabel("Output OME-TIFF:"))
        out_row.addWidget(self.txt_output, 1)
        out_row.addWidget(self.btn_browse_out)
        root.addLayout(out_row)

        self.btn_browse_out.clicked.connect(self._on_browse_out)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def set_default_output(self, path: str) -> None:
        self.txt_output.setText(str(path))

    def _on_browse_out(self):
        start = str(Path(self.txt_output.text()).parent) if self.txt_output.text() else str(Path.cwd())
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save merged OME-TIFF",
            start,
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if out_path:
            self.txt_output.setText(out_path)

    def _build_cycle_panels(self):
        # Load channel names for each file and create UI group.
        self._cycles.clear()
        for i, p in enumerate(self._paths):
            img, ch = load_multichannel_tiff(p)
            del img
            cycle_idx = i + 1

            # default registration channel: DAPI if present
            reg_default = ch[0] if ch else ""
            for nm in ch:
                if (nm or "").strip().lower() == "dapi" or "dapi" in (nm or "").strip().lower():
                    reg_default = nm
                    break

            cfg = CycleUIConfig(
                path=p,
                cycle=cycle_idx,
                channel_names=list(ch),
                marker_names=[(nm or "").strip() for nm in ch],
                antibody_names=["" for _ in ch],
                registration_channel=reg_default,
            )
            self._cycles.append(cfg)

            gb = QtWidgets.QGroupBox(f"Cycle {cycle_idx}")
            gb_layout = QtWidgets.QVBoxLayout(gb)

            gb_layout.addWidget(QtWidgets.QLabel(str(p)))

            reg_row = QtWidgets.QHBoxLayout()
            reg_row.addWidget(QtWidgets.QLabel("Registration channel:"))
            cmb = QtWidgets.QComboBox()
            cmb.addItems(ch)
            if reg_default in ch:
                cmb.setCurrentText(reg_default)
            reg_row.addWidget(cmb, 1)
            gb_layout.addLayout(reg_row)

            tbl = QtWidgets.QTableWidget()
            tbl.setColumnCount(3)
            tbl.setHorizontalHeaderLabels(["Original channel", "Marker name", "Antibody (optional)"])
            tbl.setRowCount(len(ch))
            tbl.horizontalHeader().setStretchLastSection(True)
            tbl.verticalHeader().setVisible(False)
            tbl.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
            tbl.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)

            for r, nm in enumerate(ch):
                item0 = QtWidgets.QTableWidgetItem(nm)
                item0.setFlags(item0.flags() & ~QtCore.Qt.ItemIsEditable)
                tbl.setItem(r, 0, item0)
                tbl.setItem(r, 1, QtWidgets.QTableWidgetItem((nm or "").strip()))
                tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(""))

            gb_layout.addWidget(tbl)

            # Store widgets for later readout
            gb._cycif_reg_combo = cmb  # type: ignore[attr-defined]
            gb._cycif_table = tbl  # type: ignore[attr-defined]
            self.inner_layout.addWidget(gb)

        self.inner_layout.addStretch(1)

    def get_result(self) -> dict:
        """Return a dict with tissue/species/output + per-cycle config."""
        tissue = self.txt_tissue.text().strip()
        species = self.txt_species.text().strip()
        out_path = self.txt_output.text().strip()
        if not out_path:
            raise ValueError("Output path is required")

        cycles_out: list[dict] = []
        # Iterate groupboxes in the scroll area
        for idx in range(self.inner_layout.count()):
            w = self.inner_layout.itemAt(idx).widget()
            if not isinstance(w, QtWidgets.QGroupBox):
                continue
            cmb = getattr(w, "_cycif_reg_combo", None)
            tbl = getattr(w, "_cycif_table", None)
            if cmb is None or tbl is None:
                continue

            # Cycle number is encoded in title
            title = w.title()
            cy = int(title.replace("Cycle", "").strip())
            # Find original config
            cfg = next((c for c in self._cycles if int(c.cycle) == cy), None)
            if cfg is None:
                continue

            markers: list[str] = []
            antibodies: list[str] = []
            for r in range(tbl.rowCount()):
                m = (tbl.item(r, 1).text() if tbl.item(r, 1) else "").strip()
                a = (tbl.item(r, 2).text() if tbl.item(r, 2) else "").strip()
                markers.append(m)
                antibodies.append(a)

            cycles_out.append(
                {
                    "path": cfg.path,
                    "cycle": int(cy),
                    "registration_marker": str(cmb.currentText()).strip(),
                    "channel_markers": markers,
                    "channel_antibodies": antibodies,
                }
            )

        return {
            "tissue": tissue,
            "species": species,
            "output_path": out_path,
            "cycles": cycles_out,
        }
