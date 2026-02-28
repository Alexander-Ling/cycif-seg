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

    def __init__(self, parent=None, *, paths: list[str], initial_cfg: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Merge/Register cycles")
        self.setModal(True)

        self._paths = list(paths)
        self._cycles: list[CycleUIConfig] = []
        # Optional prior configuration (e.g., from a loaded batch plan).
        # Expected shape matches get_result(): {tissue, species, output_path, cycles:[...]}
        self._initial_cfg = dict(initial_cfg or {})

        root = QtWidgets.QVBoxLayout(self)

        # Global metadata
        form = QtWidgets.QFormLayout()
        self.txt_tissue = QtWidgets.QLineEdit()
        self.txt_species = QtWidgets.QLineEdit()
        form.addRow("Tissue type:", self.txt_tissue)
        form.addRow("Species:", self.txt_species)
        root.addLayout(form)

        # Seed global metadata from an initial config (e.g., loaded plan).
        try:
            if isinstance(self._initial_cfg, dict):
                self.txt_tissue.setText(str(self._initial_cfg.get("tissue") or ""))
                self.txt_species.setText(str(self._initial_cfg.get("species") or ""))
        except Exception:
            pass

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
        # Build a mapping from filename (or full path) -> prior per-cycle config.
        init_cycles = list((self._initial_cfg.get("cycles") or []) if isinstance(self._initial_cfg, dict) else [])
        init_by_name: dict[str, dict] = {}
        init_by_path: dict[str, dict] = {}
        for d in init_cycles:
            try:
                p = str(d.get("path") or "")
                if p:
                    init_by_path[p] = d
                    init_by_name[Path(p).name] = d
            except Exception:
                continue

        # Load channel names for each file and create UI group.
        self._cycles.clear()
        for i, p in enumerate(self._paths):
            img, ch = load_multichannel_tiff(p)
            del img

            # If we have a prior config for this file, prefer it.
            prior = init_by_path.get(p) or init_by_name.get(Path(p).name)
            cycle_idx = int(prior.get("cycle")) if isinstance(prior, dict) and prior.get("cycle") else (i + 1)

            # default registration channel: use prior if available, else DAPI if present.
            reg_default = ""
            if isinstance(prior, dict):
                reg_default = str(prior.get("registration_marker") or "").strip()
            if not reg_default:
                reg_default = ch[0] if ch else ""
                for nm in ch:
                    if (nm or "").strip().lower() == "dapi" or "dapi" in (nm or "").strip().lower():
                        reg_default = nm
                        break

            # marker/antibody defaults: prior if available, else use channel names.
            prior_markers = []
            prior_abs = []
            if isinstance(prior, dict):
                try:
                    prior_markers = list(prior.get("channel_markers") or [])
                    prior_abs = list(prior.get("channel_antibodies") or [])
                except Exception:
                    prior_markers = []
                    prior_abs = []

            cfg = CycleUIConfig(
                path=p,
                cycle=cycle_idx,
                channel_names=list(ch),
                marker_names=[(prior_markers[j] if j < len(prior_markers) else (nm or "")).strip() for j, nm in enumerate(ch)],
                antibody_names=[(prior_abs[j] if j < len(prior_abs) else "").strip() for j in range(len(ch))],
                registration_channel=reg_default,
            )
            self._cycles.append(cfg)

            gb = QtWidgets.QGroupBox(f"Cycle {cycle_idx}")
            gb_layout = QtWidgets.QVBoxLayout(gb)

            gb_layout.addWidget(QtWidgets.QLabel(str(p)))

            # Allow manual cycle numbering (users often load files out of order).
            cyc_row = QtWidgets.QHBoxLayout()
            cyc_row.addWidget(QtWidgets.QLabel("Cycle number:"))
            sp_cy = QtWidgets.QSpinBox()
            sp_cy.setMinimum(1)
            sp_cy.setMaximum(999)
            sp_cy.setValue(int(cycle_idx))
            sp_cy.setToolTip("Cycle index used in output channel naming (e.g., <marker>_cy<cycle>). Must be unique.")
            cyc_row.addWidget(sp_cy)
            cyc_row.addStretch(1)
            gb_layout.addLayout(cyc_row)

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
                tbl.setItem(r, 1, QtWidgets.QTableWidgetItem((cfg.marker_names[r] if r < len(cfg.marker_names) else (nm or "")).strip()))
                tbl.setItem(r, 2, QtWidgets.QTableWidgetItem((cfg.antibody_names[r] if r < len(cfg.antibody_names) else "").strip()))

            gb_layout.addWidget(tbl)

            # Store widgets for later readout
            gb._cycif_cycle_spin = sp_cy  # type: ignore[attr-defined]
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
        seen_cycles: set[int] = set()
        # Iterate groupboxes in the scroll area
        for idx in range(self.inner_layout.count()):
            w = self.inner_layout.itemAt(idx).widget()
            if not isinstance(w, QtWidgets.QGroupBox):
                continue
            sp_cy = getattr(w, "_cycif_cycle_spin", None)
            cmb = getattr(w, "_cycif_reg_combo", None)
            tbl = getattr(w, "_cycif_table", None)
            if sp_cy is None or cmb is None or tbl is None:
                continue

            cy = int(sp_cy.value())
            if cy in seen_cycles:
                raise ValueError(f"Duplicate cycle number {cy}. Each cycle number must be unique.")
            seen_cycles.add(cy)

            # Find original config by groupbox order (stable) instead of cycle number.
            # Users can edit the cycle number, so the mapping must not depend on it.
            # The groupbox order mirrors self._paths/self._cycles.
            gb_idx = len(cycles_out)
            cfg = self._cycles[gb_idx] if gb_idx < len(self._cycles) else None
            if cfg is None:
                raise ValueError("Internal error: cycle config mismatch")

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
