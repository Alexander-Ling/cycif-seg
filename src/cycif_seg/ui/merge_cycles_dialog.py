from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets

from cycif_seg.io.ome_tiff import load_channel_names_only


@dataclass
class CycleUIConfig:
    path: str
    cycle: int
    channel_names: list[str]
    marker_names: list[str]
    antibody_names: list[str]
    registration_channel: str
    enabled: bool = True


class MergeRegisterCyclesDialog(QtWidgets.QDialog):
    """Collect step (1) metadata + per-cycle registration channel in a visual way."""

    def __init__(self, parent=None, *, paths: list[str], initial_cfg: dict | None = None, channel_name_cache: dict[str, list[str]] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Merge/Register cycles")
        self.setModal(True)

        self._paths = list(paths)
        self._channel_name_cache = dict(channel_name_cache or {})
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

        # Tiled rigid options
        self.chk_allow_rotation = QtWidgets.QCheckBox("Enable rotation during tile registration")
        self.chk_allow_rotation.setChecked(False)
        root.addWidget(self.chk_allow_rotation)
        # Seed from initial config if present
        try:
            if isinstance(self._initial_cfg, dict):
                v = self._initial_cfg.get("tiled_rigid_allow_rotation")
                if v is None:
                    v = self._initial_cfg.get("allow_rotation")
                if v is None:
                    v = self._initial_cfg.get("enable_rotation")
                if v is not None:
                    self.chk_allow_rotation.setChecked(bool(v))
        except Exception:
            pass

        tiled_form = QtWidgets.QFormLayout()
        self.spn_tile_size = QtWidgets.QSpinBox()
        self.spn_tile_size.setRange(128, 20000)
        self.spn_tile_size.setSingleStep(128)
        self.spn_tile_size.setValue(2000)
        self.spn_tile_size.setToolTip("Tile size in pixels for tile-wise rigid registration.")
        self.spn_search_factor = QtWidgets.QDoubleSpinBox()
        self.spn_search_factor.setRange(1.0, 10.0)
        self.spn_search_factor.setDecimals(2)
        self.spn_search_factor.setSingleStep(0.25)
        self.spn_search_factor.setValue(3.0)
        self.spn_search_factor.setToolTip("Search window size as a multiple of the tile size during tile-wise rigid registration.")
        tiled_form.addRow("Tile size (px):", self.spn_tile_size)
        tiled_form.addRow("Search factor:", self.spn_search_factor)
        root.addLayout(tiled_form)
        try:
            if isinstance(self._initial_cfg, dict):
                v_tile = self._initial_cfg.get("tiled_rigid_tile_size")
                if v_tile is None:
                    v_tile = self._initial_cfg.get("tile_size")
                if v_tile is not None:
                    self.spn_tile_size.setValue(max(128, int(v_tile)))
                v_sf = self._initial_cfg.get("tiled_rigid_search_factor")
                if v_sf is None:
                    v_sf = self._initial_cfg.get("search_factor")
                if v_sf is not None:
                    self.spn_search_factor.setValue(max(1.0, float(v_sf)))
        except Exception:
            pass

        self.chk_pyramidal_output = QtWidgets.QCheckBox("Write pyramidal OME-TIFF output")
        self.chk_pyramidal_output.setChecked(True)
        self.chk_pyramidal_output.setToolTip("Write step-1 output as a tiled pyramidal OME-TIFF for faster viewing in large-image viewers.")
        root.addWidget(self.chk_pyramidal_output)
        try:
            if isinstance(self._initial_cfg, dict):
                vp = self._initial_cfg.get("pyramidal_output")
                if vp is not None:
                    self.chk_pyramidal_output.setChecked(bool(vp))
        except Exception:
            pass

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
        for i, p in enumerate(self._paths, start=1):
            ch = list(self._channel_name_cache.get(p) or load_channel_names_only(p) or [])

            # If we have a prior config for this file, prefer it.
            prior = init_by_path.get(p) or init_by_name.get(Path(p).name)
            # Note: cycle 0 is valid. Only treat as missing if key is absent/None.
            cycle_idx = (
                int(prior.get("cycle"))
                if isinstance(prior, dict) and (prior.get("cycle") is not None)
                else i
            )

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

            enabled0 = True
            try:
                if isinstance(prior, dict) and (prior.get("enabled") is False):
                    enabled0 = False
            except Exception:
                enabled0 = True

            cfg = CycleUIConfig(
                path=p,
                cycle=cycle_idx,
                channel_names=list(ch),
                marker_names=[(prior_markers[j] if j < len(prior_markers) else (nm or "")).strip() for j, nm in enumerate(ch)],
                antibody_names=[(prior_abs[j] if j < len(prior_abs) else "").strip() for j in range(len(ch))],
                registration_channel=reg_default,
                enabled=bool(enabled0),
            )
            self._cycles.append(cfg)

            gb = QtWidgets.QGroupBox(f"Cycle {cycle_idx}")
            gb_layout = QtWidgets.QVBoxLayout(gb)

            gb_layout.addWidget(QtWidgets.QLabel(str(p)))

            use_row = QtWidgets.QHBoxLayout()
            chk_use = QtWidgets.QCheckBox("Use this cycle")
            chk_use.setChecked(bool(cfg.enabled))
            chk_use.setToolTip("Uncheck to skip this cycle during merge/registration (it will not be included in the output).")
            use_row.addWidget(chk_use)
            use_row.addStretch(1)
            gb_layout.addLayout(use_row)

            # Allow manual cycle numbering (users often load files out of order).
            cyc_row = QtWidgets.QHBoxLayout()
            cyc_row.addWidget(QtWidgets.QLabel("Cycle number:"))
            sp_cy = QtWidgets.QSpinBox()
            sp_cy.setMinimum(0)
            sp_cy.setMaximum(999)
            sp_cy.setValue(int(cycle_idx))
            sp_cy.setToolTip(
                "Cycle index used in output channel naming (e.g., <marker>_cy<cycle>). Must be unique. Cycle 0 is allowed."
            )
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
            gb._cycif_enabled_chk = chk_use  # type: ignore[attr-defined]
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
        reg_algorithm = "tiled_rigid"
        tiled_allow_rotation = bool(getattr(self, 'chk_allow_rotation', None) and self.chk_allow_rotation.isChecked())
        tiled_tile_size = int(getattr(self, 'spn_tile_size', None).value() if getattr(self, 'spn_tile_size', None) is not None else 2000)
        tiled_search_factor = float(getattr(self, 'spn_search_factor', None).value() if getattr(self, 'spn_search_factor', None) is not None else 3.0)
        if not out_path:
            raise ValueError("Output path is required")

        cycles_out: list[dict] = []
        seen_cycles: set[int] = set()
        # Iterate groupboxes in the scroll area
        for idx in range(self.inner_layout.count()):
            w = self.inner_layout.itemAt(idx).widget()
            if not isinstance(w, QtWidgets.QGroupBox):
                continue
            chk_use = getattr(w, "_cycif_enabled_chk", None)
            sp_cy = getattr(w, "_cycif_cycle_spin", None)
            cmb = getattr(w, "_cycif_reg_combo", None)
            tbl = getattr(w, "_cycif_table", None)
            if chk_use is None or sp_cy is None or cmb is None or tbl is None:
                continue

            cy = int(sp_cy.value())
            enabled = bool(chk_use.isChecked())
            if enabled:
                if cy in seen_cycles:
                    raise ValueError(f"Duplicate cycle number: {cy}. Cycle numbers must be unique among enabled cycles.")
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
                    "enabled": bool(enabled),
                    "registration_marker": str(cmb.currentText()).strip(),
                    "channel_markers": markers,
                    "channel_antibodies": antibodies,
                }
            )

        return {
            "tissue": tissue,
            "species": species,
            "output_path": out_path,
            "registration_algorithm": reg_algorithm,
            "tiled_rigid_allow_rotation": tiled_allow_rotation,
            "pyramidal_output": bool(self.chk_pyramidal_output.isChecked()),
            "cycles": cycles_out,
        }