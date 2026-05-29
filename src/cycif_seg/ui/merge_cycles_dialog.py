from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets

from cycif_seg.io.ome_tiff import load_channel_names_only_fast


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

        refine_form = QtWidgets.QFormLayout()
        self.spn_tile_size = QtWidgets.QSpinBox()
        self.spn_tile_size.setRange(128, 50000)
        self.spn_tile_size.setSingleStep(250)
        self.spn_tile_size.setValue(2000)
        self.spn_tile_size.setToolTip("Approximate foreground-island size in pixels. Larger values merge nearby tissue into fewer local correction regions.")
        self.spn_search_factor = QtWidgets.QDoubleSpinBox()
        self.spn_search_factor.setRange(1.0, 50.0)
        self.spn_search_factor.setDecimals(1)
        self.spn_search_factor.setSingleStep(0.5)
        self.spn_search_factor.setValue(3.0)
        self.spn_search_factor.setToolTip("Local correction search radius is tile size times this factor divided by 4.")
        refine_form.addRow("Foreground island size (px):", self.spn_tile_size)
        refine_form.addRow("Local search factor:", self.spn_search_factor)
        self.chk_fast_large_island = QtWidgets.QCheckBox("Use fast large-island refinement")
        self.chk_fast_large_island.setChecked(False)
        self.chk_fast_large_island.setToolTip("For large foreground islands, register a small set of spread-out full-resolution tiles and apply their median shift to the whole island.")
        self.spn_fast_sample_count = QtWidgets.QSpinBox()
        self.spn_fast_sample_count.setRange(1, 100)
        self.spn_fast_sample_count.setValue(5)
        self.spn_fast_sample_count.setToolTip("Number of full-resolution sample tiles to register per large foreground island when fast mode is enabled.")
        root.addLayout(refine_form)
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

        # Strip-based RAM-efficient processing
        self.chk_low_mem = QtWidgets.QCheckBox("RAM-efficient strip processing")
        self.chk_low_mem.setChecked(True)
        self.chk_low_mem.setToolTip(
            "Process the image in horizontal strips to reduce RAM usage.\n"
            "Recommended for large files (>10 GB per cycle).\n"
            "Uses downsampled images for registration and writes channel data strip by strip."
        )
        root.addWidget(self.chk_low_mem)

        strip_form = QtWidgets.QFormLayout()
        self.spn_strip_height = QtWidgets.QSpinBox()
        self.spn_strip_height.setRange(0, 200000)
        self.spn_strip_height.setSingleStep(1000)
        self.spn_strip_height.setValue(0)
        self.spn_strip_height.setSpecialValueText("Auto")
        self.spn_strip_height.setToolTip(
            "Height of each processing strip in pixels.\n"
            "\"Auto\" (0): canvas height ÷ 10, minimum 1000 rows.\n"
            "Set a specific value to override the automatic choice."
        )
        self.spn_strip_height.setEnabled(self.chk_low_mem.isChecked())
        self.chk_low_mem.toggled.connect(self.spn_strip_height.setEnabled)
        strip_form.addRow("Strip height (px):", self.spn_strip_height)
        root.addLayout(strip_form)

        try:
            if isinstance(self._initial_cfg, dict):
                vlm = self._initial_cfg.get("low_mem")
                if vlm is not None:
                    self.chk_low_mem.setChecked(bool(vlm))
                vsh = self._initial_cfg.get("strip_height")
                if vsh is not None:
                    self.spn_strip_height.setValue(max(0, int(vsh)))
        except Exception:
            pass

        # Elastic touch-up
        elastic_group = QtWidgets.QGroupBox("Elastic touch-up")
        elastic_layout = QtWidgets.QVBoxLayout(elastic_group)
        self.chk_elastic_touchup = QtWidgets.QCheckBox("Enable B-spline elastic touch-up after island refinement")
        self.chk_elastic_touchup.setChecked(True)
        self.chk_elastic_touchup.setToolTip(
            "After rigid island refinement, apply a full-resolution B-spline elastic\n"
            "correction to each foreground island using elastix (via SimpleITK).\n"
            "Helps correct residual local misregistration that rigid correction misses."
        )
        elastic_layout.addWidget(self.chk_elastic_touchup)
        elastic_form = QtWidgets.QFormLayout()
        self.spn_bspline_spacing = QtWidgets.QSpinBox()
        self.spn_bspline_spacing.setRange(4, 500)
        self.spn_bspline_spacing.setSingleStep(10)
        self.spn_bspline_spacing.setValue(50)
        self.spn_bspline_spacing.setToolTip(
            "B-spline control-point grid spacing in pixels at full resolution.\n"
            "Lower values allow more local deformation (default: 50)."
        )
        self.spn_max_iterations = QtWidgets.QSpinBox()
        self.spn_max_iterations.setRange(1, 1000)
        self.spn_max_iterations.setSingleStep(10)
        self.spn_max_iterations.setValue(100)
        self.spn_max_iterations.setToolTip(
            "Maximum elastix optimizer iterations per resolution level (default: 100)."
        )
        elastic_form.addRow("B-spline grid spacing (px):", self.spn_bspline_spacing)
        elastic_form.addRow("Max iterations:", self.spn_max_iterations)
        elastic_layout.addLayout(elastic_form)
        self.spn_bspline_spacing.setEnabled(True)
        self.spn_max_iterations.setEnabled(True)
        self.chk_elastic_touchup.toggled.connect(self.spn_bspline_spacing.setEnabled)
        self.chk_elastic_touchup.toggled.connect(self.spn_max_iterations.setEnabled)
        root.addWidget(elastic_group)
        try:
            if isinstance(self._initial_cfg, dict):
                vet = self._initial_cfg.get("elastic_touchup")
                if vet is not None:
                    self.chk_elastic_touchup.setChecked(bool(vet))
                vbs = self._initial_cfg.get("elastic_touchup_bspline_spacing")
                if vbs is not None:
                    self.spn_bspline_spacing.setValue(max(4, int(vbs)))
                vmi = self._initial_cfg.get("elastic_touchup_max_iterations")
                if vmi is not None:
                    self.spn_max_iterations.setValue(max(1, int(vmi)))
        except Exception:
            pass

        self.tabs = QtWidgets.QTabWidget()
        self.tbl_registration = QtWidgets.QTableWidget()
        self.tbl_channels = QtWidgets.QTableWidget()
        self.tabs.addTab(self.tbl_registration, "Registration")
        self.tabs.addTab(self.tbl_channels, "Channel names")
        root.addWidget(self.tabs, 1)

        self._build_cycle_tables()
        self.tbl_registration.itemChanged.connect(self._on_registration_item_changed)

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

    def _build_cycle_tables(self):
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

        # Load channel names for each file and create table rows.
        self._cycles.clear()
        for i, p in enumerate(self._paths, start=1):
            cached = self._channel_name_cache.get(p)
            if cached is None:
                ch = list(load_channel_names_only_fast(p) or [])
                self._channel_name_cache[p] = list(ch)
            else:
                ch = list(cached)

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

        self._populate_registration_table()
        self._populate_channel_table()

    def _readonly_item(self, text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _populate_registration_table(self) -> None:
        tbl = self.tbl_registration
        try:
            tbl.blockSignals(True)
        except Exception:
            pass
        tbl.setColumnCount(5)
        tbl.setHorizontalHeaderLabels([
            "Use",
            "Cycle #",
            "File",
            "Registration channel",
            "Available channels",
        ])
        tbl.setRowCount(len(self._cycles))
        tbl.verticalHeader().setVisible(False)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        tbl.horizontalHeader().setStretchLastSection(True)

        for r, cfg in enumerate(self._cycles):
            use_item = QtWidgets.QTableWidgetItem("")
            use_item.setFlags(use_item.flags() | QtCore.Qt.ItemIsUserCheckable)
            use_item.setCheckState(QtCore.Qt.Checked if cfg.enabled else QtCore.Qt.Unchecked)
            use_item.setToolTip("Uncheck to skip this cycle during merge/registration.")
            tbl.setItem(r, 0, use_item)

            cy_item = QtWidgets.QTableWidgetItem(str(int(cfg.cycle)))
            cy_item.setToolTip("Cycle index used in output channel naming. Must be unique among enabled cycles.")
            tbl.setItem(r, 1, cy_item)

            file_item = self._readonly_item(Path(cfg.path).name)
            file_item.setToolTip(str(cfg.path))
            tbl.setItem(r, 2, file_item)

            reg_item = QtWidgets.QTableWidgetItem(str(cfg.registration_channel or ""))
            reg_item.setToolTip("Type the exact channel name to use for registration. See Available channels or the Channel names tab.")
            tbl.setItem(r, 3, reg_item)

            avail_item = self._readonly_item(", ".join(cfg.channel_names))
            avail_item.setToolTip("\n".join(cfg.channel_names))
            tbl.setItem(r, 4, avail_item)

        try:
            tbl.resizeColumnsToContents()
        except Exception:
            pass
        try:
            tbl.blockSignals(False)
        except Exception:
            pass

    def _populate_channel_table(self) -> None:
        tbl = self.tbl_channels
        try:
            tbl.blockSignals(True)
        except Exception:
            pass
        rows = sum(len(cfg.channel_names) for cfg in self._cycles)
        tbl.setColumnCount(6)
        tbl.setHorizontalHeaderLabels([
            "Cycle #",
            "File",
            "Channel #",
            "Original channel",
            "Marker name",
            "Antibody (optional)",
        ])
        tbl.setRowCount(rows)
        tbl.verticalHeader().setVisible(False)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        tbl.horizontalHeader().setStretchLastSection(True)

        row = 0
        for cycle_idx, cfg in enumerate(self._cycles):
            for ch_idx, nm in enumerate(cfg.channel_names):
                cy_item = self._readonly_item(str(int(cfg.cycle)))
                cy_item.setData(QtCore.Qt.UserRole, int(cycle_idx))
                tbl.setItem(row, 0, cy_item)

                file_item = self._readonly_item(Path(cfg.path).name)
                file_item.setToolTip(str(cfg.path))
                tbl.setItem(row, 1, file_item)

                tbl.setItem(row, 2, self._readonly_item(str(int(ch_idx))))
                tbl.setItem(row, 3, self._readonly_item(str(nm)))
                tbl.setItem(row, 4, QtWidgets.QTableWidgetItem((cfg.marker_names[ch_idx] if ch_idx < len(cfg.marker_names) else (nm or "")).strip()))
                tbl.setItem(row, 5, QtWidgets.QTableWidgetItem((cfg.antibody_names[ch_idx] if ch_idx < len(cfg.antibody_names) else "").strip()))
                row += 1

        try:
            tbl.resizeColumnsToContents()
        except Exception:
            pass
        try:
            tbl.blockSignals(False)
        except Exception:
            pass

    def _on_registration_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        try:
            if int(item.column()) != 1:
                return
            cycle_idx = int(item.row())
            cycle_text = str(item.text()).strip()
            for r in range(self.tbl_channels.rowCount()):
                cy_item = self.tbl_channels.item(r, 0)
                if cy_item is not None and int(cy_item.data(QtCore.Qt.UserRole)) == cycle_idx:
                    cy_item.setText(cycle_text)
        except Exception:
            return

    def _registration_rows(self) -> list[dict]:
        rows: list[dict] = []
        tbl = self.tbl_registration
        for r, cfg in enumerate(self._cycles):
            use_item = tbl.item(r, 0)
            cy_item = tbl.item(r, 1)
            reg_item = tbl.item(r, 3)
            enabled = (use_item.checkState() == QtCore.Qt.Checked) if use_item is not None else True
            try:
                cy = int((cy_item.text() if cy_item else "").strip())
            except Exception:
                raise ValueError(f"Invalid cycle number for {Path(cfg.path).name}. Cycle numbers must be integers.")
            if cy < 0:
                raise ValueError(f"Invalid cycle number for {Path(cfg.path).name}: {cy}. Cycle numbers must be >= 0.")
            reg = (reg_item.text() if reg_item else "").strip()
            rows.append({"enabled": bool(enabled), "cycle": int(cy), "registration_marker": reg})
        return rows

    def _channel_rows_by_cycle_index(self) -> dict[int, tuple[list[str], list[str]]]:
        out: dict[int, tuple[list[str], list[str]]] = {
            i: ([""] * len(cfg.channel_names), [""] * len(cfg.channel_names))
            for i, cfg in enumerate(self._cycles)
        }
        tbl = self.tbl_channels
        counters: dict[int, int] = {}
        for r in range(tbl.rowCount()):
            cy_item = tbl.item(r, 0)
            if cy_item is None:
                continue
            idx = int(cy_item.data(QtCore.Qt.UserRole))
            pos = int(counters.get(idx, 0))
            counters[idx] = pos + 1
            markers, antibodies = out[idx]
            if pos >= len(markers):
                continue
            markers[pos] = (tbl.item(r, 4).text() if tbl.item(r, 4) else "").strip()
            antibodies[pos] = (tbl.item(r, 5).text() if tbl.item(r, 5) else "").strip()
        return out

    def get_result(self) -> dict:
        """Return a dict with tissue/species/output + per-cycle config."""
        tissue = self.txt_tissue.text().strip()
        species = self.txt_species.text().strip()
        out_path = self.txt_output.text().strip()
        global_translation_only = False
        reg_algorithm = "tiled_rigid"
        tiled_allow_rotation = False
        tiled_tile_size = int(getattr(self, 'spn_tile_size', None).value() if getattr(self, 'spn_tile_size', None) is not None else 2000)
        tiled_search_factor = float(getattr(self, 'spn_search_factor', None).value() if getattr(self, 'spn_search_factor', None) is not None else 3.0)
        fast_large_island = False
        fast_sample_count = 5
        if not out_path:
            raise ValueError("Output path is required")

        cycles_out: list[dict] = []
        seen_cycles: set[int] = set()
        reg_rows = self._registration_rows()
        channel_rows = self._channel_rows_by_cycle_index()
        for idx, cfg in enumerate(self._cycles):
            rr = reg_rows[idx]
            cy = int(rr["cycle"])
            enabled = bool(rr["enabled"])
            if enabled:
                if cy in seen_cycles:
                    raise ValueError(f"Duplicate cycle number: {cy}. Cycle numbers must be unique among enabled cycles.")
                seen_cycles.add(cy)
            reg_marker = str(rr["registration_marker"] or "").strip()
            if enabled:
                if not reg_marker:
                    raise ValueError(f"Registration channel is required for enabled cycle {cy}.")
                valid_channels = {str(v).strip().lower() for v in cfg.channel_names}
                if reg_marker.lower() not in valid_channels:
                    raise ValueError(
                        f"Registration channel {reg_marker!r} is not present in cycle {cy} ({Path(cfg.path).name})."
                    )

            markers, antibodies = channel_rows.get(idx, ([], []))

            cycles_out.append(
                {
                    "path": cfg.path,
                    "cycle": int(cy),
                    "enabled": bool(enabled),
                    "registration_marker": reg_marker,
                    "channel_markers": list(markers),
                    "channel_antibodies": list(antibodies),
                }
            )

        return {
            "tissue": tissue,
            "species": species,
            "output_path": out_path,
            "registration_algorithm": reg_algorithm,
            "global_translation_only": bool(global_translation_only),
            "tiled_rigid_allow_rotation": tiled_allow_rotation,
            "tiled_rigid_tile_size": int(tiled_tile_size),
            "tiled_rigid_search_factor": float(tiled_search_factor),
            "fast_large_island_refinement": bool(fast_large_island),
            "fast_large_island_sample_count": int(fast_sample_count),
            "pyramidal_output": bool(self.chk_pyramidal_output.isChecked()),
            "low_mem": bool(self.chk_low_mem.isChecked()),
            "strip_height": int(self.spn_strip_height.value()) if self.spn_strip_height.value() > 0 else None,
            "elastic_touchup": bool(self.chk_elastic_touchup.isChecked()),
            "elastic_touchup_bspline_spacing": int(self.spn_bspline_spacing.value()),
            "elastic_touchup_max_iterations": int(self.spn_max_iterations.value()),
            "cycles": cycles_out,
        }
