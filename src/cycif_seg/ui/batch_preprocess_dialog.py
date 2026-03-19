from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets

from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff
from cycif_seg.io.ome_tiff import load_channel_names_only_fast
from cycif_seg.ui.merge_cycles_dialog import MergeRegisterCyclesDialog


PLAN_SCHEMA_VERSION = 1

default_tiled_rigid_tile_size: int = 1000
default_tiled_rigid_search_factor: float = 5.0

def _parse_cycle_number_from_filename(name: str) -> int | None:
    """Parse cycle number from stitched file names like ``C3_sample_...ome.tiff``."""
    try:
        import re

        m = re.match(r"^c(\d+)_", str(name or "").strip(), flags=re.IGNORECASE)
        if not m:
            return None
        return int(m.group(1))
    except Exception:
        return None


def _find_stitched_cycle_files_in_sample_dir(sample_dir: Path) -> list[tuple[Path, int]]:
    """
    Discover stitched cycle OME-TIFFs using the expected layout::

        {input}/{sample_dir}/{cycle_dir}/C#_{stuff}.ome.tiff

    Rules:
    - only inspect immediate subdirectories of ``sample_dir`` as cycle dirs
    - ignore any OME-TIFF directly under ``sample_dir``
    - ignore tile files such as ``area_*.ome.tiff``
    - infer cycle number from the stitched filename prefix ``C#_``
    """
    out: list[tuple[Path, int]] = []
    try:
        cycle_dirs = [p for p in sorted(sample_dir.iterdir()) if p.is_dir()]
    except Exception:
        return out

    for cycle_dir in cycle_dirs:
        try:
            files = [p for p in sorted(cycle_dir.iterdir()) if p.is_file()]
        except Exception:
            continue
        for p in files:
            nm = p.name
            low = nm.lower()
            if ".ome." not in low:
                continue
            if not (low.endswith('.ome.tif') or low.endswith('.ome.tiff')):
                continue
            if low.startswith('area_'):
                continue
            cy = _parse_cycle_number_from_filename(nm)
            if cy is None:
                continue
            out.append((p, int(cy)))

    out.sort(key=lambda t: (int(t[1]), str(t[0]).lower()))
    return out


@dataclass
class BatchSample:
    name: str
    input_dir: Path
    files: list[Path]
    tissue: str = ""
    species: str = ""
    output_path: Path | None = None
    enabled: bool = True

    # Per-file config (same order as files)
    cycles: list[int] | None = None
    registration_markers: list[str] | None = None
    channel_markers: list[list[str]] | None = None
    channel_antibodies: list[list[str]] | None = None
    cycles_enabled: list[bool] | None = None
    registration_algorithm: str = "tiled_rigid"
    global_translation_only: bool = False
    tiled_rigid_allow_rotation: bool = False
    tiled_rigid_tile_size: int = default_tiled_rigid_tile_size
    tiled_rigid_search_factor: float = default_tiled_rigid_search_factor
    pyramidal_output: bool = True


class BatchPreprocessDialog(QtWidgets.QDialog):
    """Batch Step 1 preprocessing: merge/register cycles for many samples."""

    # Thread-safe UI update signals (emitted from worker threads).
    sig_set_status = QtCore.Signal(str)
    sig_set_cycle_progress = QtCore.Signal(int, int)  # (idx, n)
    sig_set_sample_progress = QtCore.Signal(int, int)  # (idx, n)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch preprocess (Step 1)")
        self.setModal(True)

        self._samples: list[BatchSample] = []
        self._template_cfg: dict | None = None
        self._cancel_requested: bool = False
        self._refreshing_table: bool = False
        self._channel_name_cache: dict[str, list[str]] = {}

        root = QtWidgets.QVBoxLayout(self)

        # Folder selectors
        row = QtWidgets.QGridLayout()
        self.txt_root = QtWidgets.QLineEdit()
        self.txt_outdir = QtWidgets.QLineEdit()
        self.btn_pick_root = QtWidgets.QPushButton("Select input folder…")
        self.btn_pick_out = QtWidgets.QPushButton("Select output folder…")
        self.btn_scan = QtWidgets.QPushButton("Scan samples")

        row.addWidget(QtWidgets.QLabel("Batch input folder:"), 0, 0)
        row.addWidget(self.txt_root, 0, 1)
        row.addWidget(self.btn_pick_root, 0, 2)
        row.addWidget(QtWidgets.QLabel("Batch output folder:"), 1, 0)
        row.addWidget(self.txt_outdir, 1, 1)
        row.addWidget(self.btn_pick_out, 1, 2)
        row.addWidget(self.btn_scan, 0, 3, 2, 1)
        row.setColumnStretch(1, 1)
        root.addLayout(row)

        # Defaults
        defaults = QtWidgets.QHBoxLayout()
        self.txt_default_tissue = QtWidgets.QLineEdit()
        self.txt_default_species = QtWidgets.QLineEdit()
        self.btn_apply_defaults = QtWidgets.QPushButton("Apply tissue/species to all")
        defaults.addWidget(QtWidgets.QLabel("Default tissue:"))
        defaults.addWidget(self.txt_default_tissue)
        defaults.addWidget(QtWidgets.QLabel("Default species:"))
        defaults.addWidget(self.txt_default_species)
        defaults.addWidget(self.btn_apply_defaults)
        defaults.addStretch(1)
        root.addLayout(defaults)

        # Samples table
        self.tbl = QtWidgets.QTableWidget()
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels([
            "Run",
            "Sample",
            "# cycle files",
            "Tissue",
            "Species",
            "Output .ome.tiff",
        ])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        root.addWidget(self.tbl, 1)

        # Per-cycle config controls
        cfg_row = QtWidgets.QHBoxLayout()
        self.btn_config_selected = QtWidgets.QPushButton("Configure cycles for selected sample…")
        self.btn_apply_template = QtWidgets.QPushButton("Copy cycle config to all samples")
        cfg_row.addWidget(self.btn_config_selected)
        cfg_row.addWidget(self.btn_apply_template)
        cfg_row.addStretch(1)
        root.addLayout(cfg_row)

        # Plan save/load
        plan_row = QtWidgets.QHBoxLayout()
        self.btn_save_plan = QtWidgets.QPushButton("Save plan JSON…")
        self.btn_load_plan = QtWidgets.QPushButton("Load plan JSON…")
        plan_row.addWidget(self.btn_save_plan)
        plan_row.addWidget(self.btn_load_plan)
        plan_row.addStretch(1)
        root.addLayout(plan_row)

        # Progress bars
        root.addWidget(QtWidgets.QLabel("Progress:"))
        self.prog_samples = QtWidgets.QProgressBar()
        self.prog_cycles = QtWidgets.QProgressBar()
        self.prog_samples.setVisible(False)
        self.prog_cycles.setVisible(False)
        root.addWidget(self.prog_samples)
        root.addWidget(self.prog_cycles)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # Bottom buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.btn_run = buttons.addButton("Run batch", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_stop = buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_close = buttons.addButton("Close", QtWidgets.QDialogButtonBox.RejectRole)
        root.addWidget(buttons)

        # Signals
        self.btn_pick_root.clicked.connect(self._pick_root)
        self.btn_pick_out.clicked.connect(self._pick_outdir)
        self.btn_scan.clicked.connect(self._scan)
        self.txt_root.textChanged.connect(self._update_scan_button_enabled)
        self.txt_outdir.textChanged.connect(self._on_output_dir_changed)
        self.btn_apply_defaults.clicked.connect(self._apply_defaults)
        self.btn_config_selected.clicked.connect(self._configure_selected)
        self.btn_apply_template.clicked.connect(self._apply_template_to_all)
        self.btn_save_plan.clicked.connect(self._save_plan)
        self.btn_load_plan.clicked.connect(self._load_plan)
        self.btn_run.clicked.connect(self._run_batch)
        self.btn_stop.clicked.connect(self._request_cancel)
        self.btn_close.clicked.connect(self.reject)

        # Signals -> UI slots
        self.sig_set_status.connect(self._apply_status)
        self.sig_set_cycle_progress.connect(self._apply_cycle_progress)
        self.sig_set_sample_progress.connect(self._apply_sample_progress)

        self.btn_stop.setEnabled(False)
        self._update_scan_button_enabled()

        # Keep model state synchronized with direct table edits, especially the
        # output-path column. This avoids losing the current in-place editor
        # contents if the user edits a cell and immediately runs the batch.
        self.tbl.itemChanged.connect(self._on_table_item_changed)

    # -------------------- UI helpers --------------------
    def _apply_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _set_status(self, msg: str) -> None:
        # Local helper for main-thread calls.
        self._apply_status(msg)

    def _apply_cycle_progress(self, idx: int, n: int) -> None:
        try:
            if n > 0:
                self.prog_cycles.setRange(0, n)
                self.prog_cycles.setValue(max(0, min(int(idx), int(n))))
        except Exception:
            pass

    def _apply_sample_progress(self, idx: int, n: int) -> None:
        try:
            if n > 0:
                self.prog_samples.setRange(0, n)
                self.prog_samples.setValue(max(0, min(int(idx), int(n))))
        except Exception:
            pass

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self.btn_stop.setEnabled(False)
        self._update_scan_button_enabled()
        self._set_status("Cancel requested… finishing current step.")

    def _pick_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch input folder", self.txt_root.text() or str(Path.cwd()))
        if d:
            self.txt_root.setText(d)

    def _pick_outdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch output folder", self.txt_outdir.text() or str(Path.cwd()))
        if d:
            self.txt_outdir.setText(d)

    def _update_scan_button_enabled(self) -> None:
        try:
            has_root = bool(self.txt_root.text().strip())
            has_out = bool(self.txt_outdir.text().strip())
            self.btn_scan.setEnabled(has_root and has_out)
        except Exception:
            pass

    def _rebase_sample_output_paths(self, new_outdir: str | Path) -> None:
        try:
            out_dir = Path(str(new_outdir or "").strip()).expanduser()
        except Exception:
            return
        if not str(out_dir):
            return
        for s in self._samples:
            try:
                current_name = Path(str(s.output_path)).name if s.output_path else f"{s.name}.ome.tiff"
                if not current_name:
                    current_name = f"{s.name}.ome.tiff"
                s.output_path = out_dir / current_name
            except Exception:
                continue

    def _on_output_dir_changed(self, text: str) -> None:
        self._update_scan_button_enabled()
        self._rebase_sample_output_paths(text)
        if self._samples:
            self._refresh_table()

    def _scan(self) -> None:
        root_dir = Path(self.txt_root.text().strip() or "").expanduser()
        out_dir = Path(self.txt_outdir.text().strip() or "").expanduser()
        if not root_dir.is_dir():
            show_warning("Please select a valid batch input folder.")
            return
        if not out_dir:
            show_warning("Please select a batch output folder.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        samples: list[BatchSample] = []
        n_cycle_files = 0
        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir():
                continue
            stitched = _find_stitched_cycle_files_in_sample_dir(sub)
            if not stitched:
                continue
            files = [p for (p, _cy) in stitched]
            cycles = [int(cy) for (_p, cy) in stitched]
            sname = sub.name
            n_cycle_files += len(files)
            samples.append(
                BatchSample(
                    name=sname,
                    input_dir=sub,
                    files=files,
                    tissue=self.txt_default_tissue.text().strip(),
                    species=self.txt_default_species.text().strip(),
                    output_path=(out_dir / f"{sname}.ome.tiff"),
                    enabled=True,
                    cycles=cycles,
                    registration_algorithm="tiled_rigid",
                    global_translation_only=False,
                )
            )

        self._samples = samples
        self._refresh_table()
        self._set_status(
            f"Found {len(samples)} sample(s) with {n_cycle_files} stitched cycle file(s) "
            f"under sample/cycle directories."
        )

    def _refresh_table(self) -> None:
        self._refreshing_table = True
        try:
            self.tbl.setRowCount(len(self._samples))
            for r, s in enumerate(self._samples):
                # Run checkbox
                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.Checked if s.enabled else QtCore.Qt.Unchecked)
                self.tbl.setItem(r, 0, chk)

                it_name = QtWidgets.QTableWidgetItem(s.name)
                it_name.setFlags(it_name.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 1, it_name)

                it_n = QtWidgets.QTableWidgetItem(str(len(s.files)))
                it_n.setFlags(it_n.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 2, it_n)

                self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(s.tissue or ""))
                self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(s.species or ""))
                out_name = Path(str(s.output_path)).name if s.output_path else ""
                self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(out_name))

            try:
                self.tbl.resizeColumnsToContents()
            except Exception:
                pass
        finally:
            self._refreshing_table = False

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing_table:
            return
        try:
            r = int(item.row())
            c = int(item.column())
        except Exception:
            return
        if r < 0 or r >= len(self._samples):
            return
        s = self._samples[r]
        try:
            if c == 0:
                s.enabled = (item.checkState() == QtCore.Qt.Checked)
            elif c == 3:
                s.tissue = (item.text() or "").strip()
            elif c == 4:
                s.species = (item.text() or "").strip()
            elif c == 5:
                op_name = Path((item.text() or "").strip()).name
                out_dir = Path(self.txt_outdir.text().strip() or "").expanduser() if self.txt_outdir.text().strip() else None
                s.output_path = ((out_dir / op_name) if (out_dir and op_name) else None)
        except Exception:
            return

    def _apply_defaults(self) -> None:
        t = self.txt_default_tissue.text().strip()
        sp = self.txt_default_species.text().strip()
        for s in self._samples:
            s.tissue = t
            s.species = sp
        self._refresh_table()

    def _selected_row(self) -> int | None:
        try:
            rows = self.tbl.selectionModel().selectedRows()
            if not rows:
                return None
            return int(rows[0].row())
        except Exception:
            return None

    def _configure_selected(self) -> None:
        r = self._selected_row()
        if r is None or r < 0 or r >= len(self._samples):
            show_warning("Select a sample row first.")
            return
        s = self._samples[r]
        if not s.files:
            show_warning("Selected sample has no cycle files.")
            return

        # If this sample already has configuration (from plan JSON or prior edits),
        # re-open the dialog with those settings prefilled.
        initial_cfg = self._sample_to_cfg(s)

        # Loading channel names from many OME-TIFFs can take a moment. Do it in a worker
        # with a progress dialog so the UI doesn't feel frozen.
        ch_cache: dict[str, list[str]] = {}

        prog = QtWidgets.QProgressDialog("Reading cycle channel metadata…", "Cancel", 0, max(1, len(s.files)), self)
        prog.setWindowTitle("Loading cycles…")
        prog.setWindowModality(QtCore.Qt.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        cancel_flag = {"cancel": False}
        try:
            prog.canceled.connect(lambda: cancel_flag.__setitem__("cancel", True))
        except Exception:
            pass

        @thread_worker
        def _load_channels():
            out: dict[str, list[str]] = {}
            for i, p in enumerate(s.files, start=1):
                try:
                    out[str(p)] = list(self._channel_name_cache.get(str(p)) or load_channel_names_only_fast(str(p)) or [])
                    self._channel_name_cache[str(p)] = list(out[str(p)])
                except Exception:
                    out[str(p)] = []
                self.sig_set_cycle_progress.emit(int(i), int(len(s.files)))
                self.sig_set_status.emit(f"Loaded channels for {p.name} ({i}/{len(s.files)})")
                if bool(cancel_flag.get("cancel")):
                    raise RuntimeError("Cancelled")
            return out

        # Reuse the existing per-cycle progress bar signal for this short step.
        self.sig_set_cycle_progress.emit(0, int(len(s.files)))

        w = _load_channels()

        def _open_dialog(cache: dict[str, list[str]]):
            try:
                prog.close()
            except Exception:
                pass
            nonlocal ch_cache
            ch_cache = dict(cache or {})
            dlg = MergeRegisterCyclesDialog(self, paths=[str(p) for p in s.files], initial_cfg=initial_cfg, channel_name_cache=ch_cache)

            # Default output in dialog is ignored for batch, but keeping it sensible is nice.
            try:
                dlg.set_default_output(str(s.output_path or "merged.ome.tiff"))
            except Exception:
                pass
            # Seed global metadata
            try:
                dlg.txt_tissue.setText(s.tissue or "")
                dlg.txt_species.setText(s.species or "")
            except Exception:
                pass

            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return

            cfg = dlg.get_result()
            self._template_cfg = cfg
            # Apply to this sample immediately
            self._apply_cfg_to_sample(s, cfg)
            self._refresh_table()
            self._set_status(f"Stored cycle configuration template from sample '{s.name}'.")

        def _open_err(e):
            try:
                prog.close()
            except Exception:
                pass
            msg = str(e)
            if "Cancelled" in msg:
                self._set_status("Cycle metadata loading cancelled.")
            else:
                show_warning(f"Failed to read cycle metadata: {e}")
                self._set_status(f"Failed to read cycle metadata: {e}")

        w.returned.connect(_open_dialog)
        w.errored.connect(_open_err)
        w.start()
        return

        # Default output in dialog is ignored for batch, but keeping it sensible is nice.
        try:
            dlg.set_default_output(str(s.output_path or "merged.ome.tiff"))
        except Exception:
            pass

        # Seed global metadata
        try:
            dlg.txt_tissue.setText(s.tissue or "")
            dlg.txt_species.setText(s.species or "")
        except Exception:
            pass

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        cfg = dlg.get_result()
        self._template_cfg = cfg
        # Apply to this sample immediately
        self._apply_cfg_to_sample(s, cfg)
        self._refresh_table()
        self._set_status(f"Stored cycle configuration template from sample '{s.name}'.")

    def _sample_to_cfg(self, s: BatchSample) -> dict | None:
        """Build an initial_cfg dict (MergeRegisterCyclesDialog-compatible) from a BatchSample."""
        try:
            if not s.files:
                return None
            cycles_out: list[dict] = []
            for i, p in enumerate(s.files):
                reg_markers = list(s.registration_markers or [])
                channel_markers = list(s.channel_markers or [])
                channel_antibodies = list(s.channel_antibodies or [])
                cycles_out.append(
                    {
                        "path": str(p),
                        "cycle": int(s.cycles[i] if s.cycles and i < len(s.cycles) else i),
                        "enabled": bool(s.cycles_enabled[i] if s.cycles_enabled and i < len(s.cycles_enabled) else True),
                        "registration_marker": str(reg_markers[i] if i < len(reg_markers) else ""),
                        "channel_markers": list(channel_markers[i] if i < len(channel_markers) else []),
                        "channel_antibodies": list(channel_antibodies[i] if i < len(channel_antibodies) else []),
                    }
                )
            return {
                "tissue": s.tissue,
                "species": s.species,
                "output_path": str(s.output_path or ""),
                "registration_algorithm": str(getattr(s, 'registration_algorithm', 'tiled_rigid') or 'tiled_rigid'),
                "global_translation_only": bool(getattr(s, 'global_translation_only', False)),
                "tiled_rigid_allow_rotation": bool(getattr(s, 'tiled_rigid_allow_rotation', False)),
                "tiled_rigid_tile_size": int(getattr(s, 'tiled_rigid_tile_size', default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size),
                "tiled_rigid_search_factor": float(getattr(s, 'tiled_rigid_search_factor', default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor),
                "pyramidal_output": bool(getattr(s, 'pyramidal_output', True)),
                "cycles": cycles_out,
            }
        except Exception:
            return None

    def _apply_cfg_to_sample(self, s: BatchSample, cfg: dict, preserve_paths: bool = False) -> None:
        # Update global metadata from the dialog/plan.
        try:
            if 'tissue' in cfg:
                s.tissue = str(cfg.get('tissue') or '').strip()
            if 'species' in cfg:
                s.species = str(cfg.get('species') or '').strip()
            if (not preserve_paths) and ('output_path' in cfg):
                op = str(cfg.get('output_path') or '').strip()
                s.output_path = Path(op).expanduser() if op else None
            gto = cfg.get('global_translation_only')
            if gto is None:
                gto = (str(cfg.get('registration_algorithm') or '').strip().lower() == 'translation')
            s.global_translation_only = bool(gto)
            s.registration_algorithm = 'translation' if s.global_translation_only else (str(cfg.get('registration_algorithm') or s.registration_algorithm or 'tiled_rigid').strip() or 'tiled_rigid')
            s.tiled_rigid_allow_rotation = bool(cfg.get('tiled_rigid_allow_rotation') if cfg.get('tiled_rigid_allow_rotation') is not None else getattr(s, 'tiled_rigid_allow_rotation', False))
            s.tiled_rigid_tile_size = max(128, int(cfg.get('tiled_rigid_tile_size') if cfg.get('tiled_rigid_tile_size') is not None else getattr(s, 'tiled_rigid_tile_size', default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size))
            s.tiled_rigid_search_factor = max(1.0, float(cfg.get('tiled_rigid_search_factor') if cfg.get('tiled_rigid_search_factor') is not None else getattr(s, 'tiled_rigid_search_factor', default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor))
            s.pyramidal_output = bool(cfg.get('pyramidal_output') if cfg.get('pyramidal_output') is not None else getattr(s, 'pyramidal_output', True))
        except Exception:
            pass

        cycles_cfg = cfg.get("cycles") or []
        # When applying a copied template across samples, preserve each sample's
        # own input paths by matching cycles only by position. For normal config
        # application (e.g. loading a saved plan for the same sample), allow
        # matching by basename first and then fall back to order.
        by_base = {} if preserve_paths else {Path(d.get("path") or "").name: d for d in cycles_cfg}

        cycles: list[int] = []
        enableds: list[bool] = []
        reg: list[str] = []
        chm: list[list[str]] = []
        cha: list[list[str]] = []
        for i, p in enumerate(s.files):
            d = by_base.get(p.name) if by_base else None
            if d is None and i < len(cycles_cfg):
                d = cycles_cfg[i]
            if d is None:
                continue
            # Note: cycle 0 is valid; only default when missing/None.
            cycles.append(int(d.get("cycle") if d.get("cycle") is not None else i))
            enableds.append(bool(d.get("enabled", True)))
            reg.append(str(d.get("registration_marker") or "").strip())
            chm.append(list(d.get("channel_markers") or []))
            cha.append(list(d.get("channel_antibodies") or []))

        if cycles:
            s.cycles = cycles
            s.registration_markers = reg
            s.channel_markers = chm
            s.channel_antibodies = cha
            s.cycles_enabled = enableds

    def _apply_template_to_all(self) -> None:
        if not self._template_cfg:
            show_warning("No template configured yet. Use 'Configure cycles…' first.")
            return
        for s in self._samples:
            self._apply_cfg_to_sample(s, self._template_cfg, preserve_paths=True)
        self._refresh_table()
        self._set_status("Copied cycle configuration to all samples (preserved sample input/output paths).")

    # -------------------- Plan I/O --------------------
    def _plan_dict(self) -> dict:
        root_dir = str(Path(self.txt_root.text().strip() or "").expanduser())
        out_dir = str(Path(self.txt_outdir.text().strip() or "").expanduser())
        d: dict = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "root_dir": root_dir,
            "output_dir": out_dir,
            "defaults": {
                "registration_algorithm": "tiled_rigid",
                "global_translation_only": False,
                "tiled_rigid_tile_size": default_tiled_rigid_tile_size,
                "tiled_rigid_search_factor": default_tiled_rigid_search_factor,
                "tissue": self.txt_default_tissue.text().strip(),
                "species": self.txt_default_species.text().strip(),
            },
            "samples": [],
            "registration_algorithm": "tiled_rigid",
            "global_translation_only": False,
        }
        for s in self._samples:
            rec = {
                "name": s.name,
                "input_dir": str(s.input_dir),
                "files": [str(p) for p in s.files],
                "tissue": s.tissue,
                "species": s.species,
                "output_path": str(s.output_path or ""),
                "enabled": bool(s.enabled),
                "cycles": list(s.cycles or []),
                "registration_markers": list(s.registration_markers or []),
                "channel_markers": list(s.channel_markers or []),
                "channel_antibodies": list(s.channel_antibodies or []),
                "cycles_enabled": list(s.cycles_enabled or []),
                "registration_algorithm": str(getattr(s, "registration_algorithm", "tiled_rigid") or "tiled_rigid"),
                "global_translation_only": bool(getattr(s, "global_translation_only", False)),
                "tiled_rigid_allow_rotation": bool(getattr(s, "tiled_rigid_allow_rotation", False)),
                "tiled_rigid_tile_size": int(getattr(s, "tiled_rigid_tile_size", default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size),
                "tiled_rigid_search_factor": float(getattr(s, "tiled_rigid_search_factor", default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor),
                "pyramidal_output": bool(getattr(s, "pyramidal_output", True)),
            }
            d["samples"].append(rec)
        return d

    def _save_plan(self) -> None:
        # Ensure any table edits are captured before writing JSON.
        self._sync_table_to_models()
        start = self.txt_outdir.text().strip() or self.txt_root.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save batch plan JSON", os.path.join(start, "batch_plan.json"), "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        d = self._plan_dict()
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")
        self._set_status(f"Saved plan: {path}")

    def _load_plan(self) -> None:
        start = self.txt_root.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load batch plan JSON", start, "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(d.get("schema_version") or 0) != PLAN_SCHEMA_VERSION:
            show_warning(f"Unsupported plan schema version: {d.get('schema_version')}")
            return
        self.txt_root.setText(str(d.get("root_dir") or ""))
        self.txt_outdir.setText(str(d.get("output_dir") or ""))
        defs = d.get("defaults") or {}
        self.txt_default_tissue.setText(str(defs.get("tissue") or ""))
        self.txt_default_species.setText(str(defs.get("species") or ""))

        samples: list[BatchSample] = []
        for rec in d.get("samples") or []:
            s = BatchSample(
                name=str(rec.get("name") or ""),
                input_dir=Path(str(rec.get("input_dir") or "")).expanduser(),
                files=[Path(p).expanduser() for p in (rec.get("files") or [])],
                tissue=str(rec.get("tissue") or ""),
                species=str(rec.get("species") or ""),
                output_path=Path(str(rec.get("output_path") or "")).expanduser() if rec.get("output_path") else None,
                enabled=bool(rec.get("enabled", True)),
                registration_algorithm=str(rec.get("registration_algorithm") or defs.get("registration_algorithm") or "tiled_rigid"),
                global_translation_only=bool(rec.get("global_translation_only", defs.get("global_translation_only", False))),
            )
            s.global_translation_only = bool(rec.get("global_translation_only", defs.get("global_translation_only", False)))
            if s.global_translation_only:
                s.registration_algorithm = "translation"
            s.tiled_rigid_allow_rotation = bool(rec.get("tiled_rigid_allow_rotation", False))
            s.tiled_rigid_tile_size = max(128, int(rec.get("tiled_rigid_tile_size", defs.get("tiled_rigid_tile_size", default_tiled_rigid_tile_size)) or default_tiled_rigid_tile_size))
            s.tiled_rigid_search_factor = max(1.0, float(rec.get("tiled_rigid_search_factor", defs.get("tiled_rigid_search_factor", default_tiled_rigid_search_factor)) or default_tiled_rigid_search_factor))
            s.pyramidal_output = bool(rec.get("pyramidal_output", True))
            s.cycles = list(rec.get("cycles") or []) or None
            s.registration_markers = list(rec.get("registration_markers") or []) or None
            s.channel_markers = list(rec.get("channel_markers") or []) or None
            s.channel_antibodies = list(rec.get("channel_antibodies") or []) or None
            s.cycles_enabled = list(rec.get("cycles_enabled") or []) or None
            samples.append(s)

        self._samples = samples
        self._rebase_sample_output_paths(self.txt_outdir.text().strip())
        self._refresh_table()
        self._update_scan_button_enabled()
        self._set_status(f"Loaded plan: {path}")

    # -------------------- Running --------------------
    def _sync_table_to_models(self) -> None:
        # Force any active in-place editor to commit before we read table values.
        try:
            self.tbl.clearFocus()
            self.tbl.viewport().setFocus()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        # Pull edits back from the table into self._samples.
        for r, s in enumerate(self._samples):
            try:
                it_run = self.tbl.item(r, 0)
                s.enabled = (it_run.checkState() == QtCore.Qt.Checked) if it_run else True
                s.tissue = (self.tbl.item(r, 3).text() if self.tbl.item(r, 3) else "").strip()
                s.species = (self.tbl.item(r, 4).text() if self.tbl.item(r, 4) else "").strip()
                op_name = Path((self.tbl.item(r, 5).text() if self.tbl.item(r, 5) else "").strip()).name
                out_dir = Path(self.txt_outdir.text().strip() or "").expanduser() if self.txt_outdir.text().strip() else None
                s.output_path = ((out_dir / op_name) if (out_dir and op_name) else None)
            except Exception:
                continue

    def _validate_ready(self) -> tuple[bool, str]:
        self._sync_table_to_models()

        enabled = [s for s in self._samples if s.enabled]
        if not enabled:
            return False, "No samples selected."
        for s in enabled:
            if not s.output_path:
                return False, f"Sample '{s.name}' missing output path."
            if not s.files:
                return False, f"Sample '{s.name}' has no input files."
            if not s.cycles or not s.registration_markers or not s.channel_markers or not s.channel_antibodies:
                return False, f"Sample '{s.name}' missing per-cycle configuration. Use 'Configure cycles…' and optionally 'Copy to all'."
            if s.cycles_enabled and not any(bool(x) for x in s.cycles_enabled):
                return False, f"Sample '{s.name}' has all cycles disabled. Enable at least one cycle."
        return True, ""

    def _run_batch(self) -> None:
        ok, msg = self._validate_ready()
        if not ok:
            show_warning(msg)
            return

        enabled = [s for s in self._samples if s.enabled]
        n_samp = len(enabled)

        self.prog_samples.setVisible(True)
        self.prog_cycles.setVisible(True)
        self._apply_sample_progress(0, n_samp)
        self._apply_cycle_progress(0, 1)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_config_selected.setEnabled(False)
        self.btn_apply_template.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self._cancel_requested = False

        @thread_worker
        def _worker():
            reports: list[dict] = []
            for si, s in enumerate(enabled, start=1):
                if self._cancel_requested:
                    raise RuntimeError("Cancelled")

                # sample-level progress
                self.sig_set_sample_progress.emit(int(si - 1), int(n_samp))
                out_path = Path(str(s.output_path)).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Build CycleInput list
                cycles_in: list[CycleInput] = []
                for i, p in enumerate(s.files):
                    if s.cycles_enabled and i < len(s.cycles_enabled) and (not bool(s.cycles_enabled[i])):
                        continue
                    cycles_in.append(
                        CycleInput(
                            path=str(p),
                            cycle=int(s.cycles[i] if s.cycles and i < len(s.cycles) else i),
                            tissue=(s.tissue or None),
                            species=(s.species or None),
                            registration_marker=(s.registration_markers[i] if s.registration_markers and i < len(s.registration_markers) else None),
                            channel_markers=(s.channel_markers[i] if s.channel_markers and i < len(s.channel_markers) else None),
                            channel_antibodies=(s.channel_antibodies[i] if s.channel_antibodies and i < len(s.channel_antibodies) else None),
                        )
                    )

                # Reset per-sample cycle progress before starting.
                try:
                    self.sig_set_cycle_progress.emit(0, int(len(cycles_in) + 1))
                except Exception:
                    pass

                # UI hooks
                def _ev(ev: dict) -> None:
                    try:
                        n = int(ev.get("n") or 0)
                        idx = int(ev.get("idx") or 0)
                        msg = str(ev.get("msg") or "")
                    except Exception:
                        n, idx, msg = 0, 0, ""

                    if msg:
                        self.sig_set_status.emit(f"[{si}/{n_samp}] {s.name}: {msg}")
                    if n > 0:
                        self.sig_set_cycle_progress.emit(int(idx), int(n))

                # Run merge
                rep = merge_cycles_to_ome_tiff(
                    cycles_in,
                    str(out_path),
                    default_registration_marker="DAPI",
                    registration_algorithm=str(getattr(s, 'registration_algorithm', 'tiled_rigid') or 'tiled_rigid'),
                    global_translation_only=bool(getattr(s, 'global_translation_only', False)),
                    tiled_rigid_allow_rotation=bool(getattr(s, 'tiled_rigid_allow_rotation', False)),
                    tiled_rigid_tile_size=max(128, int(getattr(s, 'tiled_rigid_tile_size', default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size)),
                    tiled_rigid_search_factor=max(1.0, float(getattr(s, 'tiled_rigid_search_factor', default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor)),
                    pyramidal_output=bool(getattr(s, 'pyramidal_output', False)),
                    progress_event_cb=_ev,
                    cancel_cb=lambda: bool(self._cancel_requested),
                    low_mem=True,
                )
                rep["sample_name"] = s.name
                reports.append(rep)

                self.sig_set_sample_progress.emit(int(si), int(n_samp))
            return reports

        w = _worker()

        @w.returned.connect
        def _done(reports: list[dict]):
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_config_selected.setEnabled(True)
            self.btn_apply_template.setEnabled(True)
            self._update_scan_button_enabled()
            self._set_status(f"Batch complete: {len(reports)} sample(s) processed.")
            show_info(f"Batch complete: {len(reports)} sample(s) processed.")
            self.btn_run.setEnabled(True)
            self.btn_close.setText("Done")
            # stash result for parent
            self._reports = reports  # type: ignore[attr-defined]

        @w.errored.connect
        def _err(e):
            msg = str(e)
            if "Cancelled" in msg:
                self._set_status("Batch cancelled.")
                show_info("Batch cancelled.")
            else:
                show_warning(f"Batch failed: {e}")
                self._set_status(f"Batch failed: {e}")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_config_selected.setEnabled(True)
            self.btn_apply_template.setEnabled(True)
            self._update_scan_button_enabled()

        w.start()

    def get_reports(self) -> list[dict]:
        return list(getattr(self, "_reports", []) or [])
