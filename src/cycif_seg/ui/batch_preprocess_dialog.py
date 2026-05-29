from __future__ import annotations

import json
import os
import gc
from pathlib import Path

from qtpy import QtCore, QtWidgets

from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from cycif_seg.preprocess.organize_cycles import CycleInput, is_preprocess_debug, is_debug_elastic_touchup, merge_cycles_to_ome_tiff
from cycif_seg.io.ome_tiff import load_channel_names_only_fast
from cycif_seg.preprocess.batch_plan import (
    BatchSample,
    default_tiled_rigid_tile_size,
    default_tiled_rigid_search_factor,
    default_fast_large_island_refinement,
    default_fast_large_island_sample_count,
    scan_root_for_samples,
    validate_sample_cycle_numbers,
    sample_has_cycle_config,
    plan_to_dict,
    plan_from_dict,
)
from cycif_seg.ui.merge_cycles_dialog import MergeRegisterCyclesDialog


class BatchPreprocessDialog(QtWidgets.QDialog):
    """Batch Step 1 preprocessing: merge/register cycles for many samples."""

    # Thread-safe UI update signals (emitted from worker threads).
    sig_set_status = QtCore.Signal(str)
    sig_set_cycle_progress = QtCore.Signal(int, int)  # (idx, n)
    sig_set_sample_progress = QtCore.Signal(int, int)  # (idx, n)
    sig_set_sample_label = QtCore.Signal(str)
    sig_set_cycle_label = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch preprocess (Step 1)")
        self.setModal(True)

        self._samples: list[BatchSample] = []
        self._template_cfg: dict | None = None
        self._cancel_requested: bool = False
        self._running: bool = False
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
        self.tbl.setColumnCount(7)
        self.tbl.setHorizontalHeaderLabels([
            "Run",
            "Config",
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
        self.lbl_sample_progress = QtWidgets.QLabel("Samples: idle")
        self.lbl_cycle_progress = QtWidgets.QLabel("Current sample: idle")
        self.prog_samples = QtWidgets.QProgressBar()
        self.prog_cycles = QtWidgets.QProgressBar()
        self.prog_samples.setVisible(False)
        self.prog_cycles.setVisible(False)
        self.lbl_sample_progress.setVisible(False)
        self.lbl_cycle_progress.setVisible(False)
        root.addWidget(self.lbl_sample_progress)
        root.addWidget(self.prog_samples)
        root.addWidget(self.lbl_cycle_progress)
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
        try:
            self.tbl.selectionModel().selectionChanged.connect(lambda *_: self._update_action_buttons())
        except Exception:
            pass

        # Signals -> UI slots
        self.sig_set_status.connect(self._apply_status)
        self.sig_set_cycle_progress.connect(self._apply_cycle_progress)
        self.sig_set_sample_progress.connect(self._apply_sample_progress)
        self.sig_set_sample_label.connect(self.lbl_sample_progress.setText)
        self.sig_set_cycle_label.connect(self.lbl_cycle_progress.setText)

        self.btn_stop.setEnabled(False)
        self._update_scan_button_enabled()
        self._update_action_buttons()

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
                self.prog_cycles.setFormat(f"%v/%m")
        except Exception:
            pass

    def _apply_sample_progress(self, idx: int, n: int) -> None:
        try:
            if n > 0:
                self.prog_samples.setRange(0, n)
                self.prog_samples.setValue(max(0, min(int(idx), int(n))))
                self.prog_samples.setFormat(f"%v/%m")
        except Exception:
            pass

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self.btn_stop.setEnabled(False)
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
            self.btn_scan.setEnabled((not self._running) and has_root and has_out)
        except Exception:
            pass

    def _update_action_buttons(self) -> None:
        try:
            has_samples = bool(self._samples)
            has_selection = self._selected_row() is not None
            self.btn_config_selected.setEnabled((not self._running) and has_samples and has_selection)
            self.btn_apply_template.setEnabled((not self._running) and has_samples and bool(self._template_cfg))
            self.btn_save_plan.setEnabled(not self._running)
            self.btn_load_plan.setEnabled(not self._running)
            self.btn_apply_defaults.setEnabled(not self._running)
            self.btn_pick_root.setEnabled(not self._running)
            self.btn_pick_out.setEnabled(not self._running)
            self.txt_root.setEnabled(not self._running)
            self.txt_outdir.setEnabled(not self._running)
            self.txt_default_tissue.setEnabled(not self._running)
            self.txt_default_species.setEnabled(not self._running)
            self.tbl.setEnabled(not self._running)
            self.btn_run.setEnabled((not self._running) and has_samples)
            self.btn_stop.setEnabled(self._running and not self._cancel_requested)
            self.btn_close.setEnabled(not self._running)
        except Exception:
            pass
        self._update_scan_button_enabled()

    def _set_running_ui(self, running: bool) -> None:
        self._running = bool(running)
        self.lbl_sample_progress.setVisible(bool(running))
        self.lbl_cycle_progress.setVisible(bool(running))
        self.prog_samples.setVisible(bool(running))
        self.prog_cycles.setVisible(bool(running))
        self._update_action_buttons()

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

        samples = scan_root_for_samples(
            root_dir, out_dir,
            tissue=self.txt_default_tissue.text().strip(),
            species=self.txt_default_species.text().strip(),
        )
        n_cycle_files = sum(len(s.files) for s in samples)
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
            style = self.style()
            ok_icon = style.standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
            warn_icon = style.standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
            for r, s in enumerate(self._samples):
                # Run checkbox
                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.Checked if s.enabled else QtCore.Qt.Unchecked)
                self.tbl.setItem(r, 0, chk)

                cfg_item = QtWidgets.QTableWidgetItem("")
                cfg_item.setFlags(cfg_item.flags() & ~QtCore.Qt.ItemIsEditable)
                cfg_state, cfg_tip = self._sample_config_state(s)
                if cfg_state == "valid":
                    cfg_item.setIcon(ok_icon)
                    cfg_item.setToolTip(cfg_tip or "Cycle configuration is valid.")
                elif cfg_state == "invalid":
                    cfg_item.setIcon(warn_icon)
                    cfg_item.setToolTip(cfg_tip or "Cycle configuration is invalid.")
                else:
                    cfg_item.setToolTip("No cycle configuration for this sample.")
                self.tbl.setItem(r, 1, cfg_item)

                it_name = QtWidgets.QTableWidgetItem(s.name)
                it_name.setFlags(it_name.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 2, it_name)

                it_n = QtWidgets.QTableWidgetItem(str(len(s.files)))
                it_n.setFlags(it_n.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 3, it_n)

                self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(s.tissue or ""))
                self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(s.species or ""))
                out_name = Path(str(s.output_path)).name if s.output_path else ""
                self.tbl.setItem(r, 6, QtWidgets.QTableWidgetItem(out_name))

            try:
                self.tbl.resizeColumnsToContents()
            except Exception:
                pass
        finally:
            self._refreshing_table = False
        self._update_action_buttons()

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
            elif c == 4:
                s.tissue = (item.text() or "").strip()
            elif c == 5:
                s.species = (item.text() or "").strip()
            elif c == 6:
                op_name = Path((item.text() or "").strip()).name
                out_dir = Path(self.txt_outdir.text().strip() or "").expanduser() if self.txt_outdir.text().strip() else None
                s.output_path = ((out_dir / op_name) if (out_dir and op_name) else None)
        except Exception:
            return
        self._update_action_buttons()

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
                    # Keep batch cycle configuration on the fast OME-XML-only
                    # channel-name path; do not swap this back to the slower
                    # full TIFF inspection helpers.
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
            snapshot = self._cycle_config_snapshot(s)
            try:
                # Apply to this sample immediately. This also validates duplicate
                # enabled cycle numbers before the config can become the template.
                self._apply_cfg_to_sample(s, cfg)
            except Exception as e:
                self._restore_cycle_config_snapshot(s, snapshot)
                show_warning(str(e))
                self._set_status(f"Invalid cycle configuration for sample '{s.name}': {e}")
                return
            self._template_cfg = cfg
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
                "fast_large_island_refinement": False,
                "fast_large_island_sample_count": int(getattr(s, 'fast_large_island_sample_count', default_fast_large_island_sample_count) or default_fast_large_island_sample_count),
                "pyramidal_output": bool(getattr(s, 'pyramidal_output', True)),
                "low_mem": bool(getattr(s, 'low_mem', True)),
                "strip_height": getattr(s, 'strip_height', None),
                "elastic_touchup": bool(getattr(s, 'elastic_touchup', False)),
                "elastic_touchup_bspline_spacing": int(getattr(s, 'elastic_touchup_bspline_spacing', 50) or 50),
                "elastic_touchup_max_iterations": int(getattr(s, 'elastic_touchup_max_iterations', 100) or 100),
                "elastic_touchup_max_step_length": float(getattr(s, 'elastic_touchup_max_step_length', 1.0) or 1.0),
                "cycles": cycles_out,
            }
        except Exception:
            return None

    def _sample_config_state(self, s: BatchSample) -> tuple[str, str]:
        if not sample_has_cycle_config(s):
            return "missing", "No cycle configuration for this sample."
        if s.cycles_enabled and not any(bool(x) for x in s.cycles_enabled):
            return "invalid", "All cycles are disabled. Enable at least one cycle."
        ok, msg = validate_sample_cycle_numbers(s)
        if not ok:
            return "invalid", msg
        return "valid", "Cycle configuration is valid."

    def _cycle_config_snapshot(self, s: BatchSample) -> tuple:
        return (
            list(s.cycles or []) if s.cycles is not None else None,
            list(s.registration_markers or []) if s.registration_markers is not None else None,
            [list(x) for x in (s.channel_markers or [])] if s.channel_markers is not None else None,
            [list(x) for x in (s.channel_antibodies or [])] if s.channel_antibodies is not None else None,
            list(s.cycles_enabled or []) if s.cycles_enabled is not None else None,
        )

    def _restore_cycle_config_snapshot(self, s: BatchSample, snapshot: tuple) -> None:
        cycles, reg, chm, cha, enableds = snapshot
        s.cycles = cycles
        s.registration_markers = reg
        s.channel_markers = chm
        s.channel_antibodies = cha
        s.cycles_enabled = enableds

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
            s.fast_large_island_refinement = False
            s.fast_large_island_sample_count = max(1, int(cfg.get('fast_large_island_sample_count') if cfg.get('fast_large_island_sample_count') is not None else getattr(s, 'fast_large_island_sample_count', default_fast_large_island_sample_count) or default_fast_large_island_sample_count))
            s.pyramidal_output = bool(cfg.get('pyramidal_output') if cfg.get('pyramidal_output') is not None else getattr(s, 'pyramidal_output', True))
            s.low_mem = bool(cfg.get('low_mem') if cfg.get('low_mem') is not None else getattr(s, 'low_mem', True))
            _sh = cfg.get('strip_height') if 'strip_height' in cfg else getattr(s, 'strip_height', None)
            s.strip_height = int(_sh) if _sh is not None and int(_sh) > 0 else None
            if 'elastic_touchup' in cfg:
                s.elastic_touchup = bool(cfg.get('elastic_touchup'))
            if 'elastic_touchup_bspline_spacing' in cfg:
                s.elastic_touchup_bspline_spacing = max(4, int(cfg.get('elastic_touchup_bspline_spacing') or 50))
            if 'elastic_touchup_max_iterations' in cfg:
                s.elastic_touchup_max_iterations = max(1, int(cfg.get('elastic_touchup_max_iterations') or 100))
            if 'elastic_touchup_max_step_length' in cfg:
                s.elastic_touchup_max_step_length = max(0.01, float(cfg.get('elastic_touchup_max_step_length') or 1.0))
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
            ok, msg = validate_sample_cycle_numbers(s)
            if not ok:
                raise ValueError(msg)

    def _apply_template_to_all(self) -> None:
        if not self._template_cfg:
            show_warning("No template configured yet. Use 'Configure cycles…' first.")
            return
        copied = 0
        skipped: list[str] = []
        for s in self._samples:
            snapshot = self._cycle_config_snapshot(s)
            try:
                self._apply_cfg_to_sample(s, self._template_cfg, preserve_paths=True)
                copied += 1
            except Exception as e:
                self._restore_cycle_config_snapshot(s, snapshot)
                skipped.append(f"{s.name}: {e}")
        self._refresh_table()
        if skipped:
            print("Cycle configuration copy skipped invalid sample(s):", flush=True)
            for item in skipped:
                print(f"  - {item}", flush=True)
            show_warning(f"Copied config to {copied} sample(s); skipped {len(skipped)} invalid sample(s). See status/console for details.")
            self._set_status(f"Copied config to {copied} sample(s); skipped {len(skipped)} invalid sample(s).")
        else:
            self._set_status(f"Copied cycle configuration to {copied} sample(s) (preserved sample input/output paths).")
        self._update_action_buttons()

    # -------------------- Plan I/O --------------------
    def _plan_dict(self) -> dict:
        root_dir = Path(self.txt_root.text().strip() or "").expanduser()
        out_dir = Path(self.txt_outdir.text().strip() or "").expanduser()
        return plan_to_dict(
            self._samples,
            root_dir=root_dir,
            output_dir=out_dir,
            default_tissue=self.txt_default_tissue.text().strip(),
            default_species=self.txt_default_species.text().strip(),
        )

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
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            result = plan_from_dict(d)
        except ValueError as e:
            show_warning(str(e))
            self._set_status(f"Invalid loaded plan: {e}")
            return

        self.txt_root.setText(result["root_dir"])
        self.txt_outdir.setText(result["output_dir"])
        defs = result["defaults"]
        self.txt_default_tissue.setText(str(defs.get("tissue") or ""))
        self.txt_default_species.setText(str(defs.get("species") or ""))
        self._samples = result["samples"]
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
                s.tissue = (self.tbl.item(r, 4).text() if self.tbl.item(r, 4) else "").strip()
                s.species = (self.tbl.item(r, 5).text() if self.tbl.item(r, 5) else "").strip()
                op_name = Path((self.tbl.item(r, 6).text() if self.tbl.item(r, 6) else "").strip()).name
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
            ok, msg = validate_sample_cycle_numbers(s)
            if not ok:
                return False, msg
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
        self.lbl_sample_progress.setVisible(True)
        self.lbl_cycle_progress.setVisible(True)
        self._apply_sample_progress(0, n_samp)
        self._apply_cycle_progress(0, 1)
        self.lbl_sample_progress.setText(f"Samples: 0/{n_samp}")
        self.lbl_cycle_progress.setText("Current sample: waiting")

        self._cancel_requested = False
        self._reports = []  # type: ignore[attr-defined]
        self._failures = []  # type: ignore[attr-defined]
        self._set_running_ui(True)

        @thread_worker
        def _worker():
            reports: list[dict] = []
            failures: list[dict] = []
            for si, s in enumerate(enabled, start=1):
                if self._cancel_requested:
                    return {"reports": reports, "failures": failures, "cancelled": True}

                # sample-level progress
                self.sig_set_sample_progress.emit(int(si - 1), int(n_samp))
                self.sig_set_sample_label.emit(f"Samples: {si - 1}/{n_samp}")
                out_path = Path(str(s.output_path)).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Build CycleInput list
                cycles_in: list[CycleInput] = []
                try:
                    ok, msg = validate_sample_cycle_numbers(s)
                    if not ok:
                        raise ValueError(msg)
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
                except Exception as e:
                    failures.append({"sample_name": str(s.name), "error": str(e)})
                    self.sig_set_status.emit(f"[{si}/{n_samp}] {s.name}: failed: {e}")
                    self.sig_set_sample_progress.emit(int(si), int(n_samp))
                    self.sig_set_sample_label.emit(f"Samples: {si}/{n_samp}")
                    continue

                # Reset per-sample cycle progress before starting.
                sample_total = 1 + max(0, len(cycles_in) - 1) * 4 + len(cycles_in) + (1 if bool(getattr(s, 'pyramidal_output', False)) else 0)
                try:
                    self.sig_set_cycle_progress.emit(0, int(max(1, sample_total)))
                    self.sig_set_cycle_label.emit(f"Current sample: {s.name} (0/{max(1, sample_total)})")
                except Exception:
                    pass

                # UI hooks
                def _progress_label(ev: dict, idx: int, n: int) -> str:
                    phase = str(ev.get("phase") or "")
                    cycle = ev.get("cycle")
                    cycle_txt = f"Cycle {int(cycle)}" if cycle is not None else "cycle"
                    action_by_phase = {
                        "load_ref": f"loading reference {cycle_txt}",
                        "load_cycle": f"loading {cycle_txt}",
                        "global_registration": f"registering {cycle_txt}: global translation",
                        "foreground_mask": f"registering {cycle_txt}: foreground mask",
                        "identify_islands": f"registering {cycle_txt}: foreground islands",
                        "foreground_island_refine": f"registering {cycle_txt}: island refinement",
                        "elastic_touchup_island": f"elastic touch-up {cycle_txt}",
                        "write_cycle": f"writing {cycle_txt}",
                        "pyramid": "building pyramid",
                    }
                    action = action_by_phase.get(phase, "working")
                    return f"Current sample: {s.name} - {action} ({max(0, min(int(idx), int(n)))}/{int(n)})"

                def _ev(ev: dict) -> None:
                    try:
                        n = int(ev.get("n") or 0)
                        idx = int(ev.get("idx") or 0)
                        msg = str(ev.get("msg") or "")
                    except Exception:
                        n, idx, msg = 0, 0, ""

                    if msg:
                        self.sig_set_status.emit(f"[{si}/{n_samp}] {s.name}: {msg}")
                    phase = str(ev.get("phase") or "")
                    top_level_phases = {
                        "load_ref",
                        "load_cycle",
                        "global_registration",
                        "foreground_mask",
                        "identify_islands",
                        "foreground_island_refine",
                        "elastic_touchup_island",
                        "write_cycle",
                        "pyramid",
                    }
                    if n > 0 and phase in top_level_phases:
                        self.sig_set_cycle_progress.emit(int(idx), int(n))
                        self.sig_set_cycle_label.emit(_progress_label(ev, idx, n))

                # Run merge
                if is_preprocess_debug():
                    _sh_disp = str(getattr(s, 'strip_height', None) or 'auto')
                    print(
                        f"[batch preprocess] starting '{s.name}'"
                        f"  cycles={len(cycles_in)}"
                        f"  low_mem={getattr(s, 'low_mem', True)}"
                        f"  strip_height={_sh_disp}"
                        f"  output={out_path}",
                        flush=True,
                    )
                try:
                    rep = merge_cycles_to_ome_tiff(
                        cycles_in,
                        str(out_path),
                        default_registration_marker="DAPI",
                        registration_algorithm=str(getattr(s, 'registration_algorithm', 'tiled_rigid') or 'tiled_rigid'),
                        global_translation_only=bool(getattr(s, 'global_translation_only', False)),
                        tiled_rigid_allow_rotation=bool(getattr(s, 'tiled_rigid_allow_rotation', False)),
                        tiled_rigid_tile_size=max(128, int(getattr(s, 'tiled_rigid_tile_size', default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size)),
                        tiled_rigid_search_factor=max(1.0, float(getattr(s, 'tiled_rigid_search_factor', default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor)),
                        fast_large_island_refinement=False,
                        fast_large_island_sample_count=max(1, int(getattr(s, 'fast_large_island_sample_count', default_fast_large_island_sample_count) or default_fast_large_island_sample_count)),
                        elastic_touchup=bool(getattr(s, 'elastic_touchup', False)),
                        elastic_touchup_tile_size=max(64, int(getattr(s, 'elastic_touchup_tile_size', 1024) or 1024)),
                        elastic_touchup_skip_corr=float(getattr(s, 'elastic_touchup_skip_corr', 0.95) or 0.95),
                        elastic_touchup_bspline_spacing=max(4, int(getattr(s, 'elastic_touchup_bspline_spacing', 50) or 50)),
                        elastic_touchup_max_iterations=max(1, int(getattr(s, 'elastic_touchup_max_iterations', 100) or 100)),
                        elastic_touchup_large_island_px=max(1, int(getattr(s, 'elastic_touchup_large_island_px', 4_000_000) or 4_000_000)),
                        elastic_touchup_workers=max(0, int(getattr(s, 'elastic_touchup_workers', 0) or 0)),
                        elastic_touchup_max_step_length=max(0.01, float(getattr(s, 'elastic_touchup_max_step_length', 1.0) or 1.0)),
                        debug_elastic_touchup=is_debug_elastic_touchup(),
                        debug_dir=str(getattr(s, 'debug_dir') or '') or None,
                        pyramidal_output=bool(getattr(s, 'pyramidal_output', False)),
                        progress_event_cb=_ev,
                        cancel_cb=lambda: bool(self._cancel_requested),
                        low_mem=bool(getattr(s, 'low_mem', True)),
                        strip_height=getattr(s, 'strip_height', None),
                    )
                    if is_preprocess_debug():
                        print(
                            f"[batch preprocess] done '{s.name}'"
                            f"  strip_height_used={rep.get('strip_height')}"
                            f"  canvas={rep.get('canvas_shape_yx')}",
                            flush=True,
                        )
                    reports.append(
                        {
                            "sample_name": str(s.name),
                            "output_path": str(rep.get("output_path") or out_path),
                            "pyramidal_output_path": rep.get("pyramidal_output_path"),
                            "canvas_yx": tuple(rep.get("canvas_shape_yx") or ()),
                            "n_cycles": int(rep.get("n_cycles") or len(cycles_in)),
                            "n_channels_total": int(rep.get("n_channels_total") or 0),
                            "cycle_global_shifts": dict(rep.get("cycle_global_shifts") or {}),
                            "inputs": [
                                {"path": str(c.path), "cycle": int(c.cycle)}
                                for c in cycles_in
                            ],
                        }
                    )
                except Exception as e:
                    if self._cancel_requested:
                        return {"reports": reports, "failures": failures, "cancelled": True}
                    failures.append({"sample_name": str(s.name), "error": str(e)})
                    self.sig_set_status.emit(f"[{si}/{n_samp}] {s.name}: failed: {e}")
                finally:
                    try:
                        del cycles_in
                    except Exception:
                        pass
                    try:
                        del rep
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass

                self.sig_set_sample_progress.emit(int(si), int(n_samp))
                self.sig_set_sample_label.emit(f"Samples: {si}/{n_samp}")
                self.sig_set_cycle_progress.emit(0, 1)
                self.sig_set_cycle_label.emit("Current sample: waiting")
            return {"reports": reports, "failures": failures}

        w = _worker()

        @w.returned.connect
        def _done(result):
            result = result or {}
            reports = list(result.get("reports") or [])
            failures = list(result.get("failures") or [])
            cancelled = bool(result.get("cancelled", False))
            self._reports = reports  # type: ignore[attr-defined]
            self._failures = failures  # type: ignore[attr-defined]
            self._set_running_ui(False)
            self.lbl_sample_progress.setVisible(True)
            self.lbl_cycle_progress.setVisible(True)
            self.prog_samples.setVisible(True)
            self.prog_cycles.setVisible(True)
            self.prog_samples.setValue(self.prog_samples.maximum())
            self.prog_cycles.setValue(0)
            self.lbl_sample_progress.setText(f"Samples: {len(reports) + len(failures)}/{n_samp}")
            self.lbl_cycle_progress.setText("Current sample: cancelled" if cancelled else "Current sample: complete")
            if cancelled:
                self._set_status("Batch cancelled.")
                show_info("Batch cancelled.")
            elif failures:
                print("Batch preprocess completed with failed sample(s):", flush=True)
                for fail in failures:
                    try:
                        print(
                            f"  - {fail.get('sample_name', '<unknown sample>')}: {fail.get('error', '<unknown error>')}",
                            flush=True,
                        )
                    except Exception:
                        print(f"  - {fail!r}", flush=True)
                self._set_status(f"Batch complete: {len(reports)} sample(s) processed, {len(failures)} failed.")
                show_warning(f"Batch complete with {len(failures)} failed sample(s). See status for details.")
            else:
                self._set_status(f"Batch complete: {len(reports)} sample(s) processed.")
                show_info(f"Batch complete: {len(reports)} sample(s) processed.")
            self.btn_close.setText("Done")

        @w.errored.connect
        def _err(e):
            msg = str(e)
            self._set_running_ui(False)
            self.lbl_sample_progress.setVisible(True)
            self.lbl_cycle_progress.setVisible(True)
            self.prog_samples.setVisible(True)
            self.prog_cycles.setVisible(True)
            if "Cancelled" in msg:
                self._set_status("Batch cancelled.")
                show_info("Batch cancelled.")
            else:
                show_warning(f"Batch failed: {e}")
                self._set_status(f"Batch failed: {e}")

        w.start()

    def get_reports(self) -> list[dict]:
        return list(getattr(self, "_reports", []) or [])
