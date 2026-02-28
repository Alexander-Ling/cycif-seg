from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets

from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff
from cycif_seg.ui.merge_cycles_dialog import MergeRegisterCyclesDialog


PLAN_SCHEMA_VERSION = 1


def _find_ome_tiffs_in_dir(d: Path) -> list[Path]:
    exts = (".ome.tif", ".ome.tiff", ".tif", ".tiff")
    out: list[Path] = []
    try:
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            nm = p.name.lower()
            if any(nm.endswith(e) for e in exts) and (".ome." in nm):
                out.append(p)
    except Exception:
        pass
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
        self._set_status("Cancel requested… finishing current step.")

    def _pick_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch input folder", self.txt_root.text() or str(Path.cwd()))
        if d:
            self.txt_root.setText(d)

    def _pick_outdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch output folder", self.txt_outdir.text() or str(Path.cwd()))
        if d:
            self.txt_outdir.setText(d)

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
        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir():
                continue
            files = _find_ome_tiffs_in_dir(sub)
            if not files:
                continue
            sname = sub.name
            samples.append(
                BatchSample(
                    name=sname,
                    input_dir=sub,
                    files=files,
                    tissue=self.txt_default_tissue.text().strip(),
                    species=self.txt_default_species.text().strip(),
                    output_path=(out_dir / f"{sname}.ome.tiff"),
                    enabled=True,
                )
            )

        self._samples = samples
        self._refresh_table()
        self._set_status(f"Found {len(samples)} sample(s) with .ome.tiff files.")

    def _refresh_table(self) -> None:
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
            self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(str(s.output_path or "")))

        try:
            self.tbl.resizeColumnsToContents()
        except Exception:
            pass

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
        dlg = MergeRegisterCyclesDialog(self, paths=[str(p) for p in s.files], initial_cfg=initial_cfg)

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
            if not (s.cycles and s.registration_markers and s.channel_markers and s.channel_antibodies):
                return None
            cycles_out: list[dict] = []
            for i, p in enumerate(s.files):
                cycles_out.append(
                    {
                        "path": str(p),
                        "cycle": int(s.cycles[i] if i < len(s.cycles) else (i + 1)),
                        "registration_marker": str(s.registration_markers[i] if i < len(s.registration_markers) else ""),
                        "channel_markers": list(s.channel_markers[i] if i < len(s.channel_markers) else []),
                        "channel_antibodies": list(s.channel_antibodies[i] if i < len(s.channel_antibodies) else []),
                    }
                )
            return {
                "tissue": s.tissue,
                "species": s.species,
                "output_path": str(s.output_path or ""),
                "cycles": cycles_out,
            }
        except Exception:
            return None

    def _apply_cfg_to_sample(self, s: BatchSample, cfg: dict) -> None:
        cycles_cfg = cfg.get("cycles") or []
        # Map by file basename when possible, else fall back to order.
        by_base = {Path(d.get("path") or "").name: d for d in cycles_cfg}

        cycles: list[int] = []
        reg: list[str] = []
        chm: list[list[str]] = []
        cha: list[list[str]] = []
        for i, p in enumerate(s.files):
            d = by_base.get(p.name)
            if d is None and i < len(cycles_cfg):
                d = cycles_cfg[i]
            if d is None:
                continue
            cycles.append(int(d.get("cycle") or (i + 1)))
            reg.append(str(d.get("registration_marker") or "").strip())
            chm.append(list(d.get("channel_markers") or []))
            cha.append(list(d.get("channel_antibodies") or []))

        if cycles:
            s.cycles = cycles
            s.registration_markers = reg
            s.channel_markers = chm
            s.channel_antibodies = cha

    def _apply_template_to_all(self) -> None:
        if not self._template_cfg:
            show_warning("No template configured yet. Use 'Configure cycles…' first.")
            return
        for s in self._samples:
            self._apply_cfg_to_sample(s, self._template_cfg)
        self._refresh_table()
        self._set_status("Copied cycle configuration to all samples.")

    # -------------------- Plan I/O --------------------
    def _plan_dict(self) -> dict:
        root_dir = str(Path(self.txt_root.text().strip() or "").expanduser())
        out_dir = str(Path(self.txt_outdir.text().strip() or "").expanduser())
        d: dict = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "root_dir": root_dir,
            "output_dir": out_dir,
            "defaults": {
                "tissue": self.txt_default_tissue.text().strip(),
                "species": self.txt_default_species.text().strip(),
            },
            "samples": [],
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
            }
            d["samples"].append(rec)
        return d

    def _save_plan(self) -> None:
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
            )
            s.cycles = list(rec.get("cycles") or []) or None
            s.registration_markers = list(rec.get("registration_markers") or []) or None
            s.channel_markers = list(rec.get("channel_markers") or []) or None
            s.channel_antibodies = list(rec.get("channel_antibodies") or []) or None
            samples.append(s)

        self._samples = samples
        self._refresh_table()
        self._set_status(f"Loaded plan: {path}")

    # -------------------- Running --------------------
    def _sync_table_to_models(self) -> None:
        # Pull edits back from the table into self._samples.
        for r, s in enumerate(self._samples):
            try:
                it_run = self.tbl.item(r, 0)
                s.enabled = (it_run.checkState() == QtCore.Qt.Checked) if it_run else True
                s.tissue = (self.tbl.item(r, 3).text() if self.tbl.item(r, 3) else "").strip()
                s.species = (self.tbl.item(r, 4).text() if self.tbl.item(r, 4) else "").strip()
                op = (self.tbl.item(r, 5).text() if self.tbl.item(r, 5) else "").strip()
                s.output_path = Path(op).expanduser() if op else None
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
                    cycles_in.append(
                        CycleInput(
                            path=str(p),
                            cycle=int(s.cycles[i] if s.cycles and i < len(s.cycles) else (i + 1)),
                            tissue=(s.tissue or None),
                            species=(s.species or None),
                            registration_marker=(s.registration_markers[i] if s.registration_markers and i < len(s.registration_markers) else None),
                            channel_markers=(s.channel_markers[i] if s.channel_markers and i < len(s.channel_markers) else None),
                            channel_antibodies=(s.channel_antibodies[i] if s.channel_antibodies and i < len(s.channel_antibodies) else None),
                        )
                    )

                # Reset per-sample cycle progress before starting.
                try:
                    self.sig_set_cycle_progress.emit(0, int(len(cycles_in)))
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
                    progress_event_cb=_ev,
                    cancel_cb=lambda: bool(self._cancel_requested),
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
            self.btn_scan.setEnabled(True)
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
            self.btn_scan.setEnabled(True)

        w.start()

    def get_reports(self) -> list[dict]:
        return list(getattr(self, "_reports", []) or [])
