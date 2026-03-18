from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtWidgets
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_warning

from cycif_seg.io.ome_tiff import load_channel_names_only
from cycif_seg.stitch.stitch_core import (
    _DEFAULT_TILE_RE,
    _DEFAULT_X_GROUP,
    _DEFAULT_Y_GROUP,
    discover_cycle_tiles,
    discover_sample_cycles,
    stitch_cycle_tiles,
)

PLAN_SCHEMA_VERSION = 2


@dataclass
class StitchSample:
    name: str
    input_dir: Path
    cycle_dirs: list[Path]
    enabled: bool = True
    stitch_channel: int = 0
    output_suffix: str = 'stitched'
    pyramidal_output: bool = True
    avg_tiles_per_cycle: float = 0.0


class BatchStitchDialog(QtWidgets.QDialog):
    sig_set_status = QtCore.Signal(str)
    sig_set_cycle_progress = QtCore.Signal(int, int)
    sig_set_sample_progress = QtCore.Signal(int, int)
    sig_show_warning = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Batch stitch (Stage 0)')
        self.setModal(True)
        self._samples: list[StitchSample] = []
        self._reports: list[dict] = []
        self._failures: list[dict] = []
        self._cancel_requested = False
        self._refreshing_table = False
        self._channel_template_index: int | None = None

        root = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QGridLayout()
        self.txt_root = QtWidgets.QLineEdit()
        self.btn_pick_root = QtWidgets.QPushButton('Select input folder…')
        self.btn_scan = QtWidgets.QPushButton('Scan samples')
        row.addWidget(QtWidgets.QLabel('Batch input folder:'), 0, 0)
        row.addWidget(self.txt_root, 0, 1)
        row.addWidget(self.btn_pick_root, 0, 2)
        row.addWidget(self.btn_scan, 0, 3)
        row.setColumnStretch(1, 1)
        root.addLayout(row)

        defaults = QtWidgets.QHBoxLayout()
        self.txt_default_suffix = QtWidgets.QLineEdit('stitched')
        self.spin_default_channel = QtWidgets.QSpinBox()
        self.spin_default_channel.setMinimum(0)
        self.spin_default_channel.setMaximum(999)
        self.chk_default_pyramidal = QtWidgets.QCheckBox('Pyramidal output')
        self.chk_default_pyramidal.setChecked(True)
        self.btn_apply_defaults = QtWidgets.QPushButton('Apply defaults to all')
        defaults.addWidget(QtWidgets.QLabel('Default suffix:'))
        defaults.addWidget(self.txt_default_suffix)
        defaults.addWidget(QtWidgets.QLabel('Default stitch channel:'))
        defaults.addWidget(self.spin_default_channel)
        defaults.addWidget(self.chk_default_pyramidal)
        defaults.addWidget(self.btn_apply_defaults)
        defaults.addStretch(1)
        root.addLayout(defaults)

        self.grp_advanced = QtWidgets.QGroupBox('Advanced tile filename parsing')
        self.grp_advanced.setCheckable(True)
        self.grp_advanced.setChecked(False)
        adv = QtWidgets.QGridLayout(self.grp_advanced)
        self.txt_tile_regex = QtWidgets.QLineEdit(_DEFAULT_TILE_RE.pattern)
        self.spin_x_group = QtWidgets.QSpinBox()
        self.spin_x_group.setMinimum(1)
        self.spin_x_group.setMaximum(99)
        self.spin_x_group.setValue(int(_DEFAULT_X_GROUP))
        self.spin_y_group = QtWidgets.QSpinBox()
        self.spin_y_group.setMinimum(1)
        self.spin_y_group.setMaximum(99)
        self.spin_y_group.setValue(int(_DEFAULT_Y_GROUP))
        adv.addWidget(QtWidgets.QLabel('Tile filename regex:'), 0, 0)
        adv.addWidget(self.txt_tile_regex, 0, 1, 1, 3)
        adv.addWidget(QtWidgets.QLabel('X capture group:'), 1, 0)
        adv.addWidget(self.spin_x_group, 1, 1)
        adv.addWidget(QtWidgets.QLabel('Y capture group:'), 1, 2)
        adv.addWidget(self.spin_y_group, 1, 3)
        adv.addWidget(
            QtWidgets.QLabel('Default behavior when advanced parsing is off: use the last two underscore-separated integer fields before .ome.tif/.ome.tiff as x and y.'),
            2, 0, 1, 4
        )
        adv.setColumnStretch(1, 1)
        root.addWidget(self.grp_advanced)

        self.tbl = QtWidgets.QTableWidget()
        self.tbl.setColumnCount(7)
        self.tbl.setHorizontalHeaderLabels(['Run', 'Sample', '# cycle dirs', 'Avg tiles/cycle', 'Stitch channel', 'Suffix', 'Pyramidal'])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        root.addWidget(self.tbl, 1)

        cfg_row = QtWidgets.QHBoxLayout()
        self.btn_choose_channel = QtWidgets.QPushButton('Choose stitch channel from selected sample…')
        self.btn_copy_channel_to_all = QtWidgets.QPushButton('Copy stitch channel to all samples')
        cfg_row.addWidget(self.btn_choose_channel)
        cfg_row.addWidget(self.btn_copy_channel_to_all)
        cfg_row.addStretch(1)
        root.addLayout(cfg_row)

        plan_row = QtWidgets.QHBoxLayout()
        self.btn_save_plan = QtWidgets.QPushButton('Save plan JSON…')
        self.btn_load_plan = QtWidgets.QPushButton('Load plan JSON…')
        plan_row.addWidget(self.btn_save_plan)
        plan_row.addWidget(self.btn_load_plan)
        plan_row.addStretch(1)
        root.addLayout(plan_row)

        root.addWidget(QtWidgets.QLabel('Progress:'))
        self.prog_samples = QtWidgets.QProgressBar()
        self.prog_cycles = QtWidgets.QProgressBar()
        self.prog_samples.setVisible(False)
        self.prog_cycles.setVisible(False)
        root.addWidget(self.prog_samples)
        root.addWidget(self.prog_cycles)

        self.lbl_status = QtWidgets.QLabel('')
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        buttons = QtWidgets.QDialogButtonBox()
        self.btn_run = buttons.addButton('Run batch', QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_stop = buttons.addButton('Stop', QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_close = buttons.addButton('Close', QtWidgets.QDialogButtonBox.RejectRole)
        root.addWidget(buttons)

        self.btn_pick_root.clicked.connect(self._pick_root)
        self.btn_scan.clicked.connect(self._scan)
        self.btn_apply_defaults.clicked.connect(self._apply_defaults)
        self.btn_choose_channel.clicked.connect(self._choose_channel_for_selected)
        self.btn_copy_channel_to_all.clicked.connect(self._copy_channel_to_all)
        self.btn_save_plan.clicked.connect(self._save_plan)
        self.btn_load_plan.clicked.connect(self._load_plan)
        self.btn_run.clicked.connect(self._run_batch)
        self.btn_stop.clicked.connect(self._request_cancel)
        self.btn_close.clicked.connect(self.reject)

        self.sig_set_status.connect(self._apply_status)
        self.sig_set_cycle_progress.connect(self._apply_cycle_progress)
        self.sig_set_sample_progress.connect(self._apply_sample_progress)
        self.sig_show_warning.connect(show_warning)
        self.tbl.itemChanged.connect(self._on_table_item_changed)
        self.btn_stop.setEnabled(False)

    def _tile_parse_settings(self) -> tuple[str | None, int, int]:
        if not self.grp_advanced.isChecked():
            return None, int(_DEFAULT_X_GROUP), int(_DEFAULT_Y_GROUP)
        return (
            self.txt_tile_regex.text().strip() or _DEFAULT_TILE_RE.pattern,
            int(self.spin_x_group.value()),
            int(self.spin_y_group.value()),
        )

    def _discover_cycle_tiles_for_dir(self, cycle_dir: Path) -> dict[tuple[int, int], Path]:
        tile_filename_regex, x_group, y_group = self._tile_parse_settings()
        return discover_cycle_tiles(
            cycle_dir,
            tile_filename_regex=tile_filename_regex,
            x_group=int(x_group),
            y_group=int(y_group),
        )

    def get_reports(self) -> list[dict]:
        return list(self._reports)

    def _apply_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _set_status(self, msg: str) -> None:
        self._apply_status(msg)

    def _apply_cycle_progress(self, idx: int, n: int) -> None:
        try:
            self.prog_cycles.setRange(0, n)
            self.prog_cycles.setValue(max(0, min(int(idx), int(n))))
        except Exception:
            pass

    def _apply_sample_progress(self, idx: int, n: int) -> None:
        try:
            self.prog_samples.setRange(0, n)
            self.prog_samples.setValue(max(0, min(int(idx), int(n))))
        except Exception:
            pass

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self.btn_stop.setEnabled(False)
        self._set_status('Cancel requested… finishing current step.')

    def _pick_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select batch input folder', self.txt_root.text() or str(Path.cwd()))
        if d:
            self.txt_root.setText(d)

    def _scan(self) -> None:
        root_dir = Path(self.txt_root.text().strip() or '').expanduser()
        if not root_dir.is_dir():
            show_warning('Please select a valid batch input folder.')
            return
        samples: list[StitchSample] = []
        n_cycles = 0
        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir():
                continue
            tile_filename_regex, x_group, y_group = self._tile_parse_settings()
            cycle_dirs = discover_sample_cycles(
                sub,
                tile_filename_regex=tile_filename_regex,
                x_group=int(x_group),
                y_group=int(y_group),
            )
            if not cycle_dirs:
                continue
            n_cycles += len(cycle_dirs)
            tile_counts = []
            for cdir in cycle_dirs:
                try:
                    tile_counts.append(len(self._discover_cycle_tiles_for_dir(cdir)))
                except Exception:
                    tile_counts.append(0)
            avg_tiles = (float(sum(tile_counts)) / float(len(tile_counts))) if tile_counts else 0.0
            samples.append(
                StitchSample(
                    name=sub.name,
                    input_dir=sub,
                    cycle_dirs=cycle_dirs,
                    enabled=True,
                    stitch_channel=int(self.spin_default_channel.value()),
                    output_suffix=(self.txt_default_suffix.text().strip() or 'stitched'),
                    pyramidal_output=bool(self.chk_default_pyramidal.isChecked()),
                    avg_tiles_per_cycle=float(avg_tiles),
                )
            )
        self._samples = samples
        self._refresh_table()
        self._set_status(f'Found {len(samples)} sample(s) with {n_cycles} cycle folder(s) containing tile files.')

    def _refresh_table(self) -> None:
        self._refreshing_table = True
        try:
            self.tbl.setRowCount(len(self._samples))
            for r, s in enumerate(self._samples):
                chk = QtWidgets.QTableWidgetItem('')
                chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.Checked if s.enabled else QtCore.Qt.Unchecked)
                self.tbl.setItem(r, 0, chk)
                name = QtWidgets.QTableWidgetItem(s.name)
                name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 1, name)
                ncy = QtWidgets.QTableWidgetItem(str(len(s.cycle_dirs)))
                ncy.setFlags(ncy.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 2, ncy)
                avg_tiles = QtWidgets.QTableWidgetItem(f"{float(s.avg_tiles_per_cycle):.1f}")
                avg_tiles.setFlags(avg_tiles.flags() & ~QtCore.Qt.ItemIsEditable)
                self.tbl.setItem(r, 3, avg_tiles)
                self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(str(int(s.stitch_channel))))
                self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(str(s.output_suffix or 'stitched')))
                pit = QtWidgets.QTableWidgetItem('')
                pit.setFlags(pit.flags() | QtCore.Qt.ItemIsUserCheckable)
                pit.setCheckState(QtCore.Qt.Checked if s.pyramidal_output else QtCore.Qt.Unchecked)
                self.tbl.setItem(r, 6, pit)
            try:
                self.tbl.resizeColumnsToContents()
            except Exception:
                pass
        finally:
            self._refreshing_table = False

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing_table:
            return
        r = int(item.row())
        c = int(item.column())
        if r < 0 or r >= len(self._samples):
            return
        s = self._samples[r]
        try:
            if c == 0:
                s.enabled = (item.checkState() == QtCore.Qt.Checked)
            elif c == 4:
                s.stitch_channel = max(0, int((item.text() or '0').strip() or '0'))
            elif c == 5:
                s.output_suffix = (item.text() or '').strip() or 'stitched'
            elif c == 6:
                s.pyramidal_output = (item.checkState() == QtCore.Qt.Checked)
        except Exception:
            return

    def _apply_defaults(self) -> None:
        suffix = self.txt_default_suffix.text().strip() or 'stitched'
        ch = int(self.spin_default_channel.value())
        pyr = bool(self.chk_default_pyramidal.isChecked())
        for s in self._samples:
            s.output_suffix = suffix
            s.stitch_channel = ch
            s.pyramidal_output = pyr
        self._refresh_table()

    def _selected_row(self) -> int | None:
        try:
            rows = self.tbl.selectionModel().selectedRows()
            if not rows:
                return None
            return int(rows[0].row())
        except Exception:
            return None

    def _choose_channel_for_selected(self) -> None:
        r = self._selected_row()
        if r is None or r < 0 or r >= len(self._samples):
            show_warning('Select a sample row first.')
            return
        s = self._samples[r]
        if not s.cycle_dirs:
            show_warning('Selected sample has no cycle folders.')
            return
        first_cycle = s.cycle_dirs[0]
        tile_map = self._discover_cycle_tiles_for_dir(first_cycle)
        tile_files = [tile_map[k] for k in sorted(tile_map.keys(), key=lambda t: (t[1], t[0]))]
        if not tile_files:
            show_warning('Selected sample has no tile files.')
            return
        try:
            ch_names = list(load_channel_names_only(str(tile_files[0])) or [])
        except Exception as e:
            show_warning(f'Failed to read channel names: {e}')
            return
        if not ch_names:
            show_warning('No channel metadata found in the first tile.')
            return
        labels = [f'[{i}] {nm or f"Channel {i}"}' for i, nm in enumerate(ch_names)]
        cur = min(max(0, int(s.stitch_channel)), max(0, len(labels) - 1))
        choice, ok = QtWidgets.QInputDialog.getItem(self, 'Choose stitch channel', f'Sample: {s.name}\nCycle: {first_cycle.name}', labels, cur, False)
        if not ok or not choice:
            return
        idx = labels.index(choice)
        s.stitch_channel = int(idx)
        self._channel_template_index = int(idx)
        self._refresh_table()
        self._set_status(f"Selected stitch channel {idx} for sample '{s.name}'.")

    def _copy_channel_to_all(self) -> None:
        if self._channel_template_index is None:
            show_warning('Choose a stitch channel from one sample first.')
            return
        for s in self._samples:
            s.stitch_channel = int(self._channel_template_index)
        self._refresh_table()
        self._set_status(f'Copied stitch channel {self._channel_template_index} to all samples.')

    def _plan_dict(self) -> dict:
        self._sync_table_to_models()
        return {
            'schema_version': PLAN_SCHEMA_VERSION,
            'root_dir': self.txt_root.text().strip(),
            'defaults': {
                'output_suffix': self.txt_default_suffix.text().strip() or 'stitched',
                'stitch_channel': int(self.spin_default_channel.value()),
                'pyramidal_output': bool(self.chk_default_pyramidal.isChecked()),
                'use_custom_filename_parsing': bool(self.grp_advanced.isChecked()),
                'tile_filename_regex': self.txt_tile_regex.text().strip() or _DEFAULT_TILE_RE.pattern,
                'x_group': int(self.spin_x_group.value()),
                'y_group': int(self.spin_y_group.value()),
            },
            'samples': [
                {
                    'name': s.name,
                    'input_dir': str(s.input_dir),
                    'cycle_dirs': [str(p) for p in s.cycle_dirs],
                    'enabled': bool(s.enabled),
                    'avg_tiles_per_cycle': float(s.avg_tiles_per_cycle),
                    'stitch_channel': int(s.stitch_channel),
                    'output_suffix': str(s.output_suffix or 'stitched'),
                    'pyramidal_output': bool(s.pyramidal_output),
                }
                for s in self._samples
            ],
        }

    def _save_plan(self) -> None:
        self._sync_table_to_models()
        start = self.txt_root.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save stitch plan JSON', os.path.join(start, 'stitch_plan.json'), 'JSON files (*.json);;All files (*.*)')
        if not path:
            return
        Path(path).write_text(json.dumps(self._plan_dict(), indent=2), encoding='utf-8')
        self._set_status(f'Saved plan: {path}')

    def _load_plan(self) -> None:
        start = self.txt_root.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load stitch plan JSON', start, 'JSON files (*.json);;All files (*.*)')
        if not path:
            return
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        schema_version = int(d.get('schema_version') or 0)
        if schema_version not in {1, PLAN_SCHEMA_VERSION}:
            show_warning(f"Unsupported plan schema version: {d.get('schema_version')}")
            return
        self.txt_root.setText(str(d.get('root_dir') or ''))
        defs = d.get('defaults') or {}
        self.txt_default_suffix.setText(str(defs.get('output_suffix') or 'stitched'))
        self.spin_default_channel.setValue(max(0, int(defs.get('stitch_channel') or 0)))
        self.chk_default_pyramidal.setChecked(bool(defs.get('pyramidal_output', True)))
        self.grp_advanced.setChecked(bool(defs.get('use_custom_filename_parsing', False)))
        self.txt_tile_regex.setText(str(defs.get('tile_filename_regex') or _DEFAULT_TILE_RE.pattern))
        self.spin_x_group.setValue(max(1, int(defs.get('x_group') or _DEFAULT_X_GROUP)))
        self.spin_y_group.setValue(max(1, int(defs.get('y_group') or _DEFAULT_Y_GROUP)))
        self._samples = [
            StitchSample(
                name=str(rec.get('name') or ''),
                input_dir=Path(str(rec.get('input_dir') or '')).expanduser(),
                cycle_dirs=[Path(p).expanduser() for p in (rec.get('cycle_dirs') or [])],
                enabled=bool(rec.get('enabled', True)),
                stitch_channel=max(0, int(rec.get('stitch_channel') or 0)),
                output_suffix=str(rec.get('output_suffix') or 'stitched'),
                pyramidal_output=bool(rec.get('pyramidal_output', True)),
                avg_tiles_per_cycle=float(rec.get('avg_tiles_per_cycle') or 0.0),
            )
            for rec in (d.get('samples') or [])
        ]
        self._refresh_table()
        self._set_status(f'Loaded plan: {path}')

    def _sync_table_to_models(self) -> None:
        try:
            self.tbl.clearFocus()
            self.tbl.viewport().setFocus()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        for r, s in enumerate(self._samples):
            try:
                it_run = self.tbl.item(r, 0)
                it_ch = self.tbl.item(r, 4)
                it_suf = self.tbl.item(r, 5)
                it_pyr = self.tbl.item(r, 6)
                s.enabled = (it_run.checkState() == QtCore.Qt.Checked) if it_run else True
                s.stitch_channel = max(0, int((it_ch.text() if it_ch else '0').strip() or '0'))
                s.output_suffix = (it_suf.text() if it_suf else 'stitched').strip() or 'stitched'
                s.pyramidal_output = (it_pyr.checkState() == QtCore.Qt.Checked) if it_pyr else True
            except Exception:
                continue

    def _validate_ready(self) -> tuple[bool, str]:
        self._sync_table_to_models()
        enabled = [s for s in self._samples if s.enabled]
        if not enabled:
            return False, 'No samples selected.'
        for s in enabled:
            if not s.cycle_dirs:
                return False, f"Sample '{s.name}' has no cycle folders."
            if not str(s.output_suffix or '').strip():
                return False, f"Sample '{s.name}' is missing an output suffix."
        return True, ''

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
        self.btn_scan.setEnabled(False)
        self.btn_choose_channel.setEnabled(False)
        self.btn_copy_channel_to_all.setEnabled(False)
        self._cancel_requested = False
        self._reports = []
        self._failures = []

        @thread_worker
        def _worker():
            reports: list[dict] = []
            failures: list[dict] = []
            for si, s in enumerate(enabled, start=1):
                if self._cancel_requested:
                    raise RuntimeError('Cancelled')
                self.sig_set_sample_progress.emit(int(si - 1), int(n_samp))
                n_cycles_this_sample = max(1, len(s.cycle_dirs))
                try:
                    self.sig_set_cycle_progress.emit(0, int(n_cycles_this_sample))
                    for ci, cdir in enumerate(s.cycle_dirs, start=1):
                        if self._cancel_requested:
                            raise RuntimeError('Cancelled')
                        self.sig_set_cycle_progress.emit(int(ci - 1), int(n_cycles_this_sample))

                        def _progress(msg: str, _si=si, _name=s.name, _cdir=cdir, _ci=ci, _nci=n_cycles_this_sample):
                            self.sig_set_status.emit(f"[{_si}/{n_samp}] {_name} [{_ci}/{_nci}] / {_cdir.name}: {msg}")

                        tile_filename_regex, x_group, y_group = self._tile_parse_settings()
                        rep = stitch_cycle_tiles(
                            cdir,
                            output_suffix=s.output_suffix,
                            stitch_channel=int(s.stitch_channel),
                            pyramidal_output=bool(s.pyramidal_output),
                            tile_filename_regex=tile_filename_regex,
                            x_group=int(x_group),
                            y_group=int(y_group),
                            progress_cb=_progress,
                            cancel_cb=lambda: bool(self._cancel_requested),
                        )
                        rep['sample_name'] = s.name
                        reports.append(rep)
                        self.sig_set_cycle_progress.emit(int(ci), int(n_cycles_this_sample))
                except Exception as e:
                    if self._cancel_requested:
                        raise
                    msg = f"Sample '{s.name}' failed during batch stitch: {e}"
                    failures.append(
                        {
                            'sample_name': str(s.name),
                            'error': str(e),
                        }
                    )
                    self.sig_show_warning.emit(msg)
                    self.sig_set_status.emit(msg)
                self.sig_set_sample_progress.emit(int(si), int(n_samp))
                self.sig_set_cycle_progress.emit(0, 1)
            return {'reports': reports, 'failures': failures}

        worker = _worker()

        @worker.returned.connect
        def _done(result):
            result = result or {}
            self._reports = list(result.get('reports') or [])
            self._failures = list(result.get('failures') or [])
            if self._failures:
                self._set_status(
                    f"Finished stitching {len(self._reports)} cycle image(s) with "
                    f"{len(self._failures)} failed sample(s)."
                )
            else:
                self._set_status(f'Finished stitching {len(self._reports)} cycle image(s).')
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_scan.setEnabled(True)
            self.btn_choose_channel.setEnabled(True)
            self.btn_copy_channel_to_all.setEnabled(True)
            self.prog_samples.setValue(self.prog_samples.maximum())
            self.prog_cycles.setValue(0)

        @worker.errored.connect
        def _err(e):
            show_warning(f'Batch stitch failed: {e}')
            self._set_status(f'Batch stitch failed: {e}')
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_scan.setEnabled(True)
            self.btn_choose_channel.setEnabled(True)
            self.btn_copy_channel_to_all.setEnabled(True)
            self.prog_cycles.setValue(0)

        worker.start()
