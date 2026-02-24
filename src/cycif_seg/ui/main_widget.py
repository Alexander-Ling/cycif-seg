from __future__ import annotations

from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    message="'where' used without 'out'",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._accelerated_triangulate_python",
)

import numpy as np
import os
from qtpy import QtWidgets, QtCore
import datetime
import pickle

# Nuclei edit ops
from skimage.draw import line as sk_line
from skimage.draw import polygon as sk_polygon
from skimage.measure import label as cc_label
from skimage.morphology import binary_erosion, disk, dilation

import napari
from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from cycif_seg.io.ome_tiff import load_multichannel_tiff, load_multichannel_tiff_lazy
from cycif_seg.project import CycIFProject, create_project, open_project, is_project_dir
from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff
from cycif_seg.ui.merge_cycles_dialog import MergeRegisterCyclesDialog
from cycif_seg.ui.nuclei_edit_controller import NucleiEditController
from cycif_seg.ui.layer_manager import LayerManager
from cycif_seg.ui.project_controller import ProjectController
from cycif_seg.ui.image_controller import ImageController
from cycif_seg.ui.rf_controller import RFController
from cycif_seg.ui.steps.step1_preprocess_panel import Step1PreprocessPanel
from cycif_seg.ui.steps.step2a_nuclei_panel import Step2aNucleiPanel
from cycif_seg.ui.steps.step2b_edit_panel import Step2bEditPanel
from cycif_seg.features.multiscale import build_features
from cycif_seg.model.rf_pixel import train_rf, predict_proba_tiled


class CycIFMVPWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        # Name for the single multichannel image layer used when 'Display selected channels only' is OFF.
        # Using a single layer (channel_axis) is much faster than one layer per channel for large OME-TIFFs.
        self._all_channels_layer_name = "All Channels"
        self.viewer = viewer

        # Centralize napari layer CRUD/update behavior outside this long widget class.
        self.layers = LayerManager(self.viewer, connect_dirty=self._connect_layer_dirty)
        self.project_ctrl = ProjectController(self)
        self.image_ctrl = ImageController(self)
        self.rf_ctrl = RFController(self)

        self.img = None            # (Y,X,C) float32
        self.ch_names = None
        self.path = None
        self.project: CycIFProject | None = None

        # Trained-but-unsaved artifacts (models, etc.)
        # Each item: {"model": obj, "meta": dict, "saved_path": Optional[str]}
        self._pending_models = []

        # Pending UI/layer restore when opening a project (applied after image load)
        self._pending_restore = None
        self._is_restoring = False
        self._prob_layers = {}
        self._scribbles_layer_name = "Scribbles (0=unlabeled,1=nuc,2=nuc_boundary,3=bg)"
        self._nuclei_edit_layer_name = "Nuclei (edit)"

        # Step 2b: nuclei edit tool controller (extracted to reduce main_widget.py size)
        self.nuclei_edit = NucleiEditController(
            self.viewer,
            get_labels_layer=self._get_nuclei_edit_layer_or_warn,
            status_cb=self.set_status,
            warn_cb=show_warning,
        )

        # Step 2b edit tracking (for training models B/C)
        # Snapshot of labels before user edits (typically the Step 2a instance labels).
        self._nuclei_edit_base: np.ndarray | None = None  # (H,W) int32
        # Pending action for the Shapes layer (auto-applies when a new shape is added)
        self._nuclei_edit_action: str | None = None  # None | 'split' | 'merge'

        layout = QtWidgets.QVBoxLayout(self)

        # Project controls (Design choice A)
        proj_row = QtWidgets.QHBoxLayout()
        self.btn_new_project = QtWidgets.QPushButton("New Project…")
        self.btn_open_project = QtWidgets.QPushButton("Open Project…")
        self.btn_save_project = QtWidgets.QPushButton("Save Project")
        self.lbl_project = QtWidgets.QLabel("(no project)")
        self.lbl_project.setWordWrap(True)
        proj_row.addWidget(self.btn_new_project)
        proj_row.addWidget(self.btn_open_project)
        proj_row.addWidget(self.btn_save_project)
        proj_row.addWidget(self.lbl_project, 1)
        layout.addLayout(proj_row)

        # ------------------------------------------------------------------
        # Tabbed workspaces (Step 1..5)
        # ------------------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # -----------------------------
        # Step 1: Preprocessing
        # -----------------------------
        self.step1_panel = Step1PreprocessPanel()
        # Back-compat: expose frequently used controls as attributes on the main widget.
        # This keeps the rest of the codebase unchanged while we incrementally refactor.
        self.btn_merge_cycles = self.step1_panel.btn_merge_cycles
        self.tabs.addTab(self.step1_panel, "Step 1: Preprocess")

        # -----------------------------
        # Step 2a: Initial nuclei segmentation (RF mask + instances)
        # -----------------------------
        self.step2a_panel = Step2aNucleiPanel()
        # Back-compat: expose frequently used controls as attributes on the main widget.
        # This keeps the rest of the codebase unchanged while we incrementally refactor.
        self.btn_load = self.step2a_panel.btn_load
        self.lbl_file = self.step2a_panel.lbl_file
        self.list_channels = self.step2a_panel.list_channels
        self.btn_all = self.step2a_panel.btn_all
        self.btn_none = self.step2a_panel.btn_none
        self.btn_apply_channels = self.step2a_panel.btn_apply_channels
        self.btn_train = self.step2a_panel.btn_train
        self.spin_nuc_thresh = self.step2a_panel.spin_nuc_thresh
        self.spin_min_nuc_area = self.step2a_panel.spin_min_nuc_area
        self.btn_nuclei = self.step2a_panel.btn_nuclei
        self.chk_show_nuc_markers = self.step2a_panel.chk_show_nuc_markers
        self.slider_alpha = self.step2a_panel.slider_alpha
        self.tabs.addTab(self.step2a_panel, "Step 2a: Nuclei (initial)")

        # -----------------------------
        # Step 2b: Nuclei touch-up (manual edits + RF propagation)
        # -----------------------------
        self.step2b_panel = Step2bEditPanel()
        # Back-compat: expose frequently used controls as attributes on the main widget.
        # This keeps the rest of the codebase unchanged while we incrementally refactor.
        self.btn_make_nuclei_edit = self.step2b_panel.btn_make_nuclei_edit
        self.btn_propagate_nuclei_edits = self.step2b_panel.btn_propagate_nuclei_edits
        self.btn_split_cut = self.step2b_panel.btn_split_cut
        self.btn_merge_lasso = self.step2b_panel.btn_merge_lasso
        self.btn_delete_nucleus = self.step2b_panel.btn_delete_nucleus
        self.btn_draw_new_nucleus = self.step2b_panel.btn_draw_new_nucleus
        self.btn_erase_nucleus = self.step2b_panel.btn_erase_nucleus
        self.spin_erode_iters = self.step2b_panel.spin_erode_iters
        self.spin_brush = self.step2b_panel.spin_brush
        self.chk_auto_regen_nuclei = self.step2b_panel.chk_auto_regen_nuclei
        self.tabs.addTab(self.step2b_panel, "Step 2b: Nuclei (touch-up)")



        # -----------------------------
        # Step 3: Wash-off detection (placeholder)
        # -----------------------------
        tab3 = QtWidgets.QWidget()
        tab3_layout = QtWidgets.QVBoxLayout(tab3)
        tab3_layout.addWidget(QtWidgets.QLabel("Step 3 (coming soon): detect nuclei wash-off across cycles."))
        tab3_layout.addStretch(1)
        self.tabs.addTab(tab3, "Step 3: Wash-off")

        # -----------------------------
        # Step 4: Marker assignment (placeholder)
        # -----------------------------
        tab4 = QtWidgets.QWidget()
        tab4_layout = QtWidgets.QVBoxLayout(tab4)
        tab4_layout.addWidget(QtWidgets.QLabel("Step 4 (coming soon): per-marker +/- classification and signal association."))
        tab4_layout.addStretch(1)
        self.tabs.addTab(tab4, "Step 4: Markers")

        # -----------------------------
        # Step 5: Batch + QC (placeholder)
        # -----------------------------
        tab5 = QtWidgets.QWidget()
        tab5_layout = QtWidgets.QVBoxLayout(tab5)
        tab5_layout.addWidget(QtWidgets.QLabel("Step 5 (coming soon): batch run steps 2–4 with QC checks."))
        tab5_layout.addStretch(1)
        self.tabs.addTab(tab5, "Step 5: Batch/QC")

        # Global status + progress (visible across tabs)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.prog = QtWidgets.QProgressBar()
        self.prog.setVisible(False)
        layout.addWidget(self.prog)
        # Throttle probability layer refresh during tile prediction.
        # Writing into self._P happens per-tile; refreshing napari layers per tile can be expensive
        # and can cause sawtooth CPU usage. We refresh at a fixed cadence instead.
        self._prob_dirty = False
        # Signals
        self.btn_new_project.clicked.connect(self.on_new_project)
        self.btn_open_project.clicked.connect(self.on_open_project)
        self.btn_save_project.clicked.connect(self.on_save_project)
        self.btn_load.clicked.connect(self.image_ctrl.on_load)
        self.btn_merge_cycles.clicked.connect(self.on_merge_cycles)
        self.tabs.currentChanged.connect(lambda _idx: self._mark_project_dirty())
        self.btn_all.clicked.connect(lambda: self.image_ctrl.set_all_channels(True))
        self.btn_none.clicked.connect(lambda: self.image_ctrl.set_all_channels(False))
        self.btn_apply_channels.clicked.connect(self.image_ctrl.on_apply_channel_selection)
        self.btn_train.clicked.connect(self.on_train_predict)
        self.btn_nuclei.clicked.connect(self.on_generate_nuclei)
        self.btn_make_nuclei_edit.clicked.connect(self.on_make_nuclei_edit_layer)
        self.btn_propagate_nuclei_edits.clicked.connect(self.on_propagate_nuclei_edits)

        # Step 2b edit tools (checkable persistent modes)
        for _btn in [
            self.btn_split_cut,
            self.btn_merge_lasso,
            self.btn_delete_nucleus,
            self.btn_draw_new_nucleus,
            self.btn_erase_nucleus,
        ]:
            _btn.setCheckable(True)

        self.btn_split_cut.toggled.connect(self.on_toggle_split_draw_mode)
        self.btn_merge_lasso.toggled.connect(self.on_toggle_merge_draw_mode)
        self.btn_delete_nucleus.toggled.connect(self.on_toggle_delete_nucleus_mode)
        self.btn_draw_new_nucleus.toggled.connect(self.on_toggle_draw_new_nucleus_mode)
        self.btn_erase_nucleus.toggled.connect(self.on_toggle_eraser_mode)


        # Edit interaction buttons
        self.slider_alpha.valueChanged.connect(self.on_alpha_change)
        self.list_channels.itemChanged.connect(lambda _: self.image_ctrl.on_channel_selection_changed())

        # Guard against closing with unsaved project changes
        self._install_close_guard()


    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    def _dialog_start_dir(self) -> str:
        return self.project_ctrl.dialog_start_dir()
    def _set_project(self, prj: CycIFProject) -> None:
        self.project_ctrl.set_project(prj)
        return
    def _update_project_label(self) -> None:
        self.project_ctrl.update_project_label()
        return
    def _mark_project_dirty(self) -> None:
        self.project_ctrl.mark_project_dirty()
        return
    def _gather_ui_state(self) -> dict:
        return self.project_ctrl.gather_ui_state()
    def _save_layer_arrays(self) -> None:
        self.project_ctrl.save_layer_arrays()
        return
    def _apply_project_restore(self, restore: dict) -> None:
        self.project_ctrl.apply_project_restore(restore)
        return
    def on_save_project(self) -> None:
        self.project_ctrl.save_project()
        return
    def _install_close_guard(self) -> None:
        self.project_ctrl.install_close_guard()
        return
    def on_new_project(self):
        self.project_ctrl.new_project()
        return
    def on_open_project(self):
        self.project_ctrl.open_project()
        return
    def on_merge_cycles(self):
        """Step (1): pick 1+ OME-TIFFs, configure metadata/registration visually, then merge.

        Notes
        -----
        Selecting a single image is supported so users can attach metadata (tissue/species)
        and rename marker/channel labels for downstream analysis.
        """
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select 1+ cycle OME-TIFFs",
            self._dialog_start_dir(),
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not paths:
            return
        if len(paths) < 1:
            return

        dlg = MergeRegisterCyclesDialog(self, paths=list(paths))

        default_out = os.path.join(self._dialog_start_dir(), "merged_cycles.ome.tiff")
        if self.project is not None:
            try:
                default_out = str(self.project.data_dir / "merged_cycles.ome.tiff")
            except Exception:
                pass

        dlg.set_default_output(default_out)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            cfg = dlg.get_result()
        except Exception as e:
            show_warning(f"Invalid merge configuration: {e}")
            return

        out_path = str(cfg.get("output_path") or "").strip()
        if not out_path:
            return

        tissue = str(cfg.get("tissue") or "").strip() or None
        species = str(cfg.get("species") or "").strip() or None

        cycles_cfg = cfg.get("cycles") or []
        if not cycles_cfg:
            show_warning("No cycles were configured.")
            return

        # Treat selection order as cycle order 1..N.
        cycles: list[CycleInput] = []
        for d in cycles_cfg:
            cycles.append(
                CycleInput(
                    path=str(d.get("path")),
                    cycle=int(d.get("cycle")),
                    tissue=tissue,
                    species=species,
                    registration_marker=str(d.get("registration_marker") or "").strip() or None,
                    channel_markers=list(d.get("channel_markers") or []),
                    channel_antibodies=list(d.get("channel_antibodies") or []),
                )
            )

        self.set_status("Merging/registering cycles (background)…")
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.prog.setValue(0)
        try:
            self.btn_merge_cycles.setEnabled(False)
        except Exception:
            pass

        @thread_worker
        def _merge_worker():
            def _progress(s: str) -> None:
                # Ensure UI updates happen on the main thread.
                QtCore.QTimer.singleShot(0, lambda: self.set_status(f"[Step 1] {s}"))

            return merge_cycles_to_ome_tiff(
                cycles,
                out_path,
                default_registration_marker="DAPI",
                progress_cb=_progress,
            )

        worker = _merge_worker()

        @worker.returned.connect
        def _done(report):
            # report contains shifts etc. Show a small summary.
            shifts = report.get("shifts_yx", {})
            msg = f"Wrote merged OME-TIFF: {report.get('output_path')} (shape={report.get('shape_yxc')}, C={report.get('n_channels_out')})."
            if shifts:
                msg += "\nPer-cycle shifts (dy,dx px): " + ", ".join(
                    f"cy{k}={tuple(round(float(vv), 2) for vv in v)}" for k, v in shifts.items()
                )
            # Record Step 1 output in the active project manifest (if any)
            if self.project is not None:
                try:
                    out_p = Path(str(report.get("output_path") or out_path))
                    cycle_inputs = []
                    for cy in cycles:
                        cycle_inputs.append(
                            {
                                "path": self.project.relpath(Path(cy.path)),
                                "cycle": int(cy.cycle),
                                "registration_marker": str(cy.registration_marker or ""),
                                "channel_markers": list(cy.channel_markers or []),
                                "channel_antibodies": list(cy.channel_antibodies or []),
                            }
                        )
                    self.project.set_merged_ome_path(
                        out_p,
                        cycle_inputs=cycle_inputs,
                        tissue=tissue,
                        species=species,
                        canvas_yx=tuple(report.get("canvas_yx") or ()),
                    )
                    self._update_project_label()
                except Exception:
                    pass

            self.set_status(msg)

            self.prog.setRange(0, 1)
            self.prog.setValue(1)
            try:
                self.btn_merge_cycles.setEnabled(True)
            except Exception:
                pass

        @worker.errored.connect
        def _err(e):
            show_warning(f"Merge cycles failed: {e}")
            self.set_status(f"Merge cycles failed: {e}")
            self.prog.setRange(0, 1)
            self.prog.setValue(0)
            try:
                self.btn_merge_cycles.setEnabled(True)
            except Exception:
                pass

        worker.start()

    def _cancel_rf_worker(self):
        # Delegated to RFController
        return self.rf_ctrl.cancel_worker()

    def set_status(self, msg: str):
        self.status.setText(msg)
        show_info(msg)

    def _refresh_prob_layers_if_dirty(self):
        # Delegated to RFController (kept for backward-compat)
        return self.rf_ctrl._refresh_prob_layers_if_dirty()

    def on_alpha_change(self, v):
        # Delegated to RFController
        return self.rf_ctrl.on_alpha_change(v)

    def _channel_layer_names(self) -> set[str]:
        return set(self.ch_names or [])


    def sync_displayed_channel_layers(self):
        return self.image_ctrl.sync_displayed_channel_layers()

    def _apply_selected_channels_now(self, indices: list[int]) -> None:
        return self.image_ctrl._apply_selected_channels_now(indices)

    def _remove_all_channels_layer(self) -> None:
        return self.image_ctrl._remove_all_channels_layer()

    def _apply_selected_channels_incremental(self, indices: list[int], *, done_cb=None) -> None:
        return self.image_ctrl._apply_selected_channels_incremental(indices, done_cb=done_cb)

    def on_channel_selection_changed(self):
        return self.image_ctrl.on_channel_selection_changed()

    def _update_apply_channels_button_state(self) -> None:
        return self.image_ctrl._update_apply_channels_button_state()

    def on_apply_channel_selection(self) -> None:
        return self.image_ctrl.on_apply_channel_selection()

    def _colormap_for_channel(self, i: int) -> str:
        palette = ["blue", "green", "red", "magenta", "cyan", "yellow"]
        return palette[i % len(palette)]

    def set_all_channels(self, checked: bool):
        return self.image_ctrl.set_all_channels(checked)

    def get_selected_channels(self) -> list[int]:
        return self.image_ctrl.get_selected_channels()

    def _connect_layer_dirty(self, layer) -> None:
        """Attach listeners so edits mark the active project as dirty."""
        try:
            if layer is None:
                return
            if getattr(layer, "_cycif_dirty_connected", False):
                return
            setattr(layer, "_cycif_dirty_connected", True)

            def _on_any(event=None):
                try:
                    if getattr(self, '_is_restoring', False):
                        return
                except Exception:
                    pass
                self._mark_project_dirty()

            ev = getattr(layer, "events", None)
            if ev is None:
                return
            for attr in ["data", "visible", "opacity", "name"]:
                try:
                    e = getattr(ev, attr, None)
                    if e is not None:
                        e.connect(_on_any)
                except Exception:
                    pass
        except Exception:
            pass

    def _add_channel_layers(self):
        """
        Add one napari Image layer per channel, named using ch_names.
        """
        assert self.img is not None and self.ch_names is not None

        # Clear viewer
        self.viewer.layers.clear()

        # Add per-channel layers
        for i, nm in enumerate(self.ch_names):
            self.viewer.add_image(
                self.img[..., i],
                name=nm,
                blending="additive",
                opacity=0.6,
                colormap=self._colormap_for_channel(i),
            )

        # Add scribbles last so it paints over everything
        if self.img is not None:
            self.layers.ensure_scribbles_layer(image_shape=self.img.shape[:2], name=self._scribbles_layer_name)

    
    def _load_image_from_path(self, path: str, *, record_input: bool = True) -> None:
        return self.image_ctrl._load_image_from_path(path, record_input=record_input)

    def on_load(self):
        return self.image_ctrl.on_load()

    def on_train_predict(self):
        # Delegated to RFController
        return self.rf_ctrl.on_train_predict()

    # ---------------------------------------------------------------------
    # Layer helpers
    # ---------------------------------------------------------------------
    # NOTE: most layer creation/update logic has been moved into
    # `cycif_seg.ui.layer_manager.LayerManager` to keep this widget smaller.

    def on_make_nuclei_edit_layer(self):
        """Create (or refresh) an editable nuclei labels layer for manual edits."""
        nuc_layer = self.layers.get("Nuclei")
        if nuc_layer is None:
            show_warning("Generate nuclei first.")
            return

        nuc = np.asarray(nuc_layer.data).astype(np.int32, copy=False)
        edit = nuc.copy()

        # Snapshot "pre-edit" labels so we can later infer what changed.
        # This enables deciding whether to train model B and/or model C.
        self._nuclei_edit_base = nuc.copy()

        layer = self.layers.set_or_update_labels(self._nuclei_edit_layer_name, edit, opacity=0.7, visible=True)

        # Make it convenient to edit
        try:
            self.viewer.layers.selection.active = self.viewer.layers[self._nuclei_edit_layer_name]
        except Exception:
            pass

        self.set_status(
            "Nuclei edit layer ready. Use Labels paint/erase; use 'New nucleus id' to create a new nucleus label."
        )

        # Ensure preview layers exist up-front so toggling tools later doesn't create
        # new layers that steal focus and make tools appear broken.
        try:
            self.nuclei_edit.ensure_preview_layers(image_shape=(int(edit.shape[0]), int(edit.shape[1])))
        except Exception:
            pass

        # Ensure a helper Shapes layer exists for split/merge interactions.
        self._ensure_nuclei_edit_shapes_layer()

    # ---------------------------------------------------------------------
    # Nuclei edit interaction helpers
    # ---------------------------------------------------------------------

    _nuclei_edit_shapes_layer_name = "Nuclei edit shapes"

    def _ensure_nuclei_edit_shapes_layer(self):
        """Create a Shapes layer used to draw cut lines / lasso regions for edit ops."""
        if self.viewer is None:
            return None
        if self._nuclei_edit_shapes_layer_name in self.viewer.layers:
            return self.viewer.layers[self._nuclei_edit_shapes_layer_name]
        try:
            shapes = self.viewer.add_shapes(
                name=self._nuclei_edit_shapes_layer_name,
                opacity=0.9,
                edge_width=2,
            )
            shapes.visible = True

            # NOTE: we intentionally do NOT attach shapes.events.data here.
            # Some napari versions emit data-change events mid-draw and can crash or behave
            # inconsistently. Split/merge are applied on mouse release by our viewer-level
            # line-draw callback.

            return shapes
        except Exception:
            return None

    def _on_nuclei_edit_shapes_changed(self, event=None):
        """Auto-apply pending split/merge action after a new shape is drawn."""
        try:
            action = self._nuclei_edit_action
            if action == "split":
                self.on_split_by_cut_line()
            elif action == "merge":
                self.on_merge_by_lasso()
        except Exception:
            # Never crash the UI from an event hook
            pass

    def _get_nuclei_edit_layer_or_warn(self):
        layer = self.layers.get(self._nuclei_edit_layer_name)
        if layer is None:
            show_warning("Create the nuclei edit layer first (Edit nuclei…).")
            return None
        return layer

    def _get_selected_nucleus_id(self, labels_layer):
        try:
            sel = int(labels_layer.selected_label)
        except Exception:
            sel = 0
        if sel <= 0:
            # fall back to whatever is under the cursor if available
            try:
                pos = self.viewer.cursor.position  # world coords
                y = int(round(pos[0]))
                x = int(round(pos[1]))
                data = np.asarray(labels_layer.data)
                if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                    sel = int(data[y, x])
            except Exception:
                pass
        return int(sel)
    def _clear_nuclei_edit_shapes(self, *, keep_action: bool = False):
        """Clear the transient Shapes feedback layer used by split/merge.

        Parameters
        ----------
        keep_action:
            If True, do not clear the current line-draw action (split/merge). This is important for
            *persistent* split/merge modes where the button remains checked and the user expects to
            draw another line immediately.
        """
        if self.viewer is None:
            return
        if self._nuclei_edit_shapes_layer_name in self.viewer.layers:
            try:
                self.viewer.layers[self._nuclei_edit_shapes_layer_name].data = []
            except Exception:
                pass

        if keep_action:
            # Keep the current action in sync with the checked tool button.
            try:
                if getattr(self, "btn_split_cut", None) is not None and self.btn_split_cut.isChecked():
                    self._nuclei_edit_action = "split"
                elif getattr(self, "btn_merge_lasso", None) is not None and self.btn_merge_lasso.isChecked():
                    self._nuclei_edit_action = "merge"
            except Exception:
                pass
        else:
            # Reset pending action after applying (only when the tool is not intended to be persistent).
            self._nuclei_edit_action = None

    def _set_active_label_fresh(self, labels_layer):
        """Select a fresh (unused) nucleus id for painting new labels."""
        try:
            data = np.asarray(labels_layer.data)
            next_id = int(data.max()) + 1 if data.size else 1
        except Exception:
            next_id = 1
        try:
            labels_layer.selected_label = int(next_id)
        except Exception:
            pass
        return int(next_id)


    def _save_labels_undo(self, labels_layer) -> None:
        """Best-effort: push the current Labels state onto napari's undo stack.

        Napari's built-in undo/redo works automatically for paint/erase strokes, but not for
        programmatic edits that assign `layer.data` directly. For split/merge/delete/draw-new,
        we call this before mutating `layer.data` so Ctrl+Z can undo those operations.
        """
        if labels_layer is None:
            return
        try:
            fn = getattr(labels_layer, "_save_history", None)
            if callable(fn):
                fn()
                return
        except Exception:
            pass
        # Fallbacks across napari versions (best effort)
        try:
            hist = getattr(labels_layer, "_history", None)
            if hist is not None and hasattr(hist, "save"):
                hist.save(labels_layer.data)
        except Exception:
            pass


    def _apply_labels_new_data(self, labels_layer, new_data: np.ndarray) -> None:
        """Apply a full labels-array update in an *undo-friendly* way.

        Napari's Labels layer records undo/redo automatically for paint/erase strokes because they
        route through `data_setitem`. Direct assignments like `layer.data = ...` often bypass that
        machinery. For programmatic edits (split/merge/delete/draw-new), we compute a diff and
        apply it via `data_setitem` when available.
        """
        if labels_layer is None:
            return
        try:
            old = np.asarray(labels_layer.data)
            new = np.asarray(new_data)
        except Exception:
            try:
                labels_self._apply_labels_new_data(layer, new_data)
            except Exception:
                pass
            return
        if old.shape != new.shape:
            try:
                labels_layer.data = new
            except Exception:
                pass
            return
        try:
            dsi = getattr(labels_layer, "data_setitem", None)
            if callable(dsi):
                changed = old != new
                if np.any(changed):
                    idx = np.nonzero(changed)
                    vals = new[changed]
                    dsi(idx, vals)
                return
        except Exception:
            pass
        # Fallback
        try:
            labels_layer.data = new
        except Exception:
            pass


    def _apply_delete_label_id(self, labels_layer, label_id: int):
        """Delete all pixels belonging to a given nucleus label id."""
        try:
            label_id = int(label_id)
        except Exception:
            return
        if label_id <= 0:
            return
        data = np.asarray(labels_layer.data)
        if data.size == 0:
            return
        if not np.any(data == label_id):
            return
        new_data = data.copy()
        new_data[new_data == label_id] = 0
        self._apply_labels_new_data(labels_layer, new_data)


    def _enable_delete_click_mode(self, labels_layer):
        """Enable click-to-delete for nuclei labels."""
        self._delete_click_mode_active = True

        # Define the callback once so we can remove it later.
        if not hasattr(self, "_delete_click_callback") or self._delete_click_callback is None:
            def _delete_click_callback(layer, event):
                # Only act on mouse press (single click).
                try:
                    if getattr(event, "type", None) != "mouse_press":
                        yield
                        return
                    # Try to read label under cursor (world coords).
                    val = None
                    try:
                        val = layer.get_value(event.position, world=True)
                    except Exception:
                        try:
                            val = layer.get_value(event.position)
                        except Exception:
                            val = None
                    if val is None:
                        yield
                        return
                    lbl = int(val)
                    if lbl > 0:
                        self._apply_delete_label_id(layer, lbl)
                except Exception:
                    # Never crash UI due to a callback
                    pass
                yield

            self._delete_click_callback = _delete_click_callback

        try:
            cbs = getattr(labels_layer, "mouse_drag_callbacks", None)
            if cbs is not None and self._delete_click_callback not in cbs:
                cbs.append(self._delete_click_callback)
        except Exception:
            pass

        # Put the labels layer in a mode that doesn't paint.
        try:
            self.viewer.layers.selection.active = labels_layer
        except Exception:
            pass
        try:
            # 'pick' exists on Labels layers in napari and is ideal for click interactions.
            labels_layer.mode = "pick"
        except Exception:
            try:
                labels_layer.mode = "pan_zoom"
            except Exception:
                pass

    def _disable_delete_click_mode(self, labels_layer):
        """Disable click-to-delete mode if active."""
        if not getattr(self, "_delete_click_mode_active", False):
            return
        self._delete_click_mode_active = False
        try:
            cbs = getattr(labels_layer, "mouse_drag_callbacks", None)
            if cbs is not None and hasattr(self, "_delete_click_callback") and self._delete_click_callback in cbs:
                cbs.remove(self._delete_click_callback)
        except Exception:
            pass
        # Restore a neutral mode if we had switched to pick.
        try:
            if getattr(labels_layer, "mode", None) == "pick":
                labels_layer.mode = "pan_zoom"
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Persistent "Draw new nucleus" tool (step 2b)
    # ---------------------------------------------------------------------

    def _set_draw_new_nucleus_checked(self, checked: bool):
        """Set the toggle state without triggering callbacks."""
        try:
            self.btn_draw_new_nucleus.blockSignals(True)
            self.btn_draw_new_nucleus.setChecked(bool(checked))
        finally:
            try:
                self.btn_draw_new_nucleus.blockSignals(False)
            except Exception:
                pass

    def _disable_other_2b_tools_for_draw_mode(self):
        """When entering draw-new mode, turn off any other mutually-exclusive modes."""
        # Currently only delete-click is a persistent alternative mode.
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return

        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)

        if layer is not None:
            self._disable_delete_click_mode(layer)


    def _ensure_draw_preview_layer(self, shape_hw):
        """Ensure an in-memory preview Labels layer exists for draw-new-nucleus strokes.

        We avoid using a Shapes polyline as a preview (it is only a rough approximation of the
        brush footprint). Instead we show the exact rasterized brush mask in a dedicated Labels
        layer that is cleared on mouse release.
        """
        if self.viewer is None:
            return None
        name = getattr(self, "_nuclei_draw_preview_layer_name", "Nuclei (draw preview)")
        self._nuclei_draw_preview_layer_name = name

        if name in self.viewer.layers:
            lyr = self.viewer.layers[name]
            # Ensure correct shape (image might have changed)
            try:
                if tuple(getattr(lyr, "data", np.zeros((0, 0))).shape) != tuple(shape_hw):
                    lyr.data = np.zeros(shape_hw, dtype=np.uint8)
            except Exception:
                pass
            return lyr

        try:
            data = np.zeros(shape_hw, dtype=np.uint8)
            lyr = self.viewer.add_labels(
                data,
                name=name,
                opacity=0.5,
            )
            # Make it non-interactive; it's purely a preview.
            try:
                lyr.editable = False
            except Exception:
                pass
            try:
                lyr.visible = True
            except Exception:
                pass
            # Neutral preview color for label=1
            try:
                lyr.color = {1: (0.6, 0.6, 0.6, 0.6)}
            except Exception:
                pass
            return lyr
        except Exception:
            return None

    def _clear_draw_preview(self):
        """Clear and hide the draw-preview layer (if present)."""
        if self.viewer is None:
            return
        name = getattr(self, "_nuclei_draw_preview_layer_name", "Nuclei (draw preview)")
        if name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        try:
            lyr.data = np.zeros_like(np.asarray(lyr.data), dtype=np.uint8)
        except Exception:
            try:
                shp = getattr(lyr.data, "shape", None)
                if shp is not None:
                    lyr.data = np.zeros(shp, dtype=np.uint8)
            except Exception:
                pass
        try:
            lyr.visible = False
        except Exception:
            pass


    def _enable_draw_new_nucleus_mode(self, labels_layer):
        """Install a mouse-drag callback that commits one stroke at a time."""
        # Ensure we have a per-session next ID counter.
        try:
            data = np.asarray(labels_layer.data)
            self._draw_new_next_id = int(data.max()) + 1
        except Exception:
            self._draw_new_next_id = None

        self._draw_new_mode_active = True

        # Force the labels layer to be active and in paint mode, so the UI behaves like a brush tool.
        try:
            self.viewer.layers.selection.active = labels_layer
        except Exception:
            pass
        try:
            labels_layer.mode = "paint"
        except Exception:
            pass
        try:
            labels_layer.brush_size = int(getattr(self, 'spin_brush', None).value() if getattr(self, 'spin_brush', None) is not None else 4)
        except Exception:
            pass
        # Ensure an exact brush-footprint preview layer exists (cleared each stroke).
        try:
            data = np.asarray(labels_layer.data)
            self._ensure_draw_preview_layer(tuple(data.shape))
        except Exception:
            pass

        # Install callback by temporarily *replacing* the layer's default painting callbacks.
        # Napari's built-in Labels paint tool would otherwise write `selected_label` directly,
        # defeating the per-stroke "expand-or-new-id" logic.
        cbs = getattr(labels_layer, "mouse_drag_callbacks", None)
        if cbs is None:
            return

        # Backup originals once per activation so we can restore on disable.
        if getattr(self, "_draw_new_saved_callbacks", None) is None:
            try:
                self._draw_new_saved_callbacks = list(cbs)
            except Exception:
                self._draw_new_saved_callbacks = None

        if not hasattr(self, "_draw_new_nucleus_callback"):
            self._draw_new_nucleus_callback = self._make_draw_new_nucleus_callback()

        try:
            labels_layer.mouse_drag_callbacks = [self._draw_new_nucleus_callback]
        except Exception:
            # Fallback: clear and append
            try:
                cbs.clear()
                cbs.append(self._draw_new_nucleus_callback)
            except Exception:
                pass


    def _disable_draw_new_nucleus_mode(self, labels_layer):
        """Remove the draw-new-nucleus mouse callback and restore default behavior."""
        self._draw_new_mode_active = False
        # Restore original callbacks if we replaced them.
        saved = getattr(self, "_draw_new_saved_callbacks", None)
        if saved is not None:
            try:
                labels_layer.mouse_drag_callbacks = list(saved)
            except Exception:
                try:
                    cbs = getattr(labels_layer, "mouse_drag_callbacks", None)
                    if cbs is not None:
                        cbs.clear()
                        cbs.extend(list(saved))
                except Exception:
                    pass
        # Ensure no stale preview remains visible.
        self._clear_draw_preview()
        self._draw_new_saved_callbacks = None

    def _make_draw_new_nucleus_callback(self):
        """Mouse-drag callback that paints a *single* stroke, then assigns a nucleus id.

        Behavior per completed stroke:
          - If stroke overlaps an existing nucleus label (>0), expand that nucleus (do not overwrite other labels).
          - Else, create a new nucleus with a fresh unique ID (paint on background only).

        A dedicated preview Labels layer shows the *exact* rasterized brush footprint during the drag.
        """
        def _rasterize(points_rc, shape_hw, brush_size):
            h, w = int(shape_hw[0]), int(shape_hw[1])
            bs = int(brush_size) if brush_size is not None else 4
            r = max(1, bs // 2)
            mask = np.zeros((h, w), dtype=bool)
            try:
                from skimage.draw import disk as _draw_disk
                use_skimage = True
            except Exception:
                use_skimage = False

            for (rr, cc) in points_rc:
                try:
                    rr_f = float(rr); cc_f = float(cc)
                except Exception:
                    continue
                rr_i = int(round(rr_f)); cc_i = int(round(cc_f))
                if rr_i < -r or rr_i >= h + r or cc_i < -r or cc_i >= w + r:
                    continue
                if use_skimage:
                    try:
                        dr, dc = _draw_disk((rr_i, cc_i), r, shape=(h, w))
                        mask[dr, dc] = True
                        continue
                    except Exception:
                        pass
                # Fallback: manual circle rasterization
                r0 = max(0, rr_i - r); r1 = min(h, rr_i + r + 1)
                c0 = max(0, cc_i - r); c1 = min(w, cc_i + r + 1)
                yy, xx = np.ogrid[r0:r1, c0:c1]
                mask[r0:r1, c0:c1] |= (yy - rr_i) ** 2 + (xx - cc_i) ** 2 <= r ** 2
            return mask

        def _cb(layer, event):
            if not getattr(self, "_draw_new_mode_active", False):
                return

            # Accumulate points for the stroke in data coords (row,col).
            pts = []
            try:
                pos = event.position
                pts.append((float(pos[-2]), float(pos[-1])))
            except Exception:
                return

            # Snapshot pre-stroke labels for overlap detection.
            try:
                pre = np.asarray(layer.data).astype(np.int32, copy=False)
            except Exception:
                pre = None

            # Ensure preview exists and is visible for this stroke.
            preview = None
            try:
                if pre is not None:
                    preview = self._ensure_draw_preview_layer(tuple(pre.shape))
                if preview is not None:
                    try:
                        preview.visible = True
                    except Exception:
                        pass
            except Exception:
                preview = None

            yield

            while event.type == "mouse_move":
                try:
                    pos = event.position
                    rr, cc = float(pos[-2]), float(pos[-1])
                    if (not pts) or (abs(rr - pts[-1][0]) + abs(cc - pts[-1][1]) >= 0.5):
                        pts.append((rr, cc))
                    if pre is not None and preview is not None:
                        bs = getattr(layer, "brush_size", None)
                        m = _rasterize(pts, pre.shape, bs)
                        preview.data = m.astype(np.uint8)
                except Exception:
                    pass
                yield

            if event.type != "mouse_release":
                return

            # Clear preview immediately so it never obscures the committed label.
            try:
                self._clear_draw_preview()
            except Exception:
                pass

            if pre is None:
                return

            try:
                bs = getattr(layer, "brush_size", None)
                stroke = _rasterize(pts, pre.shape, bs)
            except Exception:
                return
            if not np.any(stroke):
                return

            # Choose target label: existing label with max overlap, else new id.
            try:
                ov = pre[stroke]
                ov = ov[ov > 0]
            except Exception:
                ov = np.asarray([], dtype=np.int32)

            if ov.size > 0:
                # expand the nucleus with the most overlap
                vals, cnts = np.unique(ov, return_counts=True)
                target = int(vals[int(np.argmax(cnts))])
            else:
                try:
                    target = int(pre.max()) + 1
                except Exception:
                    target = 1

            # Apply without overwriting other labels: only background or the target itself.
            can_paint = (pre == 0) | (pre == target)
            apply_mask = stroke & can_paint
            if not np.any(apply_mask):
                return
            new = np.array(pre, copy=True)
            new[apply_mask] = target

            # Use undo-friendly application.
            self._apply_labels_new_data(layer, new)

            # Keep selected_label synced to the most recent target.
            try:
                layer.selected_label = int(target)
            except Exception:
                pass

        return _cb
    def _make_eraser_split_callback(self):
        """Observer callback for eraser strokes: after a stroke, split any touched labels that became disconnected.

        Important: When napari's built-in eraser runs, pixels under the cursor are often already set to 0 by the
        time our callback executes. To reliably know *which* labels were affected, we snapshot the labels image
        at the start of the drag and read touched IDs from that snapshot, then apply component-splitting on the
        post-erase data at mouse release.
        """
        def _cb(layer, event):
            if not getattr(self, "_erase_mode_active", False):
                return

            # Snapshot the labels at drag start so we can detect which labels were modified even after erasing.
            try:
                pre = np.asarray(layer.data).astype(np.int32, copy=False).copy()
            except Exception:
                pre = None

            touched: set[int] = set()

            def _touch_at(position):
                nonlocal touched, pre
                if pre is None:
                    return
                try:
                    rr, cc = int(round(float(position[-2]))), int(round(float(position[-1])))
                    if 0 <= rr < pre.shape[0] and 0 <= cc < pre.shape[1]:
                        v = int(pre[rr, cc])
                        if v > 0:
                            touched.add(v)
                except Exception:
                    return

            try:
                _touch_at(event.position)
            except Exception:
                return

            yield

            while event.type == "mouse_move":
                try:
                    _touch_at(event.position)
                except Exception:
                    pass
                yield

            if event.type != "mouse_release":
                return

            if not touched:
                return

            try:
                out, report = self._split_disconnected_labels(np.asarray(layer.data), sorted(touched))
                if out is not None and report:
                    self._apply_labels_new_data(layer, out)
                    try:
                        layer.refresh()
                    except Exception:
                        pass
            except Exception:
                # best-effort; avoid crashing UI on release
                pass

        return _cb


        
    # ---------------------------------------------------------------------
    # Line-draw interaction for split/merge (avoids napari Shapes add_path bugs)
    # ---------------------------------------------------------------------

    def _enable_line_draw_mode(self, action: str):
        """Enable a lightweight 'draw a line' mode on the Shapes layer.

        We *do not* use Shapes' interactive add_path mode (which can be brittle on some
        napari versions); instead we capture mouse drags and update a Shapes layer for
        visual feedback, then apply the action on mouse release.
        """
        shapes = self._ensure_nuclei_edit_shapes_layer()
        if shapes is None:
            return
        self._nuclei_edit_action = str(action)

        # Ensure draw-new tool is off (mutually exclusive UX).
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return

        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)

        if layer is not None:
            self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)

        # Put the shapes layer in a non-drawing mode; we'll update it ourselves.
        try:
            self.viewer.layers.selection.active = shapes
        except Exception:
            pass
        try:
            shapes.mode = "select"
        except Exception:
            pass

        # Install viewer-level drag callback
        if not hasattr(self, "_line_draw_callback"):
            self._line_draw_callback = self._make_line_draw_callback()
        cbs = getattr(self.viewer, "mouse_drag_callbacks", None)
        if cbs is not None and self._line_draw_callback not in cbs:
            cbs.append(self._line_draw_callback)

        self.set_status(
            "Draw a line on the image (release mouse to finish)." if action in ("split", "merge") else
            "Draw a line (release mouse to finish).")

    def _disable_line_draw_mode(self):
        """Disable line-draw mode."""
        if self.viewer is None:
            return
        cbs = getattr(self.viewer, "mouse_drag_callbacks", None)
        if cbs is not None and hasattr(self, "_line_draw_callback") and self._line_draw_callback in cbs:
            try:
                cbs.remove(self._line_draw_callback)
            except Exception:
                pass

    def _make_line_draw_callback(self):
        """Viewer mouse-drag callback that records a line and applies split/merge on release."""
        def _cb(viewer, event):
            action = getattr(self, "_nuclei_edit_action", None)
            if action not in ("split", "merge"):
                return

            points = []
            try:
                pos = event.position
                rr, cc = float(pos[-2]), float(pos[-1])
                points.append((rr, cc))
            except Exception:
                return

            yield

            while event.type == "mouse_move":
                try:
                    pos = event.position
                    rr, cc = float(pos[-2]), float(pos[-1])
                    if (not points) or (abs(rr - points[-1][0]) + abs(cc - points[-1][1]) >= 1.0):
                        points.append((rr, cc))
                    # Update visual feedback
                    self._update_nuclei_edit_shapes_line(points)
                except Exception:
                    pass
                yield

            if event.type != "mouse_release":
                return

            # Final update and apply
            try:
                self._update_nuclei_edit_shapes_line(points)
            except Exception:
                pass

            if len(points) < 2:
                return

            if action == "split":
                self.on_split_by_cut_line(points_rc=points)
            elif action == "merge":
                self.on_merge_by_lasso(points_rc=points)

        return _cb

    def _update_nuclei_edit_shapes_line(self, points_rc, *, edge_width=None):
        """Replace the Shapes layer content with a single polyline for feedback."""
        shapes = self._ensure_nuclei_edit_shapes_layer()
        if shapes is None:
            return
        if points_rc is None or len(points_rc) == 0:
            try:
                shapes.data = []
            except Exception:
                pass
            return
        coords = np.asarray(points_rc, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            return
        try:
            shapes.data = [coords]
            shapes.shape_type = ["path"]
            if edge_width is not None:
                try:
                    shapes.edge_width = float(edge_width)
                except Exception:
                    pass
        except Exception:
            # Older napari: setting both may fail; best-effort
            try:
                shapes.data = [coords]
            except Exception:
                pass


    
    def _shape_to_mask(self, coords: np.ndarray, shape_kind: str, out_shape, *, thickness: int = 3):
        """Rasterize provided shape coordinates into a boolean mask.

        This is the same logic as :meth:`_last_shape_as_mask`, but operates on an explicit
        coordinate array (useful for viewer-drawn polylines).
        """
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] < 2:
            return None

        H, W = out_shape
        mask = np.zeros((H, W), dtype=bool)

        if shape_kind == "line":
            ys = coords[:, 0]
            xs = coords[:, 1]
            for i in range(len(coords) - 1):
                r0 = int(round(ys[i])); c0 = int(round(xs[i]))
                r1 = int(round(ys[i + 1])); c1 = int(round(xs[i + 1]))
                rr, cc = sk_line(r0, c0, r1, c1)
                ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                mask[rr[ok], cc[ok]] = True
            if thickness and thickness > 1:
                # thickness means a thicker stroke, not a "near miss" acceptance rule.
                mask = dilation(mask, footprint=disk(int(thickness)))
        elif shape_kind == "polygon":
            rr, cc = sk_polygon(coords[:, 0], coords[:, 1], shape=(H, W))
            mask[rr, cc] = True
        else:
            raise ValueError(f"Unknown shape_kind={shape_kind!r}")

        return mask

    def _last_shape_as_mask(self, shape_kind: str, out_shape, *, thickness: int = 3):
            """Rasterize the last drawn shape in the Shapes layer into a boolean mask.

            shape_kind: 'line' or 'polygon'
            """
            shapes = self._ensure_nuclei_edit_shapes_layer()
            if shapes is None or len(shapes.data) == 0:
                show_warning("Draw a cut line / merge line first (in the 'Nuclei edit shapes' layer).")
                return None

            coords = np.asarray(shapes.data[-1])
            if coords.ndim != 2 or coords.shape[1] < 2:
                show_warning("Last drawn shape is not 2D.")
                return None

            H, W = out_shape
            mask = np.zeros((H, W), dtype=bool)

            if shape_kind == "line":
                # Treat as polyline/path
                ys = coords[:, 0]
                xs = coords[:, 1]
                for i in range(len(coords) - 1):
                    r0 = int(round(ys[i])); c0 = int(round(xs[i]))
                    r1 = int(round(ys[i + 1])); c1 = int(round(xs[i + 1]))
                    rr, cc = sk_line(r0, c0, r1, c1)
                    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    mask[rr[ok], cc[ok]] = True
                if thickness and thickness > 1:
                    mask = dilation(mask, footprint=disk(int(thickness)))
            elif shape_kind == "polygon":
                rr, cc = sk_polygon(coords[:, 0], coords[:, 1], shape=(H, W))
                mask[rr, cc] = True
            else:
                raise ValueError(f"Unknown shape_kind={shape_kind!r}")

            return mask

    # ---------------------------------------------------------------------
    # Nuclei edit interactions (buttons)
    # ---------------------------------------------------------------------

    def _set_step2b_tool_checked(self, btn, checked: bool) -> None:
        """Set a checkable Step 2b tool button without triggering its toggled handler."""
        if btn is None:
            return
        try:
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
        finally:
            try:
                btn.blockSignals(False)
            except Exception:
                pass

    def _disable_other_step2b_tools(self, active_btn) -> None:
        """Ensure Step 2b tools are mutually exclusive at the UI level.

        The controller also disables other modes defensively, but keeping the UI toggles
        consistent avoids confusing states (multiple buttons appearing active).
        """
        for btn in [
            getattr(self, "btn_split_cut", None),
            getattr(self, "btn_merge_lasso", None),
            getattr(self, "btn_delete_nucleus", None),
            getattr(self, "btn_draw_new_nucleus", None),
            getattr(self, "btn_erase_nucleus", None),
        ]:
            if btn is None or btn is active_btn:
                continue
            if getattr(btn, "isChecked", lambda: False)():
                self._set_step2b_tool_checked(btn, False)

    
    def on_toggle_split_draw_mode(self, enabled: bool):
        """Toggle split (cut line) mode."""
        if enabled:
            self._disable_other_step2b_tools(self.btn_split_cut)
        self.nuclei_edit.toggle_split(bool(enabled))



    def on_toggle_merge_draw_mode(self, enabled: bool):
        """Toggle merge (line) mode."""
        if enabled:
            self._disable_other_step2b_tools(self.btn_merge_lasso)
        self.nuclei_edit.toggle_merge(bool(enabled))



    def on_toggle_delete_nucleus_mode(self, enabled: bool):
        """Toggle click-to-delete nucleus mode."""
        if enabled:
            self._disable_other_step2b_tools(self.btn_delete_nucleus)
        self.nuclei_edit.toggle_delete(bool(enabled))



    def on_toggle_eraser_mode(self, enabled: bool):
        """Toggle erode/erase paint mode on the nuclei edit layer."""
        if enabled:
            self._disable_other_step2b_tools(self.btn_erase_nucleus)
        # Keep brush size in sync with the UI spinbox, then delegate to the controller.
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is not None and enabled:
            try:
                self.viewer.layers.selection.active = layer
            except Exception:
                pass
            try:
                if getattr(self, "spin_brush", None) is not None:
                    layer.brush_size = int(self.spin_brush.value())
            except Exception:
                pass
        self.nuclei_edit.toggle_eraser(bool(enabled))


    def on_set_split_draw_mode(self):
        """Activate 'Split (cut line)' line-draw mode."""
        # Turn off draw-new nucleus mode if enabled (mutually exclusive).
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        if layer is not None:
            self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        self._enable_line_draw_mode("split")
        self.set_status("Split (cut line): draw a cut line (release mouse to apply).")
    
    
    
    def on_split_by_cut_line(self, points_rc=None):
        """Split nuclei intersected by the drawn cut line (delegated to controller)."""
        self.nuclei_edit.apply_split(points_rc)
        return
    
    
    def on_set_merge_draw_mode(self):
        """Activate 'Merge' line-draw mode."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        if layer is not None:
            self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        self._enable_line_draw_mode("merge")
        self.set_status("Merge: draw a line crossing nuclei to merge (release mouse to apply).")
    
    
    
    def on_merge_by_lasso(self, points_rc=None):
        """Merge nuclei intersected by the drawn merge line (delegated to controller)."""
        self.nuclei_edit.apply_merge(points_rc)
        return
    
    
    def on_delete_nucleus(self):
        """Enable click-to-delete mode: click inside a nucleus to delete the entire label."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        # Turn off split/merge line-draw mode (mutually exclusive with click tools).
        self._disable_line_draw_mode()
        self._nuclei_edit_action = None
    
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        # toggle off if already active
        if getattr(self, "_delete_click_mode_active", False):
            self._disable_delete_click_mode(layer)
            self.set_status("Delete nucleus mode: off.")
            return
    
        self._enable_delete_click_mode(layer)
        self.set_status("Delete nucleus mode: click inside a nucleus to delete it (button again to turn off).")
    
    
    
    def _set_active_edit_tool(self, tool: str | None):
        """Ensure exactly one 2b edit tool button is checked (or none)."""
        mapping = {
            "split": getattr(self, "btn_split_cut", None),
            "merge": getattr(self, "btn_merge_lasso", None),
            "delete": getattr(self, "btn_delete_nucleus", None),
            "draw": getattr(self, "btn_draw_new_nucleus", None),
            "erase": getattr(self, "btn_erase_nucleus", None),
        }
        
        # If we are leaving eraser mode, make sure we remove the observer callback even if
        # we unchecked the button with signals blocked.
        try:
            if tool != "erase":
                layer = self._get_nuclei_edit_layer_or_warn()
                if layer is not None:
                    self._disable_eraser_split_mode(layer)
        except Exception:
            pass

        for key, btn in mapping.items():
            if btn is None:
                continue
            want = (tool == key)
            try:
                if btn.isChecked() != want:
                    btn.blockSignals(True)
                    btn.setChecked(want)
                    btn.blockSignals(False)
            except Exception:
                pass
    
    def on_toggle_draw_new_nucleus_mode(self, enabled: bool):
        """Enable/disable *persistent* 'Draw new nucleus' mode."""
        if enabled:
            self._disable_other_step2b_tools(getattr(self, "btn_draw_new_nucleus", None))
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is not None and enabled:
            try:
                self.viewer.layers.selection.active = layer
            except Exception:
                pass
            try:
                if getattr(self, "spin_brush", None) is not None:
                    layer.brush_size = int(self.spin_brush.value())
            except Exception:
                pass
        self.nuclei_edit.toggle_draw_new(bool(enabled))
    
    # Backward compatibility (older code paths / saved UI connections)
    def on_set_draw_new_nucleus_mode(self):
        """Legacy entrypoint: toggle draw-new-nucleus on."""
        try:
            self.btn_draw_new_nucleus.setChecked(True)
        except Exception:
            # Fallback: enable directly
            layer = self._get_nuclei_edit_layer_or_warn()
            if layer is not None:
                self._enable_draw_new_nucleus_mode(layer)
    
    def on_set_eraser_mode(self):
        """Switch to erase mode on the nuclei edit layer."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        # Turn off split/merge line-draw mode (it otherwise keeps listening to mouse drags).
        self._disable_line_draw_mode()
        self._nuclei_edit_action = None
    
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        self._disable_delete_click_mode(layer)
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "erase"
            layer.brush_size = int(self.spin_brush.value())
        except Exception:
            pass
        self.set_status("Erode (eraser): paint to erase parts of nuclei in the 'Nuclei (edit)' layer.")
    
    def on_erode_nucleus(self):
        """Morphologically erode the selected nucleus by N iterations.
    
        Important: erosion can create disconnected islands. If that happens, we split those
        islands into separate labels (keeping the largest component as the original ID).
        """
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
    
        if layer is None:
            return
    
        # Mutually exclusive with draw-new mode
        self._disable_draw_new_nucleus_mode(layer)
        self._set_draw_new_nucleus_checked(False)
        sel = self._get_selected_nucleus_id(layer)
        if sel <= 0:
            show_warning("Select a nucleus label first.")
            return
    
        data = np.asarray(layer.data).astype(np.int32, copy=False)
        m = data == sel
        if not np.any(m):
            show_warning("Selected nucleus label not present.")
            return
    
        # UI control name is spin_erode_iters (legacy code referenced spin_erode)
        iters = int(getattr(self, "spin_erode_iters", self.spin_erode_iters).value())
        st = disk(1)
        m2 = m
        for _ in range(max(1, iters)):
            m2 = binary_erosion(m2, st)
            if not np.any(m2):
                break
    
        new_data = data.copy()
        new_data[m & (~m2)] = 0
    
        # If the remaining pixels for this label are now disconnected, split into new IDs.
        m_keep = new_data == sel
        if np.any(m_keep):
            cc = cc_label(m_keep, connectivity=1)
            n_cc = int(cc.max())
            if n_cc > 1:
                sizes = [(i, int((cc == i).sum())) for i in range(1, n_cc + 1)]
                sizes.sort(key=lambda t: t[1], reverse=True)
    
                next_id = int(new_data.max()) + 1
                # Clear current label, then rebuild.
                new_data[m_keep] = 0
                new_data[cc == sizes[0][0]] = sel
                for comp_id, _sz in sizes[1:]:
                    new_data[cc == comp_id] = next_id
                    next_id += 1
                self._apply_labels_new_data(layer, new_data)
                self.set_status(f"Eroded nucleus {sel} ({iters} iters) and split into {n_cc} parts.")
                return
    
        self._apply_labels_new_data(layer, new_data)
        self.set_status(f"Eroded nucleus {sel} ({iters} iters).")
    
    
    def _split_disconnected_labels(self, data: np.ndarray, labels: np.ndarray | list[int] | None = None):
        """Ensure each label is a single connected component.
    
        If a label has multiple connected components, we keep the largest component as the
        original label ID and assign new IDs to the remaining components.
    
        Parameters
        ----------
        data:
            2D int label image.
        labels:
            Optional iterable of label IDs to check. If None, checks all labels > 0.
        """
        if labels is None:
            labels = np.unique(data)
        labels = np.asarray(labels).astype(np.int32)
        labels = labels[labels > 0]
        if labels.size == 0:
            return data, {}
    
        out = np.asarray(data).copy()
        next_id = int(out.max()) + 1
        report = {}  # lbl -> n_components
    
        for lbl in labels.tolist():
            lbl = int(lbl)
            m = out == lbl
            if not np.any(m):
                continue
            cc = cc_label(m, connectivity=1)
            n_cc = int(cc.max())
            if n_cc <= 1:
                continue
    
            sizes = [(i, int((cc == i).sum())) for i in range(1, n_cc + 1)]
            sizes.sort(key=lambda t: t[1], reverse=True)
    
            out[m] = 0
            out[cc == sizes[0][0]] = lbl
            for comp_id, _sz in sizes[1:]:
                out[cc == comp_id] = next_id
                next_id += 1
    
            report[lbl] = n_cc
    
        return out, report
    
    
    def on_propagate_nuclei_edits(self):
        """Propagate nuclei edits via RF (Step 2b)."""
        layer = self.layers.get(self._nuclei_edit_layer_name)
        if layer is None:
            show_warning("Create the nuclei edit layer first.")
            return
        nuclei_labels = np.asarray(layer.data).astype(np.int32, copy=False)
        if nuclei_labels.max() <= 0:
            show_warning("Edited nuclei layer appears empty.")
            return

        # Step 2b: decide whether edits imply training model B and/or model C (future milestones).
        base = getattr(self, "_nuclei_edit_base", None)
        if base is None:
            nuc0 = self.layers.get("Nuclei")
            if nuc0 is not None:
                try:
                    base = np.asarray(nuc0.data).astype(np.int32, copy=False)
                except Exception:
                    base = None

        if base is not None and hasattr(base, "shape") and base.shape == nuclei_labels.shape:
            diff = (nuclei_labels != base)
            need_model_b = bool(np.any(diff & (base > 0)))
            need_model_c = bool(np.any((nuclei_labels > 0) & (base == 0)))
        else:
            need_model_b = True
            need_model_c = True
        self._need_model_b = need_model_b
        self._need_model_c = need_model_c

        # Auto-fix disconnected labels created during manual edits (e.g. after erasing).
        if base is not None and hasattr(base, "shape") and base.shape == nuclei_labels.shape:
            try:
                diff = (nuclei_labels != base)
                touched_labels = np.unique(np.concatenate([base[diff].ravel(), nuclei_labels[diff].ravel()]))
                touched_labels = touched_labels[touched_labels > 0]
                if touched_labels.size > 0:
                    fixed, rep = self._split_disconnected_labels(nuclei_labels, touched_labels)
                    if rep:
                        nuclei_labels = fixed
                        self._apply_labels_new_data(layer, nuclei_labels)
                        self.set_status(
                            f"Auto-split disconnected labels after edits: {len(rep)} label(s) fixed."
                        )
            except Exception:
                pass

        self.set_status(
            "Nuclei edits detected: "
            + ("Model B" if need_model_b else "(no Model B)")
            + " + "
            + ("Model C" if need_model_c else "(no Model C)")
            + ". Starting RF propagation…"
        )

        # Resolve active channels (prefer the channel checklist from Step 2a).
        try:
            use_channels = self.get_selected_channels()
        except Exception:
            use_channels = []
        if not use_channels:
            use_channels = list(range(int(self.img.shape[2]) if self.img is not None else 0))
        if self.img is None:
            show_warning("Load an image first.")
            return

        # Delegate the RF propagation worker to RFController.
        self.rf_ctrl.propagate_nuclei_edits(
            nuclei_labels,
            use_channels,
            tile_size=getattr(self, "tile_size", 512),
            center_yx=getattr(self, "center_yx", (0.0, 0.0)),
            batch_tiles=getattr(self, "batch_tiles", 4),
            feature_workers=getattr(self, "feature_workers", 3),
            progress_every=getattr(self, "progress_every", 2),
        )

    def on_generate_nuclei(self):
        # Requires probability maps from RF prediction
        if ("P(nucleus)" not in self.viewer.layers or "P(nuc_boundary)" not in self.viewer.layers or "P(background)" not in self.viewer.layers):
            show_warning("Run Train + Predict first to create P(nucleus), P(nuc_boundary), and P(background).")
            return
    
        p_nuc = np.asarray(self.viewer.layers["P(nucleus)"].data).astype(np.float32, copy=False)
        p_nb = np.asarray(self.viewer.layers["P(nuc_boundary)"].data).astype(np.float32, copy=False)
        p_bg = np.asarray(self.viewer.layers["P(background)"].data).astype(np.float32, copy=False)
    
        params = {
            "nuc_thresh": float(self.spin_nuc_thresh.value()),
            "min_nucleus_area": int(self.spin_min_nuc_area.value()),
            # boundary-aware split defaults (can expose later)
            "boundary_thresh": 0.35,
            "boundary_dilate": 2,
            "nuc_core_thresh": max(0.05, float(self.spin_nuc_thresh.value()) + 0.05),
            "peak_min_distance": 4,
            "peak_footprint": 7,
            "bg_thresh": 0.6,
        }
    
        self.set_status("Generating nuclei (background)…")
    
        from cycif_seg.instance.workers import nuclei_instances_from_probs_worker
        worker = nuclei_instances_from_probs_worker(p_nuc, p_nb, p_bg, params)
    
        # UI feedback: show an indeterminate (busy) progress bar.
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.prog.setValue(0)
        try:
            self.btn_nuclei.setEnabled(False)
        except Exception:
            pass
    
        @worker.yielded.connect
        def _on_yielded(msg):
            # Expected messages:
            #   ("stage", text, step, total)
            #   ("nuclei", nuclei_labels)
            #   ("debug", debug_dict)
            try:
                kind = msg[0]
            except Exception:
                return
    
            if kind == "stage":
                _, text, step, total = msg
                self.status.setText(str(text))
            elif kind == "nuclei":
                _, nuclei = msg
                self.layers.set_or_update_labels("Nuclei", nuclei.astype(np.int32, copy=False))
                self.set_status(f"Nuclei generated: N={int(nuclei.max())}")
            elif kind == "debug":
                if not self.chk_show_nuc_markers.isChecked():
                    return
                _, dbg = msg
                # Show a few helpful overlays when requested
                try:
                    if "seeds" in dbg:
                        self.layers.set_or_update_labels("Nucleus seeds", dbg["seeds"].astype(np.int32, copy=False))
                    if "core_mask" in dbg:
                        self.layers.set_or_update_labels("Nucleus core mask", dbg["core_mask"].astype(np.uint8, copy=False))
                    if "boundary_mask" in dbg:
                        self.layers.set_or_update_labels("Nucleus boundary mask", dbg["boundary_mask"].astype(np.uint8, copy=False))
                except Exception:
                    pass
    
        @worker.finished.connect
        def _on_done():
            self.set_status("Nuclei generation complete.")
            try:
                self.btn_nuclei.setEnabled(True)
            except Exception:
                pass
            try:
                self.prog.setVisible(False)
            except Exception:
                pass
    
        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Nuclei generation failed: {e}")
            try:
                self.btn_nuclei.setEnabled(True)
            except Exception:
                pass
            try:
                self.prog.setVisible(False)
            except Exception:
                pass
    
        worker.start()