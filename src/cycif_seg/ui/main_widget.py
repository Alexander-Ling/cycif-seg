from __future__ import annotations

from pathlib import Path

import numpy as np
import os
from qtpy import QtWidgets, QtCore
import datetime
import pickle

# Nuclei edit ops
from skimage.draw import line as sk_line
from skimage.draw import polygon as sk_polygon
from skimage.measure import label as cc_label
from skimage.morphology import binary_dilation, binary_erosion, disk

import napari
from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from cycif_seg.io.ome_tiff import load_multichannel_tiff
from cycif_seg.project import CycIFProject, create_project, open_project, is_project_dir
from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff
from cycif_seg.ui.merge_cycles_dialog import MergeRegisterCyclesDialog
from cycif_seg.features.multiscale import build_features
from cycif_seg.model.rf_pixel import train_rf, predict_proba_tiled


class CycIFMVPWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

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

        self._rf_worker = None
        self._rf_run_id = 0

        self._prob_layers = {}
        self._scribbles_layer_name = "Scribbles (0=unlabeled,1=nuc,2=nuc_boundary,3=bg)"
        self._nuclei_edit_layer_name = "Nuclei (edit)"

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
        tab1 = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QVBoxLayout(tab1)

        file_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load TIFF/OME-TIFF…")
        self.btn_merge_cycles = QtWidgets.QPushButton("Merge/Register cycles (beta)…")
        self.lbl_file = QtWidgets.QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        file_row.addWidget(self.btn_load)
        file_row.addWidget(self.btn_merge_cycles)
        file_row.addWidget(self.lbl_file, 1)
        tab1_layout.addLayout(file_row)

        tab1_layout.addWidget(
            QtWidgets.QLabel(
                "Preprocessing (Step 1, beta): merge 2+ cycle OME-TIFFs into one co-registered OME-TIFF.\n"
                "Current MVP does translation-only registration using a chosen nuclear marker (default: DAPI)."
            )
        )
        tab1_layout.addStretch(1)
        self.tabs.addTab(tab1, "Step 1: Preprocess")

        # -----------------------------
                # Step 2a: Initial nuclei segmentation (RF mask + instances)
        # -----------------------------
        tab2a = QtWidgets.QWidget()
        tab2a_layout = QtWidgets.QVBoxLayout(tab2a)

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Channels used for RF training:"))
        self.chk_display_selected_only = QtWidgets.QCheckBox("Display selected channels only")
        header_row.addWidget(self.chk_display_selected_only)
        tab2a_layout.addLayout(header_row)

        self.list_channels = QtWidgets.QListWidget()
        self.list_channels.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        tab2a_layout.addWidget(self.list_channels, 1)

        ch_btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        ch_btn_row.addWidget(self.btn_all)
        ch_btn_row.addWidget(self.btn_none)
        tab2a_layout.addLayout(ch_btn_row)

        tab2a_layout.addWidget(
            QtWidgets.QLabel(
                "Scribbles: paint into the Labels layer using values:\n"
                "1 = nucleus, 2 = nucleus boundary/overlap, 3 = background (0 = unlabeled)\n"
                "Tip: select the Scribbles layer and set the current label value in napari."
            )
        )

        self.btn_train = QtWidgets.QPushButton("Train + Predict (RF)")
        tab2a_layout.addWidget(self.btn_train)

        grid = QtWidgets.QGridLayout()

        self.spin_nuc_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_nuc_thresh.setRange(0.0, 1.0)
        self.spin_nuc_thresh.setSingleStep(0.05)
        self.spin_nuc_thresh.setValue(0.35)

        self.spin_min_nuc_area = QtWidgets.QSpinBox()
        self.spin_min_nuc_area.setRange(0, 10_000_000)
        self.spin_min_nuc_area.setValue(30)

        grid.addWidget(QtWidgets.QLabel("Nucleus prob thresh"), 0, 0)
        grid.addWidget(self.spin_nuc_thresh, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min nucleus area (px)"), 1, 0)
        grid.addWidget(self.spin_min_nuc_area, 1, 1)
        tab2a_layout.addLayout(grid)

        self.btn_nuclei = QtWidgets.QPushButton("Generate Nuclei (instances)")
        tab2a_layout.addWidget(self.btn_nuclei)

        self.chk_show_nuc_markers = QtWidgets.QCheckBox("Show nucleus markers overlay")
        self.chk_show_nuc_markers.setChecked(False)
        tab2a_layout.addWidget(self.chk_show_nuc_markers)

        # Opacity
        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(QtWidgets.QLabel("Overlay opacity:"))
        self.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(100)
        self.slider_alpha.setValue(40)
        op_row.addWidget(self.slider_alpha, 1)
        tab2a_layout.addLayout(op_row)

        tab2a_layout.addStretch(0)
        
        self.tabs.addTab(tab2a, "Step 2a: Nuclei (initial)")

        # -----------------------------
        # Step 2b: Nuclei touch-up (manual edits + RF propagation)
        # -----------------------------
        tab2b = QtWidgets.QWidget()
        tab2b_layout = QtWidgets.QVBoxLayout(tab2b)

        tab2b_layout.addWidget(
            QtWidgets.QLabel(
                "Touch up nuclei by manually editing the instance labels and (optionally)\n"
                "training a model to propagate similar edits elsewhere."
            )
        )

        tab2b_layout.addSpacing(6)
        tab2b_layout.addWidget(QtWidgets.QLabel("Nuclei edit tools:"))

        edit_row = QtWidgets.QHBoxLayout()
        self.btn_make_nuclei_edit = QtWidgets.QPushButton("Make/Select editable nuclei")
        self.btn_new_nucleus_id = QtWidgets.QPushButton("New nucleus id")
        self.btn_propagate_nuclei_edits = QtWidgets.QPushButton("Propagate edits (RF)")
        edit_row.addWidget(self.btn_make_nuclei_edit)
        edit_row.addWidget(self.btn_new_nucleus_id)
        edit_row.addWidget(self.btn_propagate_nuclei_edits)
        tab2b_layout.addLayout(edit_row)

        # --- Edit interactions ---
        tools_row = QtWidgets.QHBoxLayout()
        self.btn_split_cut = QtWidgets.QPushButton("Split (cut line)")
        self.btn_merge_lasso = QtWidgets.QPushButton("Merge (lasso)")
        self.btn_delete_nucleus = QtWidgets.QPushButton("Delete nucleus")
        self.btn_draw_new_nucleus = QtWidgets.QPushButton("Draw new nucleus")
        self.btn_erase_nucleus = QtWidgets.QPushButton("Erode (eraser)")
        tools_row.addWidget(self.btn_split_cut)
        tools_row.addWidget(self.btn_merge_lasso)
        tools_row.addWidget(self.btn_delete_nucleus)
        tools_row.addWidget(self.btn_draw_new_nucleus)
        tools_row.addWidget(self.btn_erase_nucleus)
        tab2b_layout.addLayout(tools_row)

        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(QtWidgets.QLabel("Cut width"))
        self.spin_cut_width = QtWidgets.QSpinBox()
        self.spin_cut_width.setRange(1, 50)
        self.spin_cut_width.setValue(5)
        params_row.addWidget(self.spin_cut_width)
        params_row.addSpacing(12)
        params_row.addWidget(QtWidgets.QLabel("Erode iters"))
        self.spin_erode_iters = QtWidgets.QSpinBox()
        self.spin_erode_iters.setRange(1, 50)
        self.spin_erode_iters.setValue(2)
        params_row.addWidget(self.spin_erode_iters)
        params_row.addSpacing(12)
        params_row.addWidget(QtWidgets.QLabel("Brush"))
        self.spin_brush = QtWidgets.QSpinBox()
        self.spin_brush.setRange(1, 200)
        self.spin_brush.setValue(12)
        params_row.addWidget(self.spin_brush)
        params_row.addStretch(1)
        tab2b_layout.addLayout(params_row)

        self.chk_auto_regen_nuclei = QtWidgets.QCheckBox("Auto-regenerate nuclei after propagation")
        self.chk_auto_regen_nuclei.setChecked(True)
        tab2b_layout.addWidget(self.chk_auto_regen_nuclei)

        tab2b_layout.addWidget(QtWidgets.QLabel("Nuclei instance segmentation (boundary-aware):"))

        
        tab2b_layout.addStretch(0)
        self.tabs.addTab(tab2b, "Step 2b: Nuclei (touch-up)")



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
        self._prob_refresh_timer = QtCore.QTimer(self)
        self._prob_refresh_timer.setInterval(100)  # ms (10 Hz)
        self._prob_refresh_timer.timeout.connect(self._refresh_prob_layers_if_dirty)

        # Signals
        self.btn_new_project.clicked.connect(self.on_new_project)
        self.btn_open_project.clicked.connect(self.on_open_project)
        self.btn_save_project.clicked.connect(self.on_save_project)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_merge_cycles.clicked.connect(self.on_merge_cycles)
        self.tabs.currentChanged.connect(lambda _idx: self._mark_project_dirty())
        self.btn_all.clicked.connect(lambda: self.set_all_channels(True))
        self.btn_none.clicked.connect(lambda: self.set_all_channels(False))
        self.btn_train.clicked.connect(self.on_train_predict)
        self.btn_nuclei.clicked.connect(self.on_generate_nuclei)
        self.btn_make_nuclei_edit.clicked.connect(self.on_make_nuclei_edit_layer)
        self.btn_new_nucleus_id.clicked.connect(self.on_new_nucleus_id)
        self.btn_propagate_nuclei_edits.clicked.connect(self.on_propagate_nuclei_edits)

        # Edit interaction buttons
        self.slider_alpha.valueChanged.connect(self.on_alpha_change)
        self.chk_display_selected_only.stateChanged.connect(self.sync_displayed_channel_layers)
        self.list_channels.itemChanged.connect(lambda _: self.on_channel_selection_changed())

        # Guard against closing with unsaved project changes
        self._install_close_guard()


    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    def _dialog_start_dir(self) -> str:
        if self.project is not None:
            return str(self.project.root)
        return os.getcwd()

    def _set_project(self, prj: CycIFProject) -> None:
        # If current project has unsaved changes, warn before switching
        if self.project is not None and getattr(self.project, 'dirty', False):
            resp = QtWidgets.QMessageBox.question(
                self,
                'Unsaved project changes',
                'Current project has unsaved changes. Save before switching projects?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if resp == QtWidgets.QMessageBox.Cancel:
                return
            if resp == QtWidgets.QMessageBox.Yes:
                self.on_save_project()

        self.project = prj
        # Clear any in-memory pending artifacts when switching projects
        self._pending_models = []

        # If the project includes saved UI/layer state, restore it after image load
        ui_state = (prj.manifest.get('ui_state') or {})
        self._pending_restore = ui_state if isinstance(ui_state, dict) else None

        self._update_project_label()
        show_info(f"Project set: {prj.root}")

        # Auto-load the last image if recorded
        try:
            last = None
            if isinstance(self._pending_restore, dict):
                last = self._pending_restore.get('last_image')
            if last:
                p = Path(last)
                # allow paths saved relative to project
                if not p.is_absolute():
                    p = prj.abspath(str(last))
                if p.exists():
                    self._load_image_from_path(str(p), record_input=False)
                else:
                    # Apply whatever UI state we can even without image
                    if self._pending_restore:
                        self._apply_project_restore(self._pending_restore)
                        self._pending_restore = None
        except Exception:
            pass

    def _update_project_label(self) -> None:
        if self.project is None:
            self.lbl_project.setText("(no project)")
            return
        star = "*" if getattr(self.project, "dirty", False) else ""
        self.lbl_project.setText(f"{self.project.root}{star}")

    def _mark_project_dirty(self) -> None:
        if self.project is None:
            return
        try:
            self.project.mark_dirty()
        except Exception:
            pass
        self._update_project_label()


    def _gather_ui_state(self) -> dict:
        """Capture UI/layer state for project reload."""
        state: dict = {}
        # Channels UI
        try:
            state["display_selected_only"] = bool(self.chk_display_selected_only.isChecked())
        except Exception:
            state["display_selected_only"] = False

        selected = []
        try:
            for i in range(self.list_channels.count()):
                it = self.list_channels.item(i)
                if it is not None and it.checkState() == QtCore.Qt.Checked:
                    selected.append(str(it.text()))
        except Exception:
            pass
        state["selected_channels"] = selected

        # Which file is currently loaded
        if self.path:
            state["last_image"] = str(self.path)

        # Layer visual state (visibility, opacity, and style)
        layers = {}
        try:
            for lyr in list(self.viewer.layers):
                try:
                    st = {
                        "visible": bool(getattr(lyr, "visible", True)),
                        "opacity": float(getattr(lyr, "opacity", 1.0)),
                        "type": str(getattr(lyr, "_type_string", lyr.__class__.__name__)),
                    }
                    # Image-layer style
                    if hasattr(lyr, "colormap"):
                        try:
                            cm = getattr(lyr.colormap, "name", None) or str(lyr.colormap)
                            st["colormap"] = str(cm)
                        except Exception:
                            pass
                    if hasattr(lyr, "blending"):
                        try:
                            st["blending"] = str(getattr(lyr, "blending"))
                        except Exception:
                            pass
                    if hasattr(lyr, "contrast_limits"):
                        try:
                            cl = getattr(lyr, "contrast_limits")
                            if cl is not None and len(cl) == 2:
                                st["contrast_limits"] = [float(cl[0]), float(cl[1])]
                        except Exception:
                            pass
                    layers[str(lyr.name)] = st
                except Exception:
                    continue
        except Exception:
            pass
        state["layers"] = layers

        # Placeholder for future tab UI
        state["active_tab"] = int(self.tabs.currentIndex()) if hasattr(self, "tabs") else None
        return state

    def _save_layer_arrays(self) -> None:
        """Save scribbles/prob maps/nuclei to the project data folder and record in manifest."""
        if self.project is None:
            return

        self.project.manifest.setdefault("step2", {})
        step2 = self.project.manifest["step2"]
        step2.setdefault("layers", {})

        out = step2["layers"]
        saved_any = False

        def _save_npy(name: str, arr: np.ndarray) -> str:
            fn = f"{name}.npy".replace(" ", "_").replace("/", "_")
            p = self.project.data_dir / fn
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, arr)
            return self.project.relpath(p)

        # Scribbles
        if self._scribbles_layer_name in self.viewer.layers:
            try:
                arr = np.asarray(self.viewer.layers[self._scribbles_layer_name].data)
                out["scribbles"] = {
                    "name": self._scribbles_layer_name,
                    "path": _save_npy("scribbles", arr.astype(np.uint8, copy=False)),
                }
                saved_any = True
            except Exception:
                pass

        # Probability maps (known names)
        prob_paths = {}
        for nm in ["P(nucleus)", "P(nuc_boundary)", "P(background)"]:
            if nm in self.viewer.layers:
                try:
                    arr = np.asarray(self.viewer.layers[nm].data).astype(np.float32, copy=False)
                    prob_paths[nm] = _save_npy(nm, arr)
                    saved_any = True
                except Exception:
                    continue
        if prob_paths:
            out["probability_maps"] = prob_paths

        # Nuclei instance map
        if "Nuclei" in self.viewer.layers:
            try:
                arr = np.asarray(self.viewer.layers["Nuclei"].data).astype(np.int32, copy=False)
                out["nuclei_instances"] = {
                    "name": "Nuclei",
                    "path": _save_npy("nuclei_instances", arr),
                }
                saved_any = True
            except Exception:
                pass

        if saved_any:
            self.project.mark_dirty()

    def _apply_project_restore(self, restore: dict) -> None:
        """Restore UI, layers, and layer visibility from a saved project."""
        if not restore:
            return

        self._is_restoring = True
        try:
            # Restore channel selection + display mode (must happen after channel list is populated)
            # Restore active tab
            try:
                tab_idx = restore.get("active_tab", None)
                if tab_idx is not None and hasattr(self, "tabs"):
                    tab_idx = int(tab_idx)
                    if 0 <= tab_idx < self.tabs.count():
                        self.tabs.setCurrentIndex(tab_idx)
            except Exception:
                pass

            try:
                disp = bool(restore.get("display_selected_only", False))
                self.chk_display_selected_only.blockSignals(True)
                self.chk_display_selected_only.setChecked(disp)
                self.chk_display_selected_only.blockSignals(False)
            except Exception:
                pass

            sel = set(restore.get("selected_channels") or [])
            if sel and self.list_channels.count() > 0:
                try:
                    self.list_channels.blockSignals(True)
                    for i in range(self.list_channels.count()):
                        it = self.list_channels.item(i)
                        if it is None:
                            continue
                        it.setCheckState(QtCore.Qt.Checked if str(it.text()) in sel else QtCore.Qt.Unchecked)
                    self.list_channels.blockSignals(False)
                except Exception:
                    try:
                        self.list_channels.blockSignals(False)
                    except Exception:
                        pass

            try:
                self.sync_displayed_channel_layers()
            except Exception:
                pass

            # Restore saved layer arrays (scribbles/prob/nuclei)
            try:
                if self.project is not None:
                    step2 = (self.project.manifest.get("step2") or {})
                    layers = (step2.get("layers") or {})

                    # Scribbles
                    scrib = layers.get("scribbles")
                    if scrib and isinstance(scrib, dict):
                        p = self.project.abspath(scrib.get("path", ""))
                        if p.exists():
                            arr = np.load(p, allow_pickle=False)
                            self._set_or_update_labels_layer(self._scribbles_layer_name, arr.astype(np.uint8, copy=False))

                    # Prob maps
                    prob = layers.get("probability_maps") or {}
                    if isinstance(prob, dict):
                        for nm, relp in prob.items():
                            p = self.project.abspath(relp)
                            if p.exists():
                                arr = np.load(p, allow_pickle=False).astype(np.float32, copy=False)
                                cm_default = None
                                blend_default = None
                                if str(nm) == "P(nucleus)":
                                    cm_default, blend_default = "magenta", "additive"
                                elif str(nm) == "P(nuc_boundary)":
                                    cm_default, blend_default = "yellow", "additive"
                                elif str(nm) == "P(background)":
                                    cm_default, blend_default = "gray", "additive"
                                self._set_or_update_image_layer(str(nm), arr, colormap=cm_default, blending=blend_default)

                    # Nuclei instances
                    nuc = layers.get("nuclei_instances")
                    if nuc and isinstance(nuc, dict):
                        p = self.project.abspath(nuc.get("path", ""))
                        if p.exists():
                            arr = np.load(p, allow_pickle=False).astype(np.int32, copy=False)
                            self._set_or_update_labels_layer("Nuclei", arr)
            except Exception:
                pass

            # Restore layer visibility (+ opacity)
            vis = restore.get("layers") or {}
            if isinstance(vis, dict):
                for nm, st in vis.items():
                    if nm in self.viewer.layers:
                        try:
                            lyr = self.viewer.layers[nm]
                            if isinstance(st, dict):
                                if "visible" in st:
                                    lyr.visible = bool(st["visible"])
                                if "opacity" in st:
                                    try:
                                        lyr.opacity = float(st["opacity"])
                                    except Exception:
                                        pass
                                # Restore style for image layers
                                if "colormap" in st and hasattr(lyr, "colormap"):
                                    try:
                                        lyr.colormap = st["colormap"]
                                    except Exception:
                                        pass
                                if "blending" in st and hasattr(lyr, "blending"):
                                    try:
                                        lyr.blending = st["blending"]
                                    except Exception:
                                        pass
                                if "contrast_limits" in st and hasattr(lyr, "contrast_limits"):
                                    try:
                                        cl = st.get("contrast_limits")
                                        if isinstance(cl, (list, tuple)) and len(cl) == 2:
                                            lyr.contrast_limits = (float(cl[0]), float(cl[1]))
                                    except Exception:
                                        pass
                        except Exception:
                            continue
        finally:
            self._is_restoring = False
    def on_save_project(self) -> None:
        """Persist the project manifest and any trained-but-unsaved artifacts."""
        if self.project is None:
            show_warning("No active project. Create or open a project first.")
            return

        # Save any pending model artifacts
        for item in list(self._pending_models):
            if item.get("saved_path"):
                continue
            model = item.get("model")
            meta_in = item.get("meta") or {}
            try:
                ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
                kind = str(meta_in.get("kind") or "rf")
                out_name = f"{kind}_{ts}.joblib".replace(" ", "_")
                out_path = self.project.models_dir / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Prefer joblib if available; fallback to pickle
                try:
                    import joblib  # type: ignore
                    joblib.dump(model, out_path)
                except Exception:
                    with open(out_path, "wb") as f:
                        pickle.dump(model, f)

                item["saved_path"] = str(out_path)

                meta = dict(meta_in)
                self.project.add_model_record(
                    stage="step2",
                    name=str(meta.get("kind") or "RFModel"),
                    relpath=self.project.relpath(out_path),
                    meta=meta,
                )
            except Exception as e:
                show_warning(f"Failed to save model artifact: {e}")
                return

        # Save layer arrays + UI state
        try:
            self._save_layer_arrays()
            self.project.manifest["ui_state"] = self._gather_ui_state()
            self.project.mark_dirty()
        except Exception:
            pass

        try:
            self.project.save()
        except Exception as e:
            show_warning(f"Failed to save project: {e}")
            return

        self._update_project_label()
        show_info("Project saved.")

    def _install_close_guard(self) -> None:
        """Warn if the user closes napari with unsaved project changes."""
        try:
            qtwin = self.viewer.window._qt_window
        except Exception:
            return

        # Avoid double-wrapping
        if getattr(qtwin, "_cycif_close_guard_installed", False):
            return
        qtwin._cycif_close_guard_installed = True

        orig_close_event = qtwin.closeEvent

        def _guarded_close_event(ev):
            prj = getattr(self, "project", None)
            dirty = bool(prj is not None and getattr(prj, "dirty", False))
            if not dirty:
                return orig_close_event(ev)

            mbox = QtWidgets.QMessageBox(qtwin)
            mbox.setIcon(QtWidgets.QMessageBox.Warning)
            mbox.setWindowTitle("Unsaved project")
            mbox.setText("Your project has unsaved changes.")
            mbox.setInformativeText("Do you want to save before closing?")
            btn_save = mbox.addButton("Save", QtWidgets.QMessageBox.AcceptRole)
            btn_discard = mbox.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
            btn_cancel = mbox.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            mbox.setDefaultButton(btn_save)
            mbox.exec_()

            clicked = mbox.clickedButton()
            if clicked == btn_cancel:
                ev.ignore()
                return
            if clicked == btn_save:
                self.on_save_project()
                # If still dirty (save failed), cancel closing
                if prj is not None and getattr(prj, "dirty", False):
                    ev.ignore()
                    return
            # Discard or saved successfully
            return orig_close_event(ev)

        qtwin.closeEvent = _guarded_close_event

    def on_new_project(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Create/select project folder",
            self._dialog_start_dir(),
        )
        if not root:
            return
        try:
            prj = create_project(Path(root))
        except Exception as e:
            show_warning(f"Failed to create project: {e}")
            return
        self._set_project(prj)

    def on_open_project(self):
        # Allow selecting either a folder containing project.json or the project.json file directly.
        # Start with a folder chooser first (more common UX), then fallback to file chooser if needed.
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open project folder",
            self._dialog_start_dir(),
        )
        if not root:
            # Optional: allow direct selection of project.json
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open project.json",
                self._dialog_start_dir(),
                "Project manifest (project.json);;JSON files (*.json);;All files (*.*)",
            )
            if not path:
                return
            root = path

        try:
            prj = open_project(Path(root))
        except Exception as e:
            show_warning(f"Failed to open project: {e}")
            return
        self._set_project(prj)

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
        w = getattr(self, "_rf_worker", None)
        if w is None:
            return
        try:
            if hasattr(w, "quit"):
                w.quit()
            if hasattr(w, "cancel"):
                w.cancel()
        except Exception:
            pass
        self._rf_worker = None
        try:
            self._prob_refresh_timer.stop()
            self.prog.setVisible(False)
        except Exception:
            pass

    def set_status(self, msg: str):
        self.status.setText(msg)
        show_info(msg)

    def _refresh_prob_layers_if_dirty(self):
        if not getattr(self, "_prob_dirty", False):
            return
        self._prob_dirty = False
        try:
            for nm in ("P(nucleus)", "P(nuc_boundary)", "P(background)"):
                if nm in self.viewer.layers:
                    self.viewer.layers[nm].refresh()
        except Exception:
            pass

    def on_alpha_change(self, v):
        a = float(v) / 100.0
        for lyr in self._prob_layers.values():
            lyr.opacity = a

    def _channel_layer_names(self) -> set[str]:
        return set(self.ch_names or [])

    def sync_displayed_channel_layers(self):
        """
        If 'Display selected channels only' is checked, remove non-selected channel layers
        from the napari layer list. If unchecked, ensure all channel layers are present.
        Scribbles and probability layers are preserved.
        """
        if self.img is None or self.ch_names is None:
            return

        selected = set(self.get_selected_channels())
        display_selected_only = self.chk_display_selected_only.isChecked()

        # Determine which channel indices should be displayed
        if display_selected_only:
            should_display = {i for i in selected}
        else:
            should_display = set(range(len(self.ch_names)))

        # Remove channel layers that should NOT be displayed
        for i, nm in enumerate(self.ch_names):
            if nm in self.viewer.layers and i not in should_display:
                self.viewer.layers.remove(nm)

        # Add missing channel layers that SHOULD be displayed
        for i in sorted(should_display):
            nm = self.ch_names[i]
            if nm not in self.viewer.layers:
                self.viewer.add_image(
                    self.img[..., i],
                    name=nm,
                    blending="additive",
                    opacity=0.6,
                    colormap=self._colormap_for_channel(i),
                )
                try:
                    self._connect_layer_dirty(self.viewer.layers[nm])
                except Exception:
                    pass

        # Keep scribbles on top (re-add if needed)
        if self._scribbles_layer_name not in self.viewer.layers:
            self.ensure_scribbles_layer()
        else:
            # Move to top for painting convenience
            lyr = self.viewer.layers[self._scribbles_layer_name]
            self.viewer.layers.remove(self._scribbles_layer_name)
            self.viewer.layers.append(lyr)

    def on_channel_selection_changed(self):
        """
        Called whenever an item in the channel checklist changes.
        If display-selected-only mode is enabled, update viewer layers immediately.
        """
        if getattr(self, "chk_display_selected_only", None) and self.chk_display_selected_only.isChecked():
            self.sync_displayed_channel_layers()


    def _colormap_for_channel(self, i: int) -> str:
        palette = ["blue", "green", "red", "magenta", "cyan", "yellow"]
        return palette[i % len(palette)]

    def set_all_channels(self, checked: bool):
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def get_selected_channels(self) -> list[int]:
        idxs = []
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                idxs.append(i)
        return idxs

    
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

    def ensure_scribbles_layer(self):
        if self._scribbles_layer_name in self.viewer.layers:
            return self.viewer.layers[self._scribbles_layer_name]

        if self.img is None:
            raise RuntimeError("Load an image first.")

        H, W, _ = self.img.shape
        scrib = np.zeros((H, W), dtype=np.uint8)
        layer = self.viewer.add_labels(
            scrib,
            name=self._scribbles_layer_name,
            opacity=0.6,
        )
        self._connect_layer_dirty(layer)
        return layer

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
        self.ensure_scribbles_layer()

    
    def _load_image_from_path(self, path: str, *, record_input: bool = True) -> None:
        """Load an image from a path (shared by on_load and project restore)."""
        if not path:
            return

        self.path = path
        self.lbl_file.setText(path)

        if record_input and self.project is not None:
            try:
                self.project.add_input(Path(path))
                self._update_project_label()
            except Exception:
                pass

        # Load in a background thread so the UI stays responsive.
        self.set_status("Loading image (background)…")
        self.prog.setRange(0, 0)  # indeterminate/busy
        self.prog.setValue(0)
        try:
            self.btn_load.setEnabled(False)
        except Exception:
            pass

        @thread_worker
        def _load_worker(p):
            return load_multichannel_tiff(p)

        worker = _load_worker(path)

        @worker.returned.connect
        def _on_loaded(result):
            self.img, self.ch_names = result

            # Populate channel list with correct names
            self.list_channels.blockSignals(True)
            self.list_channels.clear()
            for nm in self.ch_names:
                it = QtWidgets.QListWidgetItem(nm)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked)  # default select all
                self.list_channels.addItem(it)
            self.list_channels.blockSignals(False)

            # Add named layers
            self._prob_layers = {}
            self._add_channel_layers()
            self.sync_displayed_channel_layers()

            # If opening a project that requested restore, apply it now
            if self._pending_restore:
                try:
                    self._apply_project_restore(self._pending_restore)
                finally:
                    self._pending_restore = None

            self.set_status(f"Loaded image {self.img.shape} with {len(self.ch_names)} channels.")

            # Done loading
            self.prog.setRange(0, 1)
            self.prog.setValue(1)
            try:
                self.btn_load.setEnabled(True)
            except Exception:
                pass

        @worker.errored.connect
        def _on_load_err(e):
            show_warning(f"Load failed: {e}")
            self.set_status(f"Load failed: {e}")
            self.prog.setRange(0, 1)
            self.prog.setValue(0)
            try:
                self.btn_load.setEnabled(True)
            except Exception:
                pass

        worker.start()

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open multi-channel TIFF/OME-TIFF",
            self._dialog_start_dir(),
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self._load_image_from_path(path, record_input=True)

    def on_train_predict(self):

        self._cancel_rf_worker()
        self._rf_run_id += 1
        my_run_id = self._rf_run_id

        def is_cancelled():
            return my_run_id != self._rf_run_id

        if self.img is None:
            show_warning("Load an image first.")
            return

        use_ch = self.get_selected_channels()
        if len(use_ch) == 0:
            show_warning("Select at least one channel.")
            return

        scrib_layer = self.ensure_scribbles_layer()
        S = np.asarray(scrib_layer.data)

        train_mask = S > 0
        n_train = int(train_mask.sum())
        if n_train < 2000:
            show_warning(f"Not enough labeled pixels ({n_train}). Paint more scribbles.")
            return

        H, W, _ = self.img.shape

        # Allocate probability volume
        self._P = np.zeros((H, W, 3), dtype=np.float32)

        alpha = float(self.slider_alpha.value()) / 100.0

        # Create probability layers if missing
        if "P(nucleus)" not in self.viewer.layers:
            self._prob_layers["P_nuc"] = self.viewer.add_image(
                self._P[..., 0],
                name="P(nucleus)",
                opacity=alpha,
                blending="additive",
                colormap="magenta",
            )
        if "P(nuc_boundary)" not in self.viewer.layers:
            self._prob_layers["P_nb"] = self.viewer.add_image(
                self._P[..., 1],
                name="P(nuc_boundary)",
                opacity=alpha,
                blending="additive",
                colormap="yellow",
            )
        if "P(background)" not in self.viewer.layers:
            self._prob_layers["P_bg"] = self.viewer.add_image(
                self._P[..., 2],
                name="P(background)",
                opacity=alpha,
                blending="additive",
                colormap="gray",
            )


        # Always reset probability layer data to the freshly-zeroed buffer.
        # (If layers already exist from a previous run, they may still be pointing at the old array.)
        name_to_idx = {
            "P(nucleus)": 0,
            "P(nuc_boundary)": 1,
            "P(background)": 2,
        }
        for nm, ii in name_to_idx.items():
            if nm in self.viewer.layers:
                try:
                    self.viewer.layers[nm].data = self._P[..., ii]
                    self.viewer.layers[nm].refresh()
                except Exception:
                    pass

        from cycif_seg.predict.workers import predict_rf_worker
        from cycif_seg.features.multiscale import build_features

        self.set_status("Training + predicting (RF)…")
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)  # indeterminate during training
        self.prog.setValue(0)

        # Choose a priority point near what the user is looking at
        try:
            # camera.center is usually (z, y, x) in 2D display; take last 2
            cy, cx = self.viewer.camera.center[-2], self.viewer.camera.center[-1]
        except Exception:
            # fall back to dims point (y, x)
            cy, cx = float(self.viewer.dims.point[0]), float(self.viewer.dims.point[1])

        center_yx = (float(cy), float(cx))

        worker = predict_rf_worker(
            self.img,
            use_ch,
            S,
            build_features,
            tile_size=512,
            center_yx=center_yx,
            run_id=my_run_id,
            is_cancelled=is_cancelled,
            progress_every=2,
            batch_tiles=4,
            # Feature prefetch to keep CPU busy while RF predicts
            feature_workers=3,
            prefetch_tiles=16,
            rf_n_jobs=12,
        )
        self._rf_worker = worker

        @worker.errored.connect
        def _on_err(e):
            show_warning(f"Prediction failed: {e}")
            self.status.setText(f"Prediction failed: {e}")
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            self.prog.setVisible(False)

        @worker.yielded.connect
        def _on_tile(result):
            kind = result[0]

            if kind == "status":
                _, run_id, msg = result
                if run_id != self._rf_run_id:
                    return
                self.status.setText(str(msg))
                return

            if kind == "trained_model":
                _, run_id, model, meta = result
                if run_id != self._rf_run_id:
                    return
                # Store as a pending artifact; user persists via "Save Project".
                meta = dict(meta or {})
                meta.setdefault("kind", "rf_pixel_model_A")
                self._pending_models.append({"model": model, "meta": meta, "saved_path": None})
                self._mark_project_dirty()
                return

            if kind == "progress_init":
                _, run_id, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                self.prog.setRange(0, int(n_tiles))
                self.prog.setValue(0)
                try:
                    self._prob_refresh_timer.start()
                except Exception:
                    pass
                return

            if kind == "progress":
                _, run_id, i, n_tiles = result
                if run_id != self._rf_run_id:
                    return
                # range already set; update value
                self.prog.setValue(int(i))
                return

            if kind == "tile":
                _, run_id, y0, y1, x0, x1, P_tile = result
                if run_id != self._rf_run_id:
                    return
                # write probabilities
                self._P[y0:y1, x0:x1, :] = P_tile
                # refresh is throttled by a QTimer to avoid per-tile repaint cost
                self._prob_dirty = True
                return
            # Unknown message type; ignore safely.
            return            

        @worker.finished.connect
        def _on_done():
            if my_run_id != self._rf_run_id:
                # stale/cancelled run
                return
            try:
                self._prob_refresh_timer.stop()
            except Exception:
                pass
            # Final refresh
            self._prob_dirty = True
            self._refresh_prob_layers_if_dirty()
            self.prog.setValue(self.prog.maximum())
            self.prog.setVisible(False)
            self.set_status("Prediction complete.")

        worker.start()

    # ---------------------------------------------------------------------
    # Layer helpers
    # ---------------------------------------------------------------------

    def _set_or_update_labels_layer(
        self,
        name: str,
        data: np.ndarray,
        *,
        opacity: float = 0.7,
        visible: bool = True,
    ):
        """
        Create or update a napari Labels layer.
        Keeps the same layer object if it already exists, to preserve user toggles.
        """
        if self.viewer is None:
            return

        # Ensure int labels
        if data.dtype != np.int32 and data.dtype != np.int64:
            data = data.astype(np.int32, copy=False)

        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            # If the existing layer isn't Labels for some reason, replace it
            try:
                layer.data = data
                layer.visible = visible
                # labels layer has opacity in napari
                if hasattr(layer, "opacity"):
                    layer.opacity = opacity
            except Exception:
                # Replace layer defensively
                self.viewer.layers.remove(layer)
                self.viewer.add_labels(data, name=name, opacity=opacity, visible=visible)
        else:
            self.viewer.add_labels(data, name=name, opacity=opacity, visible=visible)


    def _set_or_update_image_layer(
        self,
        name: str,
        data: np.ndarray,
        *,
        opacity: float = 0.7,
        visible: bool = True,
        colormap: str | None = None,
        blending: str | None = None,
        contrast_limits: tuple[float, float] | None = None,
    ):
        """Create or update a napari Image layer."""
        if self.viewer is None:
            return
        if name in self.viewer.layers:
            try:
                lyr = self.viewer.layers[name]
                lyr.data = data
                lyr.opacity = opacity
                lyr.visible = visible
                if colormap is not None:
                    try:
                        lyr.colormap = colormap
                    except Exception:
                        pass
                if blending is not None and hasattr(lyr, 'blending'):
                    try:
                        lyr.blending = blending
                    except Exception:
                        pass
                if contrast_limits is not None and hasattr(lyr, 'contrast_limits'):
                    try:
                        lyr.contrast_limits = contrast_limits
                    except Exception:
                        pass
                self._connect_layer_dirty(lyr)
                return
            except Exception:
                try:
                    del self.viewer.layers[name]
                except Exception:
                    pass
        try:
            kwargs = {'name': name, 'opacity': opacity}
            if colormap is not None:
                kwargs['colormap'] = colormap
            if blending is not None:
                kwargs['blending'] = blending
            self.viewer.add_image(data, **kwargs)
            lyr2 = self.viewer.layers[name]
            lyr2.visible = visible
            if contrast_limits is not None and hasattr(lyr2, 'contrast_limits'):
                try:
                    lyr2.contrast_limits = contrast_limits
                except Exception:
                    pass
            self._connect_layer_dirty(lyr2)
        except Exception:
            pass

    def _get_labels_layer(self, name: str):
        if self.viewer is None:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def on_make_nuclei_edit_layer(self):
        """Create (or refresh) an editable nuclei labels layer for manual edits."""
        nuc_layer = self._get_labels_layer("Nuclei")
        if nuc_layer is None:
            show_warning("Generate nuclei first.")
            return

        nuc = np.asarray(nuc_layer.data).astype(np.int32, copy=False)
        edit = nuc.copy()
        layer = self._set_or_update_labels_layer(self._nuclei_edit_layer_name, edit, opacity=0.7, visible=True)

        # Make it convenient to edit
        try:
            self.viewer.layers.selection.active = self.viewer.layers[self._nuclei_edit_layer_name]
        except Exception:
            pass

        self.set_status(
            "Nuclei edit layer ready. Use Labels paint/erase; use 'New nucleus id' to create a new nucleus label."
        )

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
            return shapes
        except Exception:
            return None

    def _get_nuclei_edit_layer_or_warn(self):
        layer = self._get_labels_layer(self._nuclei_edit_layer_name)
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

    def _clear_nuclei_edit_shapes(self):
        if self.viewer is None:
            return
        if self._nuclei_edit_shapes_layer_name in self.viewer.layers:
            try:
                self.viewer.layers[self._nuclei_edit_shapes_layer_name].data = []
            except Exception:
                pass

    def _last_shape_as_mask(self, shape_kind: str, out_shape, *, thickness: int = 3):
        """Rasterize the last drawn shape in the Shapes layer into a boolean mask.

        shape_kind: 'line' or 'polygon'
        """
        shapes = self._ensure_nuclei_edit_shapes_layer()
        if shapes is None or len(shapes.data) == 0:
            show_warning("Draw a cut line / lasso region first (in the 'Nuclei edit shapes' layer).")
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
                mask = binary_dilation(mask, disk(int(thickness)))
        elif shape_kind == "polygon":
            rr, cc = sk_polygon(coords[:, 0], coords[:, 1], shape=(H, W))
            mask[rr, cc] = True
        else:
            raise ValueError(f"Unknown shape_kind={shape_kind!r}")

        return mask

    # ---------------------------------------------------------------------
    # Nuclei edit interactions (buttons)
    # ---------------------------------------------------------------------

    def on_set_split_draw_mode(self):
        """Activate the Shapes layer for drawing a cut line."""
        shapes = self._ensure_nuclei_edit_shapes_layer()
        if shapes is None:
            return
        try:
            self.viewer.layers.selection.active = shapes
            shapes.mode = "add_path"
        except Exception:
            pass
        self.set_status("Draw a cut line on the 'Nuclei edit shapes' layer, then click 'Split by cut line'.")

    def on_split_by_cut_line(self):
        """Split a nucleus by removing a drawn cut line and relabeling connected components."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return

        data = np.asarray(layer.data).astype(np.int32, copy=False)
        H, W = data.shape

        cut_w = int(self.spin_cut_width.value())
        cut = self._last_shape_as_mask("line", (H, W), thickness=max(1, cut_w))
        if cut is None:
            return

        # Determine which label(s) are intersected by the cut; pick the most frequent nonzero.
        labs = data[cut]
        labs = labs[labs > 0]
        if labs.size == 0:
            show_warning("Cut line does not intersect any nucleus label.")
            return
        target = int(np.bincount(labs).argmax())

        m = data == target
        if not np.any(m):
            show_warning("Selected nucleus not found.")
            return

        # Remove cut pixels from the target nucleus
        m_cut = m & (~cut)
        cc = cc_label(m_cut, connectivity=1)
        n_cc = int(cc.max())
        if n_cc <= 1:
            show_warning("Cut did not split the nucleus (still one component). Try a thicker/longer cut.")
            return

        # Keep largest component as original label; assign new ids to others.
        sizes = [(i, int((cc == i).sum())) for i in range(1, n_cc + 1)]
        sizes.sort(key=lambda t: t[1], reverse=True)

        new_data = data.copy()
        new_data[m] = 0
        new_data[cc == sizes[0][0]] = target

        next_id = int(new_data.max()) + 1
        for comp_id, _sz in sizes[1:]:
            new_data[cc == comp_id] = next_id
            next_id += 1

        layer.data = new_data
        self._clear_nuclei_edit_shapes()
        self.set_status(f"Split nucleus {target} into {n_cc} parts.")

    def on_set_merge_draw_mode(self):
        """Activate the Shapes layer for drawing a lasso polygon."""
        shapes = self._ensure_nuclei_edit_shapes_layer()
        if shapes is None:
            return
        try:
            self.viewer.layers.selection.active = shapes
            shapes.mode = "add_polygon"
        except Exception:
            pass
        self.set_status("Draw a lasso polygon on 'Nuclei edit shapes', then click 'Merge by lasso'.")

    def on_merge_by_lasso(self):
        """Merge all nuclei whose labels intersect the drawn lasso polygon."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return

        data = np.asarray(layer.data).astype(np.int32, copy=False)
        H, W = data.shape

        poly = self._last_shape_as_mask("polygon", (H, W), thickness=1)
        if poly is None:
            return

        labs = np.unique(data[poly])
        labs = labs[labs > 0]
        if labs.size <= 1:
            show_warning("Lasso must include at least two nuclei to merge.")
            return

        target = int(labs.min())
        new_data = data.copy()
        for lbl in labs:
            if int(lbl) == target:
                continue
            new_data[new_data == int(lbl)] = target

        layer.data = new_data
        self._clear_nuclei_edit_shapes()
        self.set_status(f"Merged nuclei {list(map(int, labs))} -> {target}.")

    def on_delete_nucleus(self):
        """Delete the currently selected nucleus label (set to 0)."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        sel = self._get_selected_nucleus_id(layer)
        if sel <= 0:
            show_warning("Select a nucleus label first (use pick mode or click a label).")
            return
        data = np.asarray(layer.data).astype(np.int32, copy=False)
        if not np.any(data == sel):
            show_warning("Selected nucleus label not present.")
            return
        new_data = data.copy()
        new_data[new_data == sel] = 0
        layer.data = new_data
        self.set_status(f"Deleted nucleus {sel}.")

    def on_set_draw_new_nucleus_mode(self):
        """Switch to paint mode on the nuclei edit layer, selecting a fresh ID."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        self.on_new_nucleus_id()
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "paint"
            layer.brush_size = int(self.spin_brush.value())
        except Exception:
            pass
        self.set_status("Paint a new nucleus region in the 'Nuclei edits' layer.")

    def on_set_eraser_mode(self):
        """Switch to erase mode on the nuclei edit layer."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "erase"
            layer.brush_size = int(self.spin_brush.value())
        except Exception:
            pass
        self.set_status("Erase parts of a nucleus in the 'Nuclei edits' layer.")

    def on_erode_nucleus(self):
        """Morphologically erode the selected nucleus by N iterations."""
        layer = self._get_nuclei_edit_layer_or_warn()
        if layer is None:
            return
        sel = self._get_selected_nucleus_id(layer)
        if sel <= 0:
            show_warning("Select a nucleus label first.")
            return

        data = np.asarray(layer.data).astype(np.int32, copy=False)
        m = data == sel
        if not np.any(m):
            show_warning("Selected nucleus label not present.")
            return

        iters = int(self.spin_erode.value())
        st = disk(1)
        m2 = m
        for _ in range(max(1, iters)):
            m2 = binary_erosion(m2, st)
            if not np.any(m2):
                break

        new_data = data.copy()
        new_data[m & (~m2)] = 0
        layer.data = new_data
        self.set_status(f"Eroded nucleus {sel} ({iters} iters).")

    def on_new_nucleus_id(self):
        """Set the active label on the nuclei edit layer to a fresh ID."""
        layer = self._get_labels_layer(self._nuclei_edit_layer_name)
        if layer is None:
            self.on_make_nuclei_edit_layer()
            layer = self._get_labels_layer(self._nuclei_edit_layer_name)
            if layer is None:
                return

        data = np.asarray(layer.data)
        new_id = int(data.max()) + 1
        try:
            layer.selected_label = new_id
        except Exception:
            pass
        self.set_status(f"New nucleus id selected: {new_id}")

    def on_propagate_nuclei_edits(self):
        """Train a pixel RF from edited nuclei labels (as weak supervision) and update probability maps."""
        layer = self._get_labels_layer(self._nuclei_edit_layer_name)
        if layer is None:
            show_warning("Create the nuclei edit layer first (Edit nuclei…).")
            return

        nuclei_labels = np.asarray(layer.data).astype(np.int32, copy=False)
        if nuclei_labels.max() <= 0:
            show_warning("Edited nuclei layer appears empty.")
            return

        # Cancel any in-flight RF work
        self._rf_run_id += 1
        run_id = self._rf_run_id
        if self._rf_worker is not None:
            try:
                self._rf_worker.quit()
            except Exception:
                pass
            self._rf_worker = None

        if self.img is None:
            show_warning("Load an image first.")
            return

        # Ensure probability buffer matches the 3-class model (nuc, boundary, bg)
        H, W = self.img.shape[:2]
        if self._P is None or self._P.shape[:2] != (H, W) or self._P.shape[2] != 3:
            self._P = np.zeros((H, W, 3), dtype=np.float32)
        else:
            self._P[...] = 0.0

        self._ensure_prob_layers()
        self._update_prob_layers(force=True)

        # Resolve active channels
        use_channels = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
        if len(use_channels) == 0:
            use_channels = list(range(self.img.shape[2]))

        from cycif_seg.model.rf_pixel import build_features
        from cycif_seg.predict.workers import propagate_nuclei_edits_worker

        self.set_status("Propagating nuclei edits (RF)…")
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.prog.setValue(0)

        worker = propagate_nuclei_edits_worker(
            self.img,
            use_channels,
            nuclei_labels,
            build_features,
            self.tile_size,
            self.center_yx,
            run_id,
            batch_tiles=self.batch_tiles,
            feature_workers=self.feature_workers,
            progress_every=self.progress_every,
        )
        self._rf_worker = worker

        @worker.yielded.connect
        def _on_msg(msg):
            # msg is tuple tagged by kind
            if not isinstance(msg, tuple) or len(msg) == 0:
                return
            kind = msg[0]

            if kind == "stage":
                _, rid, text, step, total = msg
                if rid != self._rf_run_id:
                    return
                self.status.setText(str(text))
            elif kind == "tile":
                _, rid, y0, y1, x0, x1, P_tile = msg
                if rid != self._rf_run_id:
                    return
                # write probabilities (tile already has 3 channels)
                self._P[y0:y1, x0:x1, : P_tile.shape[2]] = P_tile
                self._prob_dirty = True
            elif kind == "progress":
                _, rid, done, total = msg
                if rid != self._rf_run_id:
                    return
                self.status.setText(f"Propagating edits… tiles {done}/{total}")
            elif kind == "trained_model":
                _, rid, model, meta = msg
                if rid != self._rf_run_id:
                    return
                meta = dict(meta or {})
                meta.setdefault("kind", "rf_from_nuclei_edits")
                self._pending_models.append({"model": model, "meta": meta, "saved_path": None})
                self._mark_project_dirty()
            elif kind == "timing":
                _, rid, which, total_s, avg_s = msg
                if rid != self._rf_run_id:
                    return
                print(f"[TIMING][edits] {which}: total={total_s:.2f}s avg_per_tile={avg_s:.4f}s")

        @worker.finished.connect
        def _on_done():
            if run_id != self._rf_run_id:
                return
            self._update_prob_layers(force=True)
            self.prog.setRange(0, 1)
            self.prog.setValue(1)
            self.prog.setVisible(False)
            self.set_status("Edits propagated. You can now Generate nuclei.")

        worker.start()
        
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
                self._set_or_update_labels_layer("Nuclei", nuclei.astype(np.int32, copy=False))
                self.set_status(f"Nuclei generated: N={int(nuclei.max())}")
            elif kind == "debug":
                if not self.chk_show_nuc_markers.isChecked():
                    return
                _, dbg = msg
                # Show a few helpful overlays when requested
                try:
                    if "seeds" in dbg:
                        self._set_or_update_labels_layer("Nucleus seeds", dbg["seeds"].astype(np.int32, copy=False))
                    if "core_mask" in dbg:
                        self._set_or_update_labels_layer("Nucleus core mask", dbg["core_mask"].astype(np.uint8, copy=False))
                    if "boundary_mask" in dbg:
                        self._set_or_update_labels_layer("Nucleus boundary mask", dbg["boundary_mask"].astype(np.uint8, copy=False))
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