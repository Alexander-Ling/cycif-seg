from __future__ import annotations

import datetime
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

from cycif_seg.project import CycIFProject, create_project, open_project
from napari.utils.notifications import show_info, show_warning


class ProjectController:
    """Owns project create/open/save + UI/layer state persistence.

    This is extracted from the (very large) CycIFMVPWidget to make future edits safer.
    The controller is intentionally UI-facing (uses Qt dialogs) but keeps napari-layer
    access through the widget's LayerManager.
    """

    def __init__(self, widget: Any):
        # widget is CycIFMVPWidget (kept as Any to avoid import cycles)
        self.w = widget

    # ----------------------------
    # Dialog helpers
    # ----------------------------
    def dialog_start_dir(self) -> str:
        prj = getattr(self.w, "project", None)
        if prj is not None:
            return str(prj.root)
        return os.getcwd()

    # ----------------------------
    # Project lifecycle
    # ----------------------------
    def set_project(self, prj: CycIFProject) -> None:
        # If current project has unsaved changes, warn before switching
        cur = getattr(self.w, "project", None)
        if cur is not None and getattr(cur, "dirty", False):
            resp = QtWidgets.QMessageBox.question(
                self.w,
                "Unsaved project changes",
                "Current project has unsaved changes. Save before switching projects?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if resp == QtWidgets.QMessageBox.Cancel:
                return
            if resp == QtWidgets.QMessageBox.Yes:
                self.save_project()

        self.w.project = prj

        # Clear pending artifacts when switching projects
        try:
            self.w._pending_models = []
        except Exception:
            pass

        # If the project includes saved UI/layer state, restore it after image load
        ui_state = (prj.manifest.get("ui_state") or {})
        self.w._pending_restore = ui_state if isinstance(ui_state, dict) else None

        self.update_project_label()
        show_info(f"Project set: {prj.root}")

        # Auto-load the last image if recorded
        try:
            last = None
            if isinstance(self.w._pending_restore, dict):
                last = self.w._pending_restore.get("last_image")
            if last:
                p = Path(last)
                if not p.is_absolute():
                    p = prj.abspath(str(last))
                if p.exists():
                    self.w._load_image_from_path(str(p), record_input=False)
                else:
                    if self.w._pending_restore:
                        self.apply_project_restore(self.w._pending_restore)
                        self.w._pending_restore = None
        except Exception:
            pass

    def update_project_label(self) -> None:
        if getattr(self.w, "project", None) is None:
            try:
                self.w.lbl_project.setText("(no project)")
            except Exception:
                pass
            return
        prj = self.w.project
        star = "*" if getattr(prj, "dirty", False) else ""
        try:
            self.w.lbl_project.setText(f"{prj.root}{star}")
        except Exception:
            pass

    def mark_project_dirty(self) -> None:
        prj = getattr(self.w, "project", None)
        if prj is None:
            return
        try:
            prj.mark_dirty()
        except Exception:
            pass
        self.update_project_label()

    # ----------------------------
    # UI/layer state capture/restore
    # ----------------------------
    def gather_ui_state(self) -> dict:
        state: dict = {}
        w = self.w

        # Channels UI
        selected: list[str] = []
        try:
            for i in range(w.list_channels.count()):
                it = w.list_channels.item(i)
                if it is not None and it.checkState() == QtCore.Qt.Checked:
                    selected.append(str(it.text()))
        except Exception:
            pass
        state["selected_channels"] = selected

        try:
            state["displayed_channel_indices"] = list(getattr(w, "_displayed_channel_indices", []) or [])
        except Exception:
            state["displayed_channel_indices"] = []

        if getattr(w, "path", None):
            state["last_image"] = str(w.path)

        # Layer visual state
        layers: dict[str, dict] = {}
        try:
            for lyr in list(w.viewer.layers):
                try:
                    st: dict[str, Any] = {
                        "visible": bool(getattr(lyr, "visible", True)),
                        "opacity": float(getattr(lyr, "opacity", 1.0)),
                        "type": str(getattr(lyr, "_type_string", lyr.__class__.__name__)),
                    }
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

        try:
            state["active_tab"] = int(w.tabs.currentIndex()) if hasattr(w, "tabs") else None
        except Exception:
            state["active_tab"] = None

        return state

    def save_layer_arrays(self) -> None:
        w = self.w
        prj = getattr(w, "project", None)
        if prj is None:
            return

        prj.manifest.setdefault("step2", {})
        step2 = prj.manifest["step2"]
        step2.setdefault("layers", {})
        out = step2["layers"]
        saved_any = False

        def _save_npy(name: str, arr: np.ndarray) -> str:
            fn = f"{name}.npy".replace(" ", "_").replace("/", "_")
            p = prj.data_dir / fn
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, arr)
            return prj.relpath(p)

        # Scribbles
        try:
            if w._scribbles_layer_name in w.viewer.layers:
                arr = np.asarray(w.viewer.layers[w._scribbles_layer_name].data)
                out["scribbles"] = {
                    "name": w._scribbles_layer_name,
                    "path": _save_npy("scribbles", arr.astype(np.uint8, copy=False)),
                }
                saved_any = True
        except Exception:
            pass

        # Probability maps
        prob_paths: dict[str, str] = {}
        for nm in ["P(nucleus)", "P(nuc_boundary)", "P(background)"]:
            if nm in w.viewer.layers:
                try:
                    arr = np.asarray(w.viewer.layers[nm].data).astype(np.float32, copy=False)
                    prob_paths[nm] = _save_npy(nm, arr)
                    saved_any = True
                except Exception:
                    continue
        if prob_paths:
            out["probability_maps"] = prob_paths

        # Nuclei
        if "Nuclei" in w.viewer.layers:
            try:
                arr = np.asarray(w.viewer.layers["Nuclei"].data).astype(np.int32, copy=False)
                out["nuclei_instances"] = {"name": "Nuclei", "path": _save_npy("nuclei_instances", arr)}
                saved_any = True
            except Exception:
                pass

        if saved_any:
            prj.mark_dirty()

    def apply_project_restore(self, restore: dict) -> None:
        w = self.w
        if not restore:
            return

        w._is_restoring = True
        try:
            # Active tab
            try:
                tab_idx = restore.get("active_tab", None)
                if tab_idx is not None and hasattr(w, "tabs"):
                    tab_idx = int(tab_idx)
                    if 0 <= tab_idx < w.tabs.count():
                        w.tabs.setCurrentIndex(tab_idx)
            except Exception:
                pass

            # Selected channels
            sel = set(restore.get("selected_channels") or [])
            if sel and w.list_channels.count() > 0:
                try:
                    w.list_channels.blockSignals(True)
                    for i in range(w.list_channels.count()):
                        it = w.list_channels.item(i)
                        if it is None:
                            continue
                        it.setCheckState(QtCore.Qt.Checked if str(it.text()) in sel else QtCore.Qt.Unchecked)
                finally:
                    try:
                        w.list_channels.blockSignals(False)
                    except Exception:
                        pass

            # Displayed channels
            try:
                disp_idx = restore.get("displayed_channel_indices", None)
                if disp_idx is not None and isinstance(disp_idx, (list, tuple)):
                    disp_set = set(int(x) for x in disp_idx)
                    try:
                        w.list_channels.blockSignals(True)
                        original_checks = [w.list_channels.item(i).checkState() for i in range(w.list_channels.count())]
                        for i in range(w.list_channels.count()):
                            it = w.list_channels.item(i)
                            if it is None:
                                continue
                            it.setCheckState(QtCore.Qt.Checked if i in disp_set else QtCore.Qt.Unchecked)
                    finally:
                        try:
                            w.list_channels.blockSignals(False)
                        except Exception:
                            pass

                    w.sync_displayed_channel_layers()

                    # Restore original checks
                    try:
                        w.list_channels.blockSignals(True)
                        for i in range(w.list_channels.count()):
                            it = w.list_channels.item(i)
                            if it is None:
                                continue
                            it.setCheckState(original_checks[i])
                    finally:
                        try:
                            w.list_channels.blockSignals(False)
                        except Exception:
                            pass

                    w._update_apply_channels_button_state()
                else:
                    w.sync_displayed_channel_layers()
            except Exception:
                try:
                    w.sync_displayed_channel_layers()
                except Exception:
                    pass

            # Restore layer arrays from manifest if present
            try:
                prj = getattr(w, "project", None)
                if prj is not None:
                    step2 = (prj.manifest.get("step2") or {})
                    layers = (step2.get("layers") or {})
                    if isinstance(layers, dict):
                        # scribbles
                        try:
                            scrib = layers.get("scribbles")
                            if isinstance(scrib, dict):
                                relp = scrib.get("path")
                                if relp:
                                    p = prj.abspath(relp)
                                    if p.exists():
                                        arr = np.load(p, allow_pickle=False).astype(np.uint8, copy=False)
                                        # LayerManager.ensure_scribbles_layer expects kwargs (image_shape, name)
                                        w.layers.ensure_scribbles_layer(image_shape=arr.shape, name=w._scribbles_layer_name)
                                        w.viewer.layers[w._scribbles_layer_name].data = arr
                        except Exception:
                            pass

                        # prob maps
                        try:
                            prob = layers.get("probability_maps") or {}
                            if isinstance(prob, dict):
                                for nm, rel in prob.items():
                                    try:
                                        p = prj.abspath(rel)
                                        if p.exists():
                                            arr = np.load(p, allow_pickle=False).astype(np.float32, copy=False)
                                            w.layers.set_or_update_image(str(nm), arr)
                                    except Exception:
                                        continue
                        except Exception:
                            pass

                        # nuclei
                        try:
                            nuc = layers.get("nuclei_instances")
                            if isinstance(nuc, dict):
                                relp = nuc.get("path")
                                if relp:
                                    p = prj.abspath(relp)
                                    if p.exists():
                                        arr = np.load(p, allow_pickle=False).astype(np.int32, copy=False)
                                        w.layers.set_or_update_labels("Nuclei", arr)
                        except Exception:
                            pass
            except Exception:
                pass

            # Restore visibility & style
            vis = restore.get("layers") or {}
            if isinstance(vis, dict):
                for nm, st in vis.items():
                    if nm in w.viewer.layers:
                        try:
                            lyr = w.viewer.layers[nm]
                            if isinstance(st, dict):
                                if "visible" in st:
                                    lyr.visible = bool(st["visible"])
                                if "opacity" in st:
                                    try:
                                        lyr.opacity = float(st["opacity"])
                                    except Exception:
                                        pass
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
            w._is_restoring = False

    # ----------------------------
    # Qt actions (wired from buttons)
    # ----------------------------
    def new_project(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self.w, "Create/select project folder", self.dialog_start_dir()
        )
        if not root:
            return
        try:
            prj = create_project(Path(root))
        except Exception as e:
            show_warning(f"Failed to create project: {e}")
            return
        self.set_project(prj)

    def open_project(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self.w, "Open project folder", self.dialog_start_dir()
        )
        if not root:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.w,
                "Open project.json",
                self.dialog_start_dir(),
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
        self.set_project(prj)

    def save_project(self) -> None:
        w = self.w
        prj = getattr(w, "project", None)
        if prj is None:
            show_warning("No active project. Create or open a project first.")
            return

        # Save pending model artifacts
        for item in list(getattr(w, "_pending_models", []) or []):
            if item.get("saved_path"):
                continue
            model = item.get("model")
            meta_in = item.get("meta") or {}
            try:
                ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
                kind = str(meta_in.get("kind") or "rf")
                out_name = f"{kind}_{ts}.joblib".replace(" ", "_")
                out_path = prj.models_dir / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    import joblib  # type: ignore
                    joblib.dump(model, out_path)
                except Exception:
                    with open(out_path, "wb") as f:
                        pickle.dump(model, f)

                item["saved_path"] = str(out_path)

                meta = dict(meta_in)
                prj.add_model_record(
                    stage="step2",
                    name=str(meta.get("kind") or "RFModel"),
                    relpath=prj.relpath(out_path),
                    meta=meta,
                )
            except Exception as e:
                show_warning(f"Failed to save model artifact: {e}")
                return

        # Save layer arrays + UI state
        try:
            self.save_layer_arrays()
            prj.manifest["ui_state"] = self.gather_ui_state()
            prj.mark_dirty()
        except Exception:
            pass

        try:
            prj.save()
        except Exception as e:
            show_warning(f"Failed to save project: {e}")
            return

        self.update_project_label()
        show_info("Project saved.")

    # ----------------------------
    # Close guard
    # ----------------------------
    def install_close_guard(self) -> None:
        w = self.w
        try:
            qtwin = w.viewer.window._qt_window
        except Exception:
            return

        if getattr(qtwin, "_cycif_close_guard_installed", False):
            return
        qtwin._cycif_close_guard_installed = True

        orig_close_event = qtwin.closeEvent

        def _guarded_close_event(ev):
            prj = getattr(w, "project", None)
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
                self.save_project()
                if prj is not None and getattr(prj, "dirty", False):
                    ev.ignore()
                    return
            return orig_close_event(ev)

        qtwin.closeEvent = _guarded_close_event
