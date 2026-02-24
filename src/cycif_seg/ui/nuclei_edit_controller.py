"""Step 2b nuclei touch-up tools.

This module keeps the Step 2b edit logic out of the very large
`CycIFMVPWidget` class, to reduce indentation/syntax regressions.

The controller is UI-agnostic: the main widget provides callbacks for status
and warnings, and a function to fetch the Labels layer used for editing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from napari.utils.notifications import show_warning

from scipy.ndimage import label as cc_label

from skimage.draw import line as sk_line
from skimage.morphology import dilation, disk


@dataclass
class _LineDrawState:
    action: str
    callback: object


class NucleiEditController:
    """Controller for Step 2b nuclei edit interactions."""

    def __init__(
        self,
        viewer,
        *,
        get_labels_layer: Callable[[], object | None],
        status_cb: Callable[[str], None],
        warn_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.viewer = viewer
        self._get_labels_layer = get_labels_layer
        self._status = status_cb
        self._warn = warn_cb or show_warning

        self._line_state: Optional[_LineDrawState] = None
        self._delete_callback = None
        self._draw_new_callback = None
        self._eraser_callback = None

        # For erode splitting: snapshot + touched labels during a stroke
        self._erode_snapshot = None
        self._erode_touched: set[int] = set()

        # Draw-new preview (a Labels layer for exact brush footprint)
        self._preview_layer_name = "Nuclei (draw preview)"
        self._preview_layer = None
        # Preserve napari/vispy overlay callbacks by never overwriting the callback lists.
        # We keep a saved copy of the original callback lists per-mode and restore them on disable.
        self._saved_mouse_drag_callbacks: dict[str, list] = {}
        self._saved_mouse_move_callbacks: dict[str, list] = {}

        # Preserve/restore Labels/Shapes layer interaction mode per tool key.
        # Without explicitly switching away from napari's paint mode, non-brush tools
        # can behave like a brush; and we want to restore the prior mode when disabling.
        self._saved_layer_modes: dict[str, object] = {}

        # Track the currently enabled tool and which layer should be active while it is enabled.
        self._active_tool: str | None = None
        self._expected_active_layer = None
        self._warned_wrong_layer: bool = False

        # Best-effort: warn if the user changes the active layer away from what the current tool expects.
        try:
            self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
        except Exception:
            pass

    def _activate_layer(self, layer) -> None:
        """Make a layer the active layer (best-effort)."""
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

    def _expect_active_layer(self, layer, *, tool: str) -> None:
        """Record which layer should be active while a tool is enabled."""
        self._active_tool = tool
        self._expected_active_layer = layer
        self._warned_wrong_layer = False

    def _clear_expected_active_layer(self, *, tool: str | None = None) -> None:
        """Clear the active-layer expectation (optionally only if tool matches)."""
        if tool is not None and self._active_tool != tool:
            return
        self._active_tool = None
        self._expected_active_layer = None
        self._warned_wrong_layer = False

    def _on_active_layer_changed(self, event=None) -> None:
        """Warn if a tool is enabled but the user switches away from the expected layer."""
        if self._active_tool is None or self._expected_active_layer is None:
            return
        try:
            active = self.viewer.layers.selection.active
        except Exception:
            return
        if active is self._expected_active_layer:
            self._warned_wrong_layer = False
            return
        # Avoid spamming: warn once per mismatch until they return.
        if not self._warned_wrong_layer:
            exp_name = getattr(self._expected_active_layer, "name", "(expected layer)")
            act_name = getattr(active, "name", "(active layer)")
            self._warn(
                f"'{self._active_tool}' is enabled but the active layer is '{act_name}'. "
                f"Select '{exp_name}' to use this tool."
            )
            self._warned_wrong_layer = True

    def _push_undo(self, layer) -> None:
        """Best-effort: record current layer state in napari's undo history."""
        try:
            # napari exposes history via private API on some versions
            if hasattr(layer, "_save_history"):
                layer._save_history()
        except Exception:
            pass

    def _get_active_layer(self):
        try:
            return getattr(self.viewer.layers.selection, "active", None)
        except Exception:
            return None

    def _restore_active_layer(self, active) -> None:
        if active is None:
            return
        try:
            self.viewer.layers.selection.active = active
        except Exception:
            pass

    def _set_layer_mode(self, layer, *, key: str, mode: str) -> None:
        """Set layer.mode while preserving previous mode for later restore."""
        try:
            if key not in self._saved_layer_modes:
                self._saved_layer_modes[key] = getattr(layer, "mode", None)
            layer.mode = mode
        except Exception:
            pass

    def _restore_layer_mode(self, layer, *, key: str) -> None:
        """Restore layer.mode if it was saved for this key."""
        if key not in self._saved_layer_modes:
            return
        prev = self._saved_layer_modes.pop(key)
        try:
            if prev is not None:
                layer.mode = prev
        except Exception:
            pass

    def _get_or_create_line_preview_layer(self):
        """Create (or fetch) a Shapes layer used to preview split/merge lines."""
        try:
            viewer = self.viewer
        except Exception:
            return None
        name = "Nuclei (line preview)"
        try:
            if name in viewer.layers:
                return viewer.layers[name]
        except Exception:
            pass
        # Creating a layer can steal focus (becoming the active layer).
        # Preserve the current active layer so tools don't appear "broken".
        prev_active = self._get_active_layer()
        try:
            # Create an empty Shapes layer for a single path.
            lyr = viewer.add_shapes(
                name=name,
                shape_type="path",
                edge_width=2,
                face_color="transparent",
            )
            return lyr
        except Exception:
            return None
        finally:
            self._restore_active_layer(prev_active)


        # Preserve/restore Labels layer interaction mode per tool.
        # If we don't switch away from "paint" for non-brush tools, everything
        # behaves like a brush. We also want to restore the user's prior mode.
        
    def _install_mouse_callbacks(self, layer, *, drag_cb=None, move_cb=None, key: str) -> None:
        """Install callbacks without clobbering existing overlay callbacks.

        Napari overlays (e.g. LabelsPolygonOverlay) rely on their callbacks remaining present in
        layer.mouse_drag_callbacks / layer.mouse_move_callbacks. Replacing the entire list
        breaks overlay teardown and can raise ValueError during viewer updates.
        """
        if drag_cb is not None:
            if key not in self._saved_mouse_drag_callbacks:
                self._saved_mouse_drag_callbacks[key] = list(getattr(layer, "mouse_drag_callbacks", []))
            cur = list(getattr(layer, "mouse_drag_callbacks", []))
            if drag_cb not in cur:
                cur.append(drag_cb)
            layer.mouse_drag_callbacks = cur

        if move_cb is not None:
            if key not in self._saved_mouse_move_callbacks:
                self._saved_mouse_move_callbacks[key] = list(getattr(layer, "mouse_move_callbacks", []))
            cur = list(getattr(layer, "mouse_move_callbacks", []))
            if move_cb not in cur:
                cur.append(move_cb)
            layer.mouse_move_callbacks = cur

    def _restore_mouse_callbacks(self, layer, *, key: str) -> None:
        """Restore callback lists saved by _install_mouse_callbacks."""
        if key in self._saved_mouse_drag_callbacks:
            layer.mouse_drag_callbacks = self._saved_mouse_drag_callbacks.pop(key)
        if key in self._saved_mouse_move_callbacks:
            layer.mouse_move_callbacks = self._saved_mouse_move_callbacks.pop(key)


    # ------------------------------------------------------------------
    # Public toggles (called by the widget)
    # ------------------------------------------------------------------

    def toggle_split(self, enabled: bool) -> None:
        layer = self._require_layer()
        if layer is None:
            return
        if enabled:
            self._disable_all_modes(layer, keep_line=False)
            self._activate_layer(layer)
            self._expect_active_layer(layer, tool="Split")
            self._enable_line_mode(layer, action="split")
            self._status("Split (cut line): draw a cut line (release mouse to apply).")
        else:
            self._disable_line_mode(layer)
            self._clear_expected_active_layer(tool="Split")
            self._status("Split mode: off.")

    def toggle_merge(self, enabled: bool) -> None:
        layer = self._require_layer()
        if layer is None:
            return
        if enabled:
            self._disable_all_modes(layer, keep_line=False)
            self._activate_layer(layer)
            self._expect_active_layer(layer, tool="Merge")
            self._enable_line_mode(layer, action="merge")
            self._status("Merge (line): draw a line crossing nuclei to merge (release mouse to apply).")
        else:
            self._disable_line_mode(layer)
            self._clear_expected_active_layer(tool="Merge")
            self._status("Merge mode: off.")

    def toggle_delete(self, enabled: bool) -> None:
        layer = self._require_layer()
        if layer is None:
            return
        if enabled:
            self._disable_all_modes(layer, keep_line=False)
            self._activate_layer(layer)
            self._expect_active_layer(layer, tool="Delete")
            self._enable_delete_mode(layer)
            self._status("Delete nucleus mode: click inside a nucleus to delete it (toggle off to stop).")
        else:
            self._disable_delete_mode(layer)
            self._clear_expected_active_layer(tool="Delete")
            self._status("Delete nucleus mode: off.")

    def toggle_draw_new(self, enabled: bool) -> None:
        layer = self._require_layer()
        if layer is None:
            return
        if enabled:
            self._disable_all_modes(layer, keep_line=False)
            self._enable_draw_new_mode(layer)
            self._status("Draw new nucleus: paint a nucleus shape (release to commit).")
        else:
            self._disable_draw_new_mode(layer)
            self._clear_expected_active_layer(tool="Draw new")
            self._status("Draw new nucleus: off.")

    def toggle_eraser(self, enabled: bool) -> None:
        layer = self._require_layer()
        if layer is None:
            return
        if enabled:
            self._disable_all_modes(layer, keep_line=False)
            self._activate_layer(layer)
            self._expect_active_layer(layer, tool="Erode")
            self._enable_erode_mode(layer)
            self._status("Erode (eraser): erase parts of nuclei; release to auto-split disconnected pieces.")
        else:
            self._disable_erode_mode(layer)
            self._clear_expected_active_layer(tool="Erode")
            self._status("Erode mode: off.")

    # ------------------------------------------------------------------
    # Line actions (called by the widget OR internally)
    # ------------------------------------------------------------------

    def apply_split(self, points_rc: list[tuple[float, float]] | None) -> None:
        layer = self._require_layer()
        if layer is None:
            return

        data = np.asarray(layer.data).astype(np.int32, copy=False)
        H, W = data.shape
        cut = self._polyline_to_mask(points_rc, (H, W), thickness=1)
        if cut is None:
            return

        touched = np.unique(data[cut])
        touched = touched[touched > 0]
        if touched.size == 0:
            self._warn("Cut line does not intersect any nucleus label.")
            return

        new_data = data.copy()
        next_id = int(new_data.max()) + 1
        reports: list[tuple[int, int]] = []  # (target, created)

        for target in map(int, touched.tolist()):
            m = new_data == target
            if not np.any(m):
                continue

            m_cut = m & (~cut)
            cc, _n = cc_label(m_cut)
            n_cc = int(cc.max())
            if n_cc <= 1:
                continue

            orig_area = int(m.sum())
            min_keep = int(max(50, 0.01 * orig_area))

            sizes = [(i, int((cc == i).sum())) for i in range(1, n_cc + 1)]
            sizes.sort(key=lambda t: t[1], reverse=True)

            new_data[m] = 0
            keep_comp = sizes[0][0]
            new_data[cc == keep_comp] = target

            created = 0
            for comp_id, sz in sizes[1:]:
                if sz < min_keep:
                    new_data[cc == comp_id] = target
                else:
                    new_data[cc == comp_id] = next_id
                    next_id += 1
                    created += 1

            reports.append((target, created))

        if not reports:
            self._warn("Cut did not split any nucleus into multiple components.")
            return

        self._set_labels(layer, new_data)
        self._status(
            "Split by cut line: "
            + ", ".join([f"{t} (+{c})" if c else f"{t} (fragments merged)" for t, c in reports])
            + "."
        )

    def apply_merge(self, points_rc: list[tuple[float, float]] | None) -> None:
        layer = self._require_layer()
        if layer is None:
            return

        data = np.asarray(layer.data).astype(np.int32, copy=False)
        H, W = data.shape
        mask = self._polyline_to_mask(points_rc, (H, W), thickness=2)
        if mask is None:
            return

        labs = np.unique(data[mask])
        labs = labs[labs > 0]
        if labs.size <= 1:
            self._warn("Merge line must cross at least two nuclei.")
            return

        target = int(labs.min())
        new_data = data.copy()
        for lbl in map(int, labs.tolist()):
            if lbl != target:
                new_data[new_data == lbl] = target

        self._set_labels(layer, new_data)
        self._status(f"Merged nuclei {list(map(int, labs))} -> {target}.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_layer(self):
        layer = self._get_labels_layer()
        if layer is None:
            self._warn("No nuclei edit layer found. Generate nuclei first.")
            return None
        return layer

    def _disable_all_modes(self, layer, *, keep_line: bool) -> None:
        self._disable_delete_mode(layer)
        self._disable_draw_new_mode(layer)
        self._disable_erode_mode(layer)
        if not keep_line:
            self._disable_line_mode(layer)

    # ----- line draw mode -----

    def _enable_line_mode(self, layer, *, action: str) -> None:
        # Disable any existing line mode and show a visible preview line layer.
        self._disable_line_mode(layer)

        # IMPORTANT: keep the viewer from panning / "move camera" while the user draws.
        # Using "pick" mode on the Labels layer is the most compatible across napari versions.
        self._set_layer_mode(layer, key="line_draw", mode="pick")

        preview = self._get_or_create_line_preview_layer()
        if preview is not None:
            try:
                preview.visible = True
                # Clear any prior preview geometry.
                preview.data = []
            except Exception:
                pass

        cb = self._make_line_draw_callback(action)
        self._line_state = _LineDrawState(action=action, callback=cb)
        self._install_mouse_callbacks(layer, drag_cb=cb, key="line_draw")

    def _disable_line_mode(self, layer) -> None:
        if self._line_state is None:
            return
        self._restore_mouse_callbacks(layer, key="line_draw")
        self._restore_layer_mode(layer, key="line_draw")
        self._line_state = None

        preview = self._get_or_create_line_preview_layer()
        if preview is not None:
            try:
                preview.data = []
                preview.visible = False
            except Exception:
                pass

    def _make_line_draw_callback(self, action: str):
        # We draw a visual preview line in a Shapes layer while dragging,
        # then apply split/merge on release.
        def _cb(layer, event):
            points: list[tuple[float, float]] = []
            preview = self._get_or_create_line_preview_layer()

            if event.type == "mouse_press":
                try:
                    pos = event.position
                    points.append((float(pos[-2]), float(pos[-1])))
                except Exception:
                    pass

                # Initialize preview
                if preview is not None:
                    try:
                        preview.data = [np.asarray(points, dtype=float)]
                    except Exception:
                        pass

            yield

            while event.type == "mouse_move":
                try:
                    pos = event.position
                    rr, cc = float(pos[-2]), float(pos[-1])
                    if (not points) or (abs(rr - points[-1][0]) + abs(cc - points[-1][1]) >= 0.5):
                        points.append((rr, cc))
                        if preview is not None:
                            try:
                                preview.data = [np.asarray(points, dtype=float)]
                            except Exception:
                                pass
                except Exception:
                    pass
                yield

            if event.type != "mouse_release":
                return

            if preview is not None:
                try:
                    preview.data = []
                except Exception:
                    pass

            if len(points) < 2:
                return

            # Undo support for the programmatic edit.
            self._push_undo(layer)

            if action == "split":
                self.apply_split(points)
            elif action == "merge":
                self.apply_merge(points)

        return _cb

    # ----- delete mode -----

    def _enable_delete_mode(self, layer) -> None:
        self._disable_delete_mode(layer)
        # Use pick (or a neutral mode) so clicks don't paint.
        self._set_layer_mode(layer, key="delete", mode="pick")
        if self._delete_callback is None:
            self._delete_callback = self._make_delete_callback()
        self._install_mouse_callbacks(layer, drag_cb=self._delete_callback, key="delete")

    def _disable_delete_mode(self, layer) -> None:
        if self._delete_callback is None:
            return
        self._restore_mouse_callbacks(layer, key="delete")
        self._restore_layer_mode(layer, key="delete")

    def _make_delete_callback(self):
        def _cb(layer, event):
            yield
            if event.type != "mouse_release":
                return
            try:
                pos = event.position
                r = int(round(float(pos[-2])))
                c = int(round(float(pos[-1])))
            except Exception:
                return
            data = np.asarray(layer.data).astype(np.int32, copy=False)
            if r < 0 or c < 0 or r >= data.shape[0] or c >= data.shape[1]:
                return
            lbl = int(data[r, c])
            if lbl <= 0:
                return
            new_data = data.copy()
            new_data[new_data == lbl] = 0
            self._set_labels(layer, new_data)
            self._status(f"Deleted nucleus {lbl}.")

        return _cb

    # ----- draw new mode -----

    def _ensure_preview_layer(self, shape: tuple[int, int]):
        if self._preview_layer is not None:
            return self._preview_layer
        if self._preview_layer_name in self.viewer.layers:
            self._preview_layer = self.viewer.layers[self._preview_layer_name]
            return self._preview_layer
        # Creating a layer can steal focus (becoming the active layer).
        # Preserve the current active layer so tools remain usable.
        prev_active = self._get_active_layer()
        self._preview_layer = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32),
            name=self._preview_layer_name,
            opacity=0.4,
        )
        try:
            self._preview_layer.editable = False
        except Exception:
            pass
        self._restore_active_layer(prev_active)
        return self._preview_layer

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def ensure_preview_layers(self, *, image_shape: tuple[int, int] | None = None) -> None:
        """Ensure preview layers exist (line preview + draw preview) without stealing focus.

        Called when the user creates/selects the nuclei edit layer so that toggling tools later
        doesn't create new layers that become active and make tools appear broken.
        """
        # Line preview (shapes)
        preview_line = self._get_or_create_line_preview_layer()
        if preview_line is not None:
            try:
                preview_line.visible = False
                preview_line.data = []
            except Exception:
                pass

        # Draw preview (labels)
        if image_shape is None:
            try:
                base = self._require_layer()
                if base is not None:
                    image_shape = (int(base.data.shape[0]), int(base.data.shape[1]))
            except Exception:
                image_shape = None
        if image_shape is not None:
            preview_draw = self._ensure_preview_layer(tuple(image_shape))
            if preview_draw is not None:
                try:
                    preview_draw.visible = False
                    preview_draw.data = 0
                except Exception:
                    pass

    def _clear_draw_preview(self) -> None:
        if self._preview_layer is None:
            return
        try:
            self._preview_layer.data = 0
        except Exception:
            try:
                self._preview_layer.data[:] = 0
            except Exception:
                pass
        try:
            self._preview_layer.visible = False
        except Exception:
            pass

    def _reset_draw_preview(self, *, keep_visible: bool = True) -> None:
        """Clear the preview mask but keep the layer visible/paintable if desired."""
        if self._preview_layer is None:
            return
        try:
            self._preview_layer.data = 0
        except Exception:
            try:
                self._preview_layer.data[:] = 0
            except Exception:
                pass
        if keep_visible:
            try:
                self._preview_layer.visible = True
            except Exception:
                pass

    def _enable_draw_new_mode(self, layer) -> None:
        # Draw-new uses native napari paint on a dedicated preview Labels layer for speed.
        self._disable_draw_new_mode(layer)

        data0 = np.asarray(layer.data)
        if data0.ndim != 2:
            # This controller currently expects 2D labels.
            self._status("Draw new nucleus currently supports 2D label layers only.")
            return

        H, W = data0.shape
        preview = self._ensure_preview_layer((H, W))
        try:
            preview.visible = True
            preview.opacity = 0.4
        except Exception:
            pass

        # Make preview paintable.
        try:
            preview.editable = True
        except Exception:
            pass
        try:
            preview.data = np.zeros((H, W), dtype=np.int32)
        except Exception:
            pass

        # Route user strokes to preview layer paint (smooth + no camera pan).
        self._set_layer_mode(preview, key="draw_new_preview", mode="paint")
        try:
            # Use label 1 in preview; mask is preview.data > 0
            preview.selected_label = 1
        except Exception:
            pass
        try:
            # Keep brush size consistent with the main edit layer if present.
            preview.brush_size = int(getattr(layer, "brush_size", getattr(preview, "brush_size", 10)))
        except Exception:
            pass

        # Make preview active so painting doesn't move the camera / affect other layers.
        try:
            self.viewer.layers.selection.active = preview
        except Exception:
            pass

        # While draw-new is enabled, the preview layer must stay active.
        self._expect_active_layer(preview, tool="Draw new")

        # Commit on mouse release.
        if self._draw_new_callback is None:
            self._draw_new_callback = self._make_draw_new_callback()
        self._install_mouse_callbacks(preview, drag_cb=self._draw_new_callback, key="draw_new")

        self._status("Draw new nucleus: paint on the preview layer; release mouse to commit.")

    def _disable_draw_new_mode(self, layer) -> None:
        # Disable callbacks on the preview layer, hide it, and restore its prior mode.
        preview = None
        try:
            preview = self._preview_layer or (self.viewer.layers[self._preview_layer_name] if self._preview_layer_name in self.viewer.layers else None)
        except Exception:
            preview = self._preview_layer

        if preview is not None:
            self._restore_mouse_callbacks(preview, key="draw_new")
            self._restore_layer_mode(preview, key="draw_new_preview")
            try:
                preview.editable = False
                preview.visible = False
            except Exception:
                pass
            try:
                preview.data = np.zeros_like(preview.data, dtype=np.int32)
            except Exception:
                pass

        self._clear_draw_preview()

    def _make_draw_new_callback(self):
        # Callback attached to the PREVIEW layer. On release, we commit preview mask
        # into the edit Labels layer and clear the preview.
        def _cb(preview_layer, event):
            yield
            while event.type == "mouse_move":
                # Let napari paint; we don't do work on move (avoids lag).
                yield

            if event.type != "mouse_release":
                return

            # Find the edit layer we are supposed to modify.
            edit_layer = self._require_layer()
            if edit_layer is None:
                return

            try:
                m = np.asarray(preview_layer.data) > 0
            except Exception:
                m = None

            # Clear preview immediately for responsiveness.
            try:
                preview_layer.data = np.zeros_like(preview_layer.data, dtype=np.int32)
            except Exception:
                pass
            # IMPORTANT: keep draw-new mode active after each commit.
            # Do not hide the preview layer; just clear its data and keep it active.
            self._reset_draw_preview(keep_visible=True)
            self._activate_layer(preview_layer)
            # Napari may revert interaction/tool after mouse_release; force paint mode again so draw-new stays active.
            self._set_layer_mode(preview_layer, key="draw_new_preview", mode="paint")
            self._expect_active_layer(preview_layer, tool="Draw new")

            if m is None or (not np.any(m)):
                return

            data = np.asarray(edit_layer.data).astype(np.int32, copy=False)
            new_data = data.copy()

            overlap = data[m]
            overlap = overlap[overlap > 0]

            # Undo support for the programmatic edit.
            self._push_undo(edit_layer)

            if overlap.size > 0:
                vals, counts = np.unique(overlap, return_counts=True)
                target = int(vals[np.argmax(counts)])
                new_data[(m) & (new_data == 0)] = target
                self._set_labels(edit_layer, new_data)
                self._status(f"Expanded nucleus {target}.")
            else:
                target = int(new_data.max()) + 1
                new_data[m] = target
                self._set_labels(edit_layer, new_data)
                self._status(f"Created nucleus {target}.")

        return _cb

    # ----- erode mode -----

    def _enable_erode_mode(self, layer) -> None:
        self._disable_erode_mode(layer)
        self._set_layer_mode(layer, key="erode", mode="erase")
        if self._eraser_callback is None:
            self._eraser_callback = self._make_eraser_callback()
        self._install_mouse_callbacks(layer, drag_cb=self._eraser_callback, key="erode")

    def _disable_erode_mode(self, layer) -> None:
        if self._eraser_callback is None:
            return
        self._restore_mouse_callbacks(layer, key="erode")
        self._restore_layer_mode(layer, key="erode")
        self._erode_snapshot = None
        self._erode_touched.clear()

    def _make_eraser_callback(self):
        def _cb(layer, event):
            if event.type == "mouse_press":
                self._erode_snapshot = np.asarray(layer.data).astype(np.int32, copy=True)
                self._erode_touched.clear()
            yield

            while event.type == "mouse_move":
                try:
                    pos = event.position
                    r = int(round(float(pos[-2])))
                    c = int(round(float(pos[-1])))
                    snap = self._erode_snapshot
                    if snap is not None and 0 <= r < snap.shape[0] and 0 <= c < snap.shape[1]:
                        lbl = int(snap[r, c])
                        if lbl > 0:
                            self._erode_touched.add(lbl)
                except Exception:
                    pass
                yield

            if event.type != "mouse_release":
                return

            if not self._erode_touched:
                return

            data = np.asarray(layer.data).astype(np.int32, copy=False)
            new_data = data.copy()
            next_id = int(new_data.max()) + 1

            for target in sorted(self._erode_touched):
                m = new_data == int(target)
                if not np.any(m):
                    continue
                cc, _n = cc_label(m)
                n_cc = int(cc.max())
                if n_cc <= 1:
                    continue
                sizes = [(i, int((cc == i).sum())) for i in range(1, n_cc + 1)]
                sizes.sort(key=lambda t: t[1], reverse=True)
                keep_comp = sizes[0][0]
                new_data[m] = 0
                new_data[cc == keep_comp] = int(target)
                for comp_id, _sz in sizes[1:]:
                    new_data[cc == comp_id] = next_id
                    next_id += 1

            self._set_labels(layer, new_data)

        return _cb

    # ----- core helpers -----

    def _set_labels(self, layer, new_data: np.ndarray) -> None:
        """Apply new label data in an undo-friendly way when possible."""
        try:
            old = np.asarray(layer.data)
            mask = old != new_data
            if hasattr(layer, "data_setitem"):
                layer.data_setitem(mask, new_data[mask])
            else:
                layer.data = new_data
        except Exception:
            layer.data = new_data

    def _polyline_to_mask(
        self,
        points_rc: list[tuple[float, float]] | None,
        out_shape: tuple[int, int],
        *,
        thickness: int = 1,
    ) -> Optional[np.ndarray]:
        if not points_rc or len(points_rc) < 2:
            self._warn("Draw a line first.")
            return None
        H, W = out_shape
        mask = np.zeros((H, W), dtype=bool)
        coords = np.asarray(points_rc, dtype=float)
        for i in range(len(coords) - 1):
            r0 = int(round(coords[i, 0])); c0 = int(round(coords[i, 1]))
            r1 = int(round(coords[i + 1, 0])); c1 = int(round(coords[i + 1, 1]))
            rr, cc = sk_line(r0, c0, r1, c1)
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            mask[rr[ok], cc[ok]] = True
        if thickness and thickness > 1:
            mask = dilation(mask, footprint=disk(int(thickness)))
        return mask
