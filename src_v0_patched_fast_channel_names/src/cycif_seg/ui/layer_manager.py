"""Napari layer utilities.

This module centralizes common "get or create" and "update" patterns for napari layers.
Keeping these helpers outside the main Qt widget dramatically reduces the size of
`main_widget.py` and lowers the risk of indentation-related regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LayerManager:
    """Small wrapper around a napari Viewer for safe layer CRUD/update operations."""

    viewer: object
    connect_dirty: Optional[Callable[[object], None]] = None

    # --------------------------
    # Generic helpers
    # --------------------------
    def get(self, name: str):
        if self.viewer is None:
            return None
        try:
            return self.viewer.layers[name]
        except Exception:
            return None

    def delete(self, name: str) -> None:
        if self.viewer is None:
            return
        try:
            if name in self.viewer.layers:
                del self.viewer.layers[name]
        except Exception:
            pass

    # --------------------------
    # Labels
    # --------------------------
    def set_or_update_labels(
        self,
        name: str,
        data,
        *,
        opacity: float = 0.7,
        visible: bool = True,
        num_colors: int | None = None,
    ):
        """Create or update a napari Labels layer."""
        if self.viewer is None:
            return None

        if name in self.viewer.layers:
            try:
                lyr = self.viewer.layers[name]
                lyr.data = data
                lyr.opacity = opacity
                lyr.visible = visible
                if num_colors is not None and hasattr(lyr, "num_colors"):
                    try:
                        lyr.num_colors = num_colors
                    except Exception:
                        pass
                if self.connect_dirty is not None:
                    self.connect_dirty(lyr)
                return lyr
            except Exception:
                # If an existing layer is corrupted, remove and recreate.
                self.delete(name)

        try:
            kwargs = {"name": name, "opacity": opacity}
            if num_colors is not None:
                kwargs["num_colors"] = num_colors
            lyr2 = self.viewer.add_labels(data, **kwargs)
            lyr2.visible = visible
            if self.connect_dirty is not None:
                self.connect_dirty(lyr2)
            return lyr2
        except Exception:
            return None

    # --------------------------
    # Image
    # --------------------------
    def set_or_update_image(
        self,
        name: str,
        data,
        *,
        opacity: float = 1.0,
        visible: bool = True,
        colormap: str | None = None,
        blending: str | None = None,
        contrast_limits: tuple[float, float] | None = None,
    ):
        """Create or update a napari Image layer."""
        if self.viewer is None:
            return None

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
                if blending is not None and hasattr(lyr, "blending"):
                    try:
                        lyr.blending = blending
                    except Exception:
                        pass
                if contrast_limits is not None and hasattr(lyr, "contrast_limits"):
                    try:
                        lyr.contrast_limits = contrast_limits
                    except Exception:
                        pass
                if self.connect_dirty is not None:
                    self.connect_dirty(lyr)
                return lyr
            except Exception:
                self.delete(name)

        try:
            kwargs = {"name": name, "opacity": opacity}
            if colormap is not None:
                kwargs["colormap"] = colormap
            if blending is not None:
                kwargs["blending"] = blending
            lyr2 = self.viewer.add_image(data, **kwargs)
            lyr2.visible = visible
            if contrast_limits is not None and hasattr(lyr2, "contrast_limits"):
                try:
                    lyr2.contrast_limits = contrast_limits
                except Exception:
                    pass
            if self.connect_dirty is not None:
                self.connect_dirty(lyr2)
            return lyr2
        except Exception:
            return None

    # --------------------------
    # Common project layers
    # --------------------------
    def ensure_scribbles_layer(
        self,
        *,
        image_shape: tuple[int, int],
        name: str,
        opacity: float = 0.6,
        dtype="uint8",
    ):
        """Ensure a Labels layer exists for scribbles, sized to (H, W)."""
        if self.viewer is None:
            return None
        if name in self.viewer.layers:
            return self.viewer.layers[name]

        try:
            import numpy as np

            H, W = int(image_shape[0]), int(image_shape[1])
            scrib = np.zeros((H, W), dtype=dtype)
            layer = self.viewer.add_labels(scrib, name=name, opacity=opacity)
            if self.connect_dirty is not None:
                self.connect_dirty(layer)
            return layer
        except Exception:
            return None
