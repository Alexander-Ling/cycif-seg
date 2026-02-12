from __future__ import annotations

import numpy as np
import tifffile

from ome_types import from_xml


def _normalize_to_yxc(arr: np.ndarray) -> np.ndarray:
    """
    Canonicalize to (Y, X, C) for common microscopy TIFF layouts:
      - (C, Y, X) -> (Y, X, C)
      - (Y, X, C) -> (Y, X, C)
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array. Got shape={arr.shape}")

    # Heuristic: treat first axis as channel if small.
    if arr.shape[0] <= 32 and arr.shape[1] > 64 and arr.shape[2] > 64:
        arr = np.moveaxis(arr, 0, -1)  # (C,Y,X)->(Y,X,C)

    if not (arr.shape[0] > 64 and arr.shape[1] > 64):
        raise ValueError(f"Unexpected shape after normalization: {arr.shape}")

    return arr


def _channel_names_from_ome(ome_xml: str, n_channels: int) -> list[str] | None:
    if not ome_xml:
        return None
    try:
        ome = from_xml(ome_xml)
        # Most OME-TIFFs store channels under the first Image/Pixels.
        img0 = ome.images[0] if ome.images else None
        px = img0.pixels if img0 else None
        if not px or not px.channels:
            return None

        names: list[str] = []
        for ch in px.channels[:n_channels]:
            nm = (ch.name or "").strip()
            names.append(nm)

        # If all empty, treat as missing
        if all(n == "" for n in names):
            return None

        # Fill blanks
        names = [n if n else "" for n in names]
        return names
    except Exception:
        return None


def load_multichannel_tiff(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load a TIFF/OME-TIFF as (Y, X, C) float32 and return (img_yxc, channel_names).
    Attempts to read OME channel names via ome-types; falls back to "Channel i".
    """
    arr = tifffile.imread(path)
    arr = _normalize_to_yxc(arr)

    img = arr.astype(np.float32, copy=False)
    n_channels = img.shape[2]

    # Try OME metadata
    ch_names = None
    try:
        with tifffile.TiffFile(path) as tf:
            ome_xml = tf.ome_metadata
        ch_names = _channel_names_from_ome(ome_xml, n_channels)
    except Exception:
        ch_names = None

    if not ch_names or len(ch_names) != n_channels:
        ch_names = [f"Channel {i}" for i in range(n_channels)]
    else:
        # Replace blanks with Channel i
        ch_names = [
            (nm if nm and nm.strip() else f"Channel {i}")
            for i, nm in enumerate(ch_names)
        ]

    return img, ch_names
