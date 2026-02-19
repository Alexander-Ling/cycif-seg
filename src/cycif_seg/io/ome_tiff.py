from __future__ import annotations

import numpy as np
import tifffile

# ome-types is nice-to-have, but keep this module usable without it.
try:
    from ome_types import from_xml  # type: ignore
except Exception:  # pragma: no cover
    from_xml = None

import xml.etree.ElementTree as ET


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

    # First try ome-types (if available)
    if from_xml is not None:
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

            if all(n == "" for n in names):
                return None
            return [n if n else "" for n in names]
        except Exception:
            pass

    # Fallback: parse OME-XML directly.
    try:
        root = ET.fromstring(ome_xml)
        # OME namespaces vary; strip them for matching.
        def _strip(tag: str) -> str:
            return tag.split("}", 1)[-1]

        channels: list[str] = []
        for el in root.iter():
            if _strip(el.tag) == "Channel":
                nm = (el.attrib.get("Name") or "").strip()
                channels.append(nm)
                if len(channels) >= n_channels:
                    break
        if not channels:
            return None
        if all(n == "" for n in channels):
            return None
        if len(channels) < n_channels:
            channels += [""] * (n_channels - len(channels))
        return channels[:n_channels]
    except Exception:
        return None


def save_ome_tiff_yxc(
    path: str,
    img_yxc: np.ndarray,
    channel_names: list[str] | None = None,
    *,
    compress: bool | int = True,
) -> None:
    """Save a (Y, X, C) array as an OME-TIFF with optional channel names.

    Notes
    -----
    tifffile's OME writer is most reliable when channels are stored in the
    leading dimension (C, Y, X) with axes="CYX".

    Some tifffile versions can raise:
        ValueError: shape does not match stored shape
    when writing with axes="YXC".

    To avoid that, we transpose to (C, Y, X) for writing.
    """
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")

    n_ch = int(img_yxc.shape[2])
    if channel_names is None or len(channel_names) != n_ch:
        channel_names = [f"Channel {i}" for i in range(n_ch)]

    # Write as (C, Y, X) to avoid OME-XML shape/axes mismatches in tifffile.
    img_cyx = np.moveaxis(img_yxc, 2, 0)

    metadata = {
        "axes": "CYX",
        "Channel": {"Name": list(channel_names)},
    }
    tifffile.imwrite(
        path,
        img_cyx,
        photometric="minisblack",
        metadata=metadata,
        ome=True,
        compression=("zlib" if compress is True else compress),
    )


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
