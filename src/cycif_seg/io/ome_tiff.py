from __future__ import annotations

import numpy as np
import tifffile

# Optional lazy-loading stack (recommended for very large OME-TIFFs)
try:  # pragma: no cover
    import dask.array as da  # type: ignore
    import zarr  # type: ignore
except Exception:  # pragma: no cover
    da = None
    zarr = None

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

    For larger images, classic TIFF uses 32-bit offsets and will fail once the
    file grows beyond ~4 GiB. In that case, we must write BigTIFF.

    To avoid OME-XML shape/axes mismatches in tifffile, we transpose to (C, Y, X)
    for writing.
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

    # Classic TIFF uses 32-bit offsets; BigTIFF is required once the written file
    # may exceed ~4 GiB. We conservatively decide based on the *uncompressed* size.
    try:
        bytes_per_px = int(np.dtype(img_cyx.dtype).itemsize)
        est_uncompressed = int(img_cyx.size) * bytes_per_px
    except Exception:
        est_uncompressed = 0

    bigtiff = bool(est_uncompressed >= (2**32 - 1))

    tifffile.imwrite(
        path,
        img_cyx,
        photometric="minisblack",
        metadata=metadata,
        ome=True,
        bigtiff=bigtiff,
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


def load_multichannel_tiff_native(path: str) -> tuple[np.ndarray, list[str]]:
    """Load a TIFF/OME-TIFF as (Y, X, C) preserving the on-disk dtype.

    This is useful for preprocessing/registration steps where we want to keep the
    original integer dtype (e.g., uint16) and only use float32 as a transient
    compute type.
    """
    arr = tifffile.imread(path)
    arr = _normalize_to_yxc(arr)
    n_channels = int(arr.shape[2])

    # Try OME metadata (same logic as eager loader)
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
        ch_names = [(nm if nm and nm.strip() else f"Channel {i}") for i, nm in enumerate(ch_names)]

    return arr, ch_names


def _normalize_to_cyx_lazy(arr) -> "da.Array":
    """Canonicalize a lazy array to (C, Y, X).

    Supported common microscopy TIFF layouts:
      - (C, Y, X) -> (C, Y, X)
      - (Y, X, C) -> (C, Y, X)
    """
    if da is None:
        raise RuntimeError(
            "Lazy loading requires dask and zarr. Install with: pip install dask[array] zarr"
        )
    if getattr(arr, "ndim", None) != 3:
        raise ValueError(f"Expected 3D array. Got shape={getattr(arr, 'shape', None)}")

    shp = tuple(int(x) for x in arr.shape)

    # Heuristic: treat first axis as channel if small.
    if shp[0] <= 32 and shp[1] > 64 and shp[2] > 64:
        return arr  # already (C,Y,X)

    # Otherwise, if last axis looks like channels, move it to front.
    if shp[2] <= 32 and shp[0] > 64 and shp[1] > 64:
        return da.moveaxis(arr, 2, 0)  # (Y,X,C)->(C,Y,X)

    # Fallback: assume (C,Y,X)
    return arr


def load_multichannel_tiff_lazy(path: str):
    """Lazy-load a TIFF/OME-TIFF and return (img_cyx, channel_names).

    The returned array is a **lazy** dask array backed by the TIFF on disk.
    No full image data are loaded into RAM until you compute slices.

    Notes
    -----
    - Requires `dask[array]` and `zarr`.
    - The returned image is canonicalized to (C, Y, X).
    """
    if da is None or zarr is None:
        raise RuntimeError(
            "Lazy loading requires dask and zarr. Install with: pip install dask[array] zarr"
        )

    # Use tifffile's Zarr interface for on-demand IO.
    store = tifffile.imread(path, aszarr=True)
    z = zarr.open(store, mode="r")
    arr = da.from_zarr(z)
    arr_cyx = _normalize_to_cyx_lazy(arr)

    # Channel names from OME metadata (same logic as eager loader)
    n_channels = int(arr_cyx.shape[0])
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
        ch_names = [
            (nm if nm and nm.strip() else f"Channel {i}")
            for i, nm in enumerate(ch_names)
        ]

    return arr_cyx, ch_names
