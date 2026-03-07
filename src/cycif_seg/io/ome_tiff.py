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



class IncrementalOmeBigTiffWriter:
    """Incrementally write a (Y, X, C) image to an on-disk OME BigTIFF.

    Uses ``tifffile.memmap`` so callers can write one channel plane at a time
    without keeping the full merged output stack in RAM.

    Notes
    -----
    - Compression is intentionally disabled because memory-mapped incremental
      writes require an uncompressed layout.
    - Internally the TIFF is stored as (C, Y, X) with OME axes="CYX".
    """

    def __init__(self, path: str, shape_yxc: tuple[int, int, int], dtype: np.dtype, channel_names: list[str] | None = None):
        if len(shape_yxc) != 3:
            raise ValueError(f"Expected output shape (Y,X,C). Got {shape_yxc}")
        y, x, c = (int(shape_yxc[0]), int(shape_yxc[1]), int(shape_yxc[2]))
        if y <= 0 or x <= 0 or c <= 0:
            raise ValueError(f"Invalid output shape: {shape_yxc}")

        self.path = path
        self.shape_yxc = (y, x, c)
        self.dtype = np.dtype(dtype)
        if channel_names is None or len(channel_names) != c:
            channel_names = [f"Channel {i}" for i in range(c)]
        self.channel_names = list(channel_names)

        metadata = {
            "axes": "CYX",
            "Channel": {"Name": list(self.channel_names)},
        }

        self._mm = tifffile.memmap(
            self.path,
            shape=(c, y, x),
            dtype=self.dtype,
            photometric="minisblack",
            metadata=metadata,
            ome=True,
            bigtiff=True,
        )

    def write_channel(self, channel_index: int, plane_yx: np.ndarray) -> None:
        idx = int(channel_index)
        if idx < 0 or idx >= int(self.shape_yxc[2]):
            raise IndexError(f"channel_index out of range: {channel_index}")
        if plane_yx.ndim != 2:
            raise ValueError(f"Expected (Y,X) plane. Got shape={plane_yx.shape}")
        exp = (int(self.shape_yxc[0]), int(self.shape_yxc[1]))
        got = (int(plane_yx.shape[0]), int(plane_yx.shape[1]))
        if got != exp:
            raise ValueError(f"Plane shape mismatch. Expected {exp}, got {got}")
        self._mm[idx, :, :] = plane_yx.astype(self.dtype, copy=False)

    def flush(self) -> None:
        try:
            self._mm.flush()
        except Exception:
            pass

    def close(self) -> None:
        self.flush()
        try:
            del self._mm
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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



def inspect_tiff_yxc(path: str) -> dict:
    """Inspect a TIFF/OME-TIFF without reading full pixel data.

    Returns a dict with keys:
      - shape_yxc: (Y, X, C)
      - dtype: numpy dtype
      - channel_names: list[str]
      - axes: original tifffile series axes string (best effort)
    """
    with tifffile.TiffFile(path) as tf:
        try:
            series = tf.series[0]
            shape = tuple(int(x) for x in series.shape)
            axes = str(getattr(series, "axes", "") or "")
            dtype = np.dtype(series.dtype)
        except Exception:
            # Fall back to pages if series metadata are unavailable.
            pages = list(tf.pages)
            if not pages:
                raise ValueError(f"No TIFF pages found in {path}")
            dtype = np.dtype(getattr(pages[0], 'dtype', np.uint16))
            axes = "CYX"
            shape = (len(pages),) + tuple(int(x) for x in pages[0].shape)

        if len(shape) != 3:
            raise ValueError(f"Expected 3D TIFF/OME-TIFF. Got shape={shape}, axes={axes!r}")

        # Convert common layouts to YXC metadata without reading pixels.
        if axes and "C" in axes:
            cpos = axes.index("C")
            if cpos == 0:
                c, y, x = shape
            elif cpos == 2:
                y, x, c = shape
            else:
                raise ValueError(f"Unsupported channel axis placement: shape={shape}, axes={axes!r}")
        else:
            # Common fallback: CYX when first dim is small, else YXC.
            if shape[0] <= 64 and shape[1] > 64 and shape[2] > 64:
                c, y, x = shape
            elif shape[2] <= 64 and shape[0] > 64 and shape[1] > 64:
                y, x, c = shape
            else:
                raise ValueError(f"Unable to infer channel axis from shape={shape}, axes={axes!r}")

        ch_names = None
        try:
            ch_names = _channel_names_from_ome(tf.ome_metadata, int(c))
        except Exception:
            ch_names = None
        if not ch_names or len(ch_names) != int(c):
            ch_names = [f"Channel {i}" for i in range(int(c))]
        else:
            ch_names = [(nm if nm and nm.strip() else f"Channel {i}") for i, nm in enumerate(ch_names)]

    return {
        "shape_yxc": (int(y), int(x), int(c)),
        "dtype": dtype,
        "channel_names": list(ch_names),
        "axes": axes,
    }


def load_single_channel_tiff_native(path: str, channel_index: int) -> np.ndarray:
    """Load a single channel plane as (Y, X), preserving native dtype.

    Uses TIFF metadata/page structure when possible to avoid reading all channels.
    Falls back conservatively to full-series read + slice for uncommon layouts.
    """
    info = inspect_tiff_yxc(path)
    y, x, c = info["shape_yxc"]
    idx = int(channel_index)
    if idx < 0 or idx >= int(c):
        raise IndexError(f"channel_index {idx} out of bounds for {c} channels")

    with tifffile.TiffFile(path) as tf:
        try:
            series = tf.series[0]
            shape = tuple(int(v) for v in series.shape)
            axes = str(getattr(series, "axes", "") or "")
        except Exception:
            series = None
            shape = ()
            axes = ""

        # Fast path: channels as separate pages / first axis.
        try:
            if len(shape) == 3 and ((axes and axes.startswith("C")) or (not axes and shape[0] == int(c))):
                arr = series.asarray(key=idx) if series is not None else tf.pages[idx].asarray()
                return np.asarray(arr)
        except Exception:
            pass

        # Fast path: sample-interleaved YXC; read one page and slice the requested channel.
        try:
            if len(shape) == 3 and ((axes and axes.endswith("C")) or (not axes and shape[2] == int(c))):
                arr = series.asarray() if series is not None else tf.asarray()
                arr = np.asarray(arr)
                return arr[..., idx]
        except Exception:
            pass

    # Conservative fallback for unusual layouts.
    arr = tifffile.imread(path)
    arr = _normalize_to_yxc(arr)
    return np.asarray(arr[..., idx])


class LazyChannelImage:
    """Lightweight on-demand YXC image wrapper backed by single-channel TIFF reads."""

    def __init__(self, path: str, *, channel_indices: list[int] | None = None, _root=None):
        self.path = str(path)
        self._root = _root or self
        if _root is None:
            info = inspect_tiff_yxc(self.path)
            self._shape_yxc_root = tuple(int(v) for v in info["shape_yxc"])
            self.dtype = np.dtype(info["dtype"])
            self.channel_names = list(info["channel_names"])
            self._cache: dict[int, np.ndarray] = {}
        else:
            self._shape_yxc_root = tuple(int(v) for v in _root._shape_yxc_root)
            self.dtype = np.dtype(_root.dtype)
            self.channel_names = list(_root.channel_names)
        if channel_indices is None:
            channel_indices = list(range(int(self._shape_yxc_root[2])))
        self._channel_indices = [int(i) for i in channel_indices]

    @property
    def shape(self):
        y, x, _ = self._shape_yxc_root
        return (int(y), int(x), int(len(self._channel_indices)))

    @property
    def ndim(self):
        return 3

    def subset(self, channel_indices: list[int]):
        mapped = [self._channel_indices[int(i)] for i in channel_indices]
        return LazyChannelImage(self.path, channel_indices=mapped, _root=self._root)

    def get_channel(self, local_index: int) -> np.ndarray:
        root = self._root
        global_index = int(self._channel_indices[int(local_index)])
        if global_index not in root._cache:
            root._cache[global_index] = np.asarray(load_single_channel_tiff_native(self.path, global_index))
        return root._cache[global_index]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        parts = list(key)
        if Ellipsis in parts:
            eidx = parts.index(Ellipsis)
            n_missing = 3 - (len(parts) - 1)
            parts = parts[:eidx] + [slice(None)] * max(0, n_missing) + parts[eidx + 1:]
        while len(parts) < 3:
            parts.append(slice(None))
        yk, xk, ck = parts[:3]

        if isinstance(ck, (int, np.integer)):
            return np.asarray(self.get_channel(int(ck))[yk, xk])

        if isinstance(ck, slice):
            local_channels = list(range(*ck.indices(len(self._channel_indices))))
        elif isinstance(ck, (list, tuple, np.ndarray)):
            local_channels = [int(i) for i in ck]
        else:
            raise TypeError(f"Unsupported channel index type: {type(ck)!r}")

        planes = [np.asarray(self.get_channel(i)[yk, xk]) for i in local_channels]
        if not planes:
            y, x = np.asarray(self.get_channel(0)[yk, xk]).shape[:2]
            return np.empty((y, x, 0), dtype=self.dtype)
        return np.stack(planes, axis=-1)


def load_channel_names_only(path: str) -> list[str]:
    """Read channel names without reading full pixel data."""
    return list(inspect_tiff_yxc(path)["channel_names"])



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
