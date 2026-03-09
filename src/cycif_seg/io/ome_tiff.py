from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Callable

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
    arr = tifffile.imread(path, series=0, level=0)
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
    arr = tifffile.imread(path, series=0, level=0)
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


def _base_series(tf: tifffile.TiffFile):
    """Return the full-resolution image series for flat or pyramidal TIFFs."""
    series0 = tf.series[0]
    try:
        levels = getattr(series0, "levels", None)
        if levels:
            return levels[0]
    except Exception:
        pass
    return series0


def inspect_tiff_pyramid(path: str) -> dict:
    """Inspect whether a TIFF/OME-TIFF contains a multi-resolution pyramid."""
    out_series: list[dict] = []
    is_pyramidal = False
    with tifffile.TiffFile(path) as tf:
        for si, series in enumerate(tf.series):
            try:
                levels = list(getattr(series, 'levels', []) or [series])
            except Exception:
                levels = [series]
            level_shapes = [tuple(int(v) for v in getattr(lvl, 'shape', ())) for lvl in levels]
            axes = str(getattr(series, 'axes', '') or '')
            try:
                subifds = int(len(getattr(series.pages[0], 'subifds', None) or ()))
            except Exception:
                subifds = 0
            if len(levels) > 1:
                is_pyramidal = True
            out_series.append({
                'index': int(si),
                'shape': tuple(int(v) for v in getattr(series, 'shape', ())),
                'axes': axes,
                'number_of_levels': int(len(levels)),
                'level_shapes': level_shapes,
                'subifds': subifds,
            })
        is_ome = bool(getattr(tf, 'is_ome', False))
    return {
        'is_ome': is_ome,
        'number_of_series': int(len(out_series)),
        'series': out_series,
        'is_pyramidal': bool(is_pyramidal),
    }


def _compute_pyramid_subifds(shape_yx: tuple[int, int], *, min_size: int = 128) -> int:
    y, x = (int(shape_yx[0]), int(shape_yx[1]))
    n = 0
    while y > int(min_size) and x > int(min_size):
        y = max(1, (y + 1) // 2)
        x = max(1, (x + 1) // 2)
        n += 1
    return int(n)


def estimate_pyramid_conversion_ticks(shape_cyx: tuple[int, int, int], *, min_level_size: int = 128, out_chunk: int = 1024) -> int:
    """Estimate progress ticks for flat->pyramid conversion."""
    _c, y, x = (int(shape_cyx[0]), int(shape_cyx[1]), int(shape_cyx[2]))
    subifds = _compute_pyramid_subifds((y, x), min_size=int(min_level_size))
    ticks = 0
    for _level in range(1, subifds + 1):
        y = max(1, (y + 1) // 2)
        x = max(1, (x + 1) // 2)
        ny = max(1, int(math.ceil(float(y) / float(max(1, int(out_chunk))))))
        nx = max(1, int(math.ceil(float(x) / float(max(1, int(out_chunk))))))
        ticks += int(ny * nx)  # build this level chunk-by-chunk
        ticks += 1             # write this level to the final TIFF
    ticks += 1  # write base level
    return int(max(1, ticks))


def _cast_like_source(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.rint(arr)
        arr = np.clip(arr, info.min, info.max)
        return arr.astype(dtype, copy=False)
    return arr.astype(dtype, copy=False)


def _block_average_2x2_cyx(arr_cyx: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Downsample a CYX array by 2x in Y/X via true block averaging."""
    if arr_cyx.ndim != 3:
        raise ValueError(f"Expected CYX array, got {arr_cyx.shape}")
    c, y, x = (int(arr_cyx.shape[0]), int(arr_cyx.shape[1]), int(arr_cyx.shape[2]))
    oy = max(1, (y + 1) // 2)
    ox = max(1, (x + 1) // 2)
    acc = np.zeros((c, oy, ox), dtype=np.float32)
    cnt = np.zeros((1, oy, ox), dtype=np.float32)
    for ys in (0, 1):
        for xs in (0, 1):
            sl = np.asarray(arr_cyx[:, ys::2, xs::2], dtype=np.float32)
            sy = sl.shape[1]
            sx = sl.shape[2]
            if sy == 0 or sx == 0:
                continue
            acc[:, :sy, :sx] += sl
            cnt[:, :sy, :sx] += 1.0
    acc /= np.maximum(cnt, 1.0)
    return _cast_like_source(acc, np.dtype(out_dtype))


def _raw_memmap(path: str, shape: tuple[int, ...], dtype: np.dtype, mode: str = 'w+') -> np.memmap:
    return np.memmap(path, dtype=np.dtype(dtype), mode=mode, shape=tuple(int(v) for v in shape))


def convert_flat_ome_to_pyramidal(
    source_path: str,
    output_path: str | None = None,
    *,
    channel_names: list[str] | None = None,
    tile_size: int = 512,
    compression: str | int | None = 'zlib',
    min_level_size: int = 128,
    out_chunk: int = 1024,
    replace_source: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    progress_step_cb: Callable[[str, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> str:
    """Convert a flat CYX OME-TIFF into a pyramidal OME-TIFF.

    This implementation is RAM-efficient:
      - the flat source is memory-mapped,
      - each pyramid level is generated chunk-by-chunk via 2x2 block averaging,
      - intermediate pyramid levels are stored as temporary raw memmaps,
      - the final TIFF is written from those memmaps.
    """
    def _check_cancel() -> None:
        try:
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError('Cancelled')
        except RuntimeError:
            raise
        except Exception:
            return

    def _step(msg: str, phase: str) -> None:
        if progress_cb:
            progress_cb(msg)
        if progress_step_cb:
            progress_step_cb(msg, phase)

    src = str(source_path)
    if output_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix='.ome.tiff', prefix='cycif_pyramid_')
        os.close(fd)
        dst = tmp_path
    else:
        dst = str(output_path)

    with tifffile.TiffFile(src) as tf:
        series0 = _base_series(tf)
        shape = tuple(int(v) for v in series0.shape)
        axes = str(getattr(series0, 'axes', '') or str(getattr(tf.series[0], 'axes', '') or ''))
        dtype = np.dtype(series0.dtype)
        ome_xml = tf.ome_metadata

    if len(shape) != 3:
        raise ValueError(f'Expected 3D CYX/YXC image. Got shape={shape}, axes={axes!r}')

    if axes and 'C' in axes:
        cpos = axes.index('C')
        if cpos == 0:
            c, y, x = shape
        elif cpos == 2:
            y, x, c = shape
        else:
            raise ValueError(f'Unsupported channel axis placement: shape={shape}, axes={axes!r}')
    else:
        if shape[0] <= 64 and shape[1] > 64 and shape[2] > 64:
            c, y, x = shape
        elif shape[2] <= 64 and shape[0] > 64 and shape[1] > 64:
            y, x, c = shape
        else:
            raise ValueError(f'Unable to infer channel axis from shape={shape}, axes={axes!r}')

    if channel_names is None or len(channel_names) != int(c):
        channel_names = _channel_names_from_ome(ome_xml, int(c)) or [f'Channel {i}' for i in range(int(c))]
    channel_names = [(nm if nm and str(nm).strip() else f'Channel {i}') for i, nm in enumerate(channel_names)]

    src_mm = tifffile.memmap(src)
    if src_mm.ndim != 3:
        raise ValueError(f'Expected 3D memmap-able TIFF. Got shape={getattr(src_mm, "shape", None)}')
    if src_mm.shape[0] == int(c):
        src_cyx = src_mm
    elif src_mm.shape[-1] == int(c):
        src_cyx = np.moveaxis(src_mm, -1, 0)
    else:
        raise ValueError(f'Unable to canonicalize source TIFF to CYX. shape={src_mm.shape}, expected channels={c}')

    subifds = _compute_pyramid_subifds((int(y), int(x)), min_size=int(min_level_size))
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': list(channel_names)},
    }

    tmpdir = Path(tempfile.mkdtemp(prefix='cycif_pyramid_levels_'))
    level_arrays: list[np.ndarray] = []
    level_paths: list[Path] = []
    try:
        prev = src_cyx
        prev_y = int(y)
        prev_x = int(x)
        for level in range(1, subifds + 1):
            _check_cancel()
            out_y = max(1, (prev_y + 1) // 2)
            out_x = max(1, (prev_x + 1) // 2)
            lvl_path = tmpdir / f'level_{level:02d}.dat'
            lvl = _raw_memmap(str(lvl_path), (int(c), int(out_y), int(out_x)), dtype, mode='w+')
            step_y = max(1, int(out_chunk))
            step_x = max(1, int(out_chunk))
            for oy0 in range(0, int(out_y), step_y):
                _check_cancel()
                oy1 = min(int(out_y), oy0 + step_y)
                for ox0 in range(0, int(out_x), step_x):
                    _check_cancel()
                    ox1 = min(int(out_x), ox0 + step_x)
                    sy0, sy1 = int(oy0 * 2), int(min(prev_y, oy1 * 2))
                    sx0, sx1 = int(ox0 * 2), int(min(prev_x, ox1 * 2))
                    chunk = np.asarray(prev[:, sy0:sy1, sx0:sx1])
                    lvl[:, oy0:oy1, ox0:ox1] = _block_average_2x2_cyx(chunk, dtype)
                    _step(
                        f'Building pyramid level {level}/{subifds} ({oy1}/{out_y} rows)',
                        'pyramid_build_chunk',
                    )
            lvl.flush()
            level_arrays.append(lvl)
            level_paths.append(lvl_path)
            prev = lvl
            prev_y, prev_x = int(out_y), int(out_x)

        options = dict(
            photometric='minisblack',
            tile=(int(tile_size), int(tile_size)),
            compression=compression,
            predictor=True if np.issubdtype(dtype, np.integer) else False,
        )

        with tifffile.TiffWriter(dst, bigtiff=True) as tif:
            _check_cancel()
            _step('Writing pyramid base level', 'pyramid_write_level')
            tif.write(src_cyx, subifds=int(subifds), metadata=metadata, **options)
            for level, lvl in enumerate(level_arrays, start=1):
                _check_cancel()
                _step(f'Writing pyramid level {level}/{subifds}', 'pyramid_write_level')
                tif.write(lvl, subfiletype=1, metadata=None, **options)
    finally:
        try:
            del src_mm
        except Exception:
            pass
        for arr in level_arrays:
            try:
                arr.flush()
            except Exception:
                pass
        for pth in level_paths:
            try:
                os.remove(pth)
            except Exception:
                pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass

    if replace_source:
        os.replace(dst, src)
        return src
    return dst


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
            series = _base_series(tf)
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
            series = _base_series(tf)
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
    try:
        if hasattr(z, 'keys'):
            keys = list(z.keys())
            if keys:
                z0 = z[keys[0]]
                if hasattr(z0, 'shape'):
                    z = z0
        elif isinstance(z, (list, tuple)) and len(z) > 0:
            z = z[0]
    except Exception:
        pass
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
