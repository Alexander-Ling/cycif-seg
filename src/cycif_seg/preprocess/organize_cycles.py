from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label

try:
    from scipy.ndimage import shift as ndi_shift  # type: ignore
    from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates  # type: ignore
except Exception:  # pragma: no cover
    ndi_shift = None  # type: ignore
    binary_dilation = None  # type: ignore
    distance_transform_edt = None  # type: ignore
    map_coordinates = None  # type: ignore

from cycif_seg.io.ome_tiff import (
    IncrementalOmeBigTiffWriter,
    IncrementalZarrRegisteredWriter,
    estimate_pyramid_conversion_ticks,
    inspect_tiff_pyramid,
    inspect_tiff_yxc,
    load_channel_names_only,
    load_channel_downsampled,
    load_channel_strip,
    load_channel_roi,
    load_physical_pixel_sizes,
    load_single_channel_tiff_native,
    convert_flat_ome_to_pyramidal,
    convert_registered_zarr_to_pyramidal,
)


_STEP1_DEBUG = str(os.environ.get("CYCIF_SEG_STEP1_DEBUG", "0")).strip().lower() not in {
    "0", "false", "no", "off", ""
}

# UI-toggled debug flag for batch preprocessing (Settings panel checkbox).
_debug_preprocess: bool = False
_debug_elastic_touchup_global: bool = False
_debug_elastic_field: bool = False


def set_preprocess_debug(enabled: bool) -> None:
    """Enable or disable verbose console output for preprocessing operations."""
    global _debug_preprocess
    _debug_preprocess = bool(enabled)


def is_preprocess_debug() -> bool:
    return _debug_preprocess


def set_debug_elastic_touchup(enabled: bool) -> None:
    """Enable or disable elastic touch-up debug image output (Settings panel checkbox)."""
    global _debug_elastic_touchup_global
    _debug_elastic_touchup_global = bool(enabled)


def is_debug_elastic_touchup() -> bool:
    return _debug_elastic_touchup_global


def set_debug_elastic_field(enabled: bool) -> None:
    """Enable or disable per-island elastic field statistics output."""
    global _debug_elastic_field
    _debug_elastic_field = bool(enabled)


def is_debug_elastic_field() -> bool:
    return _debug_elastic_field


def _dbg(msg: str) -> None:
    if _STEP1_DEBUG:
        print(f"[Step1 DEBUG] {msg}", flush=True)


@dataclass(frozen=True)
class CycleInput:
    path: str
    cycle: int
    label: str | None = None
    tissue: str | None = None
    species: str | None = None
    registration_marker: str | None = None
    channel_markers: list[str] | None = None
    channel_antibodies: list[str] | None = None


@dataclass
class _RegionTransform:
    label: int
    bbox: tuple[int, int, int, int]
    shift_y: float
    shift_x: float
    pixels: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _check_cancel(cancel_cb: Callable[[], bool] | None) -> None:
    try:
        if cancel_cb is not None and bool(cancel_cb()):
            raise RuntimeError("Cancelled")
    except RuntimeError:
        raise
    except Exception:
        return


def _cancel_futures(futures: Iterable[Future]) -> None:
    for fut in list(futures):
        fut.cancel()


def _iter_completed_futures(
    futures: Iterable[Future],
    cancel_cb: Callable[[], bool] | None,
):
    pending = set(futures)
    while pending:
        _check_cancel(cancel_cb)
        done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        if not done:
            continue
        for fut in done:
            _check_cancel(cancel_cb)
            yield fut
        _check_cancel(cancel_cb)


def _progress(
    msg: str,
    *,
    progress_cb: Callable[[str], None] | None,
    progress_event_cb: Optional[Callable[[dict], None]] = None,
    phase: str | None = None,
    idx: int | None = None,
    n: int | None = None,
    cycle: int | None = None,
) -> None:
    if progress_cb:
        progress_cb(str(msg))
    if progress_event_cb:
        evt = {"msg": str(msg)}
        if phase is not None:
            evt["phase"] = str(phase)
        if idx is not None:
            evt["idx"] = int(idx)
        if n is not None:
            evt["n"] = int(n)
        if cycle is not None:
            evt["cycle"] = int(cycle)
        progress_event_cb(evt)


def _find_channel_index(ch_names: list[str], marker: str) -> int | None:
    m = (marker or "").strip().lower()
    if not m:
        return None
    for i, nm in enumerate(ch_names):
        if (nm or "").strip().lower() == m:
            return i
    for i, nm in enumerate(ch_names):
        if m in (nm or "").strip().lower():
            return i
    return None


def _resolve_reg_channel(ci: "CycleInput", marker: str) -> int | None:
    """Find registration marker index, falling back to ci.channel_markers when
    the stitched file's own channel metadata doesn't contain the marker name."""
    from cycif_seg.io.ome_tiff import load_channel_names_only as _lcno
    idx = _find_channel_index(_lcno(ci.path), marker)
    if idx is None and ci.channel_markers:
        idx = _find_channel_index(list(ci.channel_markers), marker)
    return idx


def _tiff_info_yxc(path: str) -> tuple[tuple[int, int, int], np.dtype]:
    info = inspect_tiff_yxc(path)
    return tuple(int(v) for v in info["shape_yxc"]), np.dtype(info["dtype"])


def _pad_plane_to_canvas(plane_yx: np.ndarray, canvas_yx: tuple[int, int], *, out_dtype=np.float32) -> np.ndarray:
    if plane_yx.ndim != 2:
        raise ValueError(f"Expected (Y,X). Got shape={plane_yx.shape}")
    y, x = int(plane_yx.shape[0]), int(plane_yx.shape[1])
    Y, X = int(canvas_yx[0]), int(canvas_yx[1])
    out = np.zeros((Y, X), dtype=out_dtype)
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs] = plane_yx[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs].astype(out_dtype, copy=False)
    return out


def _cast_preserve_dtype(plane_f32: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return plane_f32.astype(dt, copy=False)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return np.clip(np.rint(plane_f32), info.min, info.max).astype(dt, copy=False)
    return plane_f32.astype(dt, copy=False)


def _downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1:
        return img.astype(np.float32, copy=False)
    y = (img.shape[0] // factor) * factor
    x = (img.shape[1] // factor) * factor
    if y <= 0 or x <= 0:
        return img.astype(np.float32, copy=False)
    trimmed = img[:y, :x].astype(np.float32, copy=False)
    return trimmed.reshape(y // factor, factor, x // factor, factor).mean(axis=(1, 3))


def _normalized_for_registration(img: np.ndarray) -> np.ndarray:
    arr = np.array(img, dtype=np.float32)  # always copy so in-place ops below don't alias caller's data
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if not np.isfinite(lo):
        lo = float(np.min(arr))
    if not np.isfinite(hi):
        hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1.0
    np.clip(arr, lo, hi, out=arr)
    arr -= lo
    arr /= (hi - lo)
    return arr


def estimate_translation(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    *,
    downsample: int = 1,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    if fixed_yx.ndim != 2 or moving_yx.ndim != 2:
        raise ValueError("estimate_translation expects 2D arrays")
    fixed_ds = _normalized_for_registration(_downsample_image(fixed_yx, downsample))
    moving_ds = _normalized_for_registration(_downsample_image(moving_yx, downsample))
    shift, _, _ = phase_cross_correlation(fixed_ds, moving_ds, upsample_factor=max(1, int(upsample_factor)))
    dy = float(shift[0]) * float(max(1, int(downsample)))
    dx = float(shift[1]) * float(max(1, int(downsample)))
    return dy, dx


def _apply_translation(plane_yx: np.ndarray, dy: float, dx: float, *, order: int = 1) -> np.ndarray:
    if ndi_shift is None:
        iy = int(round(float(dy)))
        ix = int(round(float(dx)))
        out = np.zeros_like(plane_yx)
        src_y0 = max(0, -iy)
        src_y1 = min(plane_yx.shape[0], plane_yx.shape[0] - iy) if iy >= 0 else plane_yx.shape[0]
        src_x0 = max(0, -ix)
        src_x1 = min(plane_yx.shape[1], plane_yx.shape[1] - ix) if ix >= 0 else plane_yx.shape[1]
        dst_y0 = max(0, iy)
        dst_x0 = max(0, ix)
        h = max(0, src_y1 - src_y0)
        w = max(0, src_x1 - src_x0)
        if h > 0 and w > 0:
            out[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = plane_yx[src_y0 : src_y0 + h, src_x0 : src_x0 + w]
        return out
    return ndi_shift(
        np.asarray(plane_yx),
        shift=(float(dy), float(dx)),
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=(int(order) > 1),
    )


def apply_translation_yxc(img_yxc: np.ndarray, dy: float, dx: float, *, order: int = 1) -> np.ndarray:
    out = np.zeros_like(img_yxc, dtype=np.float32)
    for c in range(int(img_yxc.shape[2])):
        out[:, :, c] = _apply_translation(img_yxc[:, :, c].astype(np.float32, copy=False), dy, dx, order=order)
    return out


def _foreground_mask(img_yx: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_yx, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    pos = arr[np.isfinite(arr)]
    if pos.size == 0:
        return np.zeros_like(arr, dtype=bool)
    thr = float(threshold_otsu(pos)) if np.any(pos > 0) else float(np.max(pos))
    mask = arr > thr
    if binary_dilation is not None and mask.any():
        mask = binary_dilation(mask, iterations=1)
    return mask.astype(bool, copy=False)


def _tile_component_masks(
    mask: np.ndarray,
    tile_hw: tuple[int, int],
    offset_yx: tuple[int, int],
    *,
    solid: bool = False,
) -> list[np.ndarray]:
    Y, X = mask.shape
    tile_h = max(1, int(tile_hw[0]))
    tile_w = max(1, int(tile_hw[1]))
    off_y = int(offset_yx[0])
    off_x = int(offset_yx[1])
    y_starts = list(range(off_y, Y, tile_h))
    x_starts = list(range(off_x, X, tile_w))
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    active = np.zeros((len(y_starts), len(x_starts)), dtype=bool)
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r, y0 in enumerate(y_starts):
        y1 = min(Y, y0 + tile_h)
        for c, x0 in enumerate(x_starts):
            x1 = min(X, x0 + tile_w)
            tile = mask[y0:y1, x0:x1]
            fg_frac = float(tile.mean()) if tile.size else 0.0
            if fg_frac > 0.005:
                active[r, c] = True
                bboxes[(r, c)] = (y0, y1, x0, x1)
    if not active.any():
        return []
    labs = sk_label(active, connectivity=1)
    comps: list[np.ndarray] = []
    for lab in range(1, int(labs.max()) + 1):
        comp = np.zeros_like(mask, dtype=bool)
        rr, cc = np.where(labs == lab)
        for r, c in zip(rr.tolist(), cc.tolist()):
            y0, y1, x0, x1 = bboxes[(r, c)]
            comp[y0:y1, x0:x1] = True
        if not solid:
            comp &= mask
        if comp.any():
            comps.append(comp)
    return comps


def _combine_component_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("No masks to combine")
    n = len(masks)
    uf = _UnionFind(n)
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            if np.any(mi & masks[j]):
                uf.union(i, j)
    merged: dict[int, np.ndarray] = {}
    for i, m in enumerate(masks):
        r = uf.find(i)
        if r not in merged:
            merged[r] = m.copy()
        else:
            merged[r] |= m
    final = np.zeros_like(masks[0], dtype=np.int32)
    lab = 0
    for m in merged.values():
        if not m.any():
            continue
        lab += 1
        final[m] = lab
    return final

def _tile_component_sets(
    mask: np.ndarray,
    tile_hw: tuple[int, int],
    offset_yx: tuple[int, int],
) -> list[list[tuple[int, int, int, int]]]:
    """Like _tile_component_masks but returns tile-bbox lists instead of full-canvas bool arrays."""
    Y, X = mask.shape
    tile_h = max(1, int(tile_hw[0]))
    tile_w = max(1, int(tile_hw[1]))
    off_y = int(offset_yx[0])
    off_x = int(offset_yx[1])
    y_starts = list(range(off_y, Y, tile_h))
    x_starts = list(range(off_x, X, tile_w))
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    active = np.zeros((len(y_starts), len(x_starts)), dtype=bool)
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r, y0 in enumerate(y_starts):
        y1 = min(Y, y0 + tile_h)
        for c, x0 in enumerate(x_starts):
            x1 = min(X, x0 + tile_w)
            tile = mask[y0:y1, x0:x1]
            fg_frac = float(tile.mean()) if tile.size else 0.0
            if fg_frac > 0.005:
                active[r, c] = True
                bboxes[(r, c)] = (y0, y1, x0, x1)
    if not active.any():
        return []
    labs = sk_label(active, connectivity=1)
    comps: list[list[tuple[int, int, int, int]]] = []
    for lab in range(1, int(labs.max()) + 1):
        rr, cc = np.where(labs == lab)
        tiles = [bboxes[(int(r), int(c))] for r, c in zip(rr.tolist(), cc.tolist())]
        comps.append(tiles)
    return comps


def _combine_tile_sets_to_label_image(
    comps: list[list[tuple[int, int, int, int]]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Merge overlapping tile-set components via union-find and write directly to a label image.

    No full-canvas bool masks are allocated — only the final int32 output.
    """
    n = len(comps)
    uf = _UnionFind(n)
    bounds: list[tuple[int, int, int, int]] = []
    for tiles in comps:
        bounds.append((
            min(t[0] for t in tiles),
            max(t[1] for t in tiles),
            min(t[2] for t in tiles),
            max(t[3] for t in tiles),
        ))
    for i in range(n):
        bi = bounds[i]
        for j in range(i + 1, n):
            bj = bounds[j]
            if bi[0] >= bj[1] or bi[1] <= bj[0] or bi[2] >= bj[3] or bi[3] <= bj[2]:
                continue
            merged = False
            for ta in comps[i]:
                if merged:
                    break
                for tb in comps[j]:
                    if ta[0] < tb[1] and ta[1] > tb[0] and ta[2] < tb[3] and ta[3] > tb[2]:
                        uf.union(i, j)
                        merged = True
                        break
    Y, X = int(shape[0]), int(shape[1])
    result = np.zeros((Y, X), dtype=np.int32)
    root_to_label: dict[int, int] = {}
    lab = 0
    for i, tiles in enumerate(comps):
        r = uf.find(i)
        if r not in root_to_label:
            lab += 1
            root_to_label[r] = lab
        lbl = root_to_label[r]
        for y0, y1, x0, x1 in tiles:
            result[y0:y1, x0:x1] = lbl
    return result


def _extract_translated_crop(
    moving: np.ndarray,
    bbox: tuple[int, int, int, int],
    dy: float,
    dx: float,
    *,
    order: int = 1,
    pad: int | None = None,
) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    if pad is None:
        pad = int(max(4, math.ceil(max(abs(float(dy)), abs(float(dx)))) + 2))
    Y, X = moving.shape
    py0 = max(0, y0 - pad)
    py1 = min(Y, y1 + pad)
    px0 = max(0, x0 - pad)
    px1 = min(X, x1 + pad)
    crop = moving[py0:py1, px0:px1]
    shifted = _apply_translation(crop, dy, dx, order=order)
    oy0 = y0 - py0
    oy1 = oy0 + (y1 - y0)
    ox0 = x0 - px0
    ox1 = ox0 + (x1 - x0)
    return shifted[oy0:oy1, ox0:ox1]


def _extract_translated_crop_from_buffer(
    moving_buffer: np.ndarray,
    buffer_origin_yx: tuple[int, int],
    bbox: tuple[int, int, int, int],
    dy: float,
    dx: float,
    *,
    order: int = 1,
) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    by0, bx0 = int(buffer_origin_yx[0]), int(buffer_origin_yx[1])
    shifted = _apply_translation(moving_buffer, dy, dx, order=order)
    oy0 = y0 - by0
    oy1 = oy0 + (y1 - y0)
    ox0 = x0 - bx0
    ox1 = ox0 + (x1 - x0)
    return shifted[oy0:oy1, ox0:ox1]


def _refine_bad_regions(
    fixed_yx,
    moving_yx,
    bad_labels,
    base_shift,
    *,
    initial_field: tuple[np.ndarray, np.ndarray] | None = None,
    search_radius,
    downsample,
    penalty_lambda: float = 0.0,
    progress_cb=None,
    progress_event_cb=None,
    cycle=None,
    improvement_threshold=0.05,
):
    global_base_dy, global_base_dx = base_shift
    fixed_n = _normalized_for_registration(fixed_yx)
    moving_n = _normalized_for_registration(moving_yx)
    regions = []
    labs = [l for l in np.unique(bad_labels) if l > 0]

    for i, lab in enumerate(labs, start=1):
        _progress(
            f"8. Refining position of poorly registered region {i}/{len(labs)}...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="bad_region_refine",
            idx=i,
            n=len(labs),
            cycle=cycle,
        )

        mask = bad_labels == lab
        bbox = _bbox_from_mask(mask, pad=max(16, int(search_radius)))
        y0, y1, x0, x1 = bbox

        f = fixed_n[y0:y1, x0:x1]
        m = moving_n[y0:y1, x0:x1]
        m_mask = mask[y0:y1, x0:x1]

        if initial_field is not None:
            fy, fx = initial_field
            if np.any(mask):
                base_dy = float(np.mean(fy[mask]))
                base_dx = float(np.mean(fx[mask]))
            else:
                base_dy = float(global_base_dy)
                base_dx = float(global_base_dx)
        else:
            base_dy = float(global_base_dy)
            base_dx = float(global_base_dx)

        moved = _apply_translation(m, base_dy, base_dx)
        base_score = _masked_corr_score(f, moved, m_mask)
        base_objective = float(base_score)
        try:
            dy, dx = estimate_translation(
                f * m_mask,
                moved * m_mask,
                downsample=downsample,
            )
        except Exception:
            dy, dx = 0.0, 0.0

        dy = np.clip(dy, -search_radius, search_radius)
        dx = np.clip(dx, -search_radius, search_radius)

        new_dy = float(base_dy + dy)
        new_dx = float(base_dx + dx)

        improved = _apply_translation(m, new_dy, new_dx)
        new_score = _masked_corr_score(f, improved, m_mask)
        dist = math.hypot(float(new_dy - global_base_dy), float(new_dx - global_base_dx))
        new_objective = float(new_score - penalty_lambda * dist)

        if new_objective > base_objective + improvement_threshold:
            regions.append(
                _RegionTransform(
                    label=int(lab),
                    bbox=bbox,
                    shift_y=float(new_dy),
                    shift_x=float(new_dx),
                    pixels=int(mask.sum()),
                )
            )

    return regions



def _identify_foreground_islands(mask: np.ndarray, tile_size: int, *, solid: bool = False) -> np.ndarray:
    half = max(100, int(round(float(tile_size) / 2.0)))
    tile_hw = (half, half)
    if solid:
        # Efficient path: components are tile-bbox lists — no full-canvas bool masks allocated.
        comps_a = _tile_component_sets(mask, tile_hw, (0, 0))
        comps_b = _tile_component_sets(mask, tile_hw, (half // 2, half // 2))
        all_comps = comps_a + comps_b
        if not all_comps:
            return np.zeros_like(mask, dtype=np.int32)
        return _combine_tile_sets_to_label_image(all_comps, mask.shape)
    comps_a = _tile_component_masks(mask, tile_hw, (0, 0), solid=False)
    comps_b = _tile_component_masks(mask, tile_hw, (half // 2, half // 2), solid=False)
    all_comps = comps_a + comps_b
    if not all_comps:
        return np.zeros_like(mask, dtype=np.int32)
    return _combine_component_masks(all_comps)


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(mask.shape[0], int(ys.max()) + 1 + int(pad))
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(mask.shape[1], int(xs.max()) + 1 + int(pad))
    return y0, y1, x0, x1


def _masked_corr_score(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    use = mask.astype(bool, copy=False)
    n = int(use.sum())
    if n < 32:
        return -1.0
    a = fixed[use].astype(np.float32, copy=False)
    b = moving[use].astype(np.float32, copy=False)
    a -= float(a.mean())
    b -= float(b.mean())
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da <= 1e-8 or db <= 1e-8:
        return -1.0
    return float(np.sum(a * b) / (da * db))


def _score_mask_for_crop(mask_crop: np.ndarray) -> np.ndarray:
    use = mask_crop.astype(bool, copy=False)
    if int(use.sum()) >= 32:
        return use
    return np.ones_like(use, dtype=bool)


def _island_sample_cells(
    region_mask: np.ndarray,
    *,
    cell_size: int,
    min_overlap_fraction: float = 0.005,
) -> list[tuple[int, int, int, int, float, float]]:
    y0, y1, x0, x1 = _bbox_from_mask(region_mask)
    size = max(1, int(cell_size))
    cells: list[tuple[int, int, int, int, float, float]] = []
    for cy0 in range(y0, y1, size):
        cy1 = min(y1, cy0 + size)
        for cx0 in range(x0, x1, size):
            cx1 = min(x1, cx0 + size)
            tile = region_mask[cy0:cy1, cx0:cx1]
            if tile.size == 0:
                continue
            if float(tile.mean()) <= float(min_overlap_fraction):
                continue
            cells.append((cy0, cy1, cx0, cx1, (cy0 + cy1 - 1) / 2.0, (cx0 + cx1 - 1) / 2.0))
    return cells


def _select_spread_cells(
    cells: list[tuple[int, int, int, int, float, float]],
    n: int,
) -> list[tuple[int, int, int, int]]:
    if not cells:
        return []
    target = max(1, int(n))
    if len(cells) <= target:
        return [(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in cells]

    pts = np.asarray([(float(c[4]), float(c[5])) for c in cells], dtype=np.float32)
    center = np.mean(pts, axis=0)
    first = int(np.argmin(np.sum((pts - center) ** 2, axis=1)))
    selected = [first]
    min_d2 = np.sum((pts - pts[first]) ** 2, axis=1)
    while len(selected) < target:
        min_d2[selected] = -1.0
        nxt = int(np.argmax(min_d2))
        if nxt in selected:
            break
        selected.append(nxt)
        min_d2 = np.minimum(min_d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
    return [(int(cells[i][0]), int(cells[i][1]), int(cells[i][2]), int(cells[i][3])) for i in selected]


def _sample_tile_registration_worker(
    fixed_crop: np.ndarray,
    moving_base_crop: np.ndarray,
    moving_buffer: np.ndarray,
    buffer_origin_yx: tuple[int, int],
    bbox: tuple[int, int, int, int],
    mask_crop: np.ndarray,
    base_shift: tuple[float, float],
    search_radius: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int, int, int]] | None:
    base_dy, base_dx = float(base_shift[0]), float(base_shift[1])
    base_score = _masked_corr_score(fixed_crop, moving_base_crop, mask_crop)
    try:
        resid_dy, resid_dx = estimate_translation(
            fixed_crop,
            moving_base_crop,
            downsample=1,
            upsample_factor=10,
        )
    except Exception:
        return None

    max_shift = float(search_radius)
    resid_dy = float(np.clip(resid_dy, -max_shift, max_shift))
    resid_dx = float(np.clip(resid_dx, -max_shift, max_shift))
    cand_dy = float(base_dy + resid_dy)
    cand_dx = float(base_dx + resid_dx)
    cand_crop = _extract_translated_crop_from_buffer(
        moving_buffer,
        buffer_origin_yx,
        bbox,
        cand_dy,
        cand_dx,
        order=1,
    )
    cand_score = _masked_corr_score(fixed_crop, cand_crop, mask_crop)
    if not (np.isfinite(base_score) and np.isfinite(cand_score)):
        return None
    return (
        resid_dy,
        resid_dx,
        float(base_score),
        fixed_crop,
        mask_crop,
        moving_buffer,
        buffer_origin_yx,
        bbox,
    )


def _estimate_sampled_region_shift(
    fixed_n: np.ndarray,
    moving_n: np.ndarray,
    region_mask: np.ndarray,
    base_shift: tuple[float, float],
    *,
    cell_size: int,
    sample_count: int,
    search_radius: int,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[float, float] | None:
    _check_cancel(cancel_cb)
    cells = _island_sample_cells(region_mask, cell_size=cell_size)
    if len(cells) <= 4:
        return None
    selected = _select_spread_cells(cells, sample_count)
    if not selected:
        return None

    base_dy, base_dx = float(base_shift[0]), float(base_shift[1])
    max_abs_shift = max(abs(base_dy), abs(base_dx)) + float(search_radius)
    buffer_pad = int(max(4, math.ceil(max_abs_shift) + 2))
    tasks = []
    for y0, y1, x0, x1 in selected:
        _check_cancel(cancel_cb)
        fixed_crop = fixed_n[y0:y1, x0:x1]
        moving_base_crop = _extract_translated_crop(moving_n, (y0, y1, x0, x1), base_dy, base_dx, order=1)
        mask_crop = _score_mask_for_crop(region_mask[y0:y1, x0:x1])
        py0 = max(0, int(y0) - buffer_pad)
        py1 = min(int(moving_n.shape[0]), int(y1) + buffer_pad)
        px0 = max(0, int(x0) - buffer_pad)
        px1 = min(int(moving_n.shape[1]), int(x1) + buffer_pad)
        moving_buffer = moving_n[py0:py1, px0:px1]
        tasks.append(
            (
                fixed_crop,
                moving_base_crop,
                moving_buffer,
                (py0, px0),
                (y0, y1, x0, x1),
                mask_crop,
                (base_dy, base_dx),
                int(search_radius),
            )
        )

    worker_count = min(len(tasks), max(1, int((os.cpu_count() or 2) - 1)))
    if worker_count <= 1:
        results = []
        for task in tasks:
            _check_cancel(cancel_cb)
            results.append(_sample_tile_registration_worker(*task))
    else:
        pool = ThreadPoolExecutor(max_workers=worker_count)
        futures: list[Future] = []
        try:
            futures = [pool.submit(_sample_tile_registration_worker, *task) for task in tasks]
            results = [fut.result() for fut in _iter_completed_futures(futures, cancel_cb)]
        except BaseException:
            _cancel_futures(futures)
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True, cancel_futures=False)

    good = [r for r in results if r is not None]
    residuals = [(float(r[0]), float(r[1])) for r in good]

    if not residuals:
        return None

    med_resid_y = float(np.median([v[0] for v in residuals]))
    med_resid_x = float(np.median([v[1] for v in residuals]))
    median_dy = float(base_dy + med_resid_y)
    median_dx = float(base_dx + med_resid_x)
    median_scores: list[float] = []
    base_scores = [float(r[2]) for r in good]
    for _resid_y, _resid_x, _base_score, fixed_crop, mask_crop, moving_buffer, buffer_origin_yx, bbox in good:
        cand_crop = _extract_translated_crop_from_buffer(
            moving_buffer,
            buffer_origin_yx,
            bbox,
            median_dy,
            median_dx,
            order=1,
        )
        score = _masked_corr_score(fixed_crop, cand_crop, mask_crop)
        if np.isfinite(score):
            median_scores.append(float(score))
    if not median_scores:
        return None
    if float(np.median(median_scores)) <= float(np.median(base_scores)) + 1e-6:
        return base_dy, base_dx
    return median_dy, median_dx


def _score_region_shift(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    region_mask: np.ndarray,
    cand_dy: float,
    cand_dx: float,
    *,
    base_dy: float,
    base_dx: float,
    penalty_lambda: float,
) -> float:
    moved_img = _apply_translation(moving_yx, cand_dy, cand_dx, order=1)
    moved_mask = _apply_translation(region_mask.astype(np.float32), cand_dy, cand_dx, order=0) > 0.5
    score = _masked_corr_score(fixed_yx, moved_img, moved_mask)
    dist = math.hypot(float(cand_dy - base_dy), float(cand_dx - base_dx))
    return float(score - penalty_lambda * dist)


def _refine_region_transforms(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    island_labels: np.ndarray,
    base_shift: tuple[float, float],
    *,
    search_radius: int,
    downsample: int,
    penalty_lambda: float,
    fast_large_island_refinement: bool,
    fast_large_island_sample_count: int,
    fast_large_island_cell_size: int,
    progress_cb: Callable[[str], None] | None,
    progress_event_cb: Optional[Callable[[dict], None]],
    cancel_cb: Callable[[], bool] | None,
    cycle: int,
) -> list[_RegionTransform]:
    regions: list[_RegionTransform] = []
    base_dy, base_dx = float(base_shift[0]), float(base_shift[1])
    labs = [int(v) for v in np.unique(island_labels) if int(v) > 0]
    total = len(labs)
    fixed_n = fixed_yx
    moving_n = moving_yx
    for i, lab in enumerate(labs, start=1):
        _check_cancel(cancel_cb)

        _progress(
            f"6. Registering Cycle {cycle}: refining foreground island {i}/{total}...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="region_refine",
            idx=i,
            n=total,
            cycle=cycle,
        )

        region_mask = island_labels == lab
        pixels = int(region_mask.sum())

        # Islands too small for reliable phase cross-correlation get the global shift.
        # search_radius^2 is a practical lower bound: a patch smaller than the search
        # window cannot distinguish a good shift from a spurious one.
        _min_reliable_px = max(32, int(search_radius) * int(search_radius))
        if pixels < _min_reliable_px:
            regions.append(_RegionTransform(
                label=int(lab),
                bbox=_bbox_from_mask(region_mask),
                shift_y=base_dy,
                shift_x=base_dx,
                pixels=pixels,
            ))
            continue

        if bool(fast_large_island_refinement):
            sampled_shift = _estimate_sampled_region_shift(
                fixed_n,
                moving_n,
                region_mask,
                (base_dy, base_dx),
                cell_size=max(128, int(fast_large_island_cell_size)),
                sample_count=max(1, int(fast_large_island_sample_count)),
                search_radius=int(search_radius),
                cancel_cb=cancel_cb,
            )
            if sampled_shift is not None:
                regions.append(
                    _RegionTransform(
                        label=int(lab),
                        bbox=_bbox_from_mask(region_mask),
                        shift_y=float(sampled_shift[0]),
                        shift_x=float(sampled_shift[1]),
                        pixels=pixels,
                    )
                )
                continue

        bbox = _bbox_from_mask(region_mask, pad=max(16, int(search_radius)))
        y0, y1, x0, x1 = bbox
        fixed_crop = fixed_n[y0:y1, x0:x1]
        moving_base_crop = _extract_translated_crop(moving_n, bbox, base_dy, base_dx, order=1)
        mask_crop = region_mask[y0:y1, x0:x1]

        base_score = _masked_corr_score(fixed_crop, moving_base_crop, mask_crop)
        base_objective = float(base_score)

        try:
            resid_dy, resid_dx = estimate_translation(
                fixed_crop,
                moving_base_crop,
                downsample=max(1, int(downsample)),
                upsample_factor=10,
            )
        except Exception:
            resid_dy, resid_dx = 0.0, 0.0

        max_shift = float(search_radius)
        resid_dy = float(np.clip(resid_dy, -max_shift, max_shift))
        resid_dx = float(np.clip(resid_dx, -max_shift, max_shift))

        cand_dy = float(base_dy + resid_dy)
        cand_dx = float(base_dx + resid_dx)
        cand_crop = _extract_translated_crop(moving_n, bbox, cand_dy, cand_dx, order=1)
        cand_score = _masked_corr_score(fixed_crop, cand_crop, mask_crop)
        dist = math.hypot(float(cand_dy - base_dy), float(cand_dx - base_dx))
        cand_objective = float(cand_score - penalty_lambda * dist)

        best_dy = base_dy
        best_dx = base_dx
        if cand_objective > base_objective + 1e-6:
            best_dy = cand_dy
            best_dx = cand_dx

        regions.append(
            _RegionTransform(
                label=int(lab),
                bbox=bbox,
                shift_y=float(best_dy),
                shift_x=float(best_dx),
                pixels=pixels,
            )
        )
    return regions


def _dense_shift_field(
    shape_yx: tuple[int, int],
    label_image: np.ndarray,
    regions: list[_RegionTransform],
    base_shift: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    field_y = np.full((Y, X), float(base_shift[0]), dtype=np.float32)
    field_x = np.full((Y, X), float(base_shift[1]), dtype=np.float32)
    if not regions:
        return field_y, field_x
    label_map = np.zeros((Y, X), dtype=np.int32)
    for reg in regions:
        y0, y1, x0, x1 = reg.bbox
        mask_crop = label_image[y0:y1, x0:x1] == int(reg.label)
        label_crop = label_map[y0:y1, x0:x1]
        fy_crop = field_y[y0:y1, x0:x1]
        fx_crop = field_x[y0:y1, x0:x1]
        label_crop[mask_crop] = int(reg.label)
        fy_crop[mask_crop] = float(reg.shift_y)
        fx_crop[mask_crop] = float(reg.shift_x)
    if distance_transform_edt is None:
        return field_y, field_x
    bg = label_map == 0
    if not np.any(bg):
        return field_y, field_x
    inds = distance_transform_edt(bg, return_distances=False, return_indices=True)
    nn_y = inds[0].astype(np.int32)
    nn_x = inds[1].astype(np.int32)
    del inds
    field_y[bg] = field_y[nn_y[bg], nn_x[bg]]
    field_x[bg] = field_x[nn_y[bg], nn_x[bg]]
    return field_y, field_x


def _piecewise_shift_field(
    shape_yx: tuple[int, int],
    label_image: np.ndarray,
    regions: list[_RegionTransform],
    base_shift: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Like _dense_shift_field but skips EDT: background pixels get the global shift."""
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    field_y = np.full((Y, X), float(base_shift[0]), dtype=np.float32)
    field_x = np.full((Y, X), float(base_shift[1]), dtype=np.float32)
    for reg in regions:
        y0, y1, x0, x1 = reg.bbox
        mask_crop = label_image[y0:y1, x0:x1] == int(reg.label)
        fy_crop = field_y[y0:y1, x0:x1]
        fx_crop = field_x[y0:y1, x0:x1]
        fy_crop[mask_crop] = float(reg.shift_y)
        fx_crop[mask_crop] = float(reg.shift_x)
    return field_y, field_x


def _warp_plane_by_field(plane_yx: np.ndarray, field_y: np.ndarray, field_x: np.ndarray, *, order: int) -> np.ndarray:
    Y, X = plane_yx.shape
    arr = np.asarray(plane_yx, dtype=np.float32)
    out = np.empty((Y, X), dtype=np.float32)
    gy = np.arange(Y, dtype=np.float32)
    gx = np.arange(X, dtype=np.float32)
    _CHUNK = 512
    for y0 in range(0, Y, _CHUNK):
        y1 = min(Y, y0 + _CHUNK)
        src_y = (gy[y0:y1, np.newaxis] - field_y[y0:y1]).ravel()
        src_x = (gx[np.newaxis, :] - field_x[y0:y1]).ravel()
        out[y0:y1] = map_coordinates(
            arr, [src_y, src_x], order=order, mode='constant', cval=0.0, prefilter=(order > 1)
        ).reshape(y1 - y0, X)
    return out


def _build_strip_shift_field(
    out_y0: int,
    out_y1: int,
    canvas_x: int,
    regs: list[_RegionTransform],
    base_dy: float,
    base_dx: float,
    island_labels_strip: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (strip_H, canvas_x) shift field for output rows [out_y0:out_y1].

    Uses only the compact list of _RegionTransform objects — no full-canvas arrays.
    Background pixels (not covered by any region bbox) get the global shift.
    If island_labels_strip is provided (shape (strip_H, canvas_x)), shifts are applied
    only to pixels matching each region's label rather than filling the entire bounding box.
    """
    strip_H = out_y1 - out_y0
    field_y = np.full((strip_H, canvas_x), float(base_dy), dtype=np.float32)
    field_x = np.full((strip_H, canvas_x), float(base_dx), dtype=np.float32)
    for reg in regs:
        ry0, ry1, rx0, rx1 = reg.bbox
        local_y0 = max(0, ry0 - out_y0)
        local_y1 = min(strip_H, ry1 - out_y0)
        rx0_c = max(0, rx0)
        rx1_c = min(canvas_x, rx1)
        if local_y0 >= local_y1 or rx0_c >= rx1_c:
            continue
        if island_labels_strip is not None:
            fy_crop = field_y[local_y0:local_y1, rx0_c:rx1_c]
            fx_crop = field_x[local_y0:local_y1, rx0_c:rx1_c]
            mask = island_labels_strip[local_y0:local_y1, rx0_c:rx1_c] == int(reg.label)
            fy_crop[mask] = float(reg.shift_y)
            fx_crop[mask] = float(reg.shift_x)
        else:
            field_y[local_y0:local_y1, rx0_c:rx1_c] = float(reg.shift_y)
            field_x[local_y0:local_y1, rx0_c:rx1_c] = float(reg.shift_x)
    return field_y, field_x


def _warp_strip_by_field(
    src_canvas_strip: np.ndarray,
    out_y0: int,
    strip_H: int,
    src_canvas_y0: int,
    field_y: np.ndarray,
    field_x: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Warp output rows [out_y0 : out_y0+strip_H] using src_canvas_strip as source.

    src_canvas_strip covers canvas rows [src_canvas_y0 : src_canvas_y0+src_H].
    field_y/field_x are shaped (strip_H, canvas_X) for the output strip.
    """
    arr = np.asarray(src_canvas_strip, dtype=np.float32)
    X = arr.shape[1]
    out = np.empty((strip_H, X), dtype=np.float32)
    gy = np.arange(out_y0, out_y0 + strip_H, dtype=np.float32)
    gx = np.arange(X, dtype=np.float32)
    _CHUNK = 512
    for j0 in range(0, strip_H, _CHUNK):
        j1 = min(strip_H, j0 + _CHUNK)
        # Canvas Y coordinates for the output rows, shifted by field -> source canvas row
        # Convert to src_canvas_strip-local index by subtracting src_canvas_y0
        src_y = (gy[j0:j1, np.newaxis] - field_y[j0:j1] - float(src_canvas_y0)).ravel()
        src_x = (gx[np.newaxis, :] - field_x[j0:j1]).ravel()
        out[j0:j1] = map_coordinates(
            arr, [src_y, src_x], order=order, mode='constant', cval=0.0, prefilter=(order > 1)
        ).reshape(j1 - j0, X)
    return out


def _scale_region_transform(r: _RegionTransform, D: int) -> _RegionTransform:
    """Scale a _RegionTransform from downsampled to full-resolution coordinates."""
    return _RegionTransform(
        label=r.label,
        bbox=(r.bbox[0] * D, r.bbox[1] * D, r.bbox[2] * D, r.bbox[3] * D),
        shift_y=r.shift_y * float(D),
        shift_x=r.shift_x * float(D),
        pixels=r.pixels * D * D,
    )


def registration_progress_sidecar_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".cyseg-registration-progress.json")


def registration_zarr_store_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".cyseg-registered.zarr")


def _cycle_write_order(cycles: list[CycleInput], reference_cycle: int) -> list[int]:
    cy = [int(ci.cycle) for ci in sorted(cycles, key=lambda x: int(x.cycle))]
    return [int(reference_cycle)] + [c for c in cy if c != int(reference_cycle)]


def _registration_layout(
    cycles: Iterable[CycleInput],
    *,
    reference_cycle: int | None = None,
) -> dict[str, Any]:
    cycles_l = sorted(list(cycles), key=lambda x: int(x.cycle))
    if not cycles_l:
        raise ValueError("No cycles provided")
    if reference_cycle is None:
        reference_cycle = int(cycles_l[0].cycle)

    infos: list[tuple[CycleInput, tuple[int, int, int], np.dtype]] = []
    canvas_y = 0
    canvas_x = 0
    total_ch = 0
    base_dtype: np.dtype | None = None
    for ci in cycles_l:
        shp, dt = _tiff_info_yxc(ci.path)
        infos.append((ci, shp, dt))
        canvas_y = max(canvas_y, int(shp[0]))
        canvas_x = max(canvas_x, int(shp[1]))
        total_ch += int(shp[2])
        if base_dtype is None:
            base_dtype = np.dtype(dt)
        elif np.dtype(dt) != np.dtype(base_dtype):
            raise ValueError(f"All cycles must have the same dtype. Got {base_dtype} vs {dt} ({ci.path})")
    assert base_dtype is not None

    merged_names: list[str] = []
    channel_offsets: dict[int, int] = {}
    cycle_n_channels: dict[int, int] = {}
    seen: dict[str, int] = {}
    out_c = 0
    for ci, shp, _ in infos:
        cycle = int(ci.cycle)
        channel_offsets[cycle] = out_c
        n_ch = int(shp[2])
        cycle_n_channels[cycle] = n_ch
        ch_names = load_channel_names_only(ci.path)
        for j in range(n_ch):
            nm = ch_names[j] if j < len(ch_names) else ""
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (nm or "").strip() or "Channel"
            stem = f"{base}_cy{ci.label if ci.label is not None else ci.cycle}"
            k = int(seen.get(stem, 0))
            out_nm = f"{stem}_{k+1}" if k else stem
            seen[stem] = k + 1
            merged_names.append(out_nm)
            out_c += 1

    return {
        "cycles": cycles_l,
        "infos": infos,
        "reference_cycle": int(reference_cycle),
        "canvas_yx": (int(canvas_y), int(canvas_x)),
        "total_ch": int(total_ch),
        "base_dtype": np.dtype(base_dtype),
        "merged_names": merged_names,
        "channel_offsets": channel_offsets,
        "cycle_n_channels": cycle_n_channels,
        "write_order": _cycle_write_order(cycles_l, int(reference_cycle)),
    }


def _registration_fingerprint(
    *,
    cycles: list[CycleInput],
    infos: list[tuple[CycleInput, tuple[int, int, int], np.dtype]],
    output_shape_yxc: tuple[int, int, int],
    dtype: np.dtype,
    merged_names: list[str],
    channel_offsets: dict[int, int],
    reference_cycle: int,
    registration_algorithm: str,
    global_translation_only: bool,
    tiled_rigid_tile_size: int,
    tiled_rigid_search_factor: float,
    low_mem: bool,
    strip_height: int | None,
) -> dict[str, Any]:
    input_records: list[dict[str, Any]] = []
    shape_by_cycle = {int(ci.cycle): tuple(int(v) for v in shp) for ci, shp, _ in infos}
    dtype_by_cycle = {int(ci.cycle): str(np.dtype(dt)) for ci, _shp, dt in infos}
    for ci in cycles:
        p = Path(ci.path)
        try:
            st = p.stat()
            size = int(st.st_size)
            mtime_ns = int(st.st_mtime_ns)
        except Exception:
            size = -1
            mtime_ns = -1
        input_records.append({
            "cycle": int(ci.cycle),
            "label": ci.label,
            "path": str(p),
            "size": size,
            "mtime_ns": mtime_ns,
            "shape_yxc": list(shape_by_cycle[int(ci.cycle)]),
            "dtype": dtype_by_cycle[int(ci.cycle)],
            "registration_marker": ci.registration_marker,
            "channel_markers": list(ci.channel_markers or []),
        })
    return {
        "schema_version": 1,
        "inputs": input_records,
        "output_shape_yxc": [int(v) for v in output_shape_yxc],
        "output_dtype": str(np.dtype(dtype)),
        "merged_names": list(merged_names),
        "channel_offsets": {str(int(k)): int(v) for k, v in channel_offsets.items()},
        "reference_cycle": int(reference_cycle),
        "registration_algorithm": str(registration_algorithm or "tiled_rigid"),
        "global_translation_only": bool(global_translation_only),
        "tiled_rigid_tile_size": int(tiled_rigid_tile_size),
        "tiled_rigid_search_factor": float(tiled_rigid_search_factor),
        "low_mem": bool(low_mem),
        "strip_height": int(strip_height) if strip_height is not None else None,
    }


def _load_registration_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_registration_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def _new_registration_manifest(
    *,
    output_path: str,
    fingerprint: dict[str, Any],
    write_order: list[int],
    channel_offsets: dict[int, int],
    cycle_n_channels: dict[int, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_path": str(output_path),
        "fingerprint": fingerprint,
        "write_order": [int(c) for c in write_order],
        "cycles": {
            str(int(c)): {
                "status": "pending",
                "channel_offset": int(channel_offsets[int(c)]),
                "n_channels": int(cycle_n_channels[int(c)]),
            }
            for c in write_order
        },
    }


def _manifest_completed_cycles(manifest: dict[str, Any], fingerprint: dict[str, Any], write_order: list[int]) -> set[int]:
    if not manifest or manifest.get("fingerprint") != fingerprint:
        return set()
    cycles_d = manifest.get("cycles") or {}
    completed: set[int] = set()
    for cy in write_order:
        rec = cycles_d.get(str(int(cy))) or {}
        if rec.get("status") == "complete":
            completed.add(int(cy))
        else:
            break
    return completed


def _channel_has_written_data(mm: np.ndarray, channel_index: int, *, row_chunk: int = 1024) -> bool:
    idx = int(channel_index)
    y = int(mm.shape[1])
    saw_nonzero = False
    global_min = None
    global_max = None
    for y0 in range(0, y, max(1, int(row_chunk))):
        y1 = min(y, y0 + max(1, int(row_chunk)))
        arr = np.asarray(mm[idx, y0:y1, :])
        if arr.size == 0:
            continue
        mn = arr.min()
        mx = arr.max()
        global_min = mn if global_min is None else min(global_min, mn)
        global_max = mx if global_max is None else max(global_max, mx)
        if bool(np.any(arr != 0)):
            saw_nonzero = True
        if saw_nonzero and global_min != global_max:
            return True
    return False


def _registered_zarr_has_expected_layout(
    store_path: Path,
    shape_yxc: tuple[int, int, int],
    dtype: np.dtype,
    channel_names: list[str],
) -> bool:
    try:
        import zarr as _zarr  # type: ignore

        arr = _zarr.open_array(str(store_path), mode="r")
        exp_cyx = (int(shape_yxc[2]), int(shape_yxc[0]), int(shape_yxc[1]))
        if tuple(int(v) for v in arr.shape) != exp_cyx:
            return False
        if np.dtype(arr.dtype) != np.dtype(dtype):
            return False
        got_names = list(arr.attrs.get("channel_names", []) or [])
        return [str(v) for v in got_names] == [str(v) for v in channel_names]
    except Exception:
        return False


def inspect_registration_flat_resume_state(
    cycles: Iterable[CycleInput],
    output_path: str | Path,
    *,
    reference_cycle: int | None = None,
    registration_algorithm: str = "tiled_rigid",
    global_translation_only: bool = False,
    tiled_rigid_tile_size: int = 2000,
    tiled_rigid_search_factor: float = 3,
    low_mem: bool = True,
    strip_height: int | None = None,
    completion: str = "hybrid",
    manifest_path: str | Path | None = None,
    force_from_cycle: int | None = None,
) -> dict[str, Any]:
    layout = _registration_layout(cycles, reference_cycle=reference_cycle)
    output_path = Path(output_path)
    shape_yxc = (int(layout["canvas_yx"][0]), int(layout["canvas_yx"][1]), int(layout["total_ch"]))
    fingerprint = _registration_fingerprint(
        cycles=layout["cycles"],
        infos=layout["infos"],
        output_shape_yxc=shape_yxc,
        dtype=layout["base_dtype"],
        merged_names=layout["merged_names"],
        channel_offsets=layout["channel_offsets"],
        reference_cycle=int(layout["reference_cycle"]),
        registration_algorithm=registration_algorithm,
        global_translation_only=global_translation_only,
        tiled_rigid_tile_size=tiled_rigid_tile_size,
        tiled_rigid_search_factor=tiled_rigid_search_factor,
        low_mem=low_mem,
        strip_height=strip_height,
    )
    write_order = [int(v) for v in layout["write_order"]]
    sidecar = Path(manifest_path) if manifest_path is not None else registration_progress_sidecar_path(output_path)
    zarr_store = registration_zarr_store_path(output_path)

    if force_from_cycle is not None:
        force = int(force_from_cycle)
        if force not in write_order:
            raise ValueError(f"--force-from-cycle {force} is not in expected write order: {write_order}")
        forced_complete = set(write_order[:write_order.index(force)])
        if forced_complete and not (output_path.is_file() or zarr_store.is_dir()):
            raise ValueError("--force-from-cycle cannot skip earlier cycles because no resumable registration output exists")
        return {
            "layout": layout,
            "fingerprint": fingerprint,
            "manifest_path": str(sidecar),
            "completed_cycles": sorted(forced_complete),
            "first_incomplete_cycle": force,
            "source": "force",
            "messages": [f"forced resume from cycle {force}"],
        }

    messages: list[str] = []
    completed: set[int] = set()
    mode = str(completion or "hybrid").strip().lower()
    manifest = _load_registration_manifest(sidecar)
    if mode in {"hybrid", "manifest"} and manifest is not None:
        completed = _manifest_completed_cycles(manifest, fingerprint, write_order)
        if completed:
            messages.append(f"trusted manifest complete cycles: {sorted(completed)}")
            if not (output_path.is_file() or zarr_store.is_dir()):
                messages.append("manifest listed complete cycles but registration output is missing; ignoring manifest completion")
                completed = set()
        elif manifest.get("fingerprint") != fingerprint:
            messages.append("manifest exists but does not match current inputs/settings")

    if mode == "manifest":
        first = next((c for c in write_order if c not in completed), None)
        return {
            "layout": layout,
            "fingerprint": fingerprint,
            "manifest_path": str(sidecar),
            "completed_cycles": sorted(completed),
            "first_incomplete_cycle": first,
            "source": "manifest",
            "messages": messages,
        }

    if not completed and zarr_store.is_dir():
        if not _registered_zarr_has_expected_layout(
            zarr_store,
            shape_yxc,
            np.dtype(layout["base_dtype"]),
            list(layout["merged_names"]),
        ):
            raise ValueError(f"Existing Zarr registration store is incompatible: {zarr_store}")
        messages.append(f"found resumable Zarr registration store: {zarr_store}")
    elif not completed and output_path.is_file():
        info = inspect_tiff_yxc(str(output_path))
        got_shape = tuple(int(v) for v in info["shape_yxc"])
        got_dtype = np.dtype(info["dtype"])
        if got_shape != shape_yxc:
            raise ValueError(f"Existing output shape mismatch. Expected {shape_yxc}, got {got_shape}")
        if got_dtype != np.dtype(layout["base_dtype"]):
            raise ValueError(f"Existing output dtype mismatch. Expected {layout['base_dtype']}, got {got_dtype}")
        if list(info.get("channel_names") or []) != list(layout["merged_names"]):
            raise ValueError("Existing output channel names do not match expected registration output")
        mm = tifffile.memmap(str(output_path), mode="r")
        try:
            if tuple(int(v) for v in mm.shape) != (shape_yxc[2], shape_yxc[0], shape_yxc[1]):
                raise ValueError(f"Existing output memmap shape mismatch: {mm.shape}")
            for cy in write_order:
                out_c0 = int(layout["channel_offsets"][int(cy)])
                n_ch = int(layout["cycle_n_channels"][int(cy)])
                ok = True
                for ch in range(n_ch):
                    if not _channel_has_written_data(mm, out_c0 + ch):
                        ok = False
                        break
                if ok:
                    completed.add(int(cy))
                else:
                    messages.append(f"pixel scan stopped at incomplete/suspicious cycle {int(cy)}")
                    break
        finally:
            try:
                from cycif_seg.io.ome_tiff import _close_memmap as _close_mm  # type: ignore
                _close_mm(mm)
            except Exception:
                pass
            try:
                del mm
            except Exception:
                pass
    elif not output_path.is_file() and not zarr_store.is_dir():
        messages.append("no existing registration output found")

    first = next((c for c in write_order if c not in completed), None)
    return {
        "layout": layout,
        "fingerprint": fingerprint,
        "manifest_path": str(sidecar),
        "completed_cycles": sorted(completed),
        "first_incomplete_cycle": first,
        "source": "pixel-scan" if completed or output_path.is_file() else "new",
        "messages": messages,
    }


# Crops larger than this in either dimension are downsampled before being
# passed to elastix, then the resulting field is upsampled back.
_ELASTIX_MAX_DIM: int = 512

# Elastix is run in a subprocess so that ITK's internal thread pool never
# initialises inside the napari/Qt process, where it hard-crashes due to a
# DLL conflict.  The worker script is written to a temp file at call time.
_ELASTIX_WORKER_SCRIPT = r"""
import sys, numpy as np, itk

d    = np.load(sys.argv[1])
ref  = d['ref'].astype(np.float32)
mov  = d['mov'].astype(np.float32)
isl  = d['isl'].astype(bool)
gs   = int(d['gs'][0])
itr  = int(d['it'][0])
msl  = float(d['msl'][0])

fixed_itk  = itk.GetImageFromArray(ref)
moving_itk = itk.GetImageFromArray(mov)

po = itk.ParameterObject.New()
pm = po.GetDefaultParameterMap('bspline')
pm['Registration']                        = ['MultiResolutionRegistration']
pm['Metric']                              = ['AdvancedNormalizedCorrelation']
pm['ImageSampler']                        = ['RandomCoordinate']
pm['NumberOfSpatialSamples']              = ['2000']
pm['MaximumNumberOfIterations']           = [str(itr)]
pm['FinalGridSpacingInPhysicalUnits']     = [str(gs)]
pm['MaximumStepLength']                   = [str(msl)]
pm['WriteResultImage']                    = ['false']
pm['WriteTransformParametersEachIteration'] = ['false']
pm['WriteResultImageAfterEachResolution'] = ['false']
po.AddParameterMap(pm)

_, tf = itk.elastix_registration_method(
    fixed_itk, moving_itk, parameter_object=po, log_to_console=False)
df_img = itk.transformix_deformation_field(moving_itk, tf, log_to_console=False)
df = itk.GetArrayFromImage(df_img)   # (H, W, 2): [0]=dx [1]=dy
fy = -df[..., 1].astype(np.float32)
fx = -df[..., 0].astype(np.float32)
fy[~isl] = 0.0
fx[~isl] = 0.0
np.savez_compressed(sys.argv[2], fy=fy, fx=fx)
"""

# Batch variant: processes N tile pairs in one subprocess, amortising startup overhead.
_ELASTIX_BATCH_WORKER_SCRIPT = r"""
import sys, numpy as np, itk

d    = np.load(sys.argv[1])
refs = d['refs'].astype(np.float32)   # (N, H, W)
movs = d['movs'].astype(np.float32)
isls = d['isls'].astype(bool)
gs   = int(d['gs'][0])
itr  = int(d['it'][0])
msl  = float(d['msl'][0])
N, H, W = refs.shape

fys = np.zeros((N, H, W), dtype=np.float32)
fxs = np.zeros((N, H, W), dtype=np.float32)
ok  = np.zeros(N, dtype=np.uint8)

po = itk.ParameterObject.New()
pm = po.GetDefaultParameterMap('bspline')
pm['Registration']                          = ['MultiResolutionRegistration']
pm['Metric']                                = ['AdvancedNormalizedCorrelation']
pm['ImageSampler']                          = ['RandomCoordinate']
pm['NumberOfSpatialSamples']                = ['2000']
pm['MaximumNumberOfIterations']             = [str(itr)]
pm['FinalGridSpacingInPhysicalUnits']       = [str(gs)]
pm['MaximumStepLength']                     = [str(msl)]
pm['WriteResultImage']                      = ['false']
pm['WriteTransformParametersEachIteration'] = ['false']
pm['WriteResultImageAfterEachResolution']   = ['false']
po.AddParameterMap(pm)

for i in range(N):
    try:
        fixed_itk  = itk.GetImageFromArray(refs[i])
        moving_itk = itk.GetImageFromArray(movs[i])
        _, tf = itk.elastix_registration_method(
            fixed_itk, moving_itk, parameter_object=po, log_to_console=False)
        df_img = itk.transformix_deformation_field(moving_itk, tf, log_to_console=False)
        df = itk.GetArrayFromImage(df_img)
        fy = -df[..., 1].astype(np.float32)
        fx = -df[..., 0].astype(np.float32)
        fy[~isls[i]] = 0.0
        fx[~isls[i]] = 0.0
        fys[i] = fy
        fxs[i] = fx
        ok[i] = 1
    except Exception:
        pass

np.savez_compressed(sys.argv[2], fys=fys, fxs=fxs, ok=ok)
"""


def _run_elastix_bspline_batch(
    tile_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    grid_spacing_px: int,
    max_iterations: int,
    max_step_length: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Run B-spline registration for N tile pairs in one subprocess.

    All pairs must have the same height; widths may differ (edge tiles are padded).
    Returns a list of (field_y, field_x) per tile, or None if that tile failed.
    """
    import sys
    import subprocess
    import tempfile

    if not tile_pairs:
        return []

    N = len(tile_pairs)
    H = int(tile_pairs[0][0].shape[0])
    W = int(tile_pairs[0][0].shape[1])
    ds = max(1, int(math.ceil(max(H, W) / _ELASTIX_MAX_DIM)))
    gs_px = max(4, grid_spacing_px // ds) if ds > 1 else int(grid_spacing_px)

    # Normalise and downsample each tile; record original (h, w) for later unpadding.
    refs_list: list[np.ndarray] = []
    movs_list: list[np.ndarray] = []
    isls_list: list[np.ndarray] = []
    orig_shapes: list[tuple[int, int]] = []

    for fixed_crop, moving_crop, mask_crop in tile_pairs:
        h_i, w_i = int(fixed_crop.shape[0]), int(fixed_crop.shape[1])
        if ds > 1:
            ref_n = _normalized_for_registration(_downsample_image(fixed_crop, ds))
            mov_n = _normalized_for_registration(_downsample_image(moving_crop, ds))
            _iy = (h_i // ds) * ds
            _ix = (w_i // ds) * ds
            isl_ds = mask_crop[:_iy, :_ix][::ds, ::ds].astype(bool)
        else:
            ref_n = _normalized_for_registration(fixed_crop)
            mov_n = _normalized_for_registration(moving_crop)
            isl_ds = mask_crop.astype(bool)
        if not np.isfinite(ref_n).all():
            ref_n = np.nan_to_num(ref_n, nan=0.0, posinf=1.0, neginf=0.0)
        if not np.isfinite(mov_n).all():
            mov_n = np.nan_to_num(mov_n, nan=0.0, posinf=1.0, neginf=0.0)
        refs_list.append(ref_n)
        movs_list.append(mov_n)
        isls_list.append(isl_ds.view(np.uint8))
        orig_shapes.append((int(ref_n.shape[0]), int(ref_n.shape[1])))

    # Pad all tiles to the max downsampled shape so they can be stacked.
    H_ds = max(s[0] for s in orig_shapes)
    W_ds = max(s[1] for s in orig_shapes)

    def _pad(arr: np.ndarray) -> np.ndarray:
        if arr.shape == (H_ds, W_ds):
            return arr
        out = np.zeros((H_ds, W_ds), dtype=arr.dtype)
        out[:arr.shape[0], :arr.shape[1]] = arr
        return out

    refs_arr = np.stack([_pad(r) for r in refs_list])
    movs_arr = np.stack([_pad(m) for m in movs_list])
    isls_arr = np.stack([_pad(s) for s in isls_list])

    try:
        with tempfile.TemporaryDirectory() as _td:
            _script = os.path.join(_td, 'w.py')
            _inp    = os.path.join(_td, 'inp.npz')
            _out    = os.path.join(_td, 'out.npz')

            with open(_script, 'w', encoding='utf-8') as _f:
                _f.write(_ELASTIX_BATCH_WORKER_SCRIPT)

            np.savez_compressed(
                _inp,
                refs=refs_arr, movs=movs_arr, isls=isls_arr,
                gs=np.array([gs_px],         dtype=np.int32),
                it=np.array([max_iterations], dtype=np.int32),
                msl=np.array([max(0.01, float(max_step_length))], dtype=np.float32),
            )

            _proc = subprocess.run(
                [sys.executable, _script, _inp, _out],
                timeout=max(300, N * 120),
                capture_output=True,
            )

            if _proc.returncode != 0 or not os.path.isfile(_out):
                if _debug_elastic_field:
                    print(
                        f"[elastic]   batch subprocess failed (N={N}): "
                        f"{_proc.stderr[-500:].decode(errors='replace') if _proc.stderr else ''}",
                        flush=True,
                    )
                return [None] * N

            import io as _io
            _npz_bytes: bytes | None = None
            for _attempt in range(5):
                try:
                    with open(_out, 'rb') as _f:
                        _npz_bytes = _f.read()
                    break
                except (PermissionError, OSError):
                    import time as _time
                    _time.sleep(0.1 * (2 ** _attempt))
            if _npz_bytes is None:
                return [None] * N

            _data = np.load(_io.BytesIO(_npz_bytes))
            fys_raw = _data['fys'].astype(np.float32)   # (N, H_ds, W_ds)
            fxs_raw = _data['fxs'].astype(np.float32)
            ok_arr  = _data['ok'].astype(bool)

    except Exception as _e:
        if _debug_elastic_field:
            print(f"[elastic]   batch subprocess error (N={N}): {_e}", flush=True)
        return [None] * N

    results: list[tuple[np.ndarray, np.ndarray] | None] = []
    for i, (h_ds, w_ds) in enumerate(orig_shapes):
        if not ok_arr[i]:
            results.append(None)
            continue
        fy_ds = fys_raw[i, :h_ds, :w_ds]
        fx_ds = fxs_raw[i, :h_ds, :w_ds]
        h_orig, w_orig = int(tile_pairs[i][0].shape[0]), int(tile_pairs[i][0].shape[1])
        if ds > 1:
            from scipy.ndimage import zoom as _sz
            field_y = (_sz(fy_ds, ds, order=1) * float(ds))[:h_orig, :w_orig]
            field_x = (_sz(fx_ds, ds, order=1) * float(ds))[:h_orig, :w_orig]
            if field_y.shape[0] < h_orig or field_y.shape[1] < w_orig:
                _fy = np.zeros((h_orig, w_orig), dtype=np.float32)
                _fx = np.zeros((h_orig, w_orig), dtype=np.float32)
                _fy[:field_y.shape[0], :field_y.shape[1]] = field_y
                _fx[:field_x.shape[0], :field_x.shape[1]] = field_x
                field_y, field_x = _fy, _fx
        else:
            field_y = fy_ds
            field_x = fx_ds
        results.append((field_y, field_x))

    return results


def _run_elastix_bspline(
    fixed_crop: np.ndarray,
    moving_crop: np.ndarray,
    island_crop: np.ndarray,
    *,
    grid_spacing_px: int,
    max_iterations: int,
    max_step_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a dense displacement field via elastix B-spline registration.

    Runs in a subprocess to keep ITK's DLL state isolated from the napari/Qt
    process.  Crops larger than _ELASTIX_MAX_DIM are downsampled first and the
    resulting field is upsampled back to the original resolution.

    Returns (field_y, field_x) in our warp convention:
        output[y, x] = moving[y - field_y[y,x], x - field_x[y,x]]
    """
    import sys
    import subprocess
    import tempfile

    H, W = int(fixed_crop.shape[0]), int(fixed_crop.shape[1])
    ds = max(1, int(math.ceil(max(H, W) / _ELASTIX_MAX_DIM)))

    if ds > 1:
        ref_n = _normalized_for_registration(_downsample_image(fixed_crop, ds))
        mov_n = _normalized_for_registration(_downsample_image(moving_crop, ds))
        # Trim island using the same floor-div footprint _downsample_image uses
        # so isl_ds shape matches ref_n exactly (avoids off-by-one on odd dims).
        _iy = (island_crop.shape[0] // ds) * ds
        _ix = (island_crop.shape[1] // ds) * ds
        isl_ds = island_crop[:_iy, :_ix][::ds, ::ds].astype(bool)
        gs_px  = max(4, grid_spacing_px // ds)
    else:
        ref_n  = _normalized_for_registration(fixed_crop)
        mov_n  = _normalized_for_registration(moving_crop)
        isl_ds = island_crop.astype(bool)
        gs_px  = int(grid_spacing_px)

    if not np.isfinite(ref_n).all():
        ref_n = np.nan_to_num(ref_n, nan=0.0, posinf=1.0, neginf=0.0)
    if not np.isfinite(mov_n).all():
        mov_n = np.nan_to_num(mov_n, nan=0.0, posinf=1.0, neginf=0.0)

    try:
        with tempfile.TemporaryDirectory() as _td:
            _script = os.path.join(_td, 'w.py')
            _inp    = os.path.join(_td, 'inp.npz')
            _out    = os.path.join(_td, 'out.npz')

            with open(_script, 'w', encoding='utf-8') as _f:
                _f.write(_ELASTIX_WORKER_SCRIPT)

            np.savez_compressed(
                _inp,
                ref=ref_n, mov=mov_n, isl=isl_ds.view(np.uint8),
                gs=np.array([gs_px],         dtype=np.int32),
                it=np.array([max_iterations], dtype=np.int32),
                msl=np.array([max(0.01, float(max_step_length))], dtype=np.float32),
            )

            _proc = subprocess.run(
                [sys.executable, _script, _inp, _out],
                timeout=300,
                capture_output=True,
            )

            if _proc.returncode != 0 or not os.path.isfile(_out):
                return None

            # Read bytes into memory before the TemporaryDirectory cleanup
            # removes the file, and to release the file handle immediately so
            # Windows file-locking (AV scan, indexer) doesn't block np.load.
            import io as _io
            _npz_bytes: bytes | None = None
            for _attempt in range(5):
                try:
                    with open(_out, 'rb') as _f:
                        _npz_bytes = _f.read()
                    break
                except (PermissionError, OSError):
                    import time as _time
                    _time.sleep(0.1 * (2 ** _attempt))
            if _npz_bytes is None:
                if _debug_elastic_field:
                    print("[elastic]     could not read output file (file lock)", flush=True)
                return None

            _data = np.load(_io.BytesIO(_npz_bytes))
            field_y_raw = _data['fy'].astype(np.float32)
            field_x_raw = _data['fx'].astype(np.float32)

    except Exception as _e:
        if _debug_elastic_field:
            print(f"[elastic]     subprocess error: {_e}", flush=True)
        return None

    if ds > 1:
        from scipy.ndimage import zoom as _sz
        field_y = (_sz(field_y_raw, ds, order=1) * float(ds))[:H, :W]
        field_x = (_sz(field_x_raw, ds, order=1) * float(ds))[:H, :W]
        if field_y.shape[0] < H or field_y.shape[1] < W:
            _fy = np.zeros((H, W), dtype=np.float32)
            _fx = np.zeros((H, W), dtype=np.float32)
            _fy[:field_y.shape[0], :field_y.shape[1]] = field_y
            _fx[:field_x.shape[0], :field_x.shape[1]] = field_x
            field_y, field_x = _fy, _fx
    else:
        field_y = field_y_raw
        field_x = field_x_raw

    island = island_crop.astype(bool)
    field_y[~island] = 0.0
    field_x[~island] = 0.0
    return field_y, field_x


def _load_elastic_touchup_crop(
    path: str,
    ch: int,
    y0_fr: int, y1_fr: int,
    x0_fr: int, x1_fr: int,
) -> np.ndarray:
    """Load float32 crop [y0_fr:y1_fr, x0_fr:x1_fr] for elastic touch-up.

    Uses load_channel_roi so only the overlapping tiles are decoded, regardless
    of whether the TIFF is uncompressed (memmap) or compressed pyramidal (zarr).
    """
    return load_channel_roi(path, ch, y0_fr, y1_fr, x0_fr, x1_fr)


def _elastic_touchup_island(
    ref_reg: np.ndarray,
    mov_reg: np.ndarray,
    island_mask: np.ndarray,
    rigid_shift: tuple[float, float],
    fg_mask: np.ndarray,
    *,
    tile_size: int,
    tile_size_h: int | None = None,
    skip_corr_threshold: float,
    min_fg_pixels: int,
    grid_spacing_px: int,
    max_iterations: int,
    large_island_px: int,
    max_step_length: float = 1.0,
    executor=None,
    cancel_cb: Callable[[], bool] | None = None,
    progress_event_cb=None,
    island_idx: int = 0,
    island_total: int = 0,
    cycle: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    _check_cancel(cancel_cb)
    n_fg = int(island_mask.sum())
    if n_fg < int(min_fg_pixels):
        if _debug_elastic_field:
            print(f"[elastic]   SKIP: too few fg pixels ({n_fg} < {min_fg_pixels})", flush=True)
        return None
    y0, y1, x0, x1 = _bbox_from_mask(island_mask)
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return None

    ref_crop = ref_reg[y0:y1, x0:x1]
    dy, dx = float(rigid_shift[0]), float(rigid_shift[1])
    mov_crop = _extract_translated_crop(mov_reg, (y0, y1, x0, x1), dy, dx)
    fg_crop = fg_mask[y0:y1, x0:x1]
    island_crop = island_mask[y0:y1, x0:x1]

    corr = _masked_corr_score(ref_crop, mov_crop, fg_crop)
    if _debug_elastic_field:
        _path = "tiled" if h * w >= int(large_island_px) else "direct"
        _verdict = "SKIP corr>threshold" if corr > float(skip_corr_threshold) else f"processing [{_path}]"
        print(
            f"[elastic]   bbox=({y0},{y1},{x0},{x1}) fg={n_fg} "
            f"corr={corr:.4f} thr={skip_corr_threshold:.4f} shift=({dy:.1f},{dx:.1f}) -> {_verdict}",
            flush=True,
        )
    if corr > float(skip_corr_threshold):
        return None

    if h * w < int(large_island_px):
        _check_cancel(cancel_cb)
        result = _run_elastix_bspline(
            ref_crop, mov_crop, island_crop,
            grid_spacing_px=grid_spacing_px, max_iterations=max_iterations,
            max_step_length=max_step_length,
        )
        if result is None:
            if _debug_elastic_field:
                print("[elastic]   -> no correction (elastix returned None)", flush=True)
            return None
        disp_y, disp_x = result
        if _debug_elastic_field:
            _mag = np.sqrt(disp_y ** 2 + disp_x ** 2)
            _isl = island_crop.astype(bool)
            _vals = _mag[_isl] if _isl.any() else _mag.ravel()
            print(
                f"[elastic]   -> direct disp: max={float(np.max(_mag)):.3f}px "
                f"mean(island)={float(np.mean(_vals)):.3f}px "
                f"p95(island)={float(np.percentile(_vals, 95)):.3f}px",
                flush=True,
            )
        return disp_y, disp_x, (y0, y1, x0, x1)

    # Tiled path: 50% stride, tent-weight blending
    _check_cancel(cancel_cb)
    _tile_h = max(1, int(tile_size_h) if tile_size_h is not None else int(tile_size))
    _tile_w = max(1, int(tile_size))
    stride_h = max(1, _tile_h // 2)
    stride_w = max(1, _tile_w // 2)
    disp_y_acc = np.zeros((h, w), dtype=np.float32)
    disp_x_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    ty_starts = list(range(0, h, stride_h))
    tx_starts = list(range(0, w, stride_w))

    # Collect qualifying tiles first so we know count before submitting
    _tile_tasks: list[tuple] = []
    _tile_corrs: list[float] = []
    _n_tiles_island_px = 0
    _n_tiles_corr_skipped = 0
    for _ty0 in ty_starts:
        _check_cancel(cancel_cb)
        _ty1 = min(h, _ty0 + _tile_h)
        _th = _ty1 - _ty0
        _tent_y = np.minimum(
            np.arange(_th, dtype=np.float32) + 1,
            np.arange(_th - 1, -1, -1, dtype=np.float32) + 1,
        ).clip(1e-6)
        for _tx0 in tx_starts:
            _check_cancel(cancel_cb)
            _tx1 = min(w, _tx0 + _tile_w)
            _tw = _tx1 - _tx0
            _tent_x = np.minimum(
                np.arange(_tw, dtype=np.float32) + 1,
                np.arange(_tw - 1, -1, -1, dtype=np.float32) + 1,
            ).clip(1e-6)
            _tent = (_tent_y[:, np.newaxis] * _tent_x[np.newaxis, :]).astype(np.float32)
            _t_island = island_crop[_ty0:_ty1, _tx0:_tx1]
            if int(_t_island.sum()) < 200:
                continue
            _n_tiles_island_px += 1
            _t_corr = _masked_corr_score(
                ref_crop[_ty0:_ty1, _tx0:_tx1],
                mov_crop[_ty0:_ty1, _tx0:_tx1],
                _t_island,
            )
            if _t_corr >= float(skip_corr_threshold):
                _n_tiles_corr_skipped += 1
                continue
            _tile_corrs.append(float(_t_corr))
            _tile_tasks.append((
                _ty0, _ty1, _tx0, _tx1, _tent,
                ref_crop[_ty0:_ty1, _tx0:_tx1].copy(),
                mov_crop[_ty0:_ty1, _tx0:_tx1].copy(),
                _t_island.copy(),
            ))

    _avg_corr_submitted = float(np.mean(_tile_corrs)) if _tile_corrs else 0.0

    if _debug_elastic_field:
        print(
            f"[elastic]   tiled: {_n_tiles_island_px} island tiles  "
            f"corr_skipped={_n_tiles_corr_skipped}  "
            f"submitting={len(_tile_tasks)}  "
            f"avg_corr={_avg_corr_submitted:.3f}  "
            f"parallel={'yes' if executor is not None and len(_tile_tasks) > 1 else 'no'}",
            flush=True,
        )

    _n_tiles_tried = len(_tile_tasks)
    _n_tiles_ok = 0
    _BAR_W = 20

    def _run_tile(_ty0, _ty1, _tx0, _tx1, _tent, _t_ref, _t_mov, _t_isl):
        _check_cancel(cancel_cb)
        _tres = _run_elastix_bspline(
            _t_ref, _t_mov, _t_isl,
            grid_spacing_px=grid_spacing_px, max_iterations=max_iterations,
            max_step_length=max_step_length,
        )
        if _tres is None:
            return None
        return _ty0, _ty1, _tx0, _tx1, _tent, _tres[0], _tres[1]

    def _accumulate(_res):
        nonlocal _n_tiles_ok
        if _res is None:
            return
        _ty0r, _ty1r, _tx0r, _tx1r, _tentr, _tdy, _tdx = _res
        _n_tiles_ok += 1
        disp_y_acc[_ty0r:_ty1r, _tx0r:_tx1r] += _tdy * _tentr
        disp_x_acc[_ty0r:_ty1r, _tx0r:_tx1r] += _tdx * _tentr
        weight_acc[_ty0r:_ty1r, _tx0r:_tx1r] += _tentr
        if progress_event_cb is not None:
            progress_event_cb({
                "msg": (
                    f"7. elastic touch-up Cycle {cycle}: "
                    f"island {island_idx}/{island_total} — "
                    f"{_n_tiles_ok}/{_n_tiles_tried} tiles"
                ),
                "phase": "elastic_touchup_tile",
            })

    def _bar(_done, _total):
        _f = min(_BAR_W, _done * _BAR_W // _total) if _total else _BAR_W
        return ('=' * _f + ('>' if _f < _BAR_W else '') + ' ' * max(0, _BAR_W - _f - 1))

    if executor is not None and len(_tile_tasks) > 1:
        _tile_futs: list[Future] = []
        _n_done = 0
        try:
            for _args in _tile_tasks:
                _check_cancel(cancel_cb)
                _tile_futs.append(executor.submit(_run_tile, *_args))
            for _fut in _iter_completed_futures(_tile_futs, cancel_cb):
                _accumulate(_fut.result())
                _n_done += 1
                if _debug_elastic_field:
                    print(
                        f"\r[elastic]   [{_bar(_n_done, _n_tiles_tried)}]"
                        f" {_n_done}/{_n_tiles_tried}  ok={_n_tiles_ok}",
                        end='', flush=True,
                    )
        except BaseException:
            _cancel_futures(_tile_futs)
            raise
        if _debug_elastic_field:
            print(flush=True)
    else:
        _n_done = 0
        for _args in _tile_tasks:
            _check_cancel(cancel_cb)
            _accumulate(_run_tile(*_args))
            _n_done += 1
            if _debug_elastic_field:
                print(
                    f"\r[elastic]   [{_bar(_n_done, _n_tiles_tried)}]"
                    f" {_n_done}/{_n_tiles_tried}  ok={_n_tiles_ok}",
                    end='', flush=True,
                )
        if _debug_elastic_field and _n_tiles_tried > 0:
            print(flush=True)

    nz = weight_acc > 0
    if not nz.any():
        if _debug_elastic_field:
            print(
                f"[elastic]   -> tiled: no corrections accumulated "
                f"(tiles_tried={_n_tiles_tried}, tiles_ok={_n_tiles_ok})",
                flush=True,
            )
        return None
    disp_y_acc[nz] /= weight_acc[nz]
    disp_x_acc[nz] /= weight_acc[nz]
    if _debug_elastic_field:
        _mag = np.sqrt(disp_y_acc[nz] ** 2 + disp_x_acc[nz] ** 2)
        _isl_nz = (island_crop.astype(bool))[nz]
        _vals = _mag[_isl_nz] if _isl_nz.any() else _mag
        print(
            f"[elastic]   -> tiled disp ({_n_tiles_ok}/{_n_tiles_tried} tiles): "
            f"max={float(np.max(_mag)):.3f}px "
            f"mean(island)={float(np.mean(_vals)):.3f}px "
            f"p95(island)={float(np.percentile(_vals, 95)):.3f}px",
            flush=True,
        )
    return disp_y_acc, disp_x_acc, (y0, y1, x0, x1)


def merge_cycles_to_ome_tiff(
    cycles: Iterable[CycleInput],
    output_path: str,
    *,
    reference_cycle: int | None = None,
    default_registration_marker: str = "DAPI",
    registration_algorithm: str = "tiled_rigid",
    global_translation_only: bool = False,
    downsample_for_registration: int = 4,
    tiled_rigid_allow_rotation: bool = False,
    tiled_rigid_tile_size: int = 2000,
    tiled_rigid_search_factor: float = 3,
    fast_large_island_refinement: bool = False,
    fast_large_island_sample_count: int = 5,
    upsample_factor: int = 10,
    low_mem: bool = False,
    strip_height: int | None = None,
    elastic_touchup: bool = False,
    elastic_touchup_tile_size: int = 2048,
    elastic_touchup_skip_corr: float = 0.95,
    elastic_touchup_bspline_spacing: int = 50,
    elastic_touchup_max_iterations: int = 10,
    elastic_touchup_large_island_px: int = 4_000_000,
    elastic_touchup_workers: int = 0,
    elastic_touchup_max_step_length: float = 1.0,
    debug_elastic_touchup: bool = False,
    debug_dir: str | None = None,
    pyramidal_output: bool = False,
    pyramidal_tile_size: int = 512,
    pyramidal_compression: str | int | None = "zlib",
    pyramidal_min_level_size: int = 128,
    pyramid_progress_chunk: int = 1024,
    resume_flat_output: bool = False,
    completed_cycles: Iterable[int] | None = None,
    registration_progress_path: str | None = None,
    registration_fingerprint: dict[str, Any] | None = None,
    progress_cb: Callable[[str], None] | None = None,
    progress_event_cb: Optional[Callable[[dict], None]] = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    cycles = list(cycles)
    if not cycles:
        raise ValueError("No cycles provided")

    if tiled_rigid_allow_rotation:
        _dbg("tiled_rigid_allow_rotation is ignored; foreground island registration is translation-only")
    if str(registration_algorithm or "").strip().lower() not in {"translation", "tiled_rigid", "tiled", "phase_correlation", "phase-correlation", "phase correlation", ""}:
        raise ValueError(f"Unknown registration_algorithm: {registration_algorithm!r}")

    cycles = sorted(cycles, key=lambda x: int(x.cycle))
    cy_vals = [int(ci.cycle) for ci in cycles]
    if any(cy < 0 for cy in cy_vals):
        raise ValueError(f"Cycle numbers must be >= 0. Got: {cy_vals}")
    if len(set(cy_vals)) != len(cy_vals):
        raise ValueError(f"Cycle numbers must be unique. Got: {cy_vals}")

    if reference_cycle is None:
        reference_cycle = int(cycles[0].cycle)

    infos: list[tuple[CycleInput, tuple[int, int, int], np.dtype]] = []
    input_pyramid_info: dict[int, dict] = {}
    canvas_y = 0
    canvas_x = 0
    total_input_ch = 0
    base_dtype: np.dtype | None = None
    for ci in cycles:
        _check_cancel(cancel_cb)
        try:
            input_pyramid_info[int(ci.cycle)] = inspect_tiff_pyramid(ci.path)
        except Exception:
            input_pyramid_info[int(ci.cycle)] = {"is_pyramidal": False, "series": []}
        shp, dt = _tiff_info_yxc(ci.path)
        infos.append((ci, shp, dt))
        canvas_y = max(canvas_y, int(shp[0]))
        canvas_x = max(canvas_x, int(shp[1]))
        total_input_ch += int(shp[2])
        if base_dtype is None:
            base_dtype = np.dtype(dt)
        elif np.dtype(dt) != np.dtype(base_dtype):
            raise ValueError(f"All cycles must have the same dtype. Got {base_dtype} vs {dt} ({ci.path})")

    assert base_dtype is not None
    canvas_yx = (int(canvas_y), int(canvas_x))

    # Strip-mode setup: activate when low_mem=True or strip_height is explicitly set.
    D = max(1, int(downsample_for_registration))
    _strip_h: int | None = strip_height
    if _strip_h is None and low_mem:
        _strip_h = max(1000, min(canvas_yx[0] // 10, 3700))
    _strip_mode = _strip_h is not None and _strip_h > 0
    # Downsampled canvas dimensions used during strip-mode registration
    _ds_canvas_yx = (max(1, canvas_yx[0] // D), max(1, canvas_yx[1] // D)) if _strip_mode else canvas_yx

    if _debug_preprocess:
        if _strip_mode:
            _n_strips = (canvas_yx[0] + _strip_h - 1) // _strip_h
            print(
                f"[preprocess] strip_mode=ON  strip_height={_strip_h} px  "
                f"canvas={canvas_yx[0]}×{canvas_yx[1]}  strips_per_cycle={_n_strips}  "
                f"reg_downsample={D}×  ds_canvas={_ds_canvas_yx[0]}×{_ds_canvas_yx[1]}",
                flush=True,
            )
        else:
            print(
                f"[preprocess] strip_mode=OFF  low_mem={low_mem}  "
                f"canvas={canvas_yx[0]}×{canvas_yx[1]}",
                flush=True,
            )

    ref_ci = next((ci for ci, _, _ in infos if int(ci.cycle) == int(reference_cycle)), None)
    if ref_ci is None:
        raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

    merged_names: list[str] = []
    channel_offsets: dict[int, int] = {}
    cycle_n_channels: dict[int, int] = {}
    seen: dict[str, int] = {}
    out_c = 0
    for ci, shp, _ in infos:
        cycle = int(ci.cycle)
        channel_offsets[cycle] = out_c
        ch_names = load_channel_names_only(ci.path)
        n_ch = int(shp[2])
        cycle_n_channels[cycle] = n_ch
        for j in range(n_ch):
            nm = ch_names[j] if j < len(ch_names) else ""
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (nm or "").strip() or "Channel"
            stem = f"{base}_cy{ci.label if ci.label is not None else ci.cycle}"
            k = int(seen.get(stem, 0))
            out_nm = f"{stem}_{k+1}" if k else stem
            seen[stem] = k + 1
            merged_names.append(out_nm)
            out_c += 1

    ref_marker = ref_ci.registration_marker or default_registration_marker
    ref_ch_idx = _resolve_reg_channel(ref_ci, ref_marker)
    if ref_ch_idx is None:
        raise ValueError(f"Could not find registration marker {ref_marker!r} in reference cycle {int(ref_ci.cycle)}")
    if _strip_mode:
        _ref_ds = load_channel_downsampled(ref_ci.path, int(ref_ch_idx), D)
        _ref_yx_ds = _pad_plane_to_canvas(_ref_ds.astype(np.float32), _ds_canvas_yx, out_dtype=np.float32)
        ref_reg = _normalized_for_registration(_ref_yx_ds)
        del _ref_ds, _ref_yx_ds
    else:
        ref_plane_native = load_single_channel_tiff_native(ref_ci.path, int(ref_ch_idx))
        ref_yx = _pad_plane_to_canvas(ref_plane_native, canvas_yx, out_dtype=np.float32)
        ref_reg = _normalized_for_registration(ref_yx)
        del ref_plane_native, ref_yx
    ref_pixel_sizes = load_physical_pixel_sizes(ref_ci.path)

    shifts: dict[int, tuple[float, float]] = {int(ref_ci.cycle): (0.0, 0.0)}
    island_counts: dict[int, int] = {int(ref_ci.cycle): 1}
    cycle_region_shift_summaries: dict[int, list[dict]] = {int(ref_ci.cycle): []}

    moving_infos = [item for item in infos if int(item[0].cycle) != int(reference_cycle)]
    total_ch = int(total_input_ch)
    total_ticks = 1 + len(moving_infos) * 4 + len(cycles) + (1 if pyramidal_output else 0)
    tick = 0
    completed_cycle_set = {int(c) for c in (completed_cycles or [])}
    write_order = _cycle_write_order(cycles, int(reference_cycle))

    _progress(
        f"1. Loading reference Cycle {int(ref_ci.cycle)}...",
        progress_cb=progress_cb,
        progress_event_cb=progress_event_cb,
        phase="load_ref",
        idx=tick,
        n=total_ticks,
        cycle=int(ref_ci.cycle),
    )
    tick += 1

    region_search_radius = max(8, int(round(max(1, tiled_rigid_tile_size) * max(1.0, float(tiled_rigid_search_factor)) / 4.0)))
    out_path = str(output_path)
    output_shape_yxc = (int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch))
    fingerprint = registration_fingerprint or _registration_fingerprint(
        cycles=cycles,
        infos=infos,
        output_shape_yxc=output_shape_yxc,
        dtype=base_dtype,
        merged_names=merged_names,
        channel_offsets=channel_offsets,
        reference_cycle=int(reference_cycle),
        registration_algorithm=registration_algorithm,
        global_translation_only=global_translation_only,
        tiled_rigid_tile_size=int(tiled_rigid_tile_size),
        tiled_rigid_search_factor=float(tiled_rigid_search_factor),
        low_mem=bool(low_mem),
        strip_height=int(_strip_h) if _strip_h is not None else None,
    )
    progress_path = Path(registration_progress_path) if registration_progress_path else registration_progress_sidecar_path(out_path)
    manifest = _load_registration_manifest(progress_path)
    if not manifest or manifest.get("fingerprint") != fingerprint:
        manifest = _new_registration_manifest(
            output_path=out_path,
            fingerprint=fingerprint,
            write_order=write_order,
            channel_offsets=channel_offsets,
            cycle_n_channels=cycle_n_channels,
        )
        for cy in completed_cycle_set:
            if str(int(cy)) in manifest["cycles"]:
                manifest["cycles"][str(int(cy))]["status"] = "complete"
    registered_zarr_path = registration_zarr_store_path(out_path)
    use_zarr_registration_store = bool(pyramidal_output)
    step1_workers = (
        max(1, int(elastic_touchup_workers))
        if elastic_touchup_workers and int(elastic_touchup_workers) > 0
        else max(1, (os.cpu_count() or 2) - 1)
    )
    if _debug_preprocess:
        print(f"[preprocess] step1_workers={step1_workers}", flush=True)
    open_existing_output = (
        bool(resume_flat_output and registered_zarr_path.is_dir())
        if use_zarr_registration_store
        else bool(resume_flat_output and Path(out_path).is_file())
    )
    _doing_debug_write = (debug_elastic_touchup or _debug_elastic_touchup_global) and elastic_touchup
    _debug_rigid_out_path: str | None = None
    if _doing_debug_write:
        _dbg_out_name = Path(output_path).name
        for _ext in ('.ome.tiff', '.ome.tif', '.tiff', '.tif'):
            if _dbg_out_name.lower().endswith(_ext.lower()):
                _dbg_out_name = _dbg_out_name[:-len(_ext)]
                break
        _dbg_out_dir = Path(debug_dir) if debug_dir else Path(output_path).parent
        _debug_rigid_out_path = str(_dbg_out_dir / f"{_dbg_out_name}_rigid_only.ome.tiff")
    with ExitStack() as _writer_stack:
        if use_zarr_registration_store:
            writer = _writer_stack.enter_context(IncrementalZarrRegisteredWriter(
                str(registered_zarr_path),
                output_shape_yxc,
                base_dtype,
                merged_names,
                physical_pixel_sizes=ref_pixel_sizes,
                chunk_yx=(int(pyramidal_tile_size), int(pyramidal_tile_size)),
                write_workers=step1_workers,
                cancel_cb=cancel_cb,
                open_existing=open_existing_output,
            ))
        else:
            writer = _writer_stack.enter_context(IncrementalOmeBigTiffWriter(
                out_path,
                output_shape_yxc,
                base_dtype,
                merged_names,
                physical_pixel_sizes=ref_pixel_sizes,
                open_existing=open_existing_output,
            ))
        debug_writer = (
            _writer_stack.enter_context(IncrementalOmeBigTiffWriter(
                _debug_rigid_out_path,
                output_shape_yxc,
                base_dtype,
                merged_names,
                physical_pixel_sizes=ref_pixel_sizes,
                open_existing=False,
            ))
            if _doing_debug_write else None
        )
        _save_registration_manifest(progress_path, manifest)

        def _set_cycle_status(cycle: int, status: str) -> None:
            rec = manifest.setdefault("cycles", {}).setdefault(str(int(cycle)), {})
            rec["status"] = str(status)
            if status == "complete":
                rec["completed_at_unix"] = time.time()
            _save_registration_manifest(progress_path, manifest)

        def _upsample_ds_mask_to_local(
            mask_ds: np.ndarray,
            *,
            label: int | None,
            y0_fr: int,
            y1_fr: int,
            x0_fr: int,
            x1_fr: int,
            downsample: int,
        ) -> np.ndarray:
            _ds = max(1, int(downsample))
            _h = max(0, int(y1_fr) - int(y0_fr))
            _w = max(0, int(x1_fr) - int(x0_fr))
            out = np.zeros((_h, _w), dtype=bool)
            if _h <= 0 or _w <= 0:
                return out
            _ds_y0 = max(0, int(y0_fr) // _ds)
            _ds_y1 = min(mask_ds.shape[0], (int(y1_fr) + _ds - 1) // _ds)
            _ds_x0 = max(0, int(x0_fr) // _ds)
            _ds_x1 = min(mask_ds.shape[1], (int(x1_fr) + _ds - 1) // _ds)
            if _ds_y0 >= _ds_y1 or _ds_x0 >= _ds_x1:
                return out
            _crop = mask_ds[_ds_y0:_ds_y1, _ds_x0:_ds_x1]
            if label is not None:
                _crop = _crop == int(label)
            else:
                _crop = _crop.astype(bool, copy=False)
            _full = np.repeat(np.repeat(_crop, _ds, axis=0), _ds, axis=1)
            _off_y = int(y0_fr) - _ds_y0 * _ds
            _off_x = int(x0_fr) - _ds_x0 * _ds
            _valid_h = min(_h, _full.shape[0] - _off_y)
            _valid_w = min(_w, _full.shape[1] - _off_x)
            if _valid_h > 0 and _valid_w > 0:
                out[:_valid_h, :_valid_w] = _full[_off_y:_off_y + _valid_h, _off_x:_off_x + _valid_w]
            return out

        def _make_strip_elastic_correction_provider(
            *,
            ci: CycleInput,
            registration_channel: int,
            moving_shape_yx: tuple[int, int],
            regs_ds: list[_RegionTransform],
            islands_ds: np.ndarray,
            downsample: int,
            cycle: int,
            base_shift_ds: tuple[float, float] = (0.0, 0.0),
        ) -> Callable[[int, int], tuple[np.ndarray, np.ndarray] | None]:
            import zarr as _zarr  # type: ignore

            _ds = max(1, int(downsample))
            _canvas_h, _canvas_w = int(canvas_yx[0]), int(canvas_yx[1])
            _n_elastic_workers = max(1, int(step1_workers))
            _et_tile_h = max(1, min(int(elastic_touchup_tile_size), int(_strip_h)))
            _et_tile_w = max(1, int(elastic_touchup_tile_size))
            _stride_h = max(1, _et_tile_h // 2)
            _stride_w = max(1, _et_tile_w // 2)
            _min_fg_px = 200
            _base_dy_fr = float(base_shift_ds[0]) * float(_ds)
            _base_dx_fr = float(base_shift_ds[1]) * float(_ds)
            _moving_shape_yx = (int(moving_shape_yx[0]), int(moving_shape_yx[1]))
            _ref_shape_yx = (int(ref_shp[0]), int(ref_shp[1]))

            _root = Path(out_path)
            _acc_path = _root.with_name(f"{_root.stem}.cyseg-elastic-cycle-{int(cycle)}.zarr")
            if _acc_path.exists():
                shutil.rmtree(_acc_path, ignore_errors=True)
            _chunk_y = max(1, min(_canvas_h, int(_strip_h)))
            _chunk_x = max(1, min(_canvas_w, int(elastic_touchup_tile_size)))
            _zg = _zarr.open_group(str(_acc_path), mode="w")

            def _create_acc_array(name: str):
                kwargs = {
                    "shape": (_canvas_h, _canvas_w),
                    "chunks": (_chunk_y, _chunk_x),
                    "dtype": "float32",
                    "fill_value": 0.0,
                }
                create = getattr(_zg, "create_array", None)
                if callable(create):
                    return create(name, **kwargs)
                return _zg.create_dataset(name, **kwargs)

            _dy_sum = _create_acc_array("dy_sum")
            _dx_sum = _create_acc_array("dx_sum")
            _wt_sum = _create_acc_array("weight")

            def _cleanup() -> None:
                try:
                    shutil.rmtree(_acc_path, ignore_errors=True)
                except Exception:
                    pass

            def _canvas_channel_roi(
                path: str,
                ch: int,
                src_shape_yx: tuple[int, int],
                y0: int,
                y1: int,
                x0: int,
                x1: int,
            ) -> np.ndarray:
                _h = max(0, int(y1) - int(y0))
                _w = max(0, int(x1) - int(x0))
                _out = np.zeros((_h, _w), dtype=np.float32)
                if _h <= 0 or _w <= 0:
                    return _out
                _src_h, _src_w = int(src_shape_yx[0]), int(src_shape_yx[1])
                _pad_y0 = max((_canvas_h - _src_h) // 2, 0)
                _pad_x0 = max((_canvas_w - _src_w) // 2, 0)
                _in_y0 = max((_src_h - _canvas_h) // 2, 0)
                _in_x0 = max((_src_w - _canvas_w) // 2, 0)
                _ys = min(_src_h - _in_y0, _canvas_h - _pad_y0)
                _xs = min(_src_w - _in_x0, _canvas_w - _pad_x0)
                _ov_y0 = max(int(y0), _pad_y0, 0)
                _ov_y1 = min(int(y1), _pad_y0 + _ys, _canvas_h)
                _ov_x0 = max(int(x0), _pad_x0, 0)
                _ov_x1 = min(int(x1), _pad_x0 + _xs, _canvas_w)
                if _ov_y0 >= _ov_y1 or _ov_x0 >= _ov_x1:
                    return _out
                _fy0 = _in_y0 + (_ov_y0 - _pad_y0)
                _fy1 = _in_y0 + (_ov_y1 - _pad_y0)
                _fx0 = _in_x0 + (_ov_x0 - _pad_x0)
                _fx1 = _in_x0 + (_ov_x1 - _pad_x0)
                _crop = _load_elastic_touchup_crop(path, ch, _fy0, _fy1, _fx0, _fx1)
                _oy0 = _ov_y0 - int(y0)
                _ox0 = _ov_x0 - int(x0)
                _out[_oy0:_oy0 + _crop.shape[0], _ox0:_ox0 + _crop.shape[1]] = _crop
                return _out

            def _iter_tiles(
                y0: int,
                y1: int,
                x0: int,
                x1: int,
                *,
                overlap: bool,
            ) -> Iterable[tuple[int, int, int, int]]:
                _step_y = _stride_h if overlap else _et_tile_h
                _step_x = _stride_w if overlap else _et_tile_w
                for _gy0 in range(int(y0), int(y1), max(1, _step_y)):
                    _gy1 = min(int(y1), _gy0 + _et_tile_h)
                    if _gy1 <= _gy0:
                        continue
                    for _gx0 in range(int(x0), int(x1), max(1, _step_x)):
                        _gx1 = min(int(x1), _gx0 + _et_tile_w)
                        if _gx1 > _gx0:
                            yield _gy0, _gy1, _gx0, _gx1

            def _island_bbox_fr(reg: _RegionTransform) -> tuple[int, int, int, int]:
                _y0 = max(0, int(reg.bbox[0]) * _ds)
                _y1 = min(_canvas_h, int(reg.bbox[1]) * _ds)
                _x0 = max(0, int(reg.bbox[2]) * _ds)
                _x1 = min(_canvas_w, int(reg.bbox[3]) * _ds)
                return _y0, _y1, _x0, _x1

            def _translated_moving_tile(
                gy0: int,
                gy1: int,
                gx0: int,
                gx1: int,
                dy_fr: float,
                dx_fr: float,
            ) -> np.ndarray:
                _pad_fr = int(max(4, math.ceil(max(abs(dy_fr), abs(dx_fr))) + 2))
                _my0 = max(0, int(gy0) - _pad_fr)
                _my1 = min(_canvas_h, int(gy1) + _pad_fr)
                _mx0 = max(0, int(gx0) - _pad_fr)
                _mx1 = min(_canvas_w, int(gx1) + _pad_fr)
                _mov_padded = _canvas_channel_roi(
                    ci.path, int(registration_channel), _moving_shape_yx,
                    _my0, _my1, _mx0, _mx1,
                )
                _loc_y0 = int(gy0) - _my0
                _loc_x0 = int(gx0) - _mx0
                _mov_crop = _extract_translated_crop(
                    _mov_padded,
                    (_loc_y0, _loc_y0 + (int(gy1) - int(gy0)), _loc_x0, _loc_x0 + (int(gx1) - int(gx0))),
                    dy_fr, dx_fr,
                )
                del _mov_padded
                return _mov_crop

            def _stream_island_corr(reg: _RegionTransform, dy_fr: float, dx_fr: float) -> float:
                _y0, _y1, _x0, _x1 = _island_bbox_fr(reg)
                _n = 0
                _sum_a = _sum_b = _sum_aa = _sum_bb = _sum_ab = 0.0
                for _gy0, _gy1, _gx0, _gx1 in _iter_tiles(_y0, _y1, _x0, _x1, overlap=False):
                    _fg = _upsample_ds_mask_to_local(
                        _ref_fg_ds_for_elastic, label=None,
                        y0_fr=_gy0, y1_fr=_gy1, x0_fr=_gx0, x1_fr=_gx1,
                        downsample=_ds,
                    )
                    if int(_fg.sum()) < 32:
                        continue
                    _ref = _canvas_channel_roi(ref_ci.path, int(ref_ch_idx), _ref_shape_yx, _gy0, _gy1, _gx0, _gx1)
                    _mov = _translated_moving_tile(_gy0, _gy1, _gx0, _gx1, dy_fr, dx_fr)
                    _use = _fg.astype(bool, copy=False)
                    _a = _ref[_use].astype(np.float64, copy=False)
                    _b = _mov[_use].astype(np.float64, copy=False)
                    _n += int(_a.size)
                    _sum_a += float(_a.sum())
                    _sum_b += float(_b.sum())
                    _sum_aa += float(np.sum(_a * _a))
                    _sum_bb += float(np.sum(_b * _b))
                    _sum_ab += float(np.sum(_a * _b))
                    del _ref, _mov, _fg, _a, _b
                if _n < 32:
                    return -1.0
                _da = _sum_aa - (_sum_a * _sum_a / float(_n))
                _db = _sum_bb - (_sum_b * _sum_b / float(_n))
                if _da <= 1e-8 or _db <= 1e-8:
                    return -1.0
                return float((_sum_ab - (_sum_a * _sum_b / float(_n))) / math.sqrt(_da * _db))

            def _tile_tent(h: int, w: int) -> np.ndarray:
                _tent_y = np.minimum(
                    np.arange(int(h), dtype=np.float32) + 1,
                    np.arange(int(h) - 1, -1, -1, dtype=np.float32) + 1,
                ).clip(1e-6)
                _tent_x = np.minimum(
                    np.arange(int(w), dtype=np.float32) + 1,
                    np.arange(int(w) - 1, -1, -1, dtype=np.float32) + 1,
                ).clip(1e-6)
                return (_tent_y[:, np.newaxis] * _tent_x[np.newaxis, :]).astype(np.float32)

            if _debug_elastic_field:
                print(
                    f"[elastic] cycle {cycle}: island-scoped elastic touch-up  "
                    f"islands={len(regs_ds)} tile={_et_tile_h}x{_et_tile_w}  "
                    f"stride={_stride_h}x{_stride_w} workers={_n_elastic_workers}  "
                    f"accumulator={_acc_path}",
                    flush=True,
                )

            _progress(
                f"7. Registering Cycle {cycle}: elastic touch-up ({len(regs_ds)} island(s))...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="elastic_touchup_island",
                idx=0,
                n=max(1, len(regs_ds)),
                cycle=cycle,
            )

            def _run_elastic_tile(
                gy0: int,
                gy1: int,
                gx0: int,
                gx1: int,
                dy_fr: float,
                dx_fr: float,
                island_label: int,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int] | None:
                _check_cancel(cancel_cb)
                _th, _tw = int(gy1) - int(gy0), int(gx1) - int(gx0)
                _island_tile = _upsample_ds_mask_to_local(
                    islands_ds, label=int(island_label),
                    y0_fr=gy0, y1_fr=gy1, x0_fr=gx0, x1_fr=gx1,
                    downsample=_ds,
                )
                if int(_island_tile.sum()) < _min_fg_px:
                    return None
                _ref_crop = _canvas_channel_roi(ref_ci.path, int(ref_ch_idx), _ref_shape_yx, gy0, gy1, gx0, gx1)
                _mov_crop = _translated_moving_tile(gy0, gy1, gx0, gx1, dy_fr, dx_fr)
                _tile_corr = _masked_corr_score(_ref_crop, _mov_crop, _island_tile)
                if _tile_corr >= float(elastic_touchup_skip_corr):
                    return None
                _res = _run_elastix_bspline(
                    _ref_crop, _mov_crop, _island_tile,
                    grid_spacing_px=int(elastic_touchup_bspline_spacing),
                    max_iterations=int(elastic_touchup_max_iterations),
                    max_step_length=float(elastic_touchup_max_step_length),
                )
                if _res is None:
                    return None
                _edy, _edx = _res
                _tent = _tile_tent(_th, _tw)
                _tent *= _island_tile.astype(np.float32, copy=False)
                return _edy * _tent, _edx * _tent, _tent, int(gy0), int(gy1), int(gx0), int(gx1)

            _n_islands_done = 0
            _n_tiles_ok = 0
            _n_tiles_tried = 0
            for _ereg in regs_ds:
                _check_cancel(cancel_cb)
                _dy_fr = float(_ereg.shift_y) * float(_ds)
                _dx_fr = float(_ereg.shift_x) * float(_ds)
                if not np.isfinite(_dy_fr) or not np.isfinite(_dx_fr):
                    _dy_fr, _dx_fr = _base_dy_fr, _base_dx_fr
                _corr = _stream_island_corr(_ereg, _dy_fr, _dx_fr)
                _iy0, _iy1, _ix0, _ix1 = _island_bbox_fr(_ereg)
                if _debug_elastic_field:
                    _verdict = "SKIP corr>threshold" if _corr > float(elastic_touchup_skip_corr) else "processing"
                    print(
                        f"[elastic]  island {_n_islands_done + 1}/{len(regs_ds)} "
                        f"label={int(_ereg.label)} bbox=({_iy0},{_iy1},{_ix0},{_ix1}) "
                        f"corr={_corr:.4f} shift=({_dy_fr:.1f},{_dx_fr:.1f}) -> {_verdict}",
                        flush=True,
                    )
                if _corr > float(elastic_touchup_skip_corr):
                    _n_islands_done += 1
                    continue
                _tiles = list(_iter_tiles(_iy0, _iy1, _ix0, _ix1, overlap=True))
                _n_tiles_tried += len(_tiles)
                _futs: list[Future] = []
                _pool = ThreadPoolExecutor(max_workers=_n_elastic_workers)
                try:
                    _futs = [
                        _pool.submit(_run_elastic_tile, _gy0, _gy1, _gx0, _gx1, _dy_fr, _dx_fr, int(_ereg.label))
                        for _gy0, _gy1, _gx0, _gx1 in _tiles
                    ]
                    for _fut in _iter_completed_futures(_futs, cancel_cb):
                        _check_cancel(cancel_cb)
                        _res = _fut.result()
                        if _res is None:
                            continue
                        _edy_w, _edx_w, _wt, _gy0, _gy1, _gx0, _gx1 = _res
                        _dy_sum[_gy0:_gy1, _gx0:_gx1] = np.asarray(_dy_sum[_gy0:_gy1, _gx0:_gx1]) + _edy_w
                        _dx_sum[_gy0:_gy1, _gx0:_gx1] = np.asarray(_dx_sum[_gy0:_gy1, _gx0:_gx1]) + _edx_w
                        _wt_sum[_gy0:_gy1, _gx0:_gx1] = np.asarray(_wt_sum[_gy0:_gy1, _gx0:_gx1]) + _wt
                        _n_tiles_ok += 1
                        del _edy_w, _edx_w, _wt
                except BaseException:
                    _cancel_futures(_futs)
                    _pool.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    _pool.shutdown(wait=True, cancel_futures=False)
                _n_islands_done += 1
                if progress_event_cb is not None:
                    progress_event_cb({
                        "msg": (
                            f"7. elastic touch-up Cycle {cycle}: "
                            f"island {_n_islands_done}/{len(regs_ds)}  tiles_ok={_n_tiles_ok}"
                        ),
                        "phase": "elastic_touchup_island",
                    })

            if _debug_elastic_field:
                print(
                    f"[elastic] cycle {cycle}: {_n_tiles_ok}/{_n_tiles_tried} tile(s) "
                    f"produced island-scoped corrections",
                    flush=True,
                )

            def _provider(out_y0: int, out_y1: int) -> tuple[np.ndarray, np.ndarray] | None:
                _out_y0, _out_y1 = int(out_y0), int(out_y1)
                if _out_y1 <= _out_y0:
                    return None
                _wt_sl = np.asarray(_wt_sum[_out_y0:_out_y1, :], dtype=np.float32)
                _nz = _wt_sl > 0
                if not _nz.any():
                    return None
                _dy = np.asarray(_dy_sum[_out_y0:_out_y1, :], dtype=np.float32)
                _dx = np.asarray(_dx_sum[_out_y0:_out_y1, :], dtype=np.float32)
                _dy[_nz] /= _wt_sl[_nz]
                _dx[_nz] /= _wt_sl[_nz]
                return _dy, _dx

            setattr(_provider, "cleanup", _cleanup)
            return _provider

        def _write_cycle_channels(
            ci: CycleInput,
            shp: tuple[int, int, int],
            field_yx: tuple[np.ndarray, np.ndarray] | None,
            target_writer=None,
        ) -> None:
            _check_cancel(cancel_cb)
            cycle = int(ci.cycle)
            out_c0 = int(channel_offsets[cycle])
            n_ch = int(shp[2])
            _fy, _fx = field_yx if field_yx is not None else (None, None)
            _w = target_writer if target_writer is not None else writer

            def _compute_ch(ch: int) -> tuple[int, np.ndarray]:
                _check_cancel(cancel_cb)
                _plane = load_single_channel_tiff_native(ci.path, int(ch))
                _check_cancel(cancel_cb)
                _plane = _pad_plane_to_canvas(_plane, canvas_yx, out_dtype=np.float32)
                if _fy is not None:
                    _plane = _warp_plane_by_field(_plane, _fy, _fx, order=1)
                _check_cancel(cancel_cb)
                return int(ch), _cast_preserve_dtype(_plane, base_dtype)

            # map_coordinates (the warp) releases the GIL, so threads give real
            # parallelism here.
            _n_workers = min(n_ch, step1_workers)
            _pool = ThreadPoolExecutor(max_workers=_n_workers)
            _futures: list[Future] = []
            try:
                _futures = [_pool.submit(_compute_ch, ch) for ch in range(n_ch)]
                for _fut in _iter_completed_futures(_futures, cancel_cb):
                    ch, _out = _fut.result()
                    _check_cancel(cancel_cb)
                    _w.write_channel(out_c0 + ch, _out)
                    _check_cancel(cancel_cb)
            except BaseException:
                _cancel_futures(_futures)
                _pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                _pool.shutdown(wait=True, cancel_futures=False)

        def _write_cycle_channels_strip(
            ci: CycleInput,
            shp: tuple[int, int, int],
            regs_fullres: list[_RegionTransform] | None,
            base_dy: float,
            base_dx: float,
            elastic_corrections: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] | None = None,
            elastic_correction_provider: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None] | None = None,
            target_writer=None,
            islands_ds: np.ndarray | None = None,
            downsample: int = 1,
        ) -> None:
            """Strip-based writer: processes the canvas in horizontal bands to minimise RAM."""
            _check_cancel(cancel_cb)
            _w = target_writer if target_writer is not None else writer
            cycle = int(ci.cycle)
            out_c0 = int(channel_offsets[cycle])
            n_ch = int(shp[2])
            source_H, source_W = int(shp[0]), int(shp[1])
            canvas_H, canvas_W = int(canvas_yx[0]), int(canvas_yx[1])

            # Canvas placement (same logic as _pad_plane_to_canvas)
            canvas_pad_y0 = max((canvas_H - source_H) // 2, 0)
            canvas_pad_x0 = max((canvas_W - source_W) // 2, 0)
            in_y0 = max((source_H - canvas_H) // 2, 0)
            in_x0 = max((source_W - canvas_W) // 2, 0)
            ys = min(source_H - in_y0, canvas_H - canvas_pad_y0)
            xs = min(source_W - in_x0, canvas_W - canvas_pad_x0)

            # Shift padding: ensure we load enough source rows for the largest shift
            shift_pad = 4
            if regs_fullres:
                shift_pad = max(shift_pad,
                    int(np.ceil(max(max(abs(r.shift_y), abs(r.shift_x)) for r in regs_fullres))) + 4)
            shift_pad = max(shift_pad, int(np.ceil(max(abs(base_dy), abs(base_dx)))) + 4)

            sh = max(1, int(_strip_h))
            _strip_starts = list(range(0, canvas_H, sh))
            # Merge any trailing strip shorter than sh//2 into the preceding strip
            # so no strip is a tiny sliver that gives elastix too little context.
            if len(_strip_starts) > 1 and (canvas_H - _strip_starts[-1]) < sh // 2:
                _strip_starts.pop()
            _dbg_n_strips = len(_strip_starts) if _debug_preprocess else 0
            if _debug_preprocess:
                print(
                    f"[preprocess] writing cycle {cycle} "
                    f"({'strip' if regs_fullres is not None else 'passthrough'} mode): "
                    f"{n_ch} channel(s) × {_dbg_n_strips} strip(s) of ≤{sh} rows",
                    flush=True,
                )
            for out_y0 in _strip_starts:
                _check_cancel(cancel_cb)
                out_y1 = canvas_H if out_y0 == _strip_starts[-1] else min(canvas_H, out_y0 + sh)
                strip_H = out_y1 - out_y0
                if _debug_preprocess:
                    _strip_idx = _strip_starts.index(out_y0) + 1
                    _pct = 100 * out_y0 // canvas_H
                    _rss = ""
                    try:
                        import psutil as _ps
                        _rss = f"  RAM={_ps.Process().memory_info().rss / 1024**2:.0f} MB"
                    except Exception:
                        pass
                    print(
                        f"[preprocess]   strip {_strip_idx}/{_dbg_n_strips}"
                        f"  rows [{out_y0}:{out_y1}]  ({_pct}%){_rss}",
                        flush=True,
                    )

                _island_lbl_strip: np.ndarray | None = None
                if islands_ds is not None and regs_fullres is not None:
                    _ds = int(downsample) or 1
                    _ds_y0 = out_y0 // _ds
                    _ds_y1 = min(islands_ds.shape[0], (out_y1 + _ds - 1) // _ds)
                    _istrip_ds = islands_ds[_ds_y0:_ds_y1]
                    _istrip_full = np.repeat(np.repeat(_istrip_ds, _ds, axis=0), _ds, axis=1)
                    _offs = out_y0 - _ds_y0 * _ds
                    _island_lbl_strip = np.zeros((strip_H, canvas_W), dtype=islands_ds.dtype)
                    _valid_h = min(strip_H, _istrip_full.shape[0] - _offs)
                    _valid_w = min(canvas_W, _istrip_full.shape[1])
                    if _valid_h > 0 and _valid_w > 0:
                        _island_lbl_strip[:_valid_h, :_valid_w] = _istrip_full[_offs:_offs + _valid_h, :_valid_w]

                if regs_fullres is not None:
                    field_y_s, field_x_s = _build_strip_shift_field(
                        out_y0, out_y1, canvas_W, regs_fullres, base_dy, base_dx,
                        island_labels_strip=_island_lbl_strip,
                    )
                else:
                    field_y_s = field_x_s = None

                if elastic_corrections and field_y_s is not None:
                    for _ec_dy, _ec_dx, (_ey0, _ey1, _ex0, _ex1) in elastic_corrections:
                        _ov_y0 = max(out_y0, _ey0)
                        _ov_y1 = min(out_y1, _ey1)
                        if _ov_y0 >= _ov_y1:
                            continue
                        _ex0_c = max(0, _ex0)
                        _ex1_c = min(canvas_W, _ex1)
                        if _ex0_c >= _ex1_c:
                            continue
                        _sl_y0 = _ov_y0 - out_y0
                        _sl_y1 = _ov_y1 - out_y0
                        _cl_y0 = _ov_y0 - _ey0
                        _cl_y1 = _ov_y1 - _ey0
                        _cl_x0 = _ex0_c - _ex0
                        _cl_x1 = _ex1_c - _ex0
                        field_y_s[_sl_y0:_sl_y1, _ex0_c:_ex1_c] += _ec_dy[_cl_y0:_cl_y1, _cl_x0:_cl_x1]
                        field_x_s[_sl_y0:_sl_y1, _ex0_c:_ex1_c] += _ec_dx[_cl_y0:_cl_y1, _cl_x0:_cl_x1]

                if elastic_correction_provider is not None and field_y_s is not None:
                    _strip_corr = elastic_correction_provider(out_y0, out_y1)
                    if _strip_corr is not None:
                        _edy_s, _edx_s = _strip_corr
                        if _edy_s.shape != field_y_s.shape or _edx_s.shape != field_x_s.shape:
                            raise ValueError(
                                "Elastic strip correction shape mismatch. "
                                f"Expected {field_y_s.shape}, got {_edy_s.shape}/{_edx_s.shape}"
                            )
                        field_y_s += _edy_s
                        field_x_s += _edx_s
                        del _edy_s, _edx_s

                # Source canvas rows needed (with shift padding)
                src_cy0 = max(0, out_y0 - shift_pad)
                src_cy1 = min(canvas_H, out_y1 + shift_pad)
                src_H = src_cy1 - src_cy0

                # Source file rows that overlap with canvas rows [src_cy0:src_cy1]
                ovlp_cy0 = max(src_cy0, canvas_pad_y0)
                ovlp_cy1 = min(src_cy1, canvas_pad_y0 + ys)

                # Process all channels for this strip in parallel.
                # load_channel_strip uses read-only memmaps (thread-safe); the
                # warp (map_coordinates) releases the GIL for true parallelism.
                # Strips are small so we allow up to cpu_count workers.
                _fys, _fxs = field_y_s, field_x_s
                _oy0, _sH, _scy0 = out_y0, strip_H, src_cy0
                _ocy0, _ocy1 = ovlp_cy0, ovlp_cy1
                _sH_src = src_H

                def _compute_strip_ch(ch: int) -> tuple[int, np.ndarray]:
                    _check_cancel(cancel_cb)
                    _src = np.zeros((_sH_src, canvas_W), dtype=np.float32)
                    if _ocy0 < _ocy1:
                        _fy0 = in_y0 + (_ocy0 - canvas_pad_y0)
                        _fy1 = in_y0 + (_ocy1 - canvas_pad_y0)
                        _raw = load_channel_strip(ci.path, ch, _fy0, _fy1)
                        _check_cancel(cancel_cb)
                        _dy0, _dy1 = _ocy0 - _scy0, _ocy1 - _scy0
                        _src[_dy0:_dy1, canvas_pad_x0:canvas_pad_x0 + xs] = (
                            _raw[:, in_x0:in_x0 + xs].astype(np.float32, copy=False)
                        )
                    if _fys is None:
                        _ry0, _ry1 = _oy0 - _scy0, (_oy0 + _sH) - _scy0
                        _check_cancel(cancel_cb)
                        return ch, _cast_preserve_dtype(_src[_ry0:_ry1], base_dtype)
                    _w = _warp_strip_by_field(_src, _oy0, _sH, _scy0, _fys, _fxs, order=1)
                    _check_cancel(cancel_cb)
                    return ch, _cast_preserve_dtype(_w, base_dtype)

                _n_workers = min(n_ch, step1_workers)
                _pool = ThreadPoolExecutor(max_workers=_n_workers)
                _strip_futs: list[Future] = []
                try:
                    _strip_futs = [_pool.submit(_compute_strip_ch, ch) for ch in range(n_ch)]
                    for _sfut in _iter_completed_futures(_strip_futs, cancel_cb):
                        _sch, _out = _sfut.result()
                        _check_cancel(cancel_cb)
                        _w.write_channel_strip(out_c0 + _sch, _out, out_y0)
                        _check_cancel(cancel_cb)
                except BaseException:
                    _cancel_futures(_strip_futs)
                    _pool.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    _pool.shutdown(wait=True, cancel_futures=False)

        ref_info = next((item for item in infos if int(item[0].cycle) == int(reference_cycle)), None)
        if ref_info is None:
            raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")
        ref_ci_write, ref_shp, _ref_dt = ref_info
        if int(reference_cycle) in completed_cycle_set:
            _progress(
                f"7. Skipping already-complete Cycle {int(reference_cycle)}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="write_cycle",
                idx=tick,
                n=total_ticks,
                cycle=int(reference_cycle),
            )
        else:
            _progress(
                f"7. Writing registered channels for Cycle {int(reference_cycle)}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="write_cycle",
                idx=tick,
                n=total_ticks,
                cycle=int(reference_cycle),
            )
            _set_cycle_status(int(reference_cycle), "in_progress")
            if _strip_mode:
                _write_cycle_channels_strip(ref_ci_write, ref_shp, None, 0.0, 0.0)
                if debug_writer is not None:
                    _write_cycle_channels_strip(ref_ci_write, ref_shp, None, 0.0, 0.0, target_writer=debug_writer)
            else:
                _write_cycle_channels(ref_ci_write, ref_shp, None)
                if debug_writer is not None:
                    _write_cycle_channels(ref_ci_write, ref_shp, None, target_writer=debug_writer)
            writer.flush_and_release()
            if debug_writer is not None:
                debug_writer.flush_and_release()
            _set_cycle_status(int(reference_cycle), "complete")
            if _debug_preprocess:
                _rss_ref = ""
                try:
                    import psutil as _ps
                    _rss_ref = f"  RAM={_ps.Process().memory_info().rss / 1024**2:.0f} MB"
                except Exception:
                    pass
                print(f"[preprocess] reference cycle written + released{_rss_ref}", flush=True)
        tick += 1

        # In strip mode, elastic touch-up loads per-island crops instead of full-canvas
        # arrays.  Pre-compute the reference foreground mask at downsampled resolution
        # once here (trivial from ref_reg already in memory) and reuse across all cycles.
        _ref_fg_ds_for_elastic: np.ndarray | None = None
        if _strip_mode and elastic_touchup:
            _ref_fg_ds_for_elastic = _foreground_mask(ref_reg)

        for _i_moving, (ci, shp, _dt) in enumerate(moving_infos):
            _check_cancel(cancel_cb)
            cycle = int(ci.cycle)
            if cycle in completed_cycle_set:
                _progress(
                    f"2-7. Skipping already-complete Cycle {cycle}...",
                    progress_cb=progress_cb,
                    progress_event_cb=progress_event_cb,
                    phase="write_cycle",
                    idx=tick,
                    n=total_ticks,
                    cycle=cycle,
                )
                tick += 5
                continue
            marker = ci.registration_marker or default_registration_marker
            ch_idx = _resolve_reg_channel(ci, marker)
            if ch_idx is None:
                raise ValueError(f"Could not find registration marker {marker!r} in cycle {cycle}")

            _progress(
                f"2. Loading Cycle {cycle}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="load_cycle",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _strip_mode:
                _mov_ds = load_channel_downsampled(ci.path, int(ch_idx), D)
                _mov_yx_ds = _pad_plane_to_canvas(_mov_ds.astype(np.float32), _ds_canvas_yx, out_dtype=np.float32)
                mov_reg = _normalized_for_registration(_mov_yx_ds)
                del _mov_ds, _mov_yx_ds
            else:
                mov_native = load_single_channel_tiff_native(ci.path, int(ch_idx))
                mov_yx = _pad_plane_to_canvas(mov_native, canvas_yx, out_dtype=np.float32)
                mov_reg = _normalized_for_registration(mov_yx)
                del mov_native, mov_yx
            if _debug_preprocess:
                _reg_shape = tuple(int(v) for v in mov_reg.shape)
                _reg_mode = f"downsampled 1/{D}" if _strip_mode else "full-resolution"
                print(
                    f"[preprocess] cycle {cycle}: reg channel loaded ({_reg_mode})  "
                    f"shape={_reg_shape}",
                    flush=True,
                )
            tick += 1

            _progress(
                f"3. Registering Cycle {cycle}: calculating global translation...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="global_registration",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _strip_mode:
                # ref_reg and mov_reg are already at 1/D scale; estimate at that scale then scale up
                _dy_ds, _dx_ds = estimate_translation(ref_reg, mov_reg, downsample=1, upsample_factor=upsample_factor)
                dy = _dy_ds * float(D)
                dx = _dx_ds * float(D)
            else:
                dy, dx = estimate_translation(ref_reg, mov_reg, downsample=max(1, int(downsample_for_registration)), upsample_factor=upsample_factor)
            shifts[cycle] = (float(dy), float(dx))
            if _debug_preprocess:
                print(
                    f"[preprocess] cycle {cycle}: global shift dy={dy:.2f} dx={dx:.2f}",
                    flush=True,
                )
            tick += 1

            _use_islands = not (bool(global_translation_only) or
                                str(registration_algorithm or "").strip().lower() in {
                                    "translation", "phase_correlation", "phase-correlation", "phase correlation"
                                })

            _progress(
                f"4. Registering Cycle {cycle}: generating foreground mask...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="foreground_mask",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _use_islands:
                moving_fg = _foreground_mask(mov_reg)
                if _strip_mode and D > 1 and binary_dilation is not None and moving_fg.any():
                    # At 1/D resolution, sparse tissue bridges (few cells per tile) can fall
                    # below the tile-activity threshold and appear split. Applying D-1 extra
                    # iterations brings the total to D, equivalent to 1 iteration at full scale.
                    moving_fg = binary_dilation(moving_fg, iterations=D - 1)
                _shift_for_mask = (_dy_ds, _dx_ds) if _strip_mode else (dy, dx)
                moved_mask = _apply_translation(moving_fg.astype(np.float32), _shift_for_mask[0], _shift_for_mask[1], order=0) > 0.5
                del moving_fg
            else:
                moved_mask = None

            _progress(
                f"5. Registering Cycle {cycle}: finding foreground islands...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="identify_islands",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _use_islands and moved_mask is not None:
                _tile_sz = max(4, int(tiled_rigid_tile_size) // D) if _strip_mode else max(100, int(tiled_rigid_tile_size))
                islands = _identify_foreground_islands(moved_mask, _tile_sz, solid=True)
                del moved_mask
                n_islands = int(np.max(islands))
            else:
                islands = None
                n_islands = 0
                if moved_mask is not None:
                    del moved_mask
            island_counts[cycle] = n_islands
            tick += 1

            _progress(
                f"6. Registering Cycle {cycle}: refining foreground islands...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="foreground_island_refine",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _use_islands and islands is not None:
                _base_shift_for_refine = (_dy_ds, _dx_ds) if _strip_mode else (dy, dx)
                _search_rad = max(2, region_search_radius // D) if _strip_mode else region_search_radius
                _ds_for_refine = 1 if _strip_mode else max(1, int(downsample_for_registration))
                regs_raw = _refine_region_transforms(
                    ref_reg,
                    mov_reg,
                    islands,
                    _base_shift_for_refine,
                    search_radius=_search_rad,
                    downsample=_ds_for_refine,
                    penalty_lambda=0.0,
                    fast_large_island_refinement=bool(fast_large_island_refinement),
                    fast_large_island_sample_count=max(1, int(fast_large_island_sample_count)),
                    fast_large_island_cell_size=max(4, int(tiled_rigid_tile_size) // D) if _strip_mode else max(128, int(tiled_rigid_tile_size)),
                    progress_cb=progress_cb,
                    progress_event_cb=progress_event_cb,
                    cancel_cb=cancel_cb,
                    cycle=cycle,
                )
                regs_raw = list(regs_raw)
                if _strip_mode:
                    regs = [_scale_region_transform(r, D) for r in regs_raw]
                else:
                    regs = regs_raw
            else:
                regs = []
            cycle_region_shift_summaries[cycle] = [
                {
                    "label": int(r.label),
                    "shift_y": float(r.shift_y),
                    "shift_x": float(r.shift_x),
                    "pixels": int(r.pixels),
                    "bbox": tuple(int(v) for v in r.bbox),
                }
                for r in regs
            ]
            if _debug_preprocess:
                print(
                    f"[preprocess] cycle {cycle}: {len(regs)} region transform(s) "
                    f"({'downsampled+scaled' if _strip_mode else 'full-res'})",
                    flush=True,
                )
            tick += 1

            # Stage 7: elastic touch-up
            _elastic_corrections: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = []
            _strip_elastic_provider: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None] | None = None
            if elastic_touchup and _use_islands and islands is not None:
                _progress(
                    f"7. Registering Cycle {cycle}: elastic touch-up...",
                    progress_cb=progress_cb,
                    progress_event_cb=progress_event_cb,
                    phase="elastic_touchup",
                    idx=tick,
                    n=total_ticks,
                    cycle=cycle,
                )
                if _strip_mode:
                    # In strip mode, each island is loaded at full extent from disk
                    # (no strip clipping) and registered exactly once here.  The
                    # provider closure is then cheap — it just slices from the
                    # cached correction arrays during the write step.
                    mov_reg = None
                    gc.collect()
                    _strip_elastic_provider = _make_strip_elastic_correction_provider(
                        ci=ci,
                        registration_channel=int(ch_idx),
                        moving_shape_yx=(int(shp[0]), int(shp[1])),
                        regs_ds=list(regs_raw),
                        islands_ds=islands,
                        downsample=D,
                        cycle=cycle,
                        base_shift_ds=(_dy_ds, _dx_ds),
                    )
                else:
                    _ref_fg_fr = _ref_full_fr = _mov_full_fr = _islands_fr = None
                    _et_ref     = ref_reg
                    _et_mov     = mov_reg
                    _et_islands = islands
                    _et_fg      = _foreground_mask(ref_reg)
                    _et_regs    = regs
                    _et_large_px = int(elastic_touchup_large_island_px)

                    _n_elastic_workers = int(step1_workers)
                    if _debug_elastic_field:
                        print(
                            f"[elastic] cycle {cycle}: starting elastic touch-up on "
                            f"{len(_et_regs)} island(s)  "
                            f"skip_corr_thr={elastic_touchup_skip_corr}  "
                            f"grid_spacing={elastic_touchup_bspline_spacing}px  "
                            f"max_iter={elastic_touchup_max_iterations}  "
                            f"large_island_px={_et_large_px}  "
                            f"workers={_n_elastic_workers}",
                            flush=True,
                        )

                    _n_et_regs = len(_et_regs)
                    _et_pool = ThreadPoolExecutor(max_workers=_n_elastic_workers)
                    try:
                        for _ereg_i, _ereg in enumerate(_et_regs, start=1):
                            _check_cancel(cancel_cb)
                            _progress(
                                f"7. Registering Cycle {cycle}: elastic touch-up island {_ereg_i}/{_n_et_regs}...",
                                progress_cb=progress_cb,
                                progress_event_cb=progress_event_cb,
                                phase="elastic_touchup_island",
                                idx=_ereg_i,
                                n=_n_et_regs,
                                cycle=cycle,
                            )
                            _et_shift = (_ereg.shift_y, _ereg.shift_x)
                            if _debug_elastic_field:
                                print(
                                    f"[elastic]  island {_ereg_i}/{_n_et_regs} "
                                    f"label={_ereg.label} pixels={_ereg.pixels}",
                                    flush=True,
                                )
                            _et_ref_arg = _et_ref
                            _et_mov_arg = _et_mov
                            _et_isl_arg = _et_islands == _ereg.label
                            _et_fg_arg  = _et_fg
                            _eresult = _elastic_touchup_island(
                                _et_ref_arg, _et_mov_arg, _et_isl_arg,
                                _et_shift, _et_fg_arg,
                                tile_size=int(elastic_touchup_tile_size),
                                skip_corr_threshold=float(elastic_touchup_skip_corr),
                                min_fg_pixels=32,
                                grid_spacing_px=int(elastic_touchup_bspline_spacing),
                                max_iterations=int(elastic_touchup_max_iterations),
                                large_island_px=_et_large_px,
                                max_step_length=float(elastic_touchup_max_step_length),
                                executor=_et_pool,
                                cancel_cb=cancel_cb,
                                progress_event_cb=progress_event_cb,
                                island_idx=_ereg_i,
                                island_total=_n_et_regs,
                                cycle=cycle,
                            )
                            if _eresult is not None:
                                _elastic_corrections.append(_eresult)
                    except BaseException:
                        _et_pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    else:
                        _et_pool.shutdown(wait=True, cancel_futures=False)

                    if _debug_elastic_field:
                        print(
                            f"[elastic] cycle {cycle}: {len(_elastic_corrections)}/{len(_et_regs)} "
                            f"island(s) produced elastic corrections",
                            flush=True,
                        )

                    del _et_fg

            if _i_moving == len(moving_infos) - 1:
                del ref_reg

            _progress(
                f"8. Writing registered channels for Cycle {cycle}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="write_cycle",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            _set_cycle_status(cycle, "in_progress")
            if _strip_mode:
                del mov_reg
                # Pass islands_ds so _build_strip_shift_field applies shifts only to
                # actual island pixels instead of entire bounding boxes (prevents
                # bbox bleed that causes duplicate-content drift artifacts).
                _islands_to_pass = islands if (_use_islands and islands is not None) else None
                # Pass regs directly (even if empty) so global shift (dy, dx) is applied;
                # None is reserved for the reference cycle (no shift at all).
                try:
                    _write_cycle_channels_strip(ci, shp, regs, dy, dx,
                                                elastic_corrections=_elastic_corrections or None,
                                                elastic_correction_provider=_strip_elastic_provider,
                                                islands_ds=_islands_to_pass,
                                                downsample=D)
                    if debug_writer is not None:
                        _write_cycle_channels_strip(ci, shp, regs, dy, dx,
                                                    elastic_corrections=None,
                                                    target_writer=debug_writer,
                                                    islands_ds=_islands_to_pass,
                                                    downsample=D)
                finally:
                    if _strip_elastic_provider is not None:
                        _cleanup = getattr(_strip_elastic_provider, "cleanup", None)
                        if callable(_cleanup):
                            _cleanup()
                if islands is not None:
                    del islands
                del regs
            else:
                # Full-plane mode: compute dense shift field then write
                _label_img = islands if (_use_islands and islands is not None) else np.zeros((1, 1), dtype=np.int32)
                field_yx = _piecewise_shift_field(canvas_yx, _label_img, regs, (dy, dx))
                _fy, _fx = field_yx
                if debug_writer is not None:
                    _write_cycle_channels(ci, shp, field_yx, target_writer=debug_writer)
                for _edy, _edx, (_ey0, _ey1, _ex0, _ex1) in _elastic_corrections:
                    _fy[_ey0:_ey1, _ex0:_ex1] += _edy
                    _fx[_ey0:_ey1, _ex0:_ex1] += _edx
                if islands is not None:
                    del islands
                del regs, mov_reg
                _write_cycle_channels(ci, shp, field_yx)
                del field_yx
            del _elastic_corrections
            writer.flush_and_release()
            if debug_writer is not None:
                debug_writer.flush_and_release()
            _set_cycle_status(cycle, "complete")
            if _debug_preprocess:
                _rss_post = ""
                try:
                    import psutil as _ps
                    _rss_post = f"  RAM={_ps.Process().memory_info().rss / 1024**2:.0f} MB"
                except Exception:
                    pass
                print(f"[preprocess] cycle {cycle} written + released{_rss_post}", flush=True)
            try:
                gc.collect()
            except Exception:
                pass
            tick += 1
    try:
        del ref_reg
    except NameError:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    pyramid_output_path: str | None = None
    if pyramidal_output:
        _check_cancel(cancel_cb)
        pyramid_output_path = out_path
        root = Path(out_path)
        tmp_pyramid_path = str(root.with_name(f"{root.stem}.__pyramid_tmp__.ome.tiff"))

        def _pyr_progress(msg: str) -> None:
            _progress(
                f"8. Building pyramidal OME-TIFF: {msg}",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="pyramid",
                idx=tick,
                n=total_ticks,
            )

        if use_zarr_registration_store:
            convert_registered_zarr_to_pyramidal(
                str(registered_zarr_path),
                tmp_pyramid_path,
                channel_names=merged_names,
                physical_pixel_sizes=ref_pixel_sizes,
                tile_size=int(pyramidal_tile_size),
                compression=pyramidal_compression,
                min_level_size=int(pyramidal_min_level_size),
                out_chunk=max(1, int(pyramid_progress_chunk)),
                progress_cb=_pyr_progress,
                cancel_cb=cancel_cb,
                build_workers=step1_workers,
            )
            os.replace(tmp_pyramid_path, out_path)
            try:
                shutil.rmtree(registered_zarr_path, ignore_errors=True)
            except Exception:
                pass
        else:
            convert_flat_ome_to_pyramidal(
                out_path,
                tmp_pyramid_path,
                tile_size=int(pyramidal_tile_size),
                compression=pyramidal_compression,
                min_level_size=int(pyramidal_min_level_size),
                out_chunk=max(1, int(pyramid_progress_chunk)),
                replace_source=True,
                progress_cb=_pyr_progress,
                build_workers=step1_workers,
            )
        tick += 1

    return {
        "output_path": out_path,
        "pyramidal_output_path": pyramid_output_path,
        "registration_work_store": str(registered_zarr_path) if use_zarr_registration_store else None,
        "reference_cycle": int(reference_cycle),
        "canvas_shape_yx": tuple(int(v) for v in canvas_yx),
        "n_cycles": int(len(cycles)),
        "n_channels_total": int(total_ch),
        "cycle_global_shifts": {int(k): (float(v[0]), float(v[1])) for k, v in shifts.items()},
        "cycle_island_counts": {int(k): int(v) for k, v in island_counts.items()},
        "cycle_region_shifts": cycle_region_shift_summaries,
        "input_pyramid_info": input_pyramid_info,
        "registration_algorithm": "global_translation_plus_foreground_island_refinement",
        "fast_large_island_refinement": bool(fast_large_island_refinement),
        "fast_large_island_sample_count": int(max(1, int(fast_large_island_sample_count))),
        "implemented_steps": [1, 2, 3, 4, 5],
        "pending_steps": [],
        "low_mem": bool(low_mem),
        "strip_height": int(_strip_h) if _strip_h is not None else None,
    }
