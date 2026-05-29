from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import gc
import json
import math
import os
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
    estimate_pyramid_conversion_ticks,
    inspect_tiff_pyramid,
    inspect_tiff_yxc,
    load_channel_names_only,
    load_channel_downsampled,
    load_channel_strip,
    load_physical_pixel_sizes,
    load_single_channel_tiff_native,
    convert_flat_ome_to_pyramidal,
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
) -> tuple[float, float] | None:
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
        results = [_sample_tile_registration_worker(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(lambda task: _sample_tile_registration_worker(*task), tasks))

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

        if bool(fast_large_island_refinement):
            sampled_shift = _estimate_sampled_region_shift(
                fixed_n,
                moving_n,
                region_mask,
                (base_dy, base_dx),
                cell_size=max(128, int(fast_large_island_cell_size)),
                sample_count=max(1, int(fast_large_island_sample_count)),
                search_radius=int(search_radius),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (strip_H, canvas_x) shift field for output rows [out_y0:out_y1].

    Uses only the compact list of _RegionTransform objects — no full-canvas arrays.
    Background pixels (not covered by any region bbox) get the global shift.
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

    if force_from_cycle is not None:
        force = int(force_from_cycle)
        if force not in write_order:
            raise ValueError(f"--force-from-cycle {force} is not in expected write order: {write_order}")
        forced_complete = set(write_order[:write_order.index(force)])
        if forced_complete and not output_path.is_file():
            raise ValueError("--force-from-cycle cannot skip earlier cycles because the flat output file does not exist")
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
            if not output_path.is_file():
                messages.append("manifest listed complete cycles but flat output is missing; ignoring manifest completion")
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

    if not completed and output_path.is_file():
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
    elif not output_path.is_file():
        messages.append("no existing flat output found")

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


def _run_elastix_bspline(
    fixed_crop: np.ndarray,
    moving_crop: np.ndarray,
    island_crop: np.ndarray,
    *,
    grid_spacing_px: int,
    max_iterations: int,
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
                gs=np.array([gs_px],        dtype=np.int32),
                it=np.array([max_iterations], dtype=np.int32),
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


def _elastic_touchup_island(
    ref_reg: np.ndarray,
    mov_reg: np.ndarray,
    island_mask: np.ndarray,
    rigid_shift: tuple[float, float],
    fg_mask: np.ndarray,
    *,
    tile_size: int,
    skip_corr_threshold: float,
    min_fg_pixels: int,
    grid_spacing_px: int,
    max_iterations: int,
    large_island_px: int,
    executor=None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
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
        result = _run_elastix_bspline(
            ref_crop, mov_crop, island_crop,
            grid_spacing_px=grid_spacing_px, max_iterations=max_iterations,
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
    stride = max(1, int(tile_size) // 2)
    disp_y_acc = np.zeros((h, w), dtype=np.float32)
    disp_x_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    ty_starts = list(range(0, h, stride))
    tx_starts = list(range(0, w, stride))

    # Collect qualifying tiles first so we know count before submitting
    _tile_tasks: list[tuple] = []
    _tile_corrs: list[float] = []
    _n_tiles_island_px = 0
    _n_tiles_corr_skipped = 0
    for _ty0 in ty_starts:
        _ty1 = min(h, _ty0 + int(tile_size))
        _th = _ty1 - _ty0
        _tent_y = np.minimum(
            np.arange(_th, dtype=np.float32) + 1,
            np.arange(_th - 1, -1, -1, dtype=np.float32) + 1,
        ).clip(1e-6)
        for _tx0 in tx_starts:
            _tx1 = min(w, _tx0 + int(tile_size))
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
        _tres = _run_elastix_bspline(
            _t_ref, _t_mov, _t_isl,
            grid_spacing_px=grid_spacing_px, max_iterations=max_iterations,
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

    def _bar(_done, _total):
        _f = min(_BAR_W, _done * _BAR_W // _total) if _total else _BAR_W
        return ('=' * _f + ('>' if _f < _BAR_W else '') + ' ' * max(0, _BAR_W - _f - 1))

    if executor is not None and len(_tile_tasks) > 1:
        _tile_futs = [executor.submit(_run_tile, *_args) for _args in _tile_tasks]
        _n_done = 0
        for _fut in as_completed(_tile_futs):
            _accumulate(_fut.result())
            _n_done += 1
            if _debug_elastic_field:
                print(
                    f"\r[elastic]   [{_bar(_n_done, _n_tiles_tried)}]"
                    f" {_n_done}/{_n_tiles_tried}  ok={_n_tiles_ok}",
                    end='', flush=True,
                )
        if _debug_elastic_field:
            print(flush=True)
    else:
        _n_done = 0
        for _args in _tile_tasks:
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
    elastic_touchup_tile_size: int = 1024,
    elastic_touchup_skip_corr: float = 0.95,
    elastic_touchup_bspline_spacing: int = 50,
    elastic_touchup_max_iterations: int = 100,
    elastic_touchup_large_island_px: int = 4_000_000,
    elastic_touchup_workers: int = 0,
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
        _strip_h = max(1000, canvas_yx[0] // 10)
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
    open_existing_output = bool(resume_flat_output and Path(out_path).is_file())
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

            def _compute_ch(ch: int) -> np.ndarray:
                _plane = load_single_channel_tiff_native(ci.path, int(ch))
                _plane = _pad_plane_to_canvas(_plane, canvas_yx, out_dtype=np.float32)
                if _fy is not None:
                    _plane = _warp_plane_by_field(_plane, _fy, _fx, order=1)
                return _cast_preserve_dtype(_plane, base_dtype)

            # map_coordinates (the warp) releases the GIL, so threads give real
            # parallelism here. Cap at 4 to bound peak in-flight plane memory.
            _n_workers = min(n_ch, 4)
            with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
                _futures = [_pool.submit(_compute_ch, ch) for ch in range(n_ch)]
                for ch, _fut in enumerate(_futures):
                    _w.write_channel(out_c0 + ch, _fut.result())

        def _write_cycle_channels_strip(
            ci: CycleInput,
            shp: tuple[int, int, int],
            regs_fullres: list[_RegionTransform] | None,
            base_dy: float,
            base_dx: float,
            elastic_corrections: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] | None = None,
            target_writer=None,
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
            _dbg_n_strips = (canvas_H + sh - 1) // sh if _debug_preprocess else 0
            if _debug_preprocess:
                print(
                    f"[preprocess] writing cycle {cycle} "
                    f"({'strip' if regs_fullres is not None else 'passthrough'} mode): "
                    f"{n_ch} channel(s) × {_dbg_n_strips} strip(s) of ≤{sh} rows",
                    flush=True,
                )
            for out_y0 in range(0, canvas_H, sh):
                _check_cancel(cancel_cb)
                out_y1 = min(canvas_H, out_y0 + sh)
                strip_H = out_y1 - out_y0
                if _debug_preprocess:
                    _strip_idx = out_y0 // sh + 1
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

                if regs_fullres is not None:
                    field_y_s, field_x_s = _build_strip_shift_field(
                        out_y0, out_y1, canvas_W, regs_fullres, base_dy, base_dx
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
                    _src = np.zeros((_sH_src, canvas_W), dtype=np.float32)
                    if _ocy0 < _ocy1:
                        _fy0 = in_y0 + (_ocy0 - canvas_pad_y0)
                        _fy1 = in_y0 + (_ocy1 - canvas_pad_y0)
                        _raw = load_channel_strip(ci.path, ch, _fy0, _fy1)
                        _dy0, _dy1 = _ocy0 - _scy0, _ocy1 - _scy0
                        _src[_dy0:_dy1, canvas_pad_x0:canvas_pad_x0 + xs] = (
                            _raw[:, in_x0:in_x0 + xs].astype(np.float32, copy=False)
                        )
                    if _fys is None:
                        _ry0, _ry1 = _oy0 - _scy0, (_oy0 + _sH) - _scy0
                        return ch, _cast_preserve_dtype(_src[_ry0:_ry1], base_dtype)
                    _w = _warp_strip_by_field(_src, _oy0, _sH, _scy0, _fys, _fxs, order=1)
                    return ch, _cast_preserve_dtype(_w, base_dtype)

                _n_workers = min(n_ch, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
                    _strip_futs = [_pool.submit(_compute_strip_ch, ch) for ch in range(n_ch)]
                    for _sfut in _strip_futs:
                        _sch, _out = _sfut.result()
                        _w.write_channel_strip(out_c0 + _sch, _out, out_y0)

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
                    # ref_reg/mov_reg are downsampled (D=4); DAPI nuclei are ~2.5 px at
                    # that scale — too small for reliable registration.  Load the
                    # registration channel at full resolution from disk for this stage.
                    from scipy.ndimage import zoom as _sz  # type: ignore
                    _ref_full_fr = _pad_plane_to_canvas(
                        load_single_channel_tiff_native(ref_ci.path, int(ref_ch_idx)),
                        canvas_yx, out_dtype=np.float32,
                    )
                    _mov_full_fr = _pad_plane_to_canvas(
                        load_single_channel_tiff_native(ci.path, int(ch_idx)),
                        canvas_yx, out_dtype=np.float32,
                    )
                    # Nearest-neighbour upsample preserves integer island labels.
                    _islands_fr = _sz(islands, D, order=0).astype(islands.dtype)
                    _ref_fg_fr  = _foreground_mask(_ref_full_fr)
                    _et_ref     = _ref_full_fr
                    _et_mov     = _mov_full_fr
                    _et_islands = _islands_fr
                    _et_fg      = _ref_fg_fr
                    _et_regs    = regs_raw
                    # Island pixel counts are D² larger at full res; scale threshold.
                    _et_large_px = int(elastic_touchup_large_island_px) * D * D
                else:
                    _ref_fg_fr = _ref_full_fr = _mov_full_fr = _islands_fr = None
                    _et_ref     = ref_reg
                    _et_mov     = mov_reg
                    _et_islands = islands
                    _et_fg      = _foreground_mask(ref_reg)
                    _et_regs    = regs
                    _et_large_px = int(elastic_touchup_large_island_px)

                _n_elastic_workers = (
                    max(1, int(elastic_touchup_workers))
                    if elastic_touchup_workers and int(elastic_touchup_workers) > 0
                    else max(1, (os.cpu_count() or 2) - 1)
                )
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

                with ThreadPoolExecutor(max_workers=_n_elastic_workers) as _et_pool:
                    for _ereg_i, _ereg in enumerate(_et_regs, start=1):
                        _et_shift = (
                            _ereg.shift_y * float(D) if _strip_mode else _ereg.shift_y,
                            _ereg.shift_x * float(D) if _strip_mode else _ereg.shift_x,
                        )
                        if _debug_elastic_field:
                            print(
                                f"[elastic]  island {_ereg_i}/{len(_et_regs)} "
                                f"label={_ereg.label} pixels={_ereg.pixels}",
                                flush=True,
                            )
                        _eresult = _elastic_touchup_island(
                            _et_ref, _et_mov, _et_islands == _ereg.label,
                            _et_shift, _et_fg,
                            tile_size=int(elastic_touchup_tile_size),
                            skip_corr_threshold=float(elastic_touchup_skip_corr),
                            min_fg_pixels=32,
                            grid_spacing_px=int(elastic_touchup_bspline_spacing),
                            max_iterations=int(elastic_touchup_max_iterations),
                            large_island_px=_et_large_px,
                            executor=_et_pool,
                        )
                        if _eresult is not None:
                            _elastic_corrections.append(_eresult)

                if _debug_elastic_field:
                    print(
                        f"[elastic] cycle {cycle}: {len(_elastic_corrections)}/{len(_et_regs)} "
                        f"island(s) produced elastic corrections",
                        flush=True,
                    )

                if not _strip_mode:
                    del _et_fg
                if _strip_mode and _ref_full_fr is not None:
                    del _ref_full_fr, _mov_full_fr, _islands_fr, _ref_fg_fr

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
                # Strip mode: islands and mov_reg not needed for writing
                if islands is not None:
                    del islands
                del mov_reg
                # Pass regs directly (even if empty) so global shift (dy, dx) is applied;
                # None is reserved for the reference cycle (no shift at all).
                _write_cycle_channels_strip(ci, shp, regs, dy, dx,
                                            elastic_corrections=_elastic_corrections or None)
                if debug_writer is not None:
                    _write_cycle_channels_strip(ci, shp, regs, dy, dx,
                                                elastic_corrections=None,
                                                target_writer=debug_writer)
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

        convert_flat_ome_to_pyramidal(
            out_path,
            tmp_pyramid_path,
            tile_size=int(pyramidal_tile_size),
            compression=pyramidal_compression,
            min_level_size=int(pyramidal_min_level_size),
            out_chunk=max(1, int(pyramid_progress_chunk)),
            replace_source=True,
            progress_cb=_pyr_progress,
        )
        tick += 1

    return {
        "output_path": out_path,
        "pyramidal_output_path": pyramid_output_path,
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
