from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from concurrent.futures import ThreadPoolExecutor
import gc
import math
import os
from pathlib import Path

import numpy as np
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


def set_preprocess_debug(enabled: bool) -> None:
    """Enable or disable verbose console output for preprocessing operations."""
    global _debug_preprocess
    _debug_preprocess = bool(enabled)


def is_preprocess_debug() -> bool:
    return _debug_preprocess


def _dbg(msg: str) -> None:
    if _STEP1_DEBUG:
        print(f"[Step1 DEBUG] {msg}", flush=True)


@dataclass(frozen=True)
class CycleInput:
    path: str
    cycle: int
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
        # Canvas Y coordinates for the output rows, shifted by field → source canvas row
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
    pyramidal_output: bool = False,
    pyramidal_tile_size: int = 512,
    pyramidal_compression: str | int | None = "zlib",
    pyramidal_min_level_size: int = 128,
    pyramid_progress_chunk: int = 1024,
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
    seen: dict[str, int] = {}
    out_c = 0
    for ci, shp, _ in infos:
        channel_offsets[int(ci.cycle)] = out_c
        ch_names = load_channel_names_only(ci.path)
        n_ch = int(shp[2])
        for j in range(n_ch):
            nm = ch_names[j] if j < len(ch_names) else ""
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (nm or "").strip() or "Channel"
            stem = f"{base}_cy{int(ci.cycle)}"
            k = int(seen.get(stem, 0))
            out_nm = f"{stem}_{k+1}" if k else stem
            seen[stem] = k + 1
            merged_names.append(out_nm)
            out_c += 1

    ref_names = load_channel_names_only(ref_ci.path)
    ref_marker = ref_ci.registration_marker or default_registration_marker
    ref_ch_idx = _find_channel_index(ref_names, ref_marker)
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
    with IncrementalOmeBigTiffWriter(
        out_path,
        (int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch)),
        base_dtype,
        merged_names,
        physical_pixel_sizes=ref_pixel_sizes,
    ) as writer:
        def _write_cycle_channels(
            ci: CycleInput,
            shp: tuple[int, int, int],
            field_yx: tuple[np.ndarray, np.ndarray] | None,
        ) -> None:
            _check_cancel(cancel_cb)
            cycle = int(ci.cycle)
            out_c0 = int(channel_offsets[cycle])
            n_ch = int(shp[2])
            for ch in range(n_ch):
                plane_native = load_single_channel_tiff_native(ci.path, int(ch))
                plane = _pad_plane_to_canvas(plane_native, canvas_yx, out_dtype=np.float32)
                del plane_native
                if field_yx is None:
                    reg_plane = plane
                else:
                    field_y, field_x = field_yx
                    reg_plane = _warp_plane_by_field(plane, field_y, field_x, order=1)
                writer.write_channel(out_c0 + ch, _cast_preserve_dtype(reg_plane, base_dtype))
                del plane, reg_plane

        def _write_cycle_channels_strip(
            ci: CycleInput,
            shp: tuple[int, int, int],
            regs_fullres: list[_RegionTransform] | None,
            base_dy: float,
            base_dx: float,
        ) -> None:
            """Strip-based writer: processes the canvas in horizontal bands to minimise RAM."""
            _check_cancel(cancel_cb)
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

                # Source canvas rows needed (with shift padding)
                src_cy0 = max(0, out_y0 - shift_pad)
                src_cy1 = min(canvas_H, out_y1 + shift_pad)
                src_H = src_cy1 - src_cy0

                # Source file rows that overlap with canvas rows [src_cy0:src_cy1]
                ovlp_cy0 = max(src_cy0, canvas_pad_y0)
                ovlp_cy1 = min(src_cy1, canvas_pad_y0 + ys)

                for ch in range(n_ch):
                    _check_cancel(cancel_cb)
                    src_canvas = np.zeros((src_H, canvas_W), dtype=np.float32)

                    if ovlp_cy0 < ovlp_cy1:
                        file_y0 = in_y0 + (ovlp_cy0 - canvas_pad_y0)
                        file_y1 = in_y0 + (ovlp_cy1 - canvas_pad_y0)
                        raw = load_channel_strip(ci.path, ch, file_y0, file_y1)
                        dest_y0 = ovlp_cy0 - src_cy0
                        dest_y1 = ovlp_cy1 - src_cy0
                        src_canvas[dest_y0:dest_y1, canvas_pad_x0:canvas_pad_x0 + xs] = \
                            raw[:, in_x0:in_x0 + xs].astype(np.float32, copy=False)
                        del raw

                    if field_y_s is None:
                        # Reference cycle or global-translation-only: just slice and cast
                        rel_y0 = out_y0 - src_cy0
                        rel_y1 = out_y1 - src_cy0
                        out_strip = _cast_preserve_dtype(src_canvas[rel_y0:rel_y1], base_dtype)
                    else:
                        warped = _warp_strip_by_field(
                            src_canvas, out_y0, strip_H, src_cy0, field_y_s, field_x_s, order=1
                        )
                        out_strip = _cast_preserve_dtype(warped, base_dtype)
                        del warped

                    writer.write_channel_strip(out_c0 + ch, out_strip, out_y0)
                    del src_canvas, out_strip

        ref_info = next((item for item in infos if int(item[0].cycle) == int(reference_cycle)), None)
        if ref_info is None:
            raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")
        ref_ci_write, ref_shp, _ref_dt = ref_info
        _progress(
            f"7. Writing registered channels for Cycle {int(reference_cycle)}...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="write_cycle",
            idx=tick,
            n=total_ticks,
            cycle=int(reference_cycle),
        )
        if _strip_mode:
            _write_cycle_channels_strip(ref_ci_write, ref_shp, None, 0.0, 0.0)
        else:
            _write_cycle_channels(ref_ci_write, ref_shp, None)
        writer.flush_and_release()
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
            marker = ci.registration_marker or default_registration_marker
            names = load_channel_names_only(ci.path)
            ch_idx = _find_channel_index(names, marker)
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
            if _i_moving == len(moving_infos) - 1:
                del ref_reg
            tick += 1

            _progress(
                f"7. Writing registered channels for Cycle {cycle}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="write_cycle",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            if _strip_mode:
                # Strip mode: islands and mov_reg not needed for writing
                if islands is not None:
                    del islands
                del mov_reg
                # Pass regs directly (even if empty) so global shift (dy, dx) is applied;
                # None is reserved for the reference cycle (no shift at all).
                _write_cycle_channels_strip(ci, shp, regs, dy, dx)
                del regs
            else:
                # Full-plane mode: compute dense shift field then write
                _label_img = islands if (_use_islands and islands is not None) else np.zeros((1, 1), dtype=np.int32)
                field_yx = _piecewise_shift_field(canvas_yx, _label_img, regs, (dy, dx))
                if islands is not None:
                    del islands
                del regs, mov_reg
                _write_cycle_channels(ci, shp, field_yx)
                del field_yx
            writer.flush_and_release()
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
