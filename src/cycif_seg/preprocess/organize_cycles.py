from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from concurrent.futures import ThreadPoolExecutor
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
    load_physical_pixel_sizes,
    load_single_channel_tiff_native,
    convert_flat_ome_to_pyramidal,
)


_STEP1_DEBUG = str(os.environ.get("CYCIF_SEG_STEP1_DEBUG", "0")).strip().lower() not in {
    "0", "false", "no", "off", ""
}


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
    mask: np.ndarray
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
    arr = np.asarray(img, dtype=np.float32)
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
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32, copy=False)


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
                    mask=mask,
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
    comps_a = _tile_component_masks(mask, tile_hw, (0, 0), solid=solid)
    comps_b = _tile_component_masks(mask, tile_hw, (half // 2, half // 2), solid=solid)
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
    moving_base_n: np.ndarray,
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
        moving_base_crop = moving_base_n[y0:y1, x0:x1]
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
    fixed_n = _normalized_for_registration(fixed_yx)
    moving_n = _normalized_for_registration(moving_yx)
    moving_base_n = _apply_translation(moving_n, base_dy, base_dx, order=1)
    for i, lab in enumerate(labs, start=1):
        _check_cancel(cancel_cb)

        _progress(
            f"6. Refining position of individual foreground island {i}/{total}...",
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
                moving_base_n,
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
                        mask=region_mask,
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
        moving_base_crop = moving_base_n[y0:y1, x0:x1]
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
                mask=region_mask,
                bbox=bbox,
                shift_y=float(best_dy),
                shift_x=float(best_dx),
                pixels=pixels,
            )
        )
    return regions


def _dense_shift_field(
    shape_yx: tuple[int, int],
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
        label_map[reg.mask] = int(reg.label)
        field_y[reg.mask] = float(reg.shift_y)
        field_x[reg.mask] = float(reg.shift_x)
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
    regions: list[_RegionTransform],
    base_shift: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Like _dense_shift_field but skips EDT: background pixels get the global shift."""
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    field_y = np.full((Y, X), float(base_shift[0]), dtype=np.float32)
    field_x = np.full((Y, X), float(base_shift[1]), dtype=np.float32)
    for reg in regions:
        field_y[reg.mask] = float(reg.shift_y)
        field_x[reg.mask] = float(reg.shift_x)
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
    fast_large_island_refinement: bool = True,
    fast_large_island_sample_count: int = 5,
    upsample_factor: int = 10,
    low_mem: bool = False,
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
    ref_ci = next((ci for ci, _, _ in infos if int(ci.cycle) == int(reference_cycle)), None)
    if ref_ci is None:
        raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

    merged_names: list[str] = []
    seen: dict[str, int] = {}
    for ci, _, _ in infos:
        ch_names = load_channel_names_only(ci.path)
        for j, nm in enumerate(ch_names):
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (nm or "").strip() or "Channel"
            stem = f"{base}_cy{int(ci.cycle)}"
            k = int(seen.get(stem, 0))
            out_nm = f"{stem}_{k+1}" if k else stem
            seen[stem] = k + 1
            merged_names.append(out_nm)

    ref_names = load_channel_names_only(ref_ci.path)
    ref_marker = ref_ci.registration_marker or default_registration_marker
    ref_ch_idx = _find_channel_index(ref_names, ref_marker)
    if ref_ch_idx is None:
        raise ValueError(f"Could not find registration marker {ref_marker!r} in reference cycle {int(ref_ci.cycle)}")
    ref_plane_native = load_single_channel_tiff_native(ref_ci.path, int(ref_ch_idx))
    ref_yx = _pad_plane_to_canvas(ref_plane_native, canvas_yx, out_dtype=np.float32)
    ref_reg = _normalized_for_registration(ref_yx)
    ref_pixel_sizes = load_physical_pixel_sizes(ref_ci.path)

    shifts: dict[int, tuple[float, float]] = {int(ref_ci.cycle): (0.0, 0.0)}
    island_counts: dict[int, int] = {int(ref_ci.cycle): 1}
    region_transforms: dict[int, list[_RegionTransform]] = {int(ref_ci.cycle): []}
    dense_fields: dict[int, tuple[np.ndarray, np.ndarray]] = {
        int(ref_ci.cycle): (
            np.zeros(canvas_yx, dtype=np.float32),
            np.zeros(canvas_yx, dtype=np.float32),
        )
    }

    moving_infos = [item for item in infos if int(item[0].cycle) != int(reference_cycle)]
    total_ch = int(total_input_ch)
    total_ticks = 1 + len(moving_infos) * 4 + len(cycles) + (1 if pyramidal_output else 0)
    tick = 0

    _progress(
        f"1. Loading Cycle {int(ref_ci.cycle)}...",
        progress_cb=progress_cb,
        progress_event_cb=progress_event_cb,
        phase="load_ref",
        idx=tick,
        n=total_ticks,
        cycle=int(ref_ci.cycle),
    )
    tick += 1

    region_search_radius = max(8, int(round(max(1, tiled_rigid_tile_size) * max(1.0, float(tiled_rigid_search_factor)) / 4.0)))
    for ci, _shp, _dt in moving_infos:
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
        mov_native = load_single_channel_tiff_native(ci.path, int(ch_idx))
        mov_yx = _pad_plane_to_canvas(mov_native, canvas_yx, out_dtype=np.float32)
        mov_reg = _normalized_for_registration(mov_yx)
        tick += 1

        _progress(
            "3. Calculating initial transform...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="global_registration",
            idx=tick,
            n=total_ticks,
            cycle=cycle,
        )
        dy, dx = estimate_translation(ref_reg, mov_reg, downsample=max(1, int(downsample_for_registration)), upsample_factor=upsample_factor)
        shifts[cycle] = (float(dy), float(dx))
        tick += 1

        _progress(
            "4. Generating foreground mask...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="foreground_mask",
            idx=tick,
            n=total_ticks,
            cycle=cycle,
        )
        moving_fg = _foreground_mask(mov_reg)
        moved_mask = _apply_translation(moving_fg.astype(np.float32), dy, dx, order=0) > 0.5

        _progress(
            "5. Finding regions of connected foreground...",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="identify_islands",
            idx=tick,
            n=total_ticks,
            cycle=cycle,
        )
        islands = _identify_foreground_islands(moved_mask, max(100, int(tiled_rigid_tile_size)), solid=True)
        n_islands = int(np.max(islands))
        island_counts[cycle] = n_islands
        _progress(
            f"5. Finding regions of connected foreground... identified {n_islands} foreground island(s)",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="identify_islands",
            idx=tick,
            n=total_ticks,
            cycle=cycle,
        )
        tick += 1

        regs = _refine_region_transforms(
            ref_reg,
            mov_reg,
            islands,
            (dy, dx),
            search_radius=region_search_radius,
            downsample=max(1, int(downsample_for_registration)),
            penalty_lambda=0.0,
            fast_large_island_refinement=bool(fast_large_island_refinement),
            fast_large_island_sample_count=max(1, int(fast_large_island_sample_count)),
            fast_large_island_cell_size=max(128, int(tiled_rigid_tile_size)),
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            cancel_cb=cancel_cb,
            cycle=cycle,
        )
        region_transforms[cycle] = list(regs)
        _progress(
            "6. Foreground island refinement complete.",
            progress_cb=progress_cb,
            progress_event_cb=progress_event_cb,
            phase="foreground_island_refine",
            idx=tick,
            n=total_ticks,
            cycle=cycle,
        )

        # Keep non-candidate pixels at the global shift; only selected foreground
        # island regions receive their accepted local shift.
        dense_fields[cycle] = _piecewise_shift_field(canvas_yx, region_transforms[cycle], (dy, dx))
        tick += 1

    out_path = str(output_path)
    with IncrementalOmeBigTiffWriter(
        out_path,
        (int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch)),
        base_dtype,
        merged_names,
        physical_pixel_sizes=ref_pixel_sizes,
    ) as writer:
        out_c = 0
        for ci, shp, _dt in infos:
            _check_cancel(cancel_cb)
            cycle = int(ci.cycle)
            field_y, field_x = dense_fields[cycle]
            n_ch = int(shp[2])
            for ch in range(n_ch):
                plane_native = load_single_channel_tiff_native(ci.path, int(ch))
                plane = _pad_plane_to_canvas(plane_native, canvas_yx, out_dtype=np.float32)
                if cycle == int(reference_cycle):
                    reg_plane = plane
                else:
                    reg_plane = _warp_plane_by_field(plane, field_y, field_x, order=1)
                writer.write_channel(out_c, _cast_preserve_dtype(reg_plane, base_dtype))
                out_c += 1
            _progress(
                f"9. Writing registered channels for Cycle {cycle}...",
                progress_cb=progress_cb,
                progress_event_cb=progress_event_cb,
                phase="write_cycle",
                idx=tick,
                n=total_ticks,
                cycle=cycle,
            )
            tick += 1
    pyramid_output_path: str | None = None
    if pyramidal_output:
        _check_cancel(cancel_cb)
        pyramid_output_path = out_path
        root = Path(out_path)
        tmp_pyramid_path = str(root.with_name(f"{root.stem}.__pyramid_tmp__.ome.tiff"))

        def _pyr_progress(msg: str) -> None:
            _progress(
                f"9. Building pyramidal OME-TIFF: {msg}",
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
        "cycle_region_shifts": {
            int(k): [
                {
                    "label": int(r.label),
                    "shift_y": float(r.shift_y),
                    "shift_x": float(r.shift_x),
                    "pixels": int(r.pixels),
                    "bbox": tuple(int(v) for v in r.bbox),
                }
                for r in regs
            ]
            for k, regs in region_transforms.items()
        },
        "input_pyramid_info": input_pyramid_info,
        "registration_algorithm": "global_translation_plus_foreground_island_refinement",
        "fast_large_island_refinement": bool(fast_large_island_refinement),
        "fast_large_island_sample_count": int(max(1, int(fast_large_island_sample_count))),
        "implemented_steps": [1, 2, 3, 4, 5],
        "pending_steps": [],
        "low_mem": bool(low_mem),
    }
