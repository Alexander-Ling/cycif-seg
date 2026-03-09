from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import math
import os

import numpy as np

from skimage.registration import phase_cross_correlation
from skimage.feature import match_template
from skimage.filters import threshold_otsu
from skimage.transform import rotate

try:
    import SimpleITK as sitk  # type: ignore
except Exception:  # pragma: no cover
    sitk = None  # type: ignore

try:
    # SciPy is faster than skimage.warp for pure translations.
    from scipy.ndimage import shift as ndi_shift  # type: ignore
except Exception:  # pragma: no cover
    ndi_shift = None

import tifffile

from cycif_seg.io.ome_tiff import (
    IncrementalOmeBigTiffWriter,
    estimate_pyramid_conversion_ticks,
    inspect_tiff_pyramid,
    inspect_tiff_yxc,
    load_channel_names_only,
    load_multichannel_tiff,
    load_multichannel_tiff_native,
    load_single_channel_tiff_native,
    save_ome_tiff_yxc,
    convert_flat_ome_to_pyramidal,
)


# Enable very fine-grained debug prints for isolating native crashes (e.g., in SimpleITK/ITK).
# Set env var CYCIF_SEG_STEP1_DEBUG=1 to turn on.
_STEP1_DEBUG = str(os.environ.get("CYCIF_SEG_STEP1_DEBUG", "0")).strip().lower() not in {"0", "false", "no", "off", ""}


def _dbg(msg: str) -> None:
    """Crash-debug prints (flush immediately)."""
    if _STEP1_DEBUG:
        print(f"[Step1 DEBUG] {msg}", flush=True)


@dataclass(frozen=True)
class CycleInput:
    """One imaging cycle file + metadata used for merging."""

    path: str
    cycle: int
    tissue: str | None = None
    species: str | None = None
    # Optional: the nuclear marker used to register this cycle.
    # If None, we'll attempt to use a marker named "DAPI" (case-insensitive).
    registration_marker: str | None = None

    # Optional: user-edited per-channel marker + antibody annotations.
    # If provided, each list should have length == n_channels in the file.
    channel_markers: list[str] | None = None
    channel_antibodies: list[str] | None = None


def _find_channel_index(ch_names: list[str], marker: str) -> int | None:
    m = (marker or "").strip().lower()
    if not m:
        return None
    for i, nm in enumerate(ch_names):
        if (nm or "").strip().lower() == m:
            return i
    # weak match: contains
    for i, nm in enumerate(ch_names):
        if m in (nm or "").strip().lower():
            return i
    return None


def _pad_to_shape(img_yx: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Zero-pad (or center-crop) a 2D image to shape_yx."""
    if img_yx.ndim != 2:
        raise ValueError(f"Expected 2D array. Got shape={img_yx.shape}")
    y, x = int(img_yx.shape[0]), int(img_yx.shape[1])
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    out = np.zeros((Y, X), dtype=np.float32)

    # center placement
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs] = img_yx[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs].astype(
        np.float32, copy=False
    )
    return out


def _pad_channels_to_canvas(img_yxc: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Zero-pad (or center-crop) a (Y,X,C) image to a common canvas (Yc,Xc,C)."""
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")
    y, x, c = img_yxc.shape
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    out = np.zeros((Y, X, c), dtype=np.float32)
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs, :] = img_yxc[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs, :].astype(
        np.float32, copy=False
    )
    return out


def _pad_plane_to_canvas(plane_yx: np.ndarray, canvas_yx: tuple[int, int], *, out_dtype=np.float32) -> np.ndarray:
    """Pad/crop a single (Y,X) plane to a common canvas."""
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
    out[y0 : y0 + ys, x0 : x0 + xs] = plane_yx[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs].astype(
        out_dtype, copy=False
    )
    return out


def _tiff_info_yxc(path: str) -> tuple[tuple[int, int, int], np.dtype]:
    """Return ((Y,X,C), dtype) without reading full pixel data."""
    info = inspect_tiff_yxc(path)
    return tuple(int(v) for v in info["shape_yxc"]), np.dtype(info["dtype"])


def _cast_preserve_dtype(plane_f32: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast float32 plane back to dtype while preserving range for integer types."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return plane_f32.astype(dt, copy=False)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return np.clip(plane_f32, info.min, info.max).astype(dt, copy=False)
    return plane_f32.astype(dt, copy=False)


def estimate_translation(
    ref_yx: np.ndarray,
    mov_yx: np.ndarray,
    *,
    downsample: int = 4,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """Return (dy, dx) translation to apply to mov to align to ref."""
    if ref_yx.ndim != 2 or mov_yx.ndim != 2:
        raise ValueError("estimate_translation expects 2D arrays")

    # phase_cross_correlation requires same shape. Pad/crop to a common canvas.
    Y = int(max(ref_yx.shape[0], mov_yx.shape[0]))
    X = int(max(ref_yx.shape[1], mov_yx.shape[1]))
    ref2 = _pad_to_shape(ref_yx, (Y, X))
    mov2 = _pad_to_shape(mov_yx, (Y, X))

    if downsample and downsample > 1:
        ref_ds = ref2[::downsample, ::downsample]
        mov_ds = mov2[::downsample, ::downsample]
    else:
        ref_ds, mov_ds = ref2, mov2

    # phase_cross_correlation returns shift to apply to mov to match ref
    shift_ds, _, _ = phase_cross_correlation(
        ref_ds,
        mov_ds,
        upsample_factor=int(upsample_factor),
        normalization=None,
    )
    dy = float(shift_ds[0]) * float(downsample)
    dx = float(shift_ds[1]) * float(downsample)
    return dy, dx


def apply_translation_yxc(
    img_yxc: np.ndarray,
    dy: float,
    dx: float,
    *,
    order: int = 1,
    preserve_range: bool = True,
) -> np.ndarray:
    """Translate (Y,X,C) image by (dy,dx)."""
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")
    out = np.empty_like(img_yxc, dtype=np.float32)

    if ndi_shift is not None:
        # SciPy: shift takes (dy, dx)
        for c in range(img_yxc.shape[2]):
            out[..., c] = ndi_shift(
                img_yxc[..., c],
                shift=(float(dy), float(dx)),
                order=int(order),
                mode="constant",
                cval=0.0,
                prefilter=(int(order) > 1),
            ).astype(np.float32, copy=False)

    return out

    # Fallback: skimage
    from skimage.transform import AffineTransform, warp

    tform = AffineTransform(translation=(dx, dy))
    for c in range(img_yxc.shape[2]):
        warped = warp(
            img_yxc[..., c],
            inverse_map=tform.inverse,
            order=int(order),
            mode="constant",
            cval=0.0,
            preserve_range=bool(preserve_range),
        )
        out[..., c] = warped.astype(np.float32, copy=False)
    return out






# --- Tiled rigid registration (translation + rotation per tile) ---
@dataclass(frozen=True)
class TileTransform:
    y0: int
    x0: int
    h: int
    w: int
    # placement in fixed canvas (top-left)
    y_fixed: int
    x_fixed: int
    angle_deg: float
    score: float


def _iter_full_tiles(canvas_yx: tuple[int, int], tile_hw: tuple[int, int]) -> Iterable[tuple[int, int, int, int]]:
    """Yield full tiles as (y0,x0,h,w). Skips partial edge tiles."""
    H, W = int(canvas_yx[0]), int(canvas_yx[1])
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    if th <= 0 or tw <= 0:
        return
    for y0 in range(0, H - th + 1, th):
        for x0 in range(0, W - tw + 1, tw):
            yield int(y0), int(x0), int(th), int(tw)


def _tile_has_signal(tile: np.ndarray, *, min_nonzero_frac: float = 0.01) -> bool:
    """Heuristic to skip mostly-background tiles."""
    if tile.size == 0:
        return False
    # Use >0 on float32 padded tiles (background is 0).
    nz = float(np.count_nonzero(tile > 0))
    frac = nz / float(tile.size)
    return frac >= float(min_nonzero_frac)


def estimate_tiled_rigid_transforms(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    *,
    tile_size: int = 2000,
    search_factor: float = 2,
    angle_deg_max: float = 5.0,
    angle_step: float = 1.0,
    min_foreground_frac: float = 0.01,
    allow_rotation: bool = False,
    on_tile_processed: Callable[[int], None] | None = None,
    smooth_iters: int = 2,
    min_score: float = 0.05,
    min_peak_ratio: float = 1.05,
) -> list[TileTransform]:
    """
    Register moving->fixed using independent tiles with optional rotation.

    Robustness improvements:
      - z-score normalize fixed/moving (global)
      - Otsu foreground masks; skip tiles with little foreground
      - smaller search window (search_factor ~2)
      - confidence gating (score + peak ratio); fallback to identity placement
      - smooth displacement field across tiles to discourage drifting
    """
    if fixed_yx.ndim != 2 or moving_yx.ndim != 2:
        raise ValueError("estimate_tiled_rigid_transforms expects 2D arrays")

    H, W = fixed_yx.shape
    th = tw = int(tile_size)
    if th <= 0 or tw <= 0:
        raise ValueError("tile_size must be positive")

    # Use *partial edge tiles* by default (ceil division).
    rows = max(1, int(math.ceil(H / th)))
    cols = max(1, int(math.ceil(W / tw)))

    def _zscore(img: np.ndarray) -> np.ndarray:
        v = img.astype(np.float32, copy=False)
        mu = float(v.mean())
        sd = float(v.std())
        if not np.isfinite(sd) or sd <= 1e-8:
            return np.zeros_like(v, dtype=np.float32)
        return (v - mu) / sd

    def _otsu_mask(img: np.ndarray) -> np.ndarray:
        # Prefer nonzero pixels if there are enough; avoids Otsu collapsing to ~0 on huge backgrounds.
        v = img.astype(np.float32, copy=False)
        nz = v[v > 0]
        try:
            if nz.size > 5000:
                thr = float(threshold_otsu(nz))
            else:
                thr = float(threshold_otsu(v))
        except Exception:
            thr = 0.0
        return (v > thr)

    fixed_z = _zscore(fixed_yx)
    moving_z = _zscore(moving_yx)

    fixed_mask = _otsu_mask(fixed_yx)
    moving_mask = _otsu_mask(moving_yx)

    angles: list[float]
    if not bool(allow_rotation):
        angles = [0.0]
    else:
        angles = []
        amax = float(angle_deg_max)
        step = float(max(angle_step, 1e-6))
        a = -amax
        while a <= amax + 1e-9:
            angles.append(float(a))
            a += step
        if 0.0 not in angles:
            angles.append(0.0)
        angles = sorted(set(angles))

    sf = float(search_factor)
    sf = max(1.0, sf)

    if _STEP1_DEBUG:
        _dbg(
            f"tiled rigid: grid={rows}x{cols} tiles={rows*cols} tile_size={tile_size} "
            f"search_factor={sf:.3f} allow_rotation={bool(allow_rotation)}"
        )

    # Storage in row-major grid
    dy = np.zeros((rows, cols), dtype=np.float32)
    dx = np.zeros((rows, cols), dtype=np.float32)
    ang = np.zeros((rows, cols), dtype=np.float32)
    wt = np.zeros((rows, cols), dtype=np.float32)
    raw_score = np.full((rows, cols), -np.inf, dtype=np.float32)

    # Parallel execution by row improves deterministic debug output and avoids "row completion" messages
    # being meaningless under as_completed ordering.
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception:  # pragma: no cover
        ThreadPoolExecutor = None  # type: ignore
        as_completed = None  # type: ignore

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    processed = 0

    def _process_one(spec: tuple[int, int, int, int]) -> tuple[tuple[int, int], float, float, float, float, float]:
        # returns (r,c), dy, dx, angle, score ; dy/dx are relative to original tile top-left
        y0, x0, h, w = spec
        r = y0 // th
        c = x0 // tw

        tile = moving_z[y0 : y0 + h, x0 : x0 + w]
        tmask = moving_mask[y0 : y0 + h, x0 : x0 + w]
        fg_frac = float(tmask.mean()) if tmask.size else 0.0
        if fg_frac < float(min_foreground_frac):
            return (r, c), 0.0, 0.0, 0.0, float("-inf"), 0.0

        # Fixed search window centered on this tile's center.
        cy = y0 + h // 2
        cx = x0 + w // 2
        win_h = int(math.ceil(sf * h))
        win_w = int(math.ceil(sf * w))
        wy0 = int(max(0, cy - win_h // 2))
        wx0 = int(max(0, cx - win_w // 2))
        wy1 = int(min(H, wy0 + win_h))
        wx1 = int(min(W, wx0 + win_w))
        if (wy1 - wy0) < h or (wx1 - wx0) < w:
            return (r, c), 0.0, 0.0, 0.0, float("-inf"), 0.0

        fixed_win = fixed_z[wy0:wy1, wx0:wx1]
        fixed_m = fixed_mask[wy0:wy1, wx0:wx1].astype(np.float32, copy=False)

        # Mask fixed window to reduce background ambiguity.
        fixed_win = fixed_win * fixed_m

        best_score = -np.inf
        second_score = -np.inf
        best_angle = 0.0
        best_y = y0
        best_x = x0

        for a in angles:
            if a == 0.0:
                tile_r = tile
                m_r = tmask.astype(np.float32, copy=False)
            else:
                tile_r = rotate(
                    tile,
                    angle=float(a),
                    resize=False,
                    preserve_range=True,
                    order=1,
                    mode="constant",
                    cval=0.0,
                ).astype(np.float32, copy=False)
                m_r = rotate(
                    tmask.astype(np.float32, copy=False),
                    angle=float(a),
                    resize=False,
                    preserve_range=True,
                    order=0,
                    mode="constant",
                    cval=0.0,
                ).astype(np.float32, copy=False)

            # Mask the tile too.
            tile_r = tile_r * m_r

            try:
                resp = match_template(fixed_win, tile_r, pad_input=False)
            except Exception:
                continue
            if resp.size == 0:
                continue

            # Find top-2 peaks for confidence gating.
            flat = resp.ravel()
            if flat.size == 1:
                top1 = float(flat[0])
                top2 = float("-inf")
                idx1 = 0
            else:
                # argpartition is O(n) and cheaper than full sort.
                k = 2
                part = np.argpartition(flat, -k)[-k:]
                # order those two
                if float(flat[part[0]]) >= float(flat[part[1]]):
                    idx1, idx2 = int(part[0]), int(part[1])
                else:
                    idx1, idx2 = int(part[1]), int(part[0])
                top1 = float(flat[idx1])
                top2 = float(flat[idx2])

            if top1 > best_score:
                # update best/second across angles
                second_score = max(second_score, best_score)
                best_score = top1
                best_angle = float(a)
                ij = np.unravel_index(idx1, resp.shape)
                best_y = int(wy0 + ij[0])
                best_x = int(wx0 + ij[1])
            else:
                second_score = max(second_score, top1)

        # Confidence gating
        if not np.isfinite(best_score) or best_score < float(min_score):
            return (r, c), 0.0, 0.0, 0.0, float("-inf"), 0.0
        denom = float(second_score) if np.isfinite(second_score) else float("-inf")
        if denom <= 1e-8:
            peak_ratio = float("inf")
        else:
            peak_ratio = float(best_score / denom)
        if peak_ratio < float(min_peak_ratio):
            return (r, c), 0.0, 0.0, 0.0, float("-inf"), 0.0

        # Weight combines match score and foreground fraction.
        weight = max(0.0, float(best_score)) * float(fg_frac)
        # store dy/dx relative to original tile position
        return (r, c), float(best_y - y0), float(best_x - x0), float(best_angle), float(best_score), float(weight)


    # Process tiles row by row.
    ex = None
    if ThreadPoolExecutor is not None:
        ex = ThreadPoolExecutor(max_workers=int(max_workers))

    try:
        for r in range(rows):
            y0 = int(r * th)
            h = int(min(th, H - y0))
            specs = []
            for c in range(cols):
                x0 = int(c * tw)
                w = int(min(tw, W - x0))
                specs.append((y0, x0, h, w))

            if ex is None:
                for spec in specs:
                    (rr, cc), ddy, ddx, aa, sc, ww = _process_one(spec)
                    dy[rr, cc] = ddy
                    dx[rr, cc] = ddx
                    ang[rr, cc] = aa
                    wt[rr, cc] = ww
                    raw_score[rr, cc] = sc
                    processed += 1
                    if on_tile_processed is not None:
                        on_tile_processed(int(processed))
            else:
                futs = [ex.submit(_process_one, spec) for spec in specs]
                for fut in as_completed(futs):
                    (rr, cc), ddy, ddx, aa, sc, ww = fut.result()
                    dy[rr, cc] = ddy
                    dx[rr, cc] = ddx
                    ang[rr, cc] = aa
                    wt[rr, cc] = ww
                    raw_score[rr, cc] = sc
                    processed += 1
                    if on_tile_processed is not None:
                        on_tile_processed(int(processed))

            if _STEP1_DEBUG:
                _dbg(f"tiled rigid: completed row {r+1}/{rows}")
    finally:
        try:
            if ex is not None:
                ex.shutdown(wait=True)
        except Exception:
            pass

    # --- Neighbor propagation: fill low-confidence/background tiles from nearby confident tiles ---
    # Any tile with wt<=0 is considered "failed" and will be filled (if possible) using the best
    # neighbor transform. This keeps the displacement field smooth and avoids reverting to global-only
    # motion in empty/background regions.
    failed = (wt <= 0.0)
    if bool(failed.any()):
        # Iteratively expand coverage so that islands of background inherit motion from the nearest
        # confident region. A few passes are sufficient for typical slides.
        for _pass in range(3):
            changed = False
            for r in range(rows):
                for c in range(cols):
                    if wt[r, c] > 0.0:
                        continue
                    best_w = 0.0
                    br = bc = -1
                    # 8-neighborhood
                    for rr in (r - 1, r, r + 1):
                        for cc in (c - 1, c, c + 1):
                            if rr == r and cc == c:
                                continue
                            if 0 <= rr < rows and 0 <= cc < cols:
                                w1 = float(wt[rr, cc])
                                if w1 > best_w:
                                    best_w = w1
                                    br, bc = rr, cc
                    if br >= 0 and best_w > 0.0:
                        dy[r, c] = dy[br, bc]
                        dx[r, c] = dx[br, bc]
                        ang[r, c] = ang[br, bc]
                        # Give filled tiles a modest weight so they can be smoothed but won't dominate.
                        wt[r, c] = max(1e-6, 0.25 * best_w)
                        raw_score[r, c] = raw_score[br, bc]
                        changed = True
            if not changed:
                break

    # --- Smoothness: discourage drifting between adjacent tiles ---
    # After filling failed tiles from neighbors, smooth the displacement field.
    # We use an 8-neighborhood and confidence-weighted averaging with a small
    # distance penalty so distant/diagonal neighbors influence slightly less.
    if int(smooth_iters) > 0:
        for _ in range(int(smooth_iters)):
            dy_new = dy.copy()
            dx_new = dx.copy()
            ang_new = ang.copy()

            for r in range(rows):
                for c in range(cols):
                    w0 = float(wt[r, c])
                    if w0 <= 0.0:
                        continue

                    acc_w = 0.0
                    acc_dy = 0.0
                    acc_dx = 0.0
                    acc_ang = 0.0

                    for rr in (r - 1, r, r + 1):
                        for cc in (c - 1, c, c + 1):
                            if 0 <= rr < rows and 0 <= cc < cols:
                                w1 = float(wt[rr, cc])
                                if w1 <= 0.0:
                                    continue
                                dr = abs(rr - r)
                                dc = abs(cc - c)
                                # distance penalty: 1 for self/axis-neighbors, sqrt(2) for diagonals
                                dist = 1.0
                                if dr == 1 and dc == 1:
                                    dist = 1.41421356237
                                w_eff = w1 / dist
                                acc_w += w_eff
                                acc_dy += float(dy[rr, cc]) * w_eff
                                acc_dx += float(dx[rr, cc]) * w_eff
                                acc_ang += float(ang[rr, cc]) * w_eff

                    if acc_w > 0.0:
                        dy_new[r, c] = acc_dy / acc_w
                        dx_new[r, c] = acc_dx / acc_w
                        ang_new[r, c] = acc_ang / acc_w

            dy, dx, ang = dy_new, dx_new, ang_new

    # Convert grid to per-tile transforms
    out: list[TileTransform] = []
    for r in range(rows):
        for c in range(cols):
            y0 = int(r * th)
            x0 = int(c * tw)
            h = int(min(th, H - y0))
            w = int(min(tw, W - x0))
            ddy = float(dy[r, c])
            ddx = float(dx[r, c])
            a = float(ang[r, c])
            sc = float(raw_score[r, c])

            y_fixed = int(round(y0 + ddy))
            x_fixed = int(round(x0 + ddx))
            # Clamp to canvas so paste region stays within bounds.
            y_fixed = max(0, min(int(H - h), y_fixed))
            x_fixed = max(0, min(int(W - w), x_fixed))
            out.append(
                TileTransform(
                    y0=y0,
                    x0=x0,
                    h=int(h),
                    w=int(w),
                    y_fixed=y_fixed,
                    x_fixed=x_fixed,
                    angle_deg=a,
                    score=sc,
                )
            )


    return out



def apply_tiled_rigid_to_plane(
    plane_yx: np.ndarray,
    *,
    canvas_yx: tuple[int, int],
    transforms: list[TileTransform],
) -> np.ndarray:
    """Apply per-tile rigid (rotation + translation via placement) transforms to a 2D plane.

    Parameters
    ----------
    plane_yx:
        2D moving plane already in the *global translated* canvas coordinates.
    canvas_yx:
        Output canvas (H, W) in pixels.
    transforms:
        List of TileTransform entries describing the source tile window (y0,x0,h,w)
        and destination placement (y_fixed,x_fixed) in the fixed canvas, plus rotation.

    Notes
    -----
    - This function does *not* attempt to create a seamless non-rigid warp. It pastes
      individually registered tiles into an output canvas.
    - When tiles overlap (e.g. if future strides < tile_size), we average values
      in the overlap region using a weight map.
    """
    H, W = int(canvas_yx[0]), int(canvas_yx[1])
    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)

    if plane_yx.dtype != np.float32:
        plane = plane_yx.astype(np.float32, copy=False)
    else:
        plane = plane_yx

    for t in transforms:
        y0, x0, h, w = int(t.y0), int(t.x0), int(t.h), int(t.w)
        if h <= 0 or w <= 0:
            continue

        # Source crop (clip to bounds)
        sy0 = max(0, y0)
        sx0 = max(0, x0)
        sy1 = min(plane.shape[0], y0 + h)
        sx1 = min(plane.shape[1], x0 + w)
        if sy1 <= sy0 or sx1 <= sx0:
            continue

        tile = plane[sy0:sy1, sx0:sx1]

        # If clipped, we also need to clip destination size consistently.
        th, tw = tile.shape[0], tile.shape[1]
        if th <= 0 or tw <= 0:
            continue

        # Rotation about tile center (preserve size).
        ang = float(getattr(t, "angle_deg", 0.0) or 0.0)
        if ang != 0.0:
            tile = rotate(
                tile,
                angle=ang,
                resize=False,
                order=1,
                mode="constant",
                cval=0.0,
                preserve_range=True,
            ).astype(np.float32, copy=False)

        # Destination placement (clip to bounds)
        dy0 = int(t.y_fixed) + (sy0 - y0)
        dx0 = int(t.x_fixed) + (sx0 - x0)
        dy1 = dy0 + th
        dx1 = dx0 + tw

        oy0 = max(0, dy0)
        ox0 = max(0, dx0)
        oy1 = min(H, dy1)
        ox1 = min(W, dx1)
        if oy1 <= oy0 or ox1 <= ox0:
            continue

        ty0 = oy0 - dy0
        tx0 = ox0 - dx0
        ty1 = ty0 + (oy1 - oy0)
        tx1 = tx0 + (ox1 - ox0)

        patch = tile[ty0:ty1, tx0:tx1]
        out[oy0:oy1, ox0:ox1] += patch
        wgt[oy0:oy1, ox0:ox1] += 1.0

    # Avoid divide-by-zero.
    m = wgt > 0
    out[m] /= wgt[m]
    return out



def _compute_demons_transform(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    *,
    initial_shift_dy_dx: tuple[float, float] | None = None,
    downsample: int = 1,
    n_iter: int = 60,
    smoothing: float = 1.5,
) -> "sitk.Transform":
    """Compute a Demons displacement field transform mapping moving -> fixed."""
    assert sitk is not None
    _dbg("_compute_demons_transform: start")
    fixed = _sitk_from_yx(fixed_yx)
    moving = _sitk_from_yx(moving_yx)

    ds = int(downsample) if downsample is not None else 1
    if ds < 1:
        ds = 1
    if ds > 1:
        _dbg(f"_compute_demons_transform: Shrink ds={ds}")
        fixed = sitk.Shrink(fixed, [ds, ds])
        moving = sitk.Shrink(moving, [ds, ds])

    if initial_shift_dy_dx is not None:
        dy, dx = initial_shift_dy_dx
        _dbg(f"_compute_demons_transform: initial translation dy={dy:.3f}, dx={dx:.3f}")
        t0 = sitk.TranslationTransform(2, (float(dx), float(dy)))
        _dbg("_compute_demons_transform: Resample for initial translation")
        moving = sitk.Resample(moving, fixed, t0, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(int(n_iter))
    demons.SetSmoothDisplacementField(True)
    demons.SetStandardDeviations(float(smoothing))

    _dbg("_compute_demons_transform: Execute")
    disp = demons.Execute(fixed, moving)
    _dbg("_compute_demons_transform: done")
    return sitk.DisplacementFieldTransform(disp)


def _apply_sitk_transform_to_plane(
    plane_yx: np.ndarray,
    fixed_ref_yx: np.ndarray,
    transform: "sitk.Transform",
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    """Resample plane_yx into fixed_ref_yx geometry using transform."""
    assert sitk is not None
    _dbg("_apply_sitk_transform_to_plane: start")
    fixed = _sitk_from_yx(fixed_ref_yx)
    mov = _sitk_from_yx(plane_yx)
    _dbg("_apply_sitk_transform_to_plane: sitk.Resample")
    out = sitk.Resample(mov, fixed, transform, sitk.sitkLinear, float(default_value), sitk.sitkFloat32)
    _dbg("_apply_sitk_transform_to_plane: done")
    return sitk.GetArrayFromImage(out)



def merge_cycles_to_ome_tiff(
    cycles: Iterable[CycleInput],
    output_path: str,
    *,
    reference_cycle: int | None = None,
    default_registration_marker: str = "DAPI",
    registration_algorithm: str = "tiled_rigid",
    downsample_for_registration: int = 4,
    tiled_rigid_allow_rotation: bool = False,
    tiled_rigid_tile_size: int = 2000,
    tiled_rigid_search_factor: float = 3,
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
    """
    Load multiple cycle OME-TIFFs, optionally rigid-translate register each cycle to a reference,
    and save one merged OME-TIFF with channels renamed as "cy{cycle}:{channel}".

    Current implementation is intentionally minimal (translation-only) as an MVP for Step (1).
    We can extend to full affine/nonlinear registration and richer metadata later.

    Returns a small report dict (useful for logging/UI).
    """
    cycles = list(cycles)
    if not cycles:
        raise ValueError("No cycles provided")

    input_pyramid_info: dict[int, dict] = {}
    for ci in cycles:
        try:
            input_pyramid_info[int(ci.cycle)] = inspect_tiff_pyramid(ci.path)
        except Exception:
            input_pyramid_info[int(ci.cycle)] = {"is_pyramidal": False, "series": []}

    if _STEP1_DEBUG:
        try:
            import faulthandler

            faulthandler.enable(all_threads=True)
            _dbg("faulthandler enabled")
        except Exception:
            pass

    reg_alg = str(registration_algorithm or "tiled_rigid").strip().lower()
    _dbg(
        f"merge_cycles_to_ome_tiff: start n_cycles={len(cycles)} low_mem={bool(low_mem)} alg={reg_alg!r} downsample={downsample_for_registration} allow_rotation={bool(tiled_rigid_allow_rotation)} tile_size={int(tiled_rigid_tile_size)} search_factor={float(tiled_rigid_search_factor):.3f}"
    )

    # Debug: report SimpleITK thread settings (helps diagnose low CPU usage).
    if sitk is not None:
        try:
            n_thr = sitk.ProcessObject_GetGlobalDefaultNumberOfThreads()
        except Exception:
            n_thr = -1
        print(f"[Step1] SimpleITK global default threads = {n_thr}")

    
    # Normalize a few UI-friendly labels
    if reg_alg in {"translation", "phase_correlation", "phase-correlation", "phase correlation"}:
        reg_alg = "translation"
    if reg_alg in {"tiled", "tiled_rigid", "tiled-rigid", "tile_rigid", "tile-rigid"}:
        reg_alg = "tiled_rigid"
    
    if reg_alg not in {"translation", "tiled_rigid"}:
        raise ValueError(f"Unknown registration_algorithm: {registration_algorithm!r}")
    
    if False and sitk is None:
        raise RuntimeError(
            "SimpleITK is required for non-rigid registration, but it is not available in this environment. "
            "Install SimpleITK or select Translation registration."
        )

    def _check_cancel() -> None:
        """Raise RuntimeError("Cancelled") if cancel_cb signals cancellation."""
        try:
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Cancelled")
        except RuntimeError:
            raise
        except Exception:
            # Do not allow a buggy cancel callback to crash preprocessing.
            return

    # Validate cycle numbering: must be non-negative and unique.
    cy_vals = [int(ci.cycle) for ci in cycles]
    if any(cy < 0 for cy in cy_vals):
        raise ValueError(f"Cycle numbers must be >= 0. Got: {cy_vals}")
    if len(set(cy_vals)) != len(cy_vals):
        raise ValueError(f"Cycle numbers must be unique. Got: {cy_vals}")

    # Sort by cycle index for stable output ordering.
    cycles = sorted(cycles, key=lambda x: int(x.cycle))

    if reference_cycle is None:
        reference_cycle = int(cycles[0].cycle)

    if bool(low_mem):
        # Low-memory merge: keep only the current cycle + reference reg channel + final merged output in RAM.
        # This avoids holding all cycles in memory at once.
        _check_cancel()

        # First pass: gather shapes/dtypes without loading all pixels.
        infos: list[tuple[CycleInput, tuple[int, int, int], np.dtype]] = []
        canvas_y = 0
        canvas_x = 0
        total_ch = 0
        base_dtype: np.dtype | None = None
        for ci in cycles:
            shp_yxc, dt = _tiff_info_yxc(ci.path)
            _dbg(f"info: cycle={int(ci.cycle)} shape={tuple(int(x) for x in shp_yxc)} dtype={np.dtype(dt)}")
            if base_dtype is None:
                base_dtype = dt
            elif np.dtype(dt) != np.dtype(base_dtype):
                raise ValueError(
                    f"All cycles must have the same dtype for low_mem merge. Got {base_dtype} vs {dt} ({ci.path})"
                )
            canvas_y = max(canvas_y, int(shp_yxc[0]))
            canvas_x = max(canvas_x, int(shp_yxc[1]))
            total_ch += int(shp_yxc[2])
            infos.append((ci, shp_yxc, dt))

        # --- Progress model (tiled-rigid) ---
        # We model progress based on expected tiles per (cycle) image.
        # Ticks = ceil(n_tiles/10) * ((n_moving_cycles)+2)
        #   - +1 segment after loading the reference
        #   - +n_moving_cycles segments during tile registration (1 tick per 10 tiles)
        #   - +1 segment after writing output
        n_cycles = int(len(infos))
        tile_size_px = max(128, int(tiled_rigid_tile_size))

        completed_ticks = 0
        assert base_dtype is not None
        canvas_yx = (int(canvas_y), int(canvas_x))

        # Tile count must include partial edge tiles (ceil division).
        tiles_y = max(1, int(math.ceil(float(canvas_yx[0]) / float(tile_size_px))))
        tiles_x = max(1, int(math.ceil(float(canvas_yx[1]) / float(tile_size_px))))
        expected_tiles = int(tiles_y * tiles_x)
        tile_ticks = max(1, int(math.ceil(float(expected_tiles) / 10.0)))

        # Progress layout:
        #   Initialization: load reference cycle -> 1 tick
        #   For each *moving* cycle:
        #       load cycle -> 1 tick
        #       initial transform -> 1 tick
        #       process 10 tiles -> 1 tick (ceil(expected_tiles/10))
        #       apply transform per channel -> 1 tick per channel
        #   Final: saving output file -> 1 tick per cycle processed (including reference)
        moving_infos = [t for t in infos if int(t[0].cycle) != int(reference_cycle)]
        total_ticks = 1  # load reference
        for (_ci, _shp_yxc, _dt) in moving_infos:
            total_ticks += 1  # load moving cycle
            total_ticks += 1  # initial transform
            total_ticks += int(tile_ticks)  # tiled registration
            total_ticks += int(_shp_yxc[2])  # apply per channel
        total_ticks += int(n_cycles)  # saving output (1 tick per cycle)
        if bool(pyramidal_output):
            total_ticks += int(estimate_pyramid_conversion_ticks((int(total_ch), int(canvas_yx[0]), int(canvas_yx[1])), min_level_size=int(pyramidal_min_level_size), out_chunk=int(pyramid_progress_chunk)))

        def _emit_tick(msg: str, *, phase: str, cycle: int | None = None) -> None:
            if progress_event_cb:
                evt = {
                    "phase": str(phase),
                    "idx": int(completed_ticks),
                    "n": int(total_ticks),
                    "msg": str(msg),
                }
                if cycle is not None:
                    evt["cycle"] = int(cycle)
                progress_event_cb(evt)

        # Identify reference cycle
        ref_ci: CycleInput | None = None
        for ci, _, _ in infos:
            if int(ci.cycle) == int(reference_cycle):
                ref_ci = ci
                break
        if ref_ci is None:
            raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

        # Build merged channel names up front so the output OME metadata can be
        # written once when the on-disk BigTIFF is initialized.
        merged_names: list[str] = []
        seen: dict[str, int] = {}

        # Load only the reference registration channel once (float32), not the full reference stack.
        ref_marker = ref_ci.registration_marker or default_registration_marker
        if progress_cb:
            progress_cb(f"Loading reference cycle {int(ref_ci.cycle)}")
        _dbg(f"load reference: cycle={int(ref_ci.cycle)} marker={ref_marker!r}")
        _emit_tick(f"Loading reference cycle {int(ref_ci.cycle)}", phase="load_ref", cycle=int(ref_ci.cycle))
        ref_names = load_channel_names_only(ref_ci.path)
        _dbg(f"loaded reference channel metadata: names={len(ref_names)}")
        ref_ch_idx = _find_channel_index(ref_names, ref_marker)
        ref_yx: np.ndarray | None
        if ref_ch_idx is None:
            ref_yx = None
        else:
            ref_plane_native = load_single_channel_tiff_native(ref_ci.path, int(ref_ch_idx))
            ref_yx = _pad_plane_to_canvas(ref_plane_native, canvas_yx, out_dtype=np.float32)
            try:
                del ref_plane_native
            except Exception:
                pass

        # After loading reference cycle
        completed_ticks += 1
        completed_ticks = min(int(completed_ticks), int(total_ticks))
        _emit_tick(f"Loaded reference cycle {int(ref_ci.cycle)}", phase="ref_loaded", cycle=int(ref_ci.cycle))

        shifts: dict[int, tuple[float, float]] = {}
        transforms: dict[int, object] = {}

        # Helper to name output channels
        def _make_output_channel_name(ci: CycleInput, orig_name: str, j: int) -> str:
            cy = int(ci.cycle)
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (orig_name or "").strip() or "Channel"
            out_nm = f"{base}_cy{cy}"
            k = int(seen.get(out_nm, 0))
            out_nm2 = f"{out_nm}_{k+1}" if k else out_nm
            seen[out_nm] = k + 1
            return out_nm2

        for ci, _shp_yxc, _dt in infos:
            ch_names_only = load_channel_names_only(ci.path)
            for j, nm in enumerate(ch_names_only):
                merged_names.append(_make_output_channel_name(ci, nm, j))

        with IncrementalOmeBigTiffWriter(
            output_path,
            (int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch)),
            base_dtype,
            merged_names,
        ) as writer:
            # Write reference cycle first (unregistered)
            out_c = 0
            for ci in cycles:
                _check_cancel()

                _dbg(f"cycle start: cycle={int(ci.cycle)}")

                ch_names = load_channel_names_only(ci.path)
                n_ch_cycle = int(len(ch_names))

                if int(ci.cycle) == int(ref_ci.cycle):
                    dy = 0.0
                    dx = 0.0
                else:
                    if progress_cb:
                        progress_cb(f"Registering cycle {int(ci.cycle)}")
                    _dbg(f"load cycle registration channel: cycle={int(ci.cycle)}")
                    _emit_tick(f"Registering cycle {int(ci.cycle)}", phase="load_cycle", cycle=int(ci.cycle))

                    # Progress: loading cycle image/registration channel
                    completed_ticks += 1
                    completed_ticks = min(int(completed_ticks), int(total_ticks))
                    _emit_tick(f"Loaded cycle {int(ci.cycle)}", phase="cycle_loaded", cycle=int(ci.cycle))

                    # Determine shift for this cycle using only its registration channel.
                    if ref_yx is None:
                        dy, dx = (0.0, 0.0)
                    else:
                        marker = ci.registration_marker or default_registration_marker
                        ch_idx = _find_channel_index(ch_names, marker)
                        if ch_idx is None:
                            dy, dx = (0.0, 0.0)
                        else:
                            mov_plane_native = load_single_channel_tiff_native(ci.path, int(ch_idx))
                            mov_yx = _pad_plane_to_canvas(mov_plane_native, canvas_yx, out_dtype=np.float32)
                            try:
                                del mov_plane_native
                            except Exception:
                                pass

                            _dbg(f"estimate translation: cycle={int(ci.cycle)}")

                            # Always compute an initial translation (for reporting + as initialization for the tiled refinement).
                            dy, dx = estimate_translation(
                                ref_yx,
                                mov_yx,
                                downsample=int(downsample_for_registration),
                                upsample_factor=int(upsample_factor),
                            )

                            _dbg(f"estimated translation: cycle={int(ci.cycle)} dy={float(dy):.3f} dx={float(dx):.3f}")

                            # Progress: initial transform
                            completed_ticks += 1
                            completed_ticks = min(int(completed_ticks), int(total_ticks))
                            _emit_tick(f"Cycle {int(ci.cycle)}: initial transform", phase="initial_transform", cycle=int(ci.cycle))

                            if reg_alg == "tiled_rigid":
                                # Tiled rigid refinement (rotation+translation per tile) on the registration channel.
                                if progress_cb:
                                    progress_cb(f"Tiled rigid registering cycle {int(ci.cycle)}")
                                _dbg(f"tiled rigid compute: cycle={int(ci.cycle)}")
                                mov_init = mov_yx
                                if dy != 0.0 or dx != 0.0:
                                    mov_init = apply_translation_yxc(mov_yx[..., None], dy=dy, dx=dx, order=1)[..., 0]

                                # Progress: 1 tick per 10 processed tiles; ensure we consume exactly tile_ticks.
                                emitted_for_cycle = 0

                                def _on_tile_processed(n_done: int) -> None:
                                    nonlocal emitted_for_cycle, completed_ticks
                                    if n_done > 0 and (n_done % 10 == 0):
                                        emitted_for_cycle += 1
                                        completed_ticks += 1
                                        completed_ticks = min(int(completed_ticks), int(total_ticks))
                                        _emit_tick(
                                            f"Cycle {int(ci.cycle)}: registered {n_done}/{expected_tiles} tiles",
                                            phase="tiles",
                                            cycle=int(ci.cycle),
                                        )

                                transforms[int(ci.cycle)] = estimate_tiled_rigid_transforms(
                                    fixed_yx=ref_yx,
                                    moving_yx=mov_init,
                                    tile_size=int(tile_size_px),
                                    search_factor=float(tiled_rigid_search_factor),
                                    angle_deg_max=5.0,
                                    angle_step=1.0,
                                    allow_rotation=bool(tiled_rigid_allow_rotation),
                                    on_tile_processed=_on_tile_processed,
                                )

                                # If the last partial batch (<10) didn't trigger a tick, advance remaining ticks.
                                # We want exactly tile_ticks ticks for this cycle.
                                need = int(tile_ticks) - int(emitted_for_cycle)
                                if need > 0:
                                    completed_ticks += int(need)
                                    completed_ticks = min(int(completed_ticks), int(total_ticks))
                                    _emit_tick(
                                        f"Cycle {int(ci.cycle)}: finished tiles",
                                        phase="tiles_done",
                                        cycle=int(ci.cycle),
                                    )
                                _dbg(f"tiled rigid done: cycle={int(ci.cycle)}")

                            try:
                                del mov_yx
                            except Exception:
                                pass

                shifts[int(ci.cycle)] = (float(dy), float(dx))

                # Register/write each channel plane directly into the final output.
                for j in range(int(n_ch_cycle)):
                    _dbg(f"plane: cycle={int(ci.cycle)} ch={j}/{int(n_ch_cycle) - 1} out_c={int(out_c)}")
                    plane_native = load_single_channel_tiff_native(ci.path, int(j))
                    plane = _pad_plane_to_canvas(plane_native, canvas_yx, out_dtype=np.float32)
                    try:
                        del plane_native
                    except Exception:
                        pass
                    if reg_alg == "translation":
                        if dy != 0.0 or dx != 0.0:
                            # Use the existing translation implementation on a 1-channel stack to reuse code.
                            plane_yxc = plane[..., None]
                            plane_reg = apply_translation_yxc(plane_yxc, dy=dy, dx=dx, order=1)[..., 0]
                        else:
                            plane_reg = plane
                    else:
                        t = transforms.get(int(ci.cycle))
                        if t is None:
                            plane_reg = plane
                        else:
                            # Apply tiled rigid reconstruction using precomputed tile transforms from the registration channel.
                            _dbg(f"apply tiled rigid: cycle={int(ci.cycle)} ch={j}")
                            # First apply the global translation (dy,dx) to the full plane, then tile-reconstruct.
                            plane_init = plane
                            if dy != 0.0 or dx != 0.0:
                                plane_init = apply_translation_yxc(plane[..., None], dy=dy, dx=dx, order=1)[..., 0]
                            plane_reg = apply_tiled_rigid_to_plane(plane_init, canvas_yx=canvas_yx, transforms=t)  # type: ignore[arg-type]
                            _dbg(f"applied tiled rigid: cycle={int(ci.cycle)} ch={j}")

                    writer.write_channel(int(out_c), _cast_preserve_dtype(plane_reg, base_dtype))

                    # Progress: applying transform to one channel (always 1 tick per channel, per spec)
                    if int(ci.cycle) != int(reference_cycle):
                        completed_ticks += 1
                        completed_ticks = min(int(completed_ticks), int(total_ticks))
                        _emit_tick(
                            f"Cycle {int(ci.cycle)}: applied channel {int(j)+1}/{int(n_ch_cycle)}",
                            phase="apply_channel",
                            cycle=int(ci.cycle),
                        )
                    out_c += 1

                # No cycle-level tick here; tile-level ticks cover registration progress.

            if progress_cb:
                progress_cb("Writing OME-TIFF")
            _dbg(f"write: finalize output_path={output_path!r}")
            _emit_tick(f"Writing output: {output_path}", phase="write_start")
            writer.flush()
            _dbg("write: done")

            # Saving output file: 1 tick per cycle processed (including reference)
            completed_ticks += int(n_cycles)
            completed_ticks = min(int(completed_ticks), int(total_ticks))
            _emit_tick(f"Wrote output: {output_path}", phase="write_done")

        pyout = {"is_pyramidal": False, "series": []}
        if bool(pyramidal_output):
            def _pstep(msg: str, phase: str) -> None:
                nonlocal completed_ticks
                completed_ticks += 1
                completed_ticks = min(int(completed_ticks), int(total_ticks))
                _emit_tick(msg, phase=phase)

            final_output_path = convert_flat_ome_to_pyramidal(
                output_path,
                output_path=None,
                channel_names=merged_names,
                tile_size=int(pyramidal_tile_size),
                compression=pyramidal_compression,
                min_level_size=int(pyramidal_min_level_size),
                out_chunk=int(pyramid_progress_chunk),
                replace_source=False,
                progress_cb=progress_cb,
                progress_step_cb=_pstep,
                cancel_cb=cancel_cb,
            )
            os.replace(final_output_path, output_path)
            try:
                pyout = inspect_tiff_pyramid(output_path)
            except Exception:
                pyout = {"is_pyramidal": True, "series": []}

        return {
            "output_path": output_path,
            "reference_cycle": int(reference_cycle),
            "default_registration_marker": default_registration_marker,
            "shifts_yx": shifts,
            "canvas_yx": canvas_yx,
            "inputs": [{"cycle": int(ci.cycle), "path": ci.path} for ci in cycles],
            "n_channels_out": int(total_ch),
            "shape_yxc": (int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch)),
            "low_mem": True,
            "pyramidal_output": bool(pyramidal_output),
            "output_pyramid_info": pyout,
            "input_pyramid_info": input_pyramid_info,
        }

    imgs: list[np.ndarray] = []
    names: list[list[str]] = []
    paths: list[str] = []
    n_cycles = int(len(cycles))
    total_steps = int(n_cycles + 1)
    for k, ci in enumerate(cycles, start=1):
        _check_cancel()
        if progress_cb:
            progress_cb(f"Registering cycle {int(ci.cycle)}")
        if progress_event_cb:
            progress_event_cb(
                {
                    "phase": "load",
                    "cycle": int(ci.cycle),
                    "idx": int(k - 1),
                    "n": int(total_steps),
                    "msg": f"Registering cycle {int(ci.cycle)}",
                }
            )
        img, ch = load_multichannel_tiff(ci.path)
        imgs.append(img)
        names.append(ch)
        paths.append(ci.path)

    # Choose a common canvas so we can merge/register cycles of different pixel dimensions.
    canvas_y = int(max(im.shape[0] for im in imgs))
    canvas_x = int(max(im.shape[1] for im in imgs))
    canvas_yx = (canvas_y, canvas_x)
    imgs = [_pad_channels_to_canvas(im, canvas_yx) for im in imgs]

    # Reference index
    ref_idx = None
    for i, ci in enumerate(cycles):
        if int(ci.cycle) == int(reference_cycle):
            ref_idx = i
            break
    if ref_idx is None:
        raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

    ref_img = imgs[ref_idx]
    ref_names = names[ref_idx]

    # Determine the channel used for registration in the reference cycle.
    ref_marker = cycles[ref_idx].registration_marker or default_registration_marker
    ref_ch_idx = _find_channel_index(ref_names, ref_marker)
    if ref_ch_idx is None:
        # If we can't find it, we skip registration entirely.
        ref_yx = None
    else:
        ref_yx = ref_img[..., ref_ch_idx]

    shifts: dict[int, tuple[float, float]] = {}
    transforms: dict[int, object] = {}
    aligned_imgs: list[np.ndarray] = []

    for i, ci in enumerate(cycles):
        _check_cancel()
        img_i = imgs[i]

        # Report progress as "cycles processed" (includes the reference cycle).
        # This maps cleanly to a determinate per-sample progress bar in the UI.
        def _emit_register_progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            if progress_event_cb:
                progress_event_cb(
                    {
                        "phase": "register",
                        "cycle": int(ci.cycle),
                        "idx": int(i + 1),
                        "n": int(total_steps),
                        "msg": msg,
                    }
                )

        if i == ref_idx or ref_yx is None:
            aligned_imgs.append(img_i.astype(np.float32, copy=False))
            shifts[int(ci.cycle)] = (0.0, 0.0)
            _emit_register_progress(f"Registered cycle {int(ci.cycle)} (reference)")
            continue

        marker = ci.registration_marker or default_registration_marker
        ch_idx = _find_channel_index(names[i], marker)
        if ch_idx is None:
            # Can't register this cycle; leave as-is.
            aligned_imgs.append(img_i.astype(np.float32, copy=False))
            shifts[int(ci.cycle)] = (0.0, 0.0)
            _emit_register_progress(
                f"Registered cycle {int(ci.cycle)} (skipped: marker '{marker}' not found)"
            )
            continue

        _emit_register_progress(f"Registering cycle {int(ci.cycle)}")

        _check_cancel()

        mov_yx = img_i[..., ch_idx]
        dy, dx = estimate_translation(
            ref_yx,
            mov_yx,
            downsample=int(downsample_for_registration),
            upsample_factor=int(upsample_factor),
        )
        aligned = apply_translation_yxc(img_i, dy=dy, dx=dx, order=1)
        aligned_imgs.append(aligned)
        shifts[int(ci.cycle)] = (dy, dx)

        _emit_register_progress(f"Registered cycle {int(ci.cycle)}")

    if progress_cb:
        progress_cb("Merging channels")
    if progress_event_cb:
        progress_event_cb(
            {
                "phase": "merge",
                "idx": int(n_cycles),
                "n": int(total_steps),
                "msg": "Merging channels",
            }
        )

    merged = np.concatenate(aligned_imgs, axis=2)

    merged_names: list[str] = []
    seen: dict[str, int] = {}
    for ci, ch in zip(cycles, names):
        cy = int(ci.cycle)
        markers = ci.channel_markers or []
        for j, orig in enumerate(ch):
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (orig or "").strip() or "Channel"
            out_nm = f"{base}_cy{cy}"
            k = int(seen.get(out_nm, 0))
            if k:
                out_nm2 = f"{out_nm}_{k+1}"
            else:
                out_nm2 = out_nm
            seen[out_nm] = k + 1
            merged_names.append(out_nm2)

    save_ome_tiff_yxc(output_path, merged, merged_names)
    if bool(pyramidal_output):
        def _simple_pstep(msg: str, phase: str) -> None:
            if progress_event_cb:
                progress_event_cb({"phase": str(phase), "idx": 0, "n": 0, "msg": str(msg)})
            if progress_cb:
                progress_cb(msg)
        convert_path = convert_flat_ome_to_pyramidal(
            output_path,
            output_path=None,
            channel_names=merged_names,
            tile_size=int(pyramidal_tile_size),
            compression=pyramidal_compression,
            min_level_size=int(pyramidal_min_level_size),
            out_chunk=int(pyramid_progress_chunk),
            replace_source=False,
            progress_cb=progress_cb,
            progress_step_cb=_simple_pstep,
            cancel_cb=cancel_cb,
        )
        os.replace(convert_path, output_path)

    if progress_event_cb:
        progress_event_cb(
            {
                "phase": "write",
                "idx": int(n_cycles),
                "n": int(total_steps),
                "msg": f"Wrote output: {output_path}",
            }
        )

    return {
        "output_path": output_path,
        "reference_cycle": int(reference_cycle),
        "default_registration_marker": default_registration_marker,
        "shifts_yx": shifts,
        "canvas_yx": canvas_yx,
        "inputs": [{"cycle": int(ci.cycle), "path": ci.path} for ci in cycles],
        "n_channels_out": int(merged.shape[2]),
        "shape_yxc": tuple(int(x) for x in merged.shape),
        "pyramidal_output": bool(pyramidal_output),
        "output_pyramid_info": inspect_tiff_pyramid(output_path) if bool(pyramidal_output) else {"is_pyramidal": False, "series": []},
        "input_pyramid_info": input_pyramid_info,
    }
