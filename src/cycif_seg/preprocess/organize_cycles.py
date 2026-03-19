from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import math
import os
import shutil
from pathlib import Path

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

try:
    from scipy.signal import correlate2d  # type: ignore
except Exception:  # pragma: no cover
    correlate2d = None  # type: ignore

import tifffile

from cycif_seg.io.ome_tiff import (
    IncrementalOmeBigTiffWriter,
    estimate_pyramid_conversion_ticks,
    inspect_tiff_pyramid,
    inspect_tiff_yxc,
    load_channel_names_only,
    load_multichannel_tiff,
    load_multichannel_tiff_native,
    load_physical_pixel_sizes,
    load_single_channel_tiff_native,
    save_ome_tiff_yxc,
    convert_flat_ome_to_pyramidal,
)


# Enable very fine-grained debug prints for isolating native crashes (e.g., in SimpleITK/ITK).
# Set env var CYCIF_SEG_STEP1_DEBUG=1 to turn on.
# In powershell: $env:CYCIF_SEG_STEP1_DEBUG = "1"
_STEP1_DEBUG = str(os.environ.get("CYCIF_SEG_STEP1_DEBUG", "0")).strip().lower() not in {"0", "false", "no", "off", ""}
_STEP1_DEBUG_OTSU = str(os.environ.get("CYCIF_SEG_STEP1_OTSU_DEBUG", "0")).strip().lower() not in {"0", "false", "no", "off", ""}
_DEBUG_WRITTEN_OTSU_MASKS: set[str] = set()


def _dbg(msg: str) -> None:
    """Crash-debug prints (flush immediately)."""
    if _STEP1_DEBUG:
        print(f"[Step1 DEBUG] {msg}", flush=True)


def _format_debug_otsu_mask_path(
    source_path: str,
    *,
    role: str,
    cycle: int | None,
    channel_index: int | None,
    channel_name: str | None,
) -> Path:
    src = Path(source_path)
    parent = src.parent if str(src.parent) else Path.cwd()

    name = src.name
    lower = name.lower()
    if lower.endswith('.ome.tiff'):
        stem = name[:-9]
    elif lower.endswith('.ome.tif'):
        stem = name[:-8]
    elif lower.endswith('.tiff'):
        stem = name[:-5]
    elif lower.endswith('.tif'):
        stem = name[:-4]
    else:
        stem = src.stem

    parts = [stem, 'otsu_mask', str(role)]
    if cycle is not None:
        parts.append(f"cycle{int(cycle)}")
    if channel_index is not None:
        parts.append(f"ch{int(channel_index)}")
    if channel_name:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(channel_name).strip())
        safe = '_'.join([p for p in safe.split('_') if p])
        if safe:
            parts.append(safe[:80])
    return parent / ('__'.join(parts) + '.tif')


def _maybe_write_debug_otsu_mask(
    *,
    source_path: str | None,
    img: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    role: str,
    cycle: int | None,
    channel_index: int | None,
    channel_name: str | None,
) -> None:
    if not _STEP1_DEBUG_OTSU or not source_path:
        return
    try:
        out_path = _format_debug_otsu_mask_path(
            source_path,
            role=role,
            cycle=cycle,
            channel_index=channel_index,
            channel_name=channel_name,
        )
        key = str(out_path.resolve())
    except Exception:
        return
    if key in _DEBUG_WRITTEN_OTSU_MASKS:
        return
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reg_img = np.asarray(img, dtype=np.float32)
        mask_img = mask.astype(np.float32, copy=False) * 255.0
        stacked = np.stack([reg_img, mask_img], axis=0)
        tifffile.imwrite(
            str(out_path),
            stacked,
            photometric='minisblack',
            metadata={
                'axes': 'CYX',
                'Channel': {'Name': ['registration_image', 'otsu_mask']},
            },
        )
        _DEBUG_WRITTEN_OTSU_MASKS.add(key)
        _dbg(
            'otsu debug stack written: '
            f"role={role} cycle={cycle if cycle is not None else 'NA'} "
            f"channel_index={channel_index if channel_index is not None else 'NA'} "
            f"channel_name={channel_name!r} threshold={float(threshold):.6g} "
            f"foreground_frac={float(mask.mean()) if mask.size else 0.0:.6f} "
            f"source={source_path!r} out={str(out_path)!r}"
        )
    except Exception as e:
        _dbg(
            'otsu debug stack write failed: '
            f"role={role} cycle={cycle if cycle is not None else 'NA'} "
            f"source={source_path!r} error={e!r}"
        )


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
    # Symmetric source/destination expansion used to fill small seams between
    # adjacent tiles after smoothing. The core registered tile remains
    # [y_fixed:y_fixed+h, x_fixed:x_fixed+w]; the extra overlap context is used
    # only to back-fill true gaps.
    overlap_px: int = 0


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

def _default_tile_overlap_px(tile_size: int) -> int:
    """Small symmetric overlap budget used to absorb inter-tile seams."""
    ts = max(1, int(tile_size))
    return max(5, int(math.ceil(0.01 * ts)))


def _limit_neighbor_offsets(
    dy: np.ndarray,
    dx: np.ndarray,
    wt: np.ndarray,
    *,
    max_offset: float,
    n_passes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp adjacent-tile displacement differences to a small overlap budget.

    This prevents smoothing from producing neighboring tile placements that
    diverge by more than the overlap region available for seam filling.
    Higher-confidence tiles move less than lower-confidence tiles.
    """
    lim = float(max_offset)
    if not np.isfinite(lim) or lim <= 0.0:
        return dy, dx

    rows, cols = dy.shape
    dy_out = dy.copy()
    dx_out = dx.copy()

    def _clamp_pair(arr: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> None:
        v1 = float(arr[r1, c1])
        v2 = float(arr[r2, c2])
        delta = v2 - v1
        if abs(delta) <= lim:
            return
        excess = abs(delta) - lim
        w1 = max(0.0, float(wt[r1, c1]))
        w2 = max(0.0, float(wt[r2, c2]))
        denom = w1 + w2
        if denom <= 0.0:
            frac1 = 0.5
            frac2 = 0.5
        else:
            # Move the lower-confidence tile more.
            frac1 = w2 / denom
            frac2 = w1 / denom
        if delta > 0.0:
            arr[r1, c1] = v1 + excess * frac1
            arr[r2, c2] = v2 - excess * frac2
        else:
            arr[r1, c1] = v1 - excess * frac1
            arr[r2, c2] = v2 + excess * frac2

    for _ in range(max(1, int(n_passes))):
        for r in range(rows):
            for c in range(cols - 1):
                _clamp_pair(dx_out, r, c, r, c + 1)
                _clamp_pair(dy_out, r, c, r, c + 1)
        for r in range(rows - 1):
            for c in range(cols):
                _clamp_pair(dx_out, r, c, r + 1, c)
                _clamp_pair(dy_out, r, c, r + 1, c)

    return dy_out, dx_out


def estimate_tiled_rigid_transforms(
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    *,
    tile_size: int = 2000,
    search_factor: float = 2,
    angle_deg_max: float = 5.0,
    angle_step: float = 1.0,
    min_foreground_frac: float = 0.001,
    allow_rotation: bool = False,
    on_tile_processed: Callable[[int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    smooth_iters: int = 2,
    min_score: float = 0.05,
    min_peak_ratio: float = 1.05,
    fixed_source_path: str | None = None,
    moving_source_path: str | None = None,
    fixed_cycle: int | None = None,
    moving_cycle: int | None = None,
    fixed_channel_index: int | None = None,
    moving_channel_index: int | None = None,
    fixed_channel_name: str | None = None,
    moving_channel_name: str | None = None,
) -> list[TileTransform]:
    """
    Register moving->fixed using independent tiles with optional rotation.

    Features:
      - z-score normalize fixed/moving (global)
      - Otsu foreground masks; skip tiles with little foreground
      - confidence gating (score + peak ratio); fallback to identity placement
      - smooth displacement field across tiles to discourage drifting
    """
    if fixed_yx.ndim != 2 or moving_yx.ndim != 2:
        raise ValueError("estimate_tiled_rigid_transforms expects 2D arrays")

    def _check_cancel_local() -> None:
        try:
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Cancelled")
        except RuntimeError:
            raise
        except Exception:
            return

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

    def _otsu_mask(img: np.ndarray) -> tuple[np.ndarray, float]:
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
        return (v > thr), thr

    def _legacy_zero_mask_score_map(
        fixed_win: np.ndarray,
        fixed_mask_win: np.ndarray,
        tile_r: np.ndarray,
        tile_mask_r: np.ndarray,
    ) -> np.ndarray:
        """Legacy scoring path: zero masked background, then score using all pixels."""
        return match_template(
            fixed_win * fixed_mask_win.astype(np.float32, copy=False),
            tile_r * tile_mask_r.astype(np.float32, copy=False),
            pad_input=False,
        )

    def _sample_mask_points(mask_bool: np.ndarray, max_points: int = 256) -> np.ndarray:
        coords = np.argwhere(mask_bool)
        if coords.shape[0] <= max_points:
            return coords.astype(np.int32, copy=False)
        # Deterministic subsampling across the foreground support.
        idx = np.linspace(0, coords.shape[0] - 1, num=max_points, dtype=np.int64)
        return coords[idx].astype(np.int32, copy=False)

    def _candidate_offsets_from_mask_points(
        fixed_mask_win: np.ndarray,
        tile_mask_r: np.ndarray,
        *,
        out_h: int,
        out_w: int,
        expected_top: tuple[int, int] | None = None,
        max_candidates: int = 100,
        max_fixed_points: int = 384,
        max_moving_points: int = 256,
    ) -> np.ndarray:
        fm_pts = _sample_mask_points(np.asarray(fixed_mask_win > 0.5, dtype=bool), max_points=max_fixed_points)
        tm_pts = _sample_mask_points(np.asarray(tile_mask_r > 0.5, dtype=bool), max_points=max_moving_points)
        if fm_pts.size == 0 or tm_pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        # Translation voting from sparse foreground point pairs.
        # Each vote corresponds to a candidate tile top-left offset (iy, ix)
        # within the fixed search window.
        dy = fm_pts[:, None, 0].astype(np.int32) - tm_pts[None, :, 0].astype(np.int32)
        dx = fm_pts[:, None, 1].astype(np.int32) - tm_pts[None, :, 1].astype(np.int32)
        valid = (dy >= 0) & (dy < out_h) & (dx >= 0) & (dx < out_w)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)

        flat = (dy[valid].astype(np.int64) * int(out_w)) + dx[valid].astype(np.int64)
        if flat.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        votes = np.bincount(flat, minlength=int(out_h) * int(out_w))
        nz = np.flatnonzero(votes)
        if nz.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        k = int(min(max_candidates, nz.size))
        if nz.size > k:
            part = np.argpartition(votes[nz], -k)[-k:]
            cand_flat = nz[part]
        else:
            cand_flat = nz

        exp_yx = expected_top if expected_top is not None else (0, 0)

        def _sort_key(flat_idx: int) -> tuple[float, float]:
            iy = int(flat_idx) // int(out_w)
            ix = int(flat_idx) % int(out_w)
            dist2 = float((iy - exp_yx[0]) ** 2 + (ix - exp_yx[1]) ** 2)
            return (float(votes[int(flat_idx)]), -dist2)

        cand_flat = np.array(sorted((int(v) for v in cand_flat), key=_sort_key, reverse=True), dtype=np.int64)
        iy = (cand_flat // int(out_w)).astype(np.int32, copy=False)
        ix = (cand_flat % int(out_w)).astype(np.int32, copy=False)
        return np.stack([iy, ix], axis=1)

    def _masked_overlap_score_map(
        fixed_win: np.ndarray,
        fixed_mask_win: np.ndarray,
        tile_r: np.ndarray,
        tile_mask_r: np.ndarray,
        *,
        expected_top: tuple[int, int] | None = None,
        max_candidates: int = 100,
        min_overlap_frac: float = 0.10,
        min_overlap_px_abs: int = 20,
        displacement_penalty: float = 1e-3,
    ) -> np.ndarray:
        """
        Score sparse-foreground tiles using sparse foreground-point voting to
        generate candidate translations, then exact masked rescoring on only the
        strongest candidates.
        """
        fz = fixed_win.astype(np.float32, copy=False)
        fm_bool = np.asarray(fixed_mask_win > 0.5, dtype=bool)
        tz = tile_r.astype(np.float32, copy=False)
        tm_bool = np.asarray(tile_mask_r > 0.5, dtype=bool)

        out_h = fz.shape[0] - tz.shape[0] + 1
        out_w = fz.shape[1] - tz.shape[1] + 1
        if out_h <= 0 or out_w <= 0:
            return np.empty((0, 0), dtype=np.float32)

        moving_fg_px = int(np.count_nonzero(tm_bool))
        if moving_fg_px <= 0:
            return np.full((out_h, out_w), -np.inf, dtype=np.float32)

        min_overlap_px = max(int(min_overlap_px_abs), int(math.ceil(float(min_overlap_frac) * float(moving_fg_px))))
        score = np.full((out_h, out_w), -np.inf, dtype=np.float32)

        candidates = _candidate_offsets_from_mask_points(
            fm_bool,
            tm_bool,
            out_h=out_h,
            out_w=out_w,
            expected_top=expected_top,
            max_candidates=max_candidates,
        )
        if candidates.size == 0:
            return score

        for iy, ix in candidates:
            iy = int(iy)
            ix = int(ix)
            fixed_patch = fz[iy : iy + tz.shape[0], ix : ix + tz.shape[1]]
            overlap_bool = fm_bool[iy : iy + tz.shape[0], ix : ix + tz.shape[1]] & tm_bool
            ov_px = int(np.count_nonzero(overlap_bool))
            if ov_px < min_overlap_px:
                continue

            fv = fixed_patch[overlap_bool].astype(np.float32, copy=False)
            tv = tz[overlap_bool].astype(np.float32, copy=False)
            if fv.size == 0 or tv.size == 0:
                continue

            fv = fv - float(fv.mean())
            tv = tv - float(tv.mean())
            denom = float(np.linalg.norm(fv) * np.linalg.norm(tv))
            if not np.isfinite(denom) or denom <= 1e-8:
                continue

            corr = float(np.dot(fv, tv) / denom)
            if expected_top is not None and displacement_penalty > 0.0:
                dist = math.hypot(float(iy - expected_top[0]), float(ix - expected_top[1]))
                corr -= float(displacement_penalty) * dist
            score[iy, ix] = np.float32(corr)
        return score

    fixed_z = _zscore(fixed_yx)
    moving_z = _zscore(moving_yx)

    fixed_mask, fixed_thr = _otsu_mask(fixed_yx)
    moving_mask, moving_thr = _otsu_mask(moving_yx)

    _maybe_write_debug_otsu_mask(
        source_path=fixed_source_path,
        img=fixed_yx,
        mask=fixed_mask,
        threshold=fixed_thr,
        role='fixed',
        cycle=fixed_cycle,
        channel_index=fixed_channel_index,
        channel_name=fixed_channel_name,
    )
    _maybe_write_debug_otsu_mask(
        source_path=moving_source_path,
        img=moving_yx,
        mask=moving_mask,
        threshold=moving_thr,
        role='moving',
        cycle=moving_cycle,
        channel_index=moving_channel_index,
        channel_name=moving_channel_name,
    )

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
    overlap_px = _default_tile_overlap_px(tile_size)

    if _STEP1_DEBUG:
        _dbg(
            f"tiled rigid: grid={rows}x{cols} tiles={rows*cols} tile_size={tile_size} "
            f"search_factor={sf:.3f} allow_rotation={bool(allow_rotation)} overlap_px={int(overlap_px)}"
        )
        _dbg(
            "tiled rigid: hybrid scoring enabled; use legacy zero-masked template matching "
            "for tiles with >5% and >200 foreground pixels; otherwise use sparse foreground-point voting "
            "to propose up to 100 candidates, then exact masked rescoring (min overlap 10%, >=20 px, mild distance prior)"
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
    # Windows stability guard:
    # When rotation is enabled, repeated skimage.rotate() + match_template()
    # calls inside worker threads can trigger a native access violation on
    # Windows (observed as a hard crash with no Python traceback).
    #
    # To avoid that crash, force serial execution for the rotation-enabled
    # path on Windows. Translation-only tiled rigid registration remains
    # threaded.
    force_serial = bool(allow_rotation) and (os.name == "nt")
    if force_serial:
        print(f"[WARNING] tiled rigid: forcing serial execution on Windows because allow_rotation=True")
    if force_serial and _STEP1_DEBUG:
        _dbg("tiled rigid: forcing serial execution on Windows because allow_rotation=True")

    max_workers = 1 if force_serial else max(1, (os.cpu_count() or 2) - 1)
    processed = 0

    def _process_one(spec: tuple[int, int, int, int]) -> tuple[tuple[int, int], float, float, float, float, float]:
        # returns (r,c), dy, dx, angle, score ; dy/dx are relative to original tile top-left
        _check_cancel_local()
        y0, x0, h, w = spec
        r = y0 // th
        c = x0 // tw

        tile = moving_z[y0 : y0 + h, x0 : x0 + w]
        tmask = moving_mask[y0 : y0 + h, x0 : x0 + w]
        fg_frac = float(tmask.mean()) if tmask.size else 0.0
        fg_px = int(np.count_nonzero(tmask)) if tmask.size else 0
        use_legacy_zero_mask_scoring = (fg_frac > 0.05) and (fg_px > 200)
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

        best_score = -np.inf
        second_score = -np.inf
        best_angle = 0.0
        best_y = y0
        best_x = x0

        for a in angles:
            _check_cancel_local()
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

            try:
                if use_legacy_zero_mask_scoring:
                    resp = _legacy_zero_mask_score_map(fixed_win, fixed_m, tile_r, m_r)
                else:
                    resp = _masked_overlap_score_map(
                        fixed_win,
                        fixed_m,
                        tile_r,
                        m_r,
                        expected_top=(int(y0 - wy0), int(x0 - wx0)),
                        max_candidates=100,
                        min_overlap_frac=0.10,
                        min_overlap_px_abs=20,
                        displacement_penalty=1e-3,
                    )
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
    if ThreadPoolExecutor is not None and max_workers > 1:
        ex = ThreadPoolExecutor(max_workers=int(max_workers))

    try:
        for r in range(rows):
            _check_cancel_local()
            y0 = int(r * th)
            h = int(min(th, H - y0))
            specs = []
            for c in range(cols):
                x0 = int(c * tw)
                w = int(min(tw, W - x0))
                specs.append((y0, x0, h, w))

            _check_cancel_local()
            if ex is None:
                for spec in specs:
                    _check_cancel_local()
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
                    _check_cancel_local()
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
                if cancel_cb is not None and bool(cancel_cb()):
                    ex.shutdown(wait=False, cancel_futures=True)
                else:
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
            _check_cancel_local()
            changed = False
            for r in range(rows):
                _check_cancel_local()
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
            _check_cancel_local()
            dy_new = dy.copy()
            dx_new = dx.copy()
            ang_new = ang.copy()

            for r in range(rows):
                _check_cancel_local()
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

            dy, dx = _limit_neighbor_offsets(dy_new, dx_new, wt, max_offset=float(overlap_px), n_passes=2)
            ang = ang_new

    # Convert grid to per-tile transforms
    out: list[TileTransform] = []
    for r in range(rows):
        _check_cancel_local()
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
                    overlap_px=int(overlap_px),
                )
            )


    return out



def apply_tiled_rigid_to_plane(
    plane_yx: np.ndarray,
    *,
    canvas_yx: tuple[int, int],
    transforms: list[TileTransform],
    cancel_cb: Callable[[], bool] | None = None,
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
    - Core registered tile regions are accumulated/averaged exactly as before.
    - A second pass uses a small symmetric overlap margin to back-fill true gaps
      created by neighboring tile translations after smoothing.
    """
    H, W = int(canvas_yx[0]), int(canvas_yx[1])
    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)

    if plane_yx.dtype != np.float32:
        plane = plane_yx.astype(np.float32, copy=False)
    else:
        plane = plane_yx

    def _check_cancel_local() -> None:
        try:
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Cancelled")
        except RuntimeError:
            raise
        except Exception:
            return

    def _render_tile(
        t: TileTransform,
        *,
        expand: int = 0,
    ) -> tuple[np.ndarray, int, int] | None:
        y0, x0, h, w = int(t.y0), int(t.x0), int(t.h), int(t.w)
        if h <= 0 or w <= 0:
            return None

        exp = max(0, int(expand))
        top_exp = min(exp, max(0, y0))
        left_exp = min(exp, max(0, x0))
        bot_exp = min(exp, max(0, plane.shape[0] - (y0 + h)))
        right_exp = min(exp, max(0, plane.shape[1] - (x0 + w)))

        sy0 = max(0, y0 - top_exp)
        sx0 = max(0, x0 - left_exp)
        sy1 = min(plane.shape[0], y0 + h + bot_exp)
        sx1 = min(plane.shape[1], x0 + w + right_exp)
        if sy1 <= sy0 or sx1 <= sx0:
            return None

        tile = plane[sy0:sy1, sx0:sx1]
        th, tw = tile.shape[0], tile.shape[1]
        if th <= 0 or tw <= 0:
            return None

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

        dy0 = int(t.y_fixed) - top_exp
        dx0 = int(t.x_fixed) - left_exp
        return tile, dy0, dx0

    # Pass 1: place the core registered tile regions exactly as before.
    for t in transforms:
        _check_cancel_local()
        rendered = _render_tile(t, expand=0)
        if rendered is None:
            continue
        tile, dy0, dx0 = rendered

        th, tw = tile.shape[0], tile.shape[1]
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

    # Pass 2: fill true seams using each tile's overlap context, but never overwrite
    # already-covered pixels from the core pass.
    if transforms:
        row_vals = sorted({int(t.y0) for t in transforms})
        col_vals = sorted({int(t.x0) for t in transforms})
        row_index = {v: i for i, v in enumerate(row_vals)}
        col_index = {v: i for i, v in enumerate(col_vals)}
        grid: dict[tuple[int, int], TileTransform] = {}
        for t in transforms:
            grid[(row_index[int(t.y0)], col_index[int(t.x0)])] = t

        for t in transforms:
            _check_cancel_local()
            overlap_px = max(0, int(getattr(t, "overlap_px", 0) or 0))
            if overlap_px <= 0:
                continue

            r = row_index[int(t.y0)]
            c = col_index[int(t.x0)]
            rendered = _render_tile(t, expand=overlap_px)
            if rendered is None:
                continue
            tile_exp, dy0_exp, dx0_exp = rendered

            # Horizontal seam: fill a gap between the tile on the left and this tile.
            if c > 0:
                t_left = grid.get((r, c - 1))
                if t_left is not None:
                    gap = int(round(int(t.x_fixed) - (int(t_left.x_fixed) + int(t_left.w))))
                    if gap > 0:
                        gap_use = min(gap, overlap_px)
                        gx0 = int(t.x_fixed) - gap_use
                        gx1 = int(t.x_fixed)
                        gy0 = int(t.y_fixed)
                        gy1 = int(t.y_fixed) + int(t.h)
                        oy0 = max(0, gy0)
                        ox0 = max(0, gx0)
                        oy1 = min(H, gy1)
                        ox1 = min(W, gx1)
                        if oy1 > oy0 and ox1 > ox0:
                            ty0 = oy0 - dy0_exp
                            tx0 = ox0 - dx0_exp
                            ty1 = ty0 + (oy1 - oy0)
                            tx1 = tx0 + (ox1 - ox0)
                            patch = tile_exp[ty0:ty1, tx0:tx1]
                            empty = (wgt[oy0:oy1, ox0:ox1] <= 0.0)
                            if bool(empty.any()):
                                out_region = out[oy0:oy1, ox0:ox1]
                                wgt_region = wgt[oy0:oy1, ox0:ox1]
                                out_region[empty] += patch[empty]
                                wgt_region[empty] += 1.0

            # Vertical seam: fill a gap between the tile above and this tile.
            if r > 0:
                t_up = grid.get((r - 1, c))
                if t_up is not None:
                    gap = int(round(int(t.y_fixed) - (int(t_up.y_fixed) + int(t_up.h))))
                    if gap > 0:
                        gap_use = min(gap, overlap_px)
                        gy0 = int(t.y_fixed) - gap_use
                        gy1 = int(t.y_fixed)
                        gx0 = int(t.x_fixed)
                        gx1 = int(t.x_fixed) + int(t.w)
                        oy0 = max(0, gy0)
                        ox0 = max(0, gx0)
                        oy1 = min(H, gy1)
                        ox1 = min(W, gx1)
                        if oy1 > oy0 and ox1 > ox0:
                            ty0 = oy0 - dy0_exp
                            tx0 = ox0 - dx0_exp
                            ty1 = ty0 + (oy1 - oy0)
                            tx1 = tx0 + (ox1 - ox0)
                            patch = tile_exp[ty0:ty1, tx0:tx1]
                            empty = (wgt[oy0:oy1, ox0:ox1] <= 0.0)
                            if bool(empty.any()):
                                out_region = out[oy0:oy1, ox0:ox1]
                                wgt_region = wgt[oy0:oy1, ox0:ox1]
                                out_region[empty] += patch[empty]
                                wgt_region[empty] += 1.0

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
    global_translation_only: bool = False,
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
    if bool(global_translation_only):
        reg_alg = "translation"
    _dbg(
        f"merge_cycles_to_ome_tiff: start n_cycles={len(cycles)} low_mem={bool(low_mem)} alg={reg_alg!r} global_translation_only={bool(global_translation_only)} downsample={downsample_for_registration} allow_rotation={bool(tiled_rigid_allow_rotation)} tile_size={int(tiled_rigid_tile_size)} search_factor={float(tiled_rigid_search_factor):.3f}"
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
        #   Optional pyramid conversion: reserve progress equal to roughly one full cycle,
        #       scaled internally across pyramid-build/write steps.
        moving_infos = [t for t in infos if int(t[0].cycle) != int(reference_cycle)]
        cycle_tick_candidates: list[int] = []
        for (_ci, _shp_yxc, _dt) in moving_infos:
            cycle_tick_candidates.append(1 + 1 + int(tile_ticks) + int(_shp_yxc[2]) + 1)
        if not cycle_tick_candidates:
            # Degenerate case: only the reference cycle is present.
            ref_nch = int(infos[0][1][2]) if infos else 1
            cycle_tick_candidates.append(1 + int(tile_ticks) + ref_nch + 1)
        pyramid_alloc_ticks = int(max(1, round(sum(cycle_tick_candidates) / len(cycle_tick_candidates))))
        pyramid_internal_ticks = int(max(1, estimate_pyramid_conversion_ticks(
            (int(total_ch), int(canvas_y), int(canvas_x)),
            min_level_size=int(pyramidal_min_level_size),
            out_chunk=int(pyramid_progress_chunk),
        )))

        total_ticks = 1  # load reference
        for (_ci, _shp_yxc, _dt) in moving_infos:
            total_ticks += 1  # load moving cycle
            total_ticks += 1  # initial transform
            total_ticks += int(tile_ticks)  # tiled registration
            total_ticks += int(_shp_yxc[2])  # apply per channel
        total_ticks += int(n_cycles)  # saving output (1 tick per cycle)
        if bool(pyramidal_output):
            total_ticks += int(pyramid_alloc_ticks)

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
        ref_physical_pixel_sizes = load_physical_pixel_sizes(ref_ci.path)
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
            physical_pixel_sizes=ref_physical_pixel_sizes,
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
                                    cancel_cb=cancel_cb,
                                    fixed_source_path=ref_ci.path if ref_ci is not None else None,
                                    moving_source_path=ci.path,
                                    fixed_cycle=int(ref_ci.cycle) if ref_ci is not None else None,
                                    moving_cycle=int(ci.cycle),
                                    fixed_channel_index=int(ref_ch_idx),
                                    moving_channel_index=int(ch_idx),
                                    fixed_channel_name=ref_marker,
                                    moving_channel_name=marker,
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
                    _check_cancel()
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
                            plane_reg = apply_tiled_rigid_to_plane(plane_init, canvas_yx=canvas_yx, transforms=t, cancel_cb=cancel_cb)  # type: ignore[arg-type]
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
            pyramid_steps_done = 0
            pyramid_ticks_emitted = 0

            def _pstep(msg: str, phase: str) -> None:
                nonlocal completed_ticks, pyramid_steps_done, pyramid_ticks_emitted
                if progress_cb:
                    progress_cb(msg)
                if phase in {"pyramid_build_chunk", "pyramid_write_level"}:
                    pyramid_steps_done = min(int(pyramid_internal_ticks), int(pyramid_steps_done) + 1)
                    scaled = int(math.floor((float(pyramid_steps_done) * float(pyramid_alloc_ticks)) / float(max(1, pyramid_internal_ticks))))
                    if scaled > int(pyramid_ticks_emitted):
                        completed_ticks += int(scaled - int(pyramid_ticks_emitted))
                        completed_ticks = min(int(completed_ticks), int(total_ticks))
                        pyramid_ticks_emitted = int(scaled)
                _emit_tick(msg, phase=phase)

            _out_parent = str(Path(output_path).resolve().parent)
            final_output_path = convert_flat_ome_to_pyramidal(
                output_path,
                output_path=None,
                channel_names=merged_names,
                physical_pixel_sizes=ref_physical_pixel_sizes,
                tile_size=int(pyramidal_tile_size),
                compression=pyramidal_compression,
                min_level_size=int(pyramidal_min_level_size),
                out_chunk=int(pyramid_progress_chunk),
                replace_source=False,
                temp_dir=_out_parent,
                progress_cb=progress_cb,
                progress_step_cb=_pstep,
                cancel_cb=cancel_cb,
            )
            try:
                os.replace(final_output_path, output_path)
            except OSError as e:
                raise OSError(
                    e.errno,
                    f'Failed to replace flat OME-TIFF with pyramidal output at {output_path!r}. '
                    f'Temporary pyramid file was {final_output_path!r}. '
                    'This usually means the output file is open in another program or the temporary file was created on a different volume.',
                ) from e
            finally:
                try:
                    shutil.rmtree(Path(final_output_path).parent, ignore_errors=True)
                except Exception:
                    pass
            try:
                pyout = inspect_tiff_pyramid(output_path)
            except Exception:
                pyout = {"is_pyramidal": True, "series": []}
            if int(pyramid_ticks_emitted) < int(pyramid_alloc_ticks):
                completed_ticks += int(pyramid_alloc_ticks - int(pyramid_ticks_emitted))
                completed_ticks = min(int(completed_ticks), int(total_ticks))
                pyramid_ticks_emitted = int(pyramid_alloc_ticks)
            _emit_tick(f"Wrote pyramidal output: {output_path}", phase="pyramid_done")

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
    ref_physical_pixel_sizes = load_physical_pixel_sizes(ref.path)
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

    save_ome_tiff_yxc(output_path, merged, merged_names, physical_pixel_sizes=ref_physical_pixel_sizes)
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
            physical_pixel_sizes=ref_physical_pixel_sizes,
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
        try:
            shutil.rmtree(Path(convert_path).parent, ignore_errors=True)
        except Exception:
            pass

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
