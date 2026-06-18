from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cycif_seg.io.ome_tiff import (
    inspect_tiff_yxc,
    load_channel_downsampled,
    load_channel_roi,
)
from cycif_seg.preprocess.batch_plan import BatchSample, plan_from_dict
from cycif_seg.preprocess.organize_cycles import (
    CycleInput,
    _apply_translation,
    _bbox_from_mask,
    _downsample_image,
    _foreground_mask,
    _identify_foreground_islands,
    _masked_corr_score,
    _normalized_for_registration,
    _refine_region_transforms,
    _resolve_reg_channel,
    _scale_region_transform,
    estimate_translation,
)


@dataclass(frozen=True)
class TileRecord:
    y0: int
    y1: int
    x0: int
    x1: int
    island_label: int
    fg_pixels: int
    rigid_dy: float
    rigid_dx: float
    residual_dy: float
    residual_dx: float
    residual_mag: float
    corr_rigid: float
    corr_elastic: float | None
    corr_local_rigid_candidate: float
    elastic_gain: float | None
    local_rigid_gain: float
    would_skip_elastic: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose CycIF preprocess elastic touch-up by scoring local residual "
            "registration on foreground tiles."
        )
    )
    parser.add_argument(
        "batch_plan",
        type=Path,
        help="Batch plan JSON used for the run.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Sample name to inspect. Defaults to the first enabled sample.",
    )
    parser.add_argument(
        "--moving-cycle",
        type=int,
        default=None,
        help="Moving cycle to diagnose. Defaults to the first enabled non-reference cycle.",
    )
    parser.add_argument(
        "--reference-cycle",
        type=int,
        default=None,
        help="Reference cycle. Defaults to the first enabled cycle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic artifacts. Defaults to <plan output_dir>/elastic_diagnostics.",
    )
    parser.add_argument(
        "--analysis-tile-size",
        type=int,
        default=1024,
        help="Full-resolution diagnostic tile size.",
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=512,
        help="Full-resolution diagnostic tile stride.",
    )
    parser.add_argument(
        "--downsample-for-registration",
        type=int,
        default=4,
        help="Registration downsample factor used by preprocess.",
    )
    parser.add_argument(
        "--max-local-residual",
        type=float,
        default=80.0,
        help="Clip local residual estimates to this many full-resolution pixels per axis.",
    )
    parser.add_argument(
        "--min-fg-pixels",
        type=int,
        default=200,
        help="Minimum foreground pixels required to score a tile.",
    )
    parser.add_argument(
        "--top-overlays",
        type=int,
        default=24,
        help="Number of worst tiles to write as RGB overlay images.",
    )
    parser.add_argument(
        "--limit-tiles",
        type=int,
        default=0,
        help="Optional maximum number of scored tiles for smoke tests.",
    )
    return parser.parse_args()


def _enabled_cycle_indices(sample: BatchSample) -> list[int]:
    return [
        i
        for i, _path in enumerate(sample.files or [])
        if bool(sample.cycles_enabled[i] if sample.cycles_enabled and i < len(sample.cycles_enabled) else True)
    ]


def _cycle_input(sample: BatchSample, index: int) -> CycleInput:
    cycles = sample.cycles or []
    markers = sample.registration_markers or []
    ch_markers = sample.channel_markers or []
    ch_abs = sample.channel_antibodies or []
    return CycleInput(
        path=str(sample.files[index]),
        cycle=int(cycles[index] if index < len(cycles) else index),
        label=str(cycles[index]) if index < len(cycles) else str(index),
        tissue=sample.tissue,
        species=sample.species,
        registration_marker=str(markers[index]) if index < len(markers) else None,
        channel_markers=list(ch_markers[index]) if index < len(ch_markers) else None,
        channel_antibodies=list(ch_abs[index]) if index < len(ch_abs) else None,
    )


def _select_sample(plan: dict, sample_name: str | None) -> BatchSample:
    samples = [s for s in plan["samples"] if bool(getattr(s, "enabled", True))]
    if sample_name:
        matches = [s for s in samples if s.name == sample_name]
        if not matches:
            raise ValueError(f"No enabled sample named {sample_name!r}")
        return matches[0]
    if not samples:
        raise ValueError("No enabled samples in plan")
    return samples[0]


def _select_cycles(
    sample: BatchSample,
    reference_cycle: int | None,
    moving_cycle: int | None,
) -> tuple[CycleInput, CycleInput]:
    enabled = _enabled_cycle_indices(sample)
    if len(enabled) < 2:
        raise ValueError("Need at least two enabled cycles to diagnose registration")
    inputs = [_cycle_input(sample, idx) for idx in enabled]
    ref = next((ci for ci in inputs if int(ci.cycle) == int(reference_cycle)), inputs[0]) if reference_cycle is not None else inputs[0]
    mov_candidates = [ci for ci in inputs if int(ci.cycle) != int(ref.cycle)]
    if moving_cycle is not None:
        mov = next((ci for ci in mov_candidates if int(ci.cycle) == int(moving_cycle)), None)
        if mov is None:
            raise ValueError(f"Moving cycle {moving_cycle} is not an enabled non-reference cycle")
    else:
        mov = mov_candidates[0]
    return ref, mov


def _pad_plane_to_canvas(plane_yx: np.ndarray, canvas_yx: tuple[int, int]) -> np.ndarray:
    y, x = int(plane_yx.shape[0]), int(plane_yx.shape[1])
    Y, X = int(canvas_yx[0]), int(canvas_yx[1])
    out = np.zeros((Y, X), dtype=np.float32)
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0:y0 + ys, x0:x0 + xs] = plane_yx[in_y0:in_y0 + ys, in_x0:in_x0 + xs].astype(np.float32, copy=False)
    return out


def _canvas_channel_roi(
    path: str,
    ch: int,
    src_shape_yx: tuple[int, int],
    canvas_yx: tuple[int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    h = max(0, int(y1) - int(y0))
    w = max(0, int(x1) - int(x0))
    out = np.zeros((h, w), dtype=np.float32)
    if h <= 0 or w <= 0:
        return out
    canvas_h, canvas_w = int(canvas_yx[0]), int(canvas_yx[1])
    src_h, src_w = int(src_shape_yx[0]), int(src_shape_yx[1])
    pad_y0 = max((canvas_h - src_h) // 2, 0)
    pad_x0 = max((canvas_w - src_w) // 2, 0)
    in_y0 = max((src_h - canvas_h) // 2, 0)
    in_x0 = max((src_w - canvas_w) // 2, 0)
    ys = min(src_h - in_y0, canvas_h - pad_y0)
    xs = min(src_w - in_x0, canvas_w - pad_x0)
    ov_y0 = max(int(y0), pad_y0, 0)
    ov_y1 = min(int(y1), pad_y0 + ys, canvas_h)
    ov_x0 = max(int(x0), pad_x0, 0)
    ov_x1 = min(int(x1), pad_x0 + xs, canvas_w)
    if ov_y0 >= ov_y1 or ov_x0 >= ov_x1:
        return out
    fy0 = in_y0 + (ov_y0 - pad_y0)
    fy1 = in_y0 + (ov_y1 - pad_y0)
    fx0 = in_x0 + (ov_x0 - pad_x0)
    fx1 = in_x0 + (ov_x1 - pad_x0)
    crop = load_channel_roi(path, ch, fy0, fy1, fx0, fx1).astype(np.float32, copy=False)
    oy0 = ov_y0 - int(y0)
    ox0 = ov_x0 - int(x0)
    out[oy0:oy0 + crop.shape[0], ox0:ox0 + crop.shape[1]] = crop
    return out


def _translated_moving_tile(
    path: str,
    ch: int,
    src_shape_yx: tuple[int, int],
    canvas_yx: tuple[int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    dy: float,
    dx: float,
) -> np.ndarray:
    iy = int(round(float(dy)))
    ix = int(round(float(dx)))
    fy = float(dy) - iy
    fx = float(dx) - ix
    pad = int(max(4, math.ceil(max(abs(fy), abs(fx))) + 2))
    wy0 = int(y0) - iy - pad
    wy1 = int(y1) - iy + pad
    wx0 = int(x0) - ix - pad
    wx1 = int(x1) - ix + pad
    padded = _canvas_channel_roi(path, ch, src_shape_yx, canvas_yx, wy0, wy1, wx0, wx1)
    crop = _apply_translation(
        padded,
        fy,
        fx,
        order=1,
    )[pad:pad + (int(y1) - int(y0)), pad:pad + (int(x1) - int(x0))]
    return crop.astype(np.float32, copy=False)


def _upsample_labels_to_tile(
    labels_ds: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    downsample: int,
) -> np.ndarray:
    ds = max(1, int(downsample))
    h = max(0, int(y1) - int(y0))
    w = max(0, int(x1) - int(x0))
    out = np.zeros((h, w), dtype=labels_ds.dtype)
    ds_y0 = max(0, int(y0) // ds)
    ds_y1 = min(labels_ds.shape[0], (int(y1) + ds - 1) // ds)
    ds_x0 = max(0, int(x0) // ds)
    ds_x1 = min(labels_ds.shape[1], (int(x1) + ds - 1) // ds)
    if ds_y0 >= ds_y1 or ds_x0 >= ds_x1:
        return out
    full = np.repeat(np.repeat(labels_ds[ds_y0:ds_y1, ds_x0:ds_x1], ds, axis=0), ds, axis=1)
    off_y = int(y0) - ds_y0 * ds
    off_x = int(x0) - ds_x0 * ds
    valid_h = min(h, full.shape[0] - off_y)
    valid_w = min(w, full.shape[1] - off_x)
    if valid_h > 0 and valid_w > 0:
        out[:valid_h, :valid_w] = full[off_y:off_y + valid_h, off_x:off_x + valid_w]
    return out


def _iter_tiles(canvas_yx: tuple[int, int], tile_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    canvas_h, canvas_w = int(canvas_yx[0]), int(canvas_yx[1])
    tile = max(1, int(tile_size))
    step = max(1, int(stride))
    for y0 in range(0, canvas_h, step):
        y1 = min(canvas_h, y0 + tile)
        if y1 <= y0:
            continue
        for x0 in range(0, canvas_w, step):
            x1 = min(canvas_w, x0 + tile)
            if x1 > x0:
                yield y0, y1, x0, x1


def _stretch_u8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(np.round(out * 255.0), dtype=np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(path)
    except Exception:
        import tifffile

        fallback = path.with_suffix(".tiff")
        tifffile.imwrite(str(fallback), rgb, photometric="rgb")


def _write_overlay(
    path: Path,
    ref_crop: np.ndarray,
    moving_crop: np.ndarray,
    title_values: dict[str, float | int | str],
) -> None:
    red = _stretch_u8(moving_crop)
    green = _stretch_u8(ref_crop)
    blue = np.zeros_like(red)
    rgb = np.stack([red, green, blue], axis=2)
    _save_rgb(path, rgb)
    txt = path.with_suffix(".txt")
    txt.write_text(
        "\n".join(f"{k}: {v}" for k, v in title_values.items()) + "\n",
        encoding="utf-8",
    )


def _write_heatmap(path: Path, records: list[TileRecord], canvas_yx: tuple[int, int], stride: int) -> None:
    h = max(1, int(math.ceil(canvas_yx[0] / max(1, stride))))
    w = max(1, int(math.ceil(canvas_yx[1] / max(1, stride))))
    mag = np.full((h, w), np.nan, dtype=np.float32)
    gain = np.full((h, w), np.nan, dtype=np.float32)
    for rec in records:
        yy = min(h - 1, max(0, int(rec.y0 // max(1, stride))))
        xx = min(w - 1, max(0, int(rec.x0 // max(1, stride))))
        mag[yy, xx] = max(float(rec.residual_mag), float(mag[yy, xx]) if np.isfinite(mag[yy, xx]) else -1.0)
        if rec.elastic_gain is not None:
            gain[yy, xx] = float(rec.elastic_gain)
    mag_u8 = _stretch_u8(np.nan_to_num(mag, nan=0.0))
    gain_vis = np.nan_to_num(gain, nan=0.0)
    gain_pos = _stretch_u8(np.maximum(gain_vis, 0.0))
    gain_neg = _stretch_u8(np.maximum(-gain_vis, 0.0))
    rgb = np.stack([mag_u8, gain_pos, gain_neg], axis=2)
    _save_rgb(path, rgb)


def _write_records_csv(path: Path, records: list[TileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TileRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow({k: getattr(rec, k) for k in writer.fieldnames})


def _write_sweep_candidates(path: Path) -> None:
    rows = []
    for tile, spacing, iters, skip in itertools.product(
        [1024, 1536, 2048, 3500],
        [25, 35, 50],
        [10, 25, 50],
        [0.95, 0.98, 0.99],
    ):
        rows.append({
            "elastic_touchup_tile_size": tile,
            "elastic_touchup_bspline_spacing": spacing,
            "elastic_touchup_max_iterations": iters,
            "elastic_touchup_skip_corr": skip,
            "priority": (
                "high"
                if tile in {1024, 1536} and spacing in {25, 35} and iters in {25, 50}
                else "normal"
            ),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _estimate_masked_residual_shift(
    fixed_crop: np.ndarray,
    moving_crop: np.ndarray,
    mask: np.ndarray,
    *,
    max_shift: float,
    target_dim: int = 256,
) -> tuple[float, float, float]:
    h, w = int(fixed_crop.shape[0]), int(fixed_crop.shape[1])
    if h <= 0 or w <= 0 or int(mask.sum()) < 32:
        return 0.0, 0.0, _masked_corr_score(fixed_crop, moving_crop, mask)

    factor = max(1, int(math.ceil(max(h, w) / max(32, int(target_dim)))))
    fixed_ds = _normalized_for_registration(_downsample_image(fixed_crop, factor))
    moving_ds = _normalized_for_registration(_downsample_image(moving_crop, factor))
    mask_f = mask.astype(np.float32, copy=False)
    if factor > 1:
        mask_ds = _downsample_image(mask_f, factor) > 0.2
    else:
        mask_ds = mask.astype(bool, copy=False)
    if int(mask_ds.sum()) < 32:
        mask_ds = np.ones_like(fixed_ds, dtype=bool)

    radius = max(1, int(math.ceil(float(max_shift) / float(factor))))
    step = max(1, int(math.ceil(radius / 8.0)))
    best_dy = 0
    best_dx = 0
    best_score = _masked_corr_score(fixed_ds, moving_ds, mask_ds)

    def _score(dy_i: int, dx_i: int) -> float:
        shifted = _apply_translation(moving_ds, float(dy_i), float(dx_i), order=1)
        return _masked_corr_score(fixed_ds, shifted, mask_ds)

    while True:
        y_min = max(-radius, best_dy - step * 4)
        y_max = min(radius, best_dy + step * 4)
        x_min = max(-radius, best_dx - step * 4)
        x_max = min(radius, best_dx + step * 4)
        improved = False
        for dy_i in range(y_min, y_max + 1, step):
            for dx_i in range(x_min, x_max + 1, step):
                score = _score(dy_i, dx_i)
                if score > best_score + 1e-6:
                    best_score = float(score)
                    best_dy = int(dy_i)
                    best_dx = int(dx_i)
                    improved = True
        if step <= 1:
            break
        step = max(1, step // 2)
        if not improved and step == 1:
            continue

    return float(best_dy * factor), float(best_dx * factor), float(best_score)


def main() -> int:
    args = _parse_args()
    plan_d = json.loads(args.batch_plan.read_text(encoding="utf-8"))
    plan = plan_from_dict(plan_d)
    sample = _select_sample(plan, args.sample)
    ref_ci, mov_ci = _select_cycles(sample, args.reference_cycle, args.moving_cycle)
    out_dir = args.output_dir
    if out_dir is None:
        default_root = plan_d.get("output_dir")
        if not default_root and sample.output_path:
            default_root = sample.output_path.parent
        if not default_root:
            default_root = args.batch_plan.parent
        out_root = Path(str(default_root))
        out_dir = out_root / "elastic_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_info = inspect_tiff_yxc(ref_ci.path)
    mov_info = inspect_tiff_yxc(mov_ci.path)
    ref_shape = tuple(int(v) for v in ref_info["shape_yxc"])
    mov_shape = tuple(int(v) for v in mov_info["shape_yxc"])
    canvas_yx = (max(ref_shape[0], mov_shape[0]), max(ref_shape[1], mov_shape[1]))
    ref_ch = _resolve_reg_channel(ref_ci, ref_ci.registration_marker or "DAPI")
    mov_ch = _resolve_reg_channel(mov_ci, mov_ci.registration_marker or "DAPI")
    if ref_ch is None or mov_ch is None:
        raise ValueError(f"Could not resolve registration channels: ref={ref_ch}, moving={mov_ch}")

    D = max(1, int(args.downsample_for_registration))
    ds_canvas_yx = (max(1, canvas_yx[0] // D), max(1, canvas_yx[1] // D))
    ref_ds = _pad_plane_to_canvas(load_channel_downsampled(ref_ci.path, int(ref_ch), D), ds_canvas_yx)
    mov_ds = _pad_plane_to_canvas(load_channel_downsampled(mov_ci.path, int(mov_ch), D), ds_canvas_yx)
    ref_reg = _normalized_for_registration(ref_ds)
    mov_reg = _normalized_for_registration(mov_ds)

    dy_ds, dx_ds = estimate_translation(ref_reg, mov_reg, downsample=1, upsample_factor=10)
    dy = float(dy_ds) * float(D)
    dx = float(dx_ds) * float(D)
    moving_fg = _foreground_mask(mov_reg)
    moved_mask = _apply_translation(moving_fg.astype(np.float32), dy_ds, dx_ds, order=0) > 0.5
    tile_sz_ds = max(4, int(getattr(sample, "tiled_rigid_tile_size", 2000)) // D)
    islands = _identify_foreground_islands(moved_mask, tile_sz_ds, solid=True)
    n_islands = int(np.max(islands)) if islands.size else 0
    search_radius = max(
        2,
        max(8, int(round(max(1, int(getattr(sample, "tiled_rigid_tile_size", 2000))) * max(1.0, float(getattr(sample, "tiled_rigid_search_factor", 3.0))) / 4.0))) // D,
    )
    regs_raw = _refine_region_transforms(
        ref_reg,
        mov_reg,
        islands,
        (float(dy_ds), float(dx_ds)),
        search_radius=search_radius,
        downsample=1,
        penalty_lambda=0.0,
        fast_large_island_refinement=bool(getattr(sample, "fast_large_island_refinement", False)),
        fast_large_island_sample_count=max(1, int(getattr(sample, "fast_large_island_sample_count", 5))),
        fast_large_island_cell_size=max(4, int(getattr(sample, "tiled_rigid_tile_size", 2000)) // D),
        progress_cb=None,
        progress_event_cb=None,
        cancel_cb=None,
        cycle=int(mov_ci.cycle),
    )
    regs_full = [_scale_region_transform(r, D) for r in regs_raw]
    shift_by_label = {int(r.label): (float(r.shift_y), float(r.shift_x)) for r in regs_full}
    base_shift = (dy, dx)

    ref_fg_ds = _foreground_mask(ref_reg)
    elastic_output_path = Path(str(sample.output_path)) if sample.output_path else None
    has_elastic_output = bool(elastic_output_path and elastic_output_path.is_file())
    ref_output_ch = 0
    mov_output_ch = int(ref_shape[2])
    records: list[TileRecord] = []
    overlay_cache: list[tuple[TileRecord, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

    for y0, y1, x0, x1 in _iter_tiles(canvas_yx, args.analysis_tile_size, args.analysis_stride):
        labels_tile = _upsample_labels_to_tile(islands, y0, y1, x0, x1, D)
        fg_tile = _upsample_labels_to_tile(ref_fg_ds.astype(np.int32), y0, y1, x0, x1, D).astype(bool)
        if int(fg_tile.sum()) < int(args.min_fg_pixels):
            continue
        labs, counts = np.unique(labels_tile[labels_tile > 0], return_counts=True)
        if labs.size == 0:
            continue
        island_label = int(labs[int(np.argmax(counts))])
        island_mask = labels_tile == island_label
        use_mask = fg_tile & island_mask
        if int(use_mask.sum()) < int(args.min_fg_pixels):
            continue
        rigid_dy, rigid_dx = shift_by_label.get(island_label, base_shift)
        ref_crop = _canvas_channel_roi(ref_ci.path, int(ref_ch), ref_shape[:2], canvas_yx, y0, y1, x0, x1)
        rigid_crop = _translated_moving_tile(
            mov_ci.path,
            int(mov_ch),
            mov_shape[:2],
            canvas_yx,
            y0,
            y1,
            x0,
            x1,
            rigid_dy,
            rigid_dx,
        )
        corr_rigid = _masked_corr_score(ref_crop, rigid_crop, use_mask)
        max_resid = float(args.max_local_residual)
        resid_dy, resid_dx, _best_local_score = _estimate_masked_residual_shift(
            ref_crop,
            rigid_crop,
            use_mask,
            max_shift=max_resid,
        )
        resid_dy = float(np.clip(resid_dy, -max_resid, max_resid))
        resid_dx = float(np.clip(resid_dx, -max_resid, max_resid))
        candidate_crop = _translated_moving_tile(
            mov_ci.path,
            int(mov_ch),
            mov_shape[:2],
            canvas_yx,
            y0,
            y1,
            x0,
            x1,
            rigid_dy + resid_dy,
            rigid_dx + resid_dx,
        )
        corr_candidate = _masked_corr_score(ref_crop, candidate_crop, use_mask)
        corr_elastic = None
        elastic_gain = None
        elastic_crop = None
        if has_elastic_output and elastic_output_path is not None:
            try:
                out_ref = load_channel_roi(str(elastic_output_path), ref_output_ch, y0, y1, x0, x1).astype(np.float32, copy=False)
                elastic_crop = load_channel_roi(str(elastic_output_path), mov_output_ch, y0, y1, x0, x1).astype(np.float32, copy=False)
                corr_elastic = _masked_corr_score(out_ref, elastic_crop, use_mask)
                elastic_gain = float(corr_elastic - corr_rigid)
            except Exception:
                corr_elastic = None
                elastic_gain = None
                elastic_crop = None
        rec = TileRecord(
            y0=int(y0),
            y1=int(y1),
            x0=int(x0),
            x1=int(x1),
            island_label=island_label,
            fg_pixels=int(use_mask.sum()),
            rigid_dy=float(rigid_dy),
            rigid_dx=float(rigid_dx),
            residual_dy=float(resid_dy),
            residual_dx=float(resid_dx),
            residual_mag=float(math.hypot(resid_dy, resid_dx)),
            corr_rigid=float(corr_rigid),
            corr_elastic=corr_elastic,
            corr_local_rigid_candidate=float(corr_candidate),
            elastic_gain=elastic_gain,
            local_rigid_gain=float(corr_candidate - corr_rigid),
            would_skip_elastic=bool(corr_rigid >= float(getattr(sample, "elastic_touchup_skip_corr", 0.95))),
        )
        records.append(rec)
        overlay_cache.append((rec, ref_crop, rigid_crop, candidate_crop, elastic_crop))
        if args.limit_tiles and len(records) >= int(args.limit_tiles):
            break

    records.sort(
        key=lambda r: (
            -(r.residual_mag),
            r.corr_elastic if r.corr_elastic is not None else r.corr_rigid,
            -r.local_rigid_gain,
        )
    )
    _write_records_csv(out_dir / "tile_residuals.csv", records)
    _write_heatmap(out_dir / "residual_heatmap.png", records, canvas_yx, args.analysis_stride)
    _write_sweep_candidates(out_dir / "settings_sweep_candidates.csv")

    cached_by_key = {
        (r.y0, r.x0): (r, ref, rigid, candidate, elastic)
        for r, ref, rigid, candidate, elastic in overlay_cache
    }
    for i, rec in enumerate(records[:max(0, int(args.top_overlays))], start=1):
        cached = cached_by_key.get((rec.y0, rec.x0))
        if cached is None:
            continue
        _rec, ref_crop, rigid_crop, candidate_crop, elastic_crop = cached
        _write_overlay(
            out_dir / "overlays" / f"{i:03d}_rigid_y{rec.y0}_x{rec.x0}.png",
            ref_crop,
            rigid_crop,
            {
                "cycle": int(mov_ci.cycle),
                "tile": f"{rec.y0}:{rec.y1},{rec.x0}:{rec.x1}",
                "island": rec.island_label,
                "corr_rigid": f"{rec.corr_rigid:.4f}",
                "residual": f"{rec.residual_dy:.2f},{rec.residual_dx:.2f}",
                "residual_mag": f"{rec.residual_mag:.2f}",
                "corr_local_rigid_candidate": f"{rec.corr_local_rigid_candidate:.4f}",
            },
        )
        _write_overlay(
            out_dir / "overlays" / f"{i:03d}_local_rigid_candidate_y{rec.y0}_x{rec.x0}.png",
            ref_crop,
            candidate_crop,
            {
                "cycle": int(mov_ci.cycle),
                "tile": f"{rec.y0}:{rec.y1},{rec.x0}:{rec.x1}",
                "island": rec.island_label,
                "corr_local_rigid_candidate": f"{rec.corr_local_rigid_candidate:.4f}",
                "local_rigid_gain": f"{rec.local_rigid_gain:.4f}",
                "residual": f"{rec.residual_dy:.2f},{rec.residual_dx:.2f}",
            },
        )
        if elastic_crop is not None:
            _write_overlay(
                out_dir / "overlays" / f"{i:03d}_elastic_y{rec.y0}_x{rec.x0}.png",
                ref_crop,
                elastic_crop,
                {
                    "cycle": int(mov_ci.cycle),
                    "tile": f"{rec.y0}:{rec.y1},{rec.x0}:{rec.x1}",
                    "island": rec.island_label,
                    "corr_elastic": f"{rec.corr_elastic:.4f}" if rec.corr_elastic is not None else "",
                    "elastic_gain": f"{rec.elastic_gain:.4f}" if rec.elastic_gain is not None else "",
                },
            )

    summary = {
        "sample": sample.name,
        "reference_cycle": int(ref_ci.cycle),
        "moving_cycle": int(mov_ci.cycle),
        "reference_channel": int(ref_ch),
        "moving_channel": int(mov_ch),
        "canvas_yx": list(canvas_yx),
        "downsample_for_registration": int(D),
        "global_shift_fullres_yx": [float(dy), float(dx)],
        "n_islands": int(n_islands),
        "n_region_transforms": int(len(regs_full)),
        "n_tiles_scored": int(len(records)),
        "elastic_output_found": bool(has_elastic_output),
        "settings": {
            "tiled_rigid_tile_size": int(getattr(sample, "tiled_rigid_tile_size", 2000)),
            "tiled_rigid_search_factor": float(getattr(sample, "tiled_rigid_search_factor", 3.0)),
            "elastic_touchup_tile_size": int(getattr(sample, "elastic_touchup_tile_size", 2048)),
            "elastic_touchup_skip_corr": float(getattr(sample, "elastic_touchup_skip_corr", 0.95)),
            "elastic_touchup_bspline_spacing": int(getattr(sample, "elastic_touchup_bspline_spacing", 50)),
            "elastic_touchup_max_iterations": int(getattr(sample, "elastic_touchup_max_iterations", 10)),
            "strip_height": int(getattr(sample, "strip_height", 0) or 0),
            "low_mem": bool(getattr(sample, "low_mem", True)),
        },
        "worst_tiles": [
            {
                "y0": r.y0,
                "x0": r.x0,
                "island_label": r.island_label,
                "residual_mag": r.residual_mag,
                "corr_rigid": r.corr_rigid,
                "corr_elastic": r.corr_elastic,
                "corr_local_rigid_candidate": r.corr_local_rigid_candidate,
                "would_skip_elastic": r.would_skip_elastic,
            }
            for r in records[:10]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
