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
class DiagnosticRun:
    sample_name: str
    output_path: Path | None
    output_dir_hint: Path
    ref_ci: CycleInput
    mov_ci: CycleInput
    ref_output_ch: int
    mov_output_ch: int
    tiled_rigid_tile_size: int
    tiled_rigid_search_factor: float
    elastic_touchup_tile_size: int
    elastic_touchup_skip_corr: float
    elastic_touchup_bspline_spacing: int
    elastic_touchup_max_iterations: int
    elastic_touchup_rigid_max_shift: float
    strip_height: int | None
    low_mem: bool


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


@dataclass(frozen=True)
class TopBandTileRecord:
    y0: int
    y1: int
    x0: int
    x1: int
    fg_pixels: int
    corr_final_output: float | None
    corr_reconstructed_rigid: float
    best_bound: float
    best_dy: float
    best_dx: float
    best_residual_mag: float
    best_corr_local_rigid: float
    best_gain_vs_rigid: float
    best_gain_vs_final: float | None
    hit_128_bound: bool
    improves_beyond_128: bool
    would_skip_elastic: bool
    classification: str


@dataclass(frozen=True)
class BoundSweepRecord:
    y0: int
    x0: int
    bound: float
    dy: float
    dx: float
    residual_mag: float
    corr_local_rigid: float
    gain_vs_rigid: float
    hit_bound: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose CycIF preprocess elastic touch-up by scoring local residual "
            "registration on foreground tiles."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Batch plan JSON or .cyseg-registration-progress.json sidecar used for the run.",
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
        "--tile-size",
        type=int,
        default=None,
        help="Full-resolution diagnostic tile size.",
    )
    parser.add_argument(
        "--analysis-stride",
        "--stride",
        type=int,
        default=None,
        help="Full-resolution diagnostic tile stride.",
    )
    parser.add_argument(
        "--y-range",
        default=None,
        help="Full-resolution Y range as start:end, e.g. 0:12000. Defaults to full height.",
    )
    parser.add_argument(
        "--x-range",
        default=None,
        help="Full-resolution X range as start:end or 'auto'. Defaults to full width.",
    )
    parser.add_argument(
        "--rigid-bound-sweep",
        default="128,256,512,1024",
        help="Comma-separated local rigid bounds to test in full-resolution pixels.",
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


def _parse_int_range(text: str | None, max_value: int) -> tuple[int, int]:
    if text is None or str(text).strip().lower() in {"", "auto", "all"}:
        return 0, int(max_value)
    raw = str(text).strip()
    if ":" not in raw:
        raise ValueError(f"Range must be start:end, got {raw!r}")
    a, b = raw.split(":", 1)
    start = int(a) if a.strip() else 0
    end = int(b) if b.strip() else int(max_value)
    start = max(0, min(int(max_value), start))
    end = max(start, min(int(max_value), end))
    return start, end


def _parse_float_list(text: str) -> list[float]:
    values = [float(v.strip()) for v in str(text or "").split(",") if v.strip()]
    if not values:
        raise ValueError("At least one rigid bound is required")
    return sorted({float(v) for v in values})


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


def _run_from_batch_plan(config_path: Path, sample_name: str | None, reference_cycle: int | None, moving_cycle: int | None) -> DiagnosticRun:
    plan_d = json.loads(config_path.read_text(encoding="utf-8"))
    plan = plan_from_dict(plan_d)
    sample = _select_sample(plan, sample_name)
    ref_ci, mov_ci = _select_cycles(sample, reference_cycle, moving_cycle)
    ref_info = inspect_tiff_yxc(ref_ci.path)
    ref_shape = tuple(int(v) for v in ref_info["shape_yxc"])
    output_dir_hint = Path(str(plan_d.get("output_dir") or (sample.output_path.parent if sample.output_path else config_path.parent)))
    return DiagnosticRun(
        sample_name=str(sample.name),
        output_path=Path(str(sample.output_path)) if sample.output_path else None,
        output_dir_hint=output_dir_hint,
        ref_ci=ref_ci,
        mov_ci=mov_ci,
        ref_output_ch=0,
        mov_output_ch=int(ref_shape[2]),
        tiled_rigid_tile_size=int(getattr(sample, "tiled_rigid_tile_size", 2000) or 2000),
        tiled_rigid_search_factor=float(getattr(sample, "tiled_rigid_search_factor", 3.0) or 3.0),
        elastic_touchup_tile_size=int(getattr(sample, "elastic_touchup_tile_size", 2048) or 2048),
        elastic_touchup_skip_corr=float(getattr(sample, "elastic_touchup_skip_corr", 0.95) or 0.95),
        elastic_touchup_bspline_spacing=int(getattr(sample, "elastic_touchup_bspline_spacing", 50) or 50),
        elastic_touchup_max_iterations=int(getattr(sample, "elastic_touchup_max_iterations", 10) or 10),
        elastic_touchup_rigid_max_shift=float(getattr(sample, "elastic_touchup_rigid_max_shift", 512.0) or 512.0),
        strip_height=int(getattr(sample, "strip_height", 0) or 0) or None,
        low_mem=bool(getattr(sample, "low_mem", True)),
    )


def _run_from_sidecar(config_path: Path, reference_cycle: int | None, moving_cycle: int | None) -> DiagnosticRun:
    sidecar = json.loads(config_path.read_text(encoding="utf-8"))
    fp = sidecar.get("fingerprint") or {}
    inputs = list(fp.get("inputs") or [])
    if len(inputs) < 2:
        raise ValueError("Sidecar fingerprint must contain at least two input records")
    ref_cycle = int(reference_cycle if reference_cycle is not None else fp.get("reference_cycle", inputs[0].get("cycle", 0)))
    ref_rec = next((rec for rec in inputs if int(rec.get("cycle")) == ref_cycle), None)
    if ref_rec is None:
        raise ValueError(f"Reference cycle {ref_cycle} not found in sidecar inputs")
    mov_candidates = [rec for rec in inputs if int(rec.get("cycle")) != ref_cycle]
    if moving_cycle is not None:
        mov_rec = next((rec for rec in mov_candidates if int(rec.get("cycle")) == int(moving_cycle)), None)
        if mov_rec is None:
            raise ValueError(f"Moving cycle {moving_cycle} not found in sidecar inputs")
    else:
        mov_rec = mov_candidates[0]

    def _ci(rec: dict) -> CycleInput:
        return CycleInput(
            path=str(rec.get("path") or ""),
            cycle=int(rec.get("cycle") or 0),
            label=str(rec.get("label") if rec.get("label") is not None else rec.get("cycle")),
            registration_marker=str(rec.get("registration_marker") or "DAPI"),
            channel_markers=list(rec.get("channel_markers") or []),
        )

    ref_shape = tuple(int(v) for v in (ref_rec.get("shape_yxc") or inspect_tiff_yxc(str(ref_rec.get("path")))["shape_yxc"]))
    channel_offsets = {int(k): int(v) for k, v in (fp.get("channel_offsets") or {}).items()}
    ref_output_ch = int(channel_offsets.get(int(ref_rec.get("cycle")), 0))
    mov_output_ch = int(channel_offsets.get(int(mov_rec.get("cycle")), ref_output_ch + int(ref_shape[2])))
    output_path = Path(str(sidecar.get("output_path") or ""))
    sample_name = output_path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if sample_name.lower().endswith(suffix):
            sample_name = sample_name[: -len(suffix)]
            break
    return DiagnosticRun(
        sample_name=sample_name or config_path.stem,
        output_path=output_path if str(output_path) else None,
        output_dir_hint=output_path.parent if str(output_path) else config_path.parent,
        ref_ci=_ci(ref_rec),
        mov_ci=_ci(mov_rec),
        ref_output_ch=ref_output_ch,
        mov_output_ch=mov_output_ch,
        tiled_rigid_tile_size=int(fp.get("tiled_rigid_tile_size") or 2000),
        tiled_rigid_search_factor=float(fp.get("tiled_rigid_search_factor") or 3.0),
        elastic_touchup_tile_size=2048,
        elastic_touchup_skip_corr=0.95,
        elastic_touchup_bspline_spacing=50,
        elastic_touchup_max_iterations=10,
        elastic_touchup_rigid_max_shift=float(fp.get("elastic_touchup_rigid_max_shift", 512.0) or 512.0),
        strip_height=int(fp.get("strip_height") or 0) or None,
        low_mem=bool(fp.get("low_mem", True)),
    )


def _load_run(config_path: Path, sample_name: str | None, reference_cycle: int | None, moving_cycle: int | None) -> DiagnosticRun:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "fingerprint" in data and "output_path" in data:
        return _run_from_sidecar(config_path, reference_cycle, moving_cycle)
    return _run_from_batch_plan(config_path, sample_name, reference_cycle, moving_cycle)


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


def _iter_tiles(
    canvas_yx: tuple[int, int],
    tile_size: int,
    stride: int,
    *,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
) -> Iterable[tuple[int, int, int, int]]:
    canvas_h, canvas_w = int(canvas_yx[0]), int(canvas_yx[1])
    tile = max(1, int(tile_size))
    step = max(1, int(stride))
    yr = y_range or (0, canvas_h)
    xr = x_range or (0, canvas_w)
    y_start, y_stop = max(0, int(yr[0])), min(canvas_h, int(yr[1]))
    x_start, x_stop = max(0, int(xr[0])), min(canvas_w, int(xr[1]))
    for y0 in range(y_start, y_stop, step):
        y1 = min(y_stop, y0 + tile)
        if y1 <= y0:
            continue
        for x0 in range(x_start, x_stop, step):
            x1 = min(x_stop, x0 + tile)
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


def _write_dataclass_csv(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: getattr(rec, k) for k in fieldnames})


def _classify_tile(
    *,
    corr_final: float | None,
    corr_rigid: float,
    best_corr: float,
    hit_128_bound: bool,
    improves_beyond_128: bool,
    would_skip: bool,
) -> str:
    if corr_final is not None and corr_final >= 0.8:
        return "well_registered"
    if hit_128_bound or improves_beyond_128:
        return "rigid_bound_limited"
    if would_skip and corr_final is not None and corr_final < 0.6:
        return "skipped_incorrectly_suspect"
    if best_corr < corr_rigid + 0.05:
        return "upstream_rigid_or_low_texture_suspect"
    return "elastic_or_blending_suspect"


def _write_top_band_heatmap(path: Path, records: list[TopBandTileRecord], canvas_yx: tuple[int, int], stride: int) -> None:
    h = max(1, int(math.ceil(canvas_yx[0] / max(1, stride))))
    w = max(1, int(math.ceil(canvas_yx[1] / max(1, stride))))
    mag = np.full((h, w), np.nan, dtype=np.float32)
    gain = np.full((h, w), np.nan, dtype=np.float32)
    bound = np.full((h, w), np.nan, dtype=np.float32)
    for rec in records:
        yy = min(h - 1, max(0, int(rec.y0 // max(1, stride))))
        xx = min(w - 1, max(0, int(rec.x0 // max(1, stride))))
        mag[yy, xx] = float(rec.best_residual_mag)
        gain[yy, xx] = float(rec.best_gain_vs_rigid)
        bound[yy, xx] = float(rec.best_bound)
    rgb = np.stack([
        _stretch_u8(np.nan_to_num(mag, nan=0.0)),
        _stretch_u8(np.maximum(np.nan_to_num(gain, nan=0.0), 0.0)),
        _stretch_u8(np.nan_to_num(bound, nan=0.0)),
    ], axis=2)
    _save_rgb(path, rgb)


def _write_adjacent_seam_scores(path: Path, records: list[TopBandTileRecord], stride: int) -> list[dict[str, float | int | str]]:
    by_pos = {(r.y0, r.x0): r for r in records}
    rows: list[dict[str, float | int | str]] = []
    for rec in records:
        for direction, key in (("right", (rec.y0, rec.x0 + stride)), ("down", (rec.y0 + stride, rec.x0))):
            other = by_pos.get(key)
            if other is None:
                continue
            shift_jump = float(math.hypot(rec.best_dy - other.best_dy, rec.best_dx - other.best_dx))
            final_drop = None
            if rec.corr_final_output is not None and other.corr_final_output is not None:
                final_drop = float(abs(rec.corr_final_output - other.corr_final_output))
            rows.append({
                "y0": int(rec.y0),
                "x0": int(rec.x0),
                "neighbor_y0": int(other.y0),
                "neighbor_x0": int(other.x0),
                "direction": direction,
                "shift_jump": shift_jump,
                "final_corr_delta": final_drop if final_drop is not None else "",
                "seam_suspect": bool(shift_jump >= 64.0 and (final_drop is None or final_drop >= 0.15)),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["y0", "x0", "neighbor_y0", "neighbor_x0", "direction", "shift_jump", "final_corr_delta", "seam_suspect"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


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
    run = _load_run(args.config_path, args.sample, args.reference_cycle, args.moving_cycle)
    out_dir = args.output_dir or (run.output_dir_hint / f"{run.sample_name}_elastic_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_ci, mov_ci = run.ref_ci, run.mov_ci
    ref_info = inspect_tiff_yxc(ref_ci.path)
    mov_info = inspect_tiff_yxc(mov_ci.path)
    ref_shape = tuple(int(v) for v in ref_info["shape_yxc"])
    mov_shape = tuple(int(v) for v in mov_info["shape_yxc"])
    canvas_yx = (max(ref_shape[0], mov_shape[0]), max(ref_shape[1], mov_shape[1]))
    ref_ch = _resolve_reg_channel(ref_ci, ref_ci.registration_marker or "DAPI")
    mov_ch = _resolve_reg_channel(mov_ci, mov_ci.registration_marker or "DAPI")
    if ref_ch is None or mov_ch is None:
        raise ValueError(f"Could not resolve registration channels: ref={ref_ch}, moving={mov_ch}")

    tile_size = int(args.analysis_tile_size or run.elastic_touchup_tile_size or 2048)
    stride = int(args.analysis_stride or max(1, tile_size // 2))
    y_range = _parse_int_range(args.y_range, canvas_yx[0])
    x_range = _parse_int_range(args.x_range, canvas_yx[1])
    bounds = _parse_float_list(args.rigid_bound_sweep)
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
    tile_sz_ds = max(4, int(run.tiled_rigid_tile_size) // D)
    islands = _identify_foreground_islands(moved_mask, tile_sz_ds, solid=True)
    n_islands = int(np.max(islands)) if islands.size else 0
    search_radius = max(2, max(8, int(round(max(1, int(run.tiled_rigid_tile_size)) * max(1.0, float(run.tiled_rigid_search_factor)) / 4.0))) // D)
    regs_raw = _refine_region_transforms(
        ref_reg,
        mov_reg,
        islands,
        (float(dy_ds), float(dx_ds)),
        search_radius=search_radius,
        downsample=1,
        penalty_lambda=0.0,
        fast_large_island_refinement=False,
        fast_large_island_sample_count=5,
        fast_large_island_cell_size=tile_sz_ds,
        progress_cb=None,
        progress_event_cb=None,
        cancel_cb=None,
        cycle=int(mov_ci.cycle),
    )
    regs_full = [_scale_region_transform(r, D) for r in regs_raw]
    shift_by_label = {int(r.label): (float(r.shift_y), float(r.shift_x)) for r in regs_full}
    base_shift = (dy, dx)
    ref_fg_ds = _foreground_mask(ref_reg)
    has_output = bool(run.output_path and run.output_path.is_file())

    top_records: list[TopBandTileRecord] = []
    sweep_records: list[BoundSweepRecord] = []
    overlay_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}

    for y0, y1, x0, x1 in _iter_tiles(canvas_yx, tile_size, stride, y_range=y_range, x_range=x_range):
        labels_tile = _upsample_labels_to_tile(islands, y0, y1, x0, x1, D)
        fg_tile = _upsample_labels_to_tile(ref_fg_ds.astype(np.int32), y0, y1, x0, x1, D).astype(bool)
        labs, counts = np.unique(labels_tile[labels_tile > 0], return_counts=True)
        if labs.size == 0:
            use_mask = fg_tile
            island_label = 0
        else:
            island_label = int(labs[int(np.argmax(counts))])
            use_mask = fg_tile & (labels_tile == island_label)
        if int(use_mask.sum()) < int(args.min_fg_pixels):
            continue

        rigid_dy, rigid_dx = shift_by_label.get(island_label, base_shift)
        ref_crop = _canvas_channel_roi(ref_ci.path, int(ref_ch), ref_shape[:2], canvas_yx, y0, y1, x0, x1)
        rigid_crop = _translated_moving_tile(mov_ci.path, int(mov_ch), mov_shape[:2], canvas_yx, y0, y1, x0, x1, rigid_dy, rigid_dx)
        corr_rigid = _masked_corr_score(ref_crop, rigid_crop, use_mask)
        corr_final = None
        final_crop = None
        if has_output and run.output_path is not None:
            try:
                out_ref = load_channel_roi(str(run.output_path), int(run.ref_output_ch), y0, y1, x0, x1).astype(np.float32, copy=False)
                final_crop = load_channel_roi(str(run.output_path), int(run.mov_output_ch), y0, y1, x0, x1).astype(np.float32, copy=False)
                corr_final = _masked_corr_score(out_ref, final_crop, use_mask)
            except Exception:
                corr_final = None
                final_crop = None

        best = None
        best_crop = rigid_crop
        corr_at_128 = None
        for bound in bounds:
            rdy, rdx, _score = _estimate_masked_residual_shift(ref_crop, rigid_crop, use_mask, max_shift=float(bound))
            rdy = float(np.clip(rdy, -float(bound), float(bound)))
            rdx = float(np.clip(rdx, -float(bound), float(bound)))
            cand_crop = _translated_moving_tile(mov_ci.path, int(mov_ch), mov_shape[:2], canvas_yx, y0, y1, x0, x1, rigid_dy + rdy, rigid_dx + rdx)
            corr_candidate = _masked_corr_score(ref_crop, cand_crop, use_mask)
            mag = float(math.hypot(rdy, rdx))
            hit_bound = bool(abs(abs(rdy) - float(bound)) <= 1e-6 or abs(abs(rdx) - float(bound)) <= 1e-6)
            sweep_records.append(BoundSweepRecord(
                y0=int(y0),
                x0=int(x0),
                bound=float(bound),
                dy=float(rdy),
                dx=float(rdx),
                residual_mag=mag,
                corr_local_rigid=float(corr_candidate),
                gain_vs_rigid=float(corr_candidate - corr_rigid),
                hit_bound=hit_bound,
            ))
            if int(bound) == 128:
                corr_at_128 = float(corr_candidate)
            if best is None or corr_candidate > best[0]:
                best = (float(corr_candidate), float(bound), float(rdy), float(rdx), mag, hit_bound)
                best_crop = cand_crop

        if best is None:
            continue
        best_corr, best_bound, best_dy, best_dx, best_mag, best_hit = best
        hit_128 = any(r.y0 == y0 and r.x0 == x0 and int(r.bound) == 128 and r.hit_bound for r in sweep_records)
        improves_beyond_128 = bool(corr_at_128 is not None and best_bound > 128.0 and best_corr >= corr_at_128 + 0.05)
        would_skip = bool(corr_rigid >= float(run.elastic_touchup_skip_corr))
        classification = _classify_tile(
            corr_final=corr_final,
            corr_rigid=float(corr_rigid),
            best_corr=float(best_corr),
            hit_128_bound=hit_128,
            improves_beyond_128=improves_beyond_128,
            would_skip=would_skip,
        )
        rec = TopBandTileRecord(
            y0=int(y0),
            y1=int(y1),
            x0=int(x0),
            x1=int(x1),
            fg_pixels=int(use_mask.sum()),
            corr_final_output=corr_final,
            corr_reconstructed_rigid=float(corr_rigid),
            best_bound=float(best_bound),
            best_dy=float(best_dy),
            best_dx=float(best_dx),
            best_residual_mag=float(best_mag),
            best_corr_local_rigid=float(best_corr),
            best_gain_vs_rigid=float(best_corr - corr_rigid),
            best_gain_vs_final=float(best_corr - corr_final) if corr_final is not None else None,
            hit_128_bound=bool(hit_128),
            improves_beyond_128=bool(improves_beyond_128),
            would_skip_elastic=bool(would_skip),
            classification=classification,
        )
        top_records.append(rec)
        overlay_cache[(rec.y0, rec.x0)] = (ref_crop, rigid_crop, best_crop, final_crop)
        if args.limit_tiles and len(top_records) >= int(args.limit_tiles):
            break

    top_records.sort(key=lambda r: (r.classification != "rigid_bound_limited", -(r.best_gain_vs_final if r.best_gain_vs_final is not None else r.best_gain_vs_rigid), -r.best_residual_mag))
    _write_dataclass_csv(out_dir / "tile_residuals_top_band.csv", top_records)
    _write_dataclass_csv(out_dir / "rigid_bound_sweep.csv", sweep_records)
    seam_rows = _write_adjacent_seam_scores(out_dir / "adjacent_tile_seam_scores.csv", top_records, stride)
    _write_top_band_heatmap(out_dir / "residual_bound_heatmap.png", top_records, canvas_yx, stride)
    _write_sweep_candidates(out_dir / "settings_sweep_candidates.csv")

    for i, rec in enumerate(top_records[:max(0, int(args.top_overlays))], start=1):
        cached = overlay_cache.get((rec.y0, rec.x0))
        if cached is None:
            continue
        ref_crop, rigid_crop, best_crop, final_crop = cached
        info = {
            "tile": f"{rec.y0}:{rec.y1},{rec.x0}:{rec.x1}",
            "classification": rec.classification,
            "best_bound": rec.best_bound,
            "best_shift": f"{rec.best_dy:.1f},{rec.best_dx:.1f}",
            "corr_rigid": f"{rec.corr_reconstructed_rigid:.4f}",
            "corr_best": f"{rec.best_corr_local_rigid:.4f}",
            "corr_final": f"{rec.corr_final_output:.4f}" if rec.corr_final_output is not None else "",
        }
        _write_overlay(out_dir / "overlays" / f"{i:03d}_reconstructed_rigid_y{rec.y0}_x{rec.x0}.png", ref_crop, rigid_crop, info)
        _write_overlay(out_dir / "overlays" / f"{i:03d}_best_local_rigid_y{rec.y0}_x{rec.x0}.png", ref_crop, best_crop, info)
        if final_crop is not None:
            _write_overlay(out_dir / "overlays" / f"{i:03d}_current_output_y{rec.y0}_x{rec.x0}.png", ref_crop, final_crop, info)

    class_counts: dict[str, int] = {}
    for rec in top_records:
        class_counts[rec.classification] = class_counts.get(rec.classification, 0) + 1
    summary = {
        "sample": run.sample_name,
        "config_path": str(args.config_path),
        "reference_cycle": int(ref_ci.cycle),
        "moving_cycle": int(mov_ci.cycle),
        "reference_channel": int(ref_ch),
        "moving_channel": int(mov_ch),
        "canvas_yx": list(canvas_yx),
        "analysis_y_range": list(y_range),
        "analysis_x_range": list(x_range),
        "analysis_tile_size": int(tile_size),
        "analysis_stride": int(stride),
        "rigid_bound_sweep": [float(v) for v in bounds],
        "downsample_for_registration": int(D),
        "global_shift_fullres_yx": [float(dy), float(dx)],
        "n_islands": int(n_islands),
        "n_region_transforms": int(len(regs_full)),
        "n_tiles_scored": int(len(top_records)),
        "n_seam_pairs": int(len(seam_rows)),
        "n_seam_suspect_pairs": int(sum(1 for r in seam_rows if bool(r.get("seam_suspect")))),
        "classification_counts": class_counts,
        "output_found": bool(has_output),
        "settings": {
            "tiled_rigid_tile_size": int(run.tiled_rigid_tile_size),
            "tiled_rigid_search_factor": float(run.tiled_rigid_search_factor),
            "elastic_touchup_tile_size": int(run.elastic_touchup_tile_size),
            "elastic_touchup_skip_corr": float(run.elastic_touchup_skip_corr),
            "elastic_touchup_rigid_max_shift": float(run.elastic_touchup_rigid_max_shift),
            "strip_height": int(run.strip_height or 0),
            "low_mem": bool(run.low_mem),
        },
        "worst_tiles": [rec.__dict__ for rec in top_records[:10]],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
