from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from skimage.registration import phase_cross_correlation

try:
    from skimage.filters import threshold_otsu
except Exception:  # pragma: no cover
    threshold_otsu = None

from cycif_seg.io.ome_tiff import (
    IncrementalOmeBigTiffWriter,
    convert_flat_ome_to_pyramidal,
    inspect_tiff_yxc,
    load_physical_pixel_sizes,
    load_single_channel_tiff_native,
)

_DEFAULT_TILE_RE = re.compile(r"^(?:area|raw).*_(\d+)_(\d+)\.ome\.tiff?$", re.IGNORECASE)
_DEFAULT_X_GROUP = 1
_DEFAULT_Y_GROUP = 2


@dataclass(frozen=True)
class TileRecord:
    x: int
    y: int
    path: Path


@dataclass(frozen=True)
class NeighborEstimate:
    axis: str  # 'x' or 'y'
    src: tuple[int, int]
    dst: tuple[int, int]
    dy: float
    dx: float
    overlap_px: float
    score: float
    used_fallback: bool = False
    used_unmasked: bool = False




@dataclass(frozen=True)
class DirectedEdge:
    src: tuple[int, int]
    dst: tuple[int, int]
    dy: float
    dx: float
    weight: float
    overlap_px: float
    score: float
    used_fallback: bool = False
    used_unmasked: bool = False


class RunningOverlap:
    def __init__(self, tile_h: int, tile_w: int, frac: float = 0.05):
        self._x: list[float] = [max(8.0, float(tile_w) * float(frac))]
        self._y: list[float] = [max(8.0, float(tile_h) * float(frac))]

    @property
    def overlap_x(self) -> float:
        return float(sum(self._x) / max(1, len(self._x)))

    @property
    def overlap_y(self) -> float:
        return float(sum(self._y) / max(1, len(self._y)))

    def add(self, axis: str, overlap_px: float) -> None:
        if not np.isfinite(overlap_px):
            return
        if axis == 'x':
            self._x.append(float(overlap_px))
        else:
            self._y.append(float(overlap_px))


def _compile_tile_regex(tile_filename_regex: str | None) -> re.Pattern[str]:
    if tile_filename_regex is None or not str(tile_filename_regex).strip():
        return _DEFAULT_TILE_RE
    return re.compile(str(tile_filename_regex), re.IGNORECASE)


def _parse_tile_xy(
    filename: str,
    *,
    tile_re: re.Pattern[str],
    x_group: int,
    y_group: int,
) -> tuple[int, int] | None:
    m = tile_re.match(filename)
    if not m:
        return None
    try:
        x = int(m.group(int(x_group)))
        y = int(m.group(int(y_group)))
    except Exception:
        return None
    return (x, y)


def discover_cycle_tiles(
    cycle_dir: str | Path,
    *,
    tile_filename_regex: str | None = None,
    x_group: int = _DEFAULT_X_GROUP,
    y_group: int = _DEFAULT_Y_GROUP,
) -> dict[tuple[int, int], Path]:
    out: dict[tuple[int, int], Path] = {}
    cdir = Path(cycle_dir)
    tile_re = _compile_tile_regex(tile_filename_regex)
    for p in sorted(cdir.iterdir()):
        if not p.is_file():
            continue
        xy = _parse_tile_xy(
            p.name,
            tile_re=tile_re,
            x_group=int(x_group),
            y_group=int(y_group),
        )
        if xy is None:
            continue
        x, y = xy
        if (x, y) in out:
            continue
        out[(x, y)] = p
    return out


def discover_sample_cycles(
    sample_dir: str | Path,
    *,
    tile_filename_regex: str | None = None,
    x_group: int = _DEFAULT_X_GROUP,
    y_group: int = _DEFAULT_Y_GROUP,
) -> list[Path]:
    out: list[Path] = []
    sdir = Path(sample_dir)
    for p in sorted(sdir.iterdir()):
        if not p.is_dir():
            continue
        if discover_cycle_tiles(
            p,
            tile_filename_regex=tile_filename_regex,
            x_group=int(x_group),
            y_group=int(y_group),
        ):
            out.append(p)
    return out


def _sample_global_threshold(tile_paths: list[Path], channel_index: int, *, stride: int = 8, max_vals: int = 2_000_000) -> float:
    vals: list[np.ndarray] = []
    total = 0
    for p in tile_paths:
        arr = np.asarray(load_single_channel_tiff_native(str(p), int(channel_index)))
        samp = np.asarray(arr[::stride, ::stride], dtype=np.float32)
        if samp.size == 0:
            continue
        vals.append(samp.reshape(-1))
        total += int(samp.size)
        if total >= int(max_vals):
            break
    if not vals:
        return 0.0
    data = np.concatenate(vals, axis=0)
    if data.size > int(max_vals):
        data = data[: int(max_vals)]
    if threshold_otsu is not None:
        try:
            return float(threshold_otsu(data))
        except Exception:
            pass
    return float(np.percentile(data, 75.0))


def _foreground_fraction(region: np.ndarray, thr: float) -> float:
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region > thr)) / float(region.size)


def _shift_int(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    y0_src = max(0, -dy)
    y1_src = min(img.shape[0], img.shape[0] - dy) if dy >= 0 else img.shape[0]
    x0_src = max(0, -dx)
    x1_src = min(img.shape[1], img.shape[1] - dx) if dx >= 0 else img.shape[1]

    y0_dst = max(0, dy)
    y1_dst = y0_dst + max(0, y1_src - y0_src)
    x0_dst = max(0, dx)
    x1_dst = x0_dst + max(0, x1_src - x0_src)
    if y1_src > y0_src and x1_src > x0_src and y1_dst > y0_dst and x1_dst > x0_dst:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out


def _normalized_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return -1.0
    aa = a[mask]
    bb = b[mask]
    if aa.size < 64:
        return -1.0
    aa = aa - float(aa.mean())
    bb = bb - float(bb.mean())
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0.0:
        return -1.0
    return float(np.dot(aa, bb) / den)


def _estimate_strip_pair(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    axis: str,
    nominal_overlap_px: float,
    thr: float,
    min_fg_frac: float = 0.01,
    max_search_frac: float = 0.20,
) -> NeighborEstimate | None:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    h, w = src.shape
    if src.shape != dst.shape:
        return None
    if axis not in {'x', 'y'}:
        return None

    if axis == 'x':
        base = w
    else:
        base = h
    min_ov = max(8, int(round(base * 0.02)))
    max_ov = max(min_ov + 1, int(round(base * float(max_search_frac))))
    nom = int(round(nominal_overlap_px))
    nom = min(max(nom, min_ov), max_ov)

    candidates: list[int] = sorted(set([nom] + list(range(min_ov, max_ov + 1, max(1, base // 100)))))
    best: tuple[float, float, float, float] | None = None  # objective, overlap, dy, dx
    best_score = -1.0
    used_unmasked = False

    def _search(*, require_foreground: bool) -> tuple[tuple[float, float, float, float] | None, float]:
        local_best: tuple[float, float, float, float] | None = None
        local_best_score = -1.0
        for ov in candidates:
            if axis == 'x':
                a = src[:, w - ov : w]
                b = dst[:, :ov]
            else:
                a = src[h - ov : h, :]
                b = dst[:ov, :]
            if a.size == 0 or b.size == 0:
                continue
            if require_foreground and (_foreground_fraction(a, thr) < min_fg_frac or _foreground_fraction(b, thr) < min_fg_frac):
                continue
            try:
                shift, _err, _ph = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
                dy = float(shift[0])
                dx = float(shift[1])
            except Exception:
                dy = 0.0
                dx = 0.0
            sc = _normalized_score(a, _shift_int(b, int(round(dy)), int(round(dx))))
            if axis == 'x':
                objective = abs(dx) + 0.25 * abs(dy)
            else:
                objective = abs(dy) + 0.25 * abs(dx)
            if (local_best is None) or (objective < local_best[0] - 1e-6) or (abs(objective - local_best[0]) <= 1e-6 and sc > local_best_score):
                local_best = (float(objective), float(ov), float(dy), float(dx))
                local_best_score = float(sc)
        return local_best, float(local_best_score)

    best, best_score = _search(require_foreground=True)
    if best is None:
        best, best_score = _search(require_foreground=False)
        used_unmasked = best is not None

    if best is None:
        return None

    _obj, ov, dy, dx = best
    if axis == 'x':
        global_dx = float(w - ov + dx)
        global_dy = float(dy)
        overlap = float(w - global_dx)
    else:
        global_dy = float(h - ov + dy)
        global_dx = float(dx)
        overlap = float(h - global_dy)
    return NeighborEstimate(
        axis=axis,
        src=(0, 0),
        dst=(0, 0),
        dy=global_dy,
        dx=global_dx,
        overlap_px=float(overlap),
        score=float(best_score),
        used_fallback=False,
        used_unmasked=bool(used_unmasked),
    )


def _feather_weights(h: int, w: int) -> np.ndarray:
    yy = np.minimum(np.arange(h, dtype=np.float32) + 1.0, np.arange(h, dtype=np.float32)[::-1] + 1.0)
    xx = np.minimum(np.arange(w, dtype=np.float32) + 1.0, np.arange(w, dtype=np.float32)[::-1] + 1.0)
    wy = yy / float(max(1.0, yy.max()))
    wx = xx / float(max(1.0, xx.max()))
    out = np.outer(wy, wx).astype(np.float32, copy=False)
    out[out <= 0] = 1e-3
    return out


def _build_neighbor_estimates(
    tiles: dict[tuple[int, int], Path],
    channel_index: int,
    threshold: float,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[dict[tuple[tuple[int, int], tuple[int, int]], NeighborEstimate], RunningOverlap, int, int, int]:
    if not tiles:
        raise ValueError('No tile files found')
    first = next(iter(tiles.values()))
    info = inspect_tiff_yxc(str(first))
    tile_h, tile_w, n_channels = (int(info['shape_yxc'][0]), int(info['shape_yxc'][1]), int(info['shape_yxc'][2]))
    running = RunningOverlap(tile_h, tile_w, frac=0.05)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def _img(xy: tuple[int, int]) -> np.ndarray:
        if xy not in cache:
            cache[xy] = np.asarray(load_single_channel_tiff_native(str(tiles[xy]), int(channel_index)), dtype=np.float32)
        return cache[xy]

    pairs: list[tuple[float, str, tuple[int, int], tuple[int, int]]] = []
    nom_x = max(8, int(round(tile_w * 0.05)))
    nom_y = max(8, int(round(tile_h * 0.05)))
    for (x, y), p in sorted(tiles.items()):
        if (x + 1, y) in tiles:
            a = _img((x, y))[:, tile_w - nom_x : tile_w]
            b = _img((x + 1, y))[:, :nom_x]
            score = _foreground_fraction(a, threshold) + _foreground_fraction(b, threshold)
            pairs.append((score, 'x', (x, y), (x + 1, y)))
        if (x, y + 1) in tiles:
            a = _img((x, y))[tile_h - nom_y : tile_h, :]
            b = _img((x, y + 1))[:nom_y, :]
            score = _foreground_fraction(a, threshold) + _foreground_fraction(b, threshold)
            pairs.append((score, 'y', (x, y), (x, y + 1)))
    pairs.sort(key=lambda t: (-float(t[0]), t[1], t[2], t[3]))

    estimates: dict[tuple[tuple[int, int], tuple[int, int]], NeighborEstimate] = {}
    any_signal = False
    any_unmasked = False
    for _score, axis, src_xy, dst_xy in pairs:
        src = _img(src_xy)
        dst = _img(dst_xy)
        nominal = running.overlap_x if axis == 'x' else running.overlap_y
        est = _estimate_strip_pair(src, dst, axis=axis, nominal_overlap_px=nominal, thr=threshold)
        if est is None:
            if axis == 'x':
                overlap = float(running.overlap_x)
                dy, dx = 0.0, float(tile_w - overlap)
            else:
                overlap = float(running.overlap_y)
                dy, dx = float(tile_h - overlap), 0.0
            est = NeighborEstimate(axis=axis, src=src_xy, dst=dst_xy, dy=dy, dx=dx, overlap_px=overlap, score=-1.0, used_fallback=True)
        else:
            est = NeighborEstimate(
                axis=axis,
                src=src_xy,
                dst=dst_xy,
                dy=est.dy,
                dx=est.dx,
                overlap_px=est.overlap_px,
                score=est.score,
                used_fallback=False,
                used_unmasked=bool(getattr(est, 'used_unmasked', False)),
            )
            running.add(axis, est.overlap_px)
            any_signal = True
            any_unmasked = any_unmasked or bool(est.used_unmasked)
        estimates[(src_xy, dst_xy)] = est
        if progress_cb is not None:
            try:
                extra = ' [unmasked]' if bool(getattr(est, 'used_unmasked', False)) else ''
                progress_cb(f"Estimated tile offset {src_xy} -> {dst_xy} (axis={axis}, overlap≈{est.overlap_px:.1f}px){extra}")
            except Exception:
                pass
    if not any_signal:
        raise RuntimeError('Unable to estimate tile overlaps for this cycle/channel, including unmasked fallback registration.')
    return estimates, running, tile_h, tile_w, n_channels




def _edge_weight(est: NeighborEstimate) -> float:
    overlap_term = max(0.0, float(est.overlap_px))
    score = float(est.score)
    score_term = max(0.0, min(1.0, (score + 1.0) / 2.0))
    if score < -0.5:
        score_term *= 0.1
    weight = max(1.0, overlap_term) * (0.25 + 0.75 * score_term)
    if bool(est.used_unmasked):
        weight *= 0.85
    if bool(est.used_fallback):
        weight *= 0.05
    return float(max(weight, 1e-6))


def _is_valid_voting_edge(est: NeighborEstimate) -> bool:
    if bool(est.used_fallback):
        return False
    if not np.isfinite(est.dy) or not np.isfinite(est.dx):
        return False
    if float(est.overlap_px) <= 0.0:
        return False
    if float(est.score) < -0.35:
        return False
    return True


def _build_adjacency(
    estimates: dict[tuple[tuple[int, int], tuple[int, int]], NeighborEstimate],
) -> dict[tuple[int, int], list[DirectedEdge]]:
    adj: dict[tuple[int, int], list[DirectedEdge]] = {}
    for (src_xy, dst_xy), est in estimates.items():
        w = _edge_weight(est)
        adj.setdefault(src_xy, []).append(
            DirectedEdge(
                src=src_xy,
                dst=dst_xy,
                dy=float(est.dy),
                dx=float(est.dx),
                weight=w,
                overlap_px=float(est.overlap_px),
                score=float(est.score),
                used_fallback=bool(est.used_fallback),
                used_unmasked=bool(getattr(est, 'used_unmasked', False)),
            )
        )
        adj.setdefault(dst_xy, []).append(
            DirectedEdge(
                src=dst_xy,
                dst=src_xy,
                dy=float(-est.dy),
                dx=float(-est.dx),
                weight=w,
                overlap_px=float(est.overlap_px),
                score=float(est.score),
                used_fallback=bool(est.used_fallback),
                used_unmasked=bool(getattr(est, 'used_unmasked', False)),
            )
        )
    return adj


def _choose_seed_tile(
    tiles: dict[tuple[int, int], Path],
    adjacency: dict[tuple[int, int], list[DirectedEdge]],
    channel_index: int,
    threshold: float,
) -> tuple[int, int]:
    if not tiles:
        raise ValueError('No tile files found')

    fg_cache: dict[tuple[int, int], float] = {}

    def _tile_fg_fraction(xy: tuple[int, int]) -> float:
        if xy not in fg_cache:
            arr = np.asarray(load_single_channel_tiff_native(str(tiles[xy]), int(channel_index)), dtype=np.float32)
            fg_cache[xy] = _foreground_fraction(arr, float(threshold))
        return float(fg_cache[xy])

    best_xy: tuple[int, int] | None = None
    best_key: tuple[float, float, float, float, float, int, int] | None = None
    for xy in sorted(tiles.keys(), key=lambda t: (t[1], t[0])):
        edges = adjacency.get(xy, [])
        valid_edges = [e for e in edges if _is_valid_voting_edge(NeighborEstimate(
            axis='x', src=e.src, dst=e.dst, dy=e.dy, dx=e.dx, overlap_px=e.overlap_px, score=e.score,
            used_fallback=e.used_fallback, used_unmasked=e.used_unmasked
        ))]
        valid_degree = len(valid_edges)
        support_sum = float(sum(e.weight for e in valid_edges))
        overlap_sum = float(sum(max(0.0, e.overlap_px) for e in valid_edges))
        raw_degree = len(edges)
        fg_frac = _tile_fg_fraction(xy)
        key = (
            float(valid_degree),
            float(support_sum),
            float(overlap_sum),
            float(raw_degree),
            float(fg_frac),
            -int(xy[1]),
            -int(xy[0]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_xy = xy
    assert best_xy is not None
    return best_xy


def _initial_positions_from_seed(
    tiles: dict[tuple[int, int], Path],
    estimates: dict[tuple[tuple[int, int], tuple[int, int]], NeighborEstimate],
    seed_xy: tuple[int, int],
) -> dict[tuple[int, int], tuple[float, float]]:
    pos: dict[tuple[int, int], tuple[float, float]] = {seed_xy: (0.0, 0.0)}
    changed = True
    while changed:
        changed = False
        for (src_xy, dst_xy), est in estimates.items():
            if src_xy in pos and dst_xy not in pos:
                y0, x0 = pos[src_xy]
                pos[dst_xy] = (float(y0 + est.dy), float(x0 + est.dx))
                changed = True
            elif dst_xy in pos and src_xy not in pos:
                y1, x1 = pos[dst_xy]
                pos[src_xy] = (float(y1 - est.dy), float(x1 - est.dx))
                changed = True
    seeds = sorted(tiles.keys(), key=lambda t: (t[1], t[0]))
    for xy in seeds:
        if xy not in pos:
            x, y = xy
            guesses: list[tuple[float, float]] = []
            if (x - 1, y) in pos and (((x - 1, y), (x, y)) in estimates):
                p = pos[(x - 1, y)]
                e = estimates[((x - 1, y), (x, y))]
                guesses.append((p[0] + e.dy, p[1] + e.dx))
            if (x, y - 1) in pos and (((x, y - 1), (x, y)) in estimates):
                p = pos[(x, y - 1)]
                e = estimates[((x, y - 1), (x, y))]
                guesses.append((p[0] + e.dy, p[1] + e.dx))
            if guesses:
                pos[xy] = (float(np.mean([g[0] for g in guesses])), float(np.mean([g[1] for g in guesses])))
            else:
                pos[xy] = (0.0, 0.0)
    return pos


def _refine_positions_multi_neighbor(
    positions: dict[tuple[int, int], tuple[float, float]],
    adjacency: dict[tuple[int, int], list[DirectedEdge]],
    seed_xy: tuple[int, int],
    *,
    n_iter: int = 8,
    damping: float = 0.7,
    outlier_px: float = 12.0,
) -> dict[tuple[int, int], tuple[float, float]]:
    pos = {k: (float(v[0]), float(v[1])) for k, v in positions.items()}
    if seed_xy not in pos:
        pos[seed_xy] = (0.0, 0.0)
    for _ in range(max(1, int(n_iter))):
        updated = dict(pos)
        for xy in pos.keys():
            if xy == seed_xy:
                updated[xy] = (0.0, 0.0)
                continue
            edges = adjacency.get(xy, [])
            if not edges:
                continue
            candidates_real: list[tuple[float, float, float]] = []
            candidates_fallback: list[tuple[float, float, float]] = []
            for edge in edges:
                if edge.dst not in pos:
                    continue
                nbr_y, nbr_x = pos[edge.dst]
                cand_y = float(nbr_y - edge.dy)
                cand_x = float(nbr_x - edge.dx)
                item = (cand_y, cand_x, float(edge.weight))
                if _is_valid_voting_edge(NeighborEstimate(axis='x', src=edge.src, dst=edge.dst, dy=edge.dy, dx=edge.dx, overlap_px=edge.overlap_px, score=edge.score, used_fallback=edge.used_fallback, used_unmasked=edge.used_unmasked)):
                    candidates_real.append(item)
                else:
                    candidates_fallback.append(item)
            candidates = candidates_real if candidates_real else candidates_fallback
            if not candidates:
                continue
            if len(candidates) >= 3:
                ys = np.asarray([c[0] for c in candidates], dtype=np.float32)
                xs = np.asarray([c[1] for c in candidates], dtype=np.float32)
                med_y = float(np.median(ys))
                med_x = float(np.median(xs))
                filtered = []
                for cand_y, cand_x, cand_w in candidates:
                    dist = math.hypot(float(cand_y - med_y), float(cand_x - med_x))
                    if dist <= float(outlier_px):
                        filtered.append((cand_y, cand_x, cand_w))
                if filtered:
                    candidates = filtered
            weights = np.asarray([max(1e-6, c[2]) for c in candidates], dtype=np.float64)
            ys = np.asarray([c[0] for c in candidates], dtype=np.float64)
            xs = np.asarray([c[1] for c in candidates], dtype=np.float64)
            target_y = float(np.average(ys, weights=weights))
            target_x = float(np.average(xs, weights=weights))
            old_y, old_x = pos[xy]
            new_y = float((1.0 - damping) * old_y + damping * target_y)
            new_x = float((1.0 - damping) * old_x + damping * target_x)
            updated[xy] = (new_y, new_x)
        pos = updated
        seed_y, seed_x = pos.get(seed_xy, (0.0, 0.0))
        if seed_y != 0.0 or seed_x != 0.0:
            for xy, (yy, xx) in list(pos.items()):
                pos[xy] = (float(yy - seed_y), float(xx - seed_x))
            pos[seed_xy] = (0.0, 0.0)
    return pos


def _solve_positions(
    tiles: dict[tuple[int, int], Path],
    estimates: dict[tuple[tuple[int, int], tuple[int, int]], NeighborEstimate],
    *,
    channel_index: int,
    threshold: float,
) -> dict[tuple[int, int], tuple[int, int]]:
    adjacency = _build_adjacency(estimates)
    seed_xy = _choose_seed_tile(tiles, adjacency, int(channel_index), float(threshold))
    pos = _initial_positions_from_seed(tiles, estimates, seed_xy)
    pos = _refine_positions_multi_neighbor(pos, adjacency, seed_xy, n_iter=8, damping=0.7, outlier_px=12.0)
    min_y = min(float(v[0]) for v in pos.values())
    min_x = min(float(v[1]) for v in pos.values())
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for xy, (yy, xx) in pos.items():
        out[xy] = (int(round(yy - min_y)), int(round(xx - min_x)))
    return out


def stitch_cycle_tiles(
    cycle_dir: str | Path,
    *,
    output_suffix: str = 'stitched',
    stitch_channel: int = 0,
    pyramidal_output: bool = True,
    tile_filename_regex: str | None = None,
    x_group: int = _DEFAULT_X_GROUP,
    y_group: int = _DEFAULT_Y_GROUP,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    def _check_cancel() -> None:
        if cancel_cb is not None and bool(cancel_cb()):
            raise RuntimeError('Cancelled')

    cycle_dir = Path(cycle_dir)
    tiles = discover_cycle_tiles(
        cycle_dir,
        tile_filename_regex=tile_filename_regex,
        x_group=int(x_group),
        y_group=int(y_group),
    )
    if not tiles:
        if tile_filename_regex is None or not str(tile_filename_regex).strip():
            raise ValueError(
                f'No tile files found in {cycle_dir} with filenames ending in underscore-separated integer fields for x and y, e.g. *_<x>_<y>.ome.tif[f]'
            )
        raise ValueError(
            f'No tile files found in {cycle_dir} matching the custom tile filename regex.'
        )

    tile_paths = [tiles[k] for k in sorted(tiles.keys(), key=lambda t: (t[1], t[0]))]
    first = tile_paths[0]
    info = inspect_tiff_yxc(str(first))
    tile_h, tile_w, n_channels = (int(info['shape_yxc'][0]), int(info['shape_yxc'][1]), int(info['shape_yxc'][2]))
    channel_names = list(info.get('channel_names') or [])
    if len(channel_names) != int(n_channels):
        channel_names = [f'Channel {i}' for i in range(int(n_channels))]
    phys = load_physical_pixel_sizes(str(first))

    if int(stitch_channel) < 0 or int(stitch_channel) >= int(n_channels):
        raise IndexError(f'stitch_channel {stitch_channel} out of range for {n_channels} channel(s)')

    if progress_cb is not None:
        progress_cb(f'Computing global foreground threshold for cycle {cycle_dir.name}…')
    thr = _sample_global_threshold(tile_paths, int(stitch_channel))
    _check_cancel()

    estimates, running, tile_h, tile_w, n_channels = _build_neighbor_estimates(
        tiles,
        int(stitch_channel),
        float(thr),
        progress_cb=progress_cb,
    )
    positions = _solve_positions(tiles, estimates, channel_index=int(stitch_channel), threshold=float(thr))

    max_y = max(int(v[0]) for v in positions.values()) + int(tile_h)
    max_x = max(int(v[1]) for v in positions.values()) + int(tile_w)
    shape_yxc = (int(max_y), int(max_x), int(n_channels))

    base_name = f"{cycle_dir.name}_{str(output_suffix or 'stitched').strip()}.ome.tiff"
    out_flat = cycle_dir / f".{base_name}.flat_tmp.ome.tiff"
    out_final = cycle_dir / base_name
    feather = _feather_weights(tile_h, tile_w)

    used_fallback_pairs = sum(1 for e in estimates.values() if bool(e.used_fallback))
    used_unmasked_pairs = sum(1 for e in estimates.values() if bool(getattr(e, 'used_unmasked', False)))
    try:
        with IncrementalOmeBigTiffWriter(str(out_flat), shape_yxc, info['dtype'], channel_names=channel_names, physical_pixel_sizes=phys) as writer:
            for ch in range(int(n_channels)):
                _check_cancel()
                if progress_cb is not None:
                    progress_cb(f'Stitching channel {ch + 1}/{n_channels} for cycle {cycle_dir.name}…')
                acc = np.zeros((shape_yxc[0], shape_yxc[1]), dtype=np.float32)
                wsum = np.zeros((shape_yxc[0], shape_yxc[1]), dtype=np.float32)
                for xy in sorted(tiles.keys(), key=lambda t: (t[1], t[0])):
                    _check_cancel()
                    tile = np.asarray(load_single_channel_tiff_native(str(tiles[xy]), int(ch)), dtype=np.float32)
                    y0, x0 = positions[xy]
                    y1 = y0 + tile.shape[0]
                    x1 = x0 + tile.shape[1]
                    w = feather[: tile.shape[0], : tile.shape[1]]
                    acc[y0:y1, x0:x1] += tile * w
                    wsum[y0:y1, x0:x1] += w
                plane = np.zeros_like(acc)
                nz = wsum > 0
                plane[nz] = acc[nz] / wsum[nz]
                writer.write_channel(ch, plane)
                writer.flush()
        if pyramidal_output:
            _check_cancel()
            if progress_cb is not None:
                progress_cb(f'Converting stitched flat TIFF to pyramidal OME-TIFF for {cycle_dir.name}…')
            convert_flat_ome_to_pyramidal(
                str(out_flat),
                output_path=str(out_final),
                channel_names=channel_names,
                physical_pixel_sizes=phys,
                replace_source=False,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            try:
                os.remove(out_flat)
            except Exception:
                pass
        else:
            if out_final.exists():
                out_final.unlink()
            out_flat.replace(out_final)
    except Exception:
        try:
            if out_flat.exists():
                out_flat.unlink()
        except Exception:
            pass
        raise

    return {
        'cycle_dir': str(cycle_dir),
        'output_path': str(out_final),
        'shape_yxc': tuple(int(v) for v in shape_yxc),
        'tile_shape_yx': (int(tile_h), int(tile_w)),
        'n_tiles': int(len(tiles)),
        'n_channels': int(n_channels),
        'stitch_channel': int(stitch_channel),
        'tile_filename_regex': str(tile_filename_regex) if tile_filename_regex is not None else '',
        'x_group': int(x_group),
        'y_group': int(y_group),
        'threshold': float(thr),
        'running_overlap_x_px': float(running.overlap_x),
        'running_overlap_y_px': float(running.overlap_y),
        'used_fallback_pairs': int(used_fallback_pairs),
        'used_unmasked_pairs': int(used_unmasked_pairs),
        'positions': {f'{k[0]},{k[1]}': [int(v[0]), int(v[1])] for k, v in positions.items()},
    }
