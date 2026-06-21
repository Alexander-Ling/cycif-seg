from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

import numpy as np
import tifffile

from cycif_seg.io.ome_tiff import load_single_channel_tiff_native
from cycif_seg.preprocess.organize_cycles import (
    CycleInput,
    _apply_translation,
    _elastic_tile_trust_weight,
    _estimate_masked_rigid_touchup,
    _resolve_borrowed_rigid_touchups,
    _RigidTouchupTile,
    _smooth_rigid_prior_for_tile,
    _run_elastix_bspline,
    _strip_source_row_bounds_for_field,
    merge_cycles_to_ome_tiff,
)


VISUAL_OUTPUT_DIR = Path(__file__).resolve().parent / "elastic_touchup_visuals"


def _disk(y: np.ndarray, x: np.ndarray, cy: float, cx: float, r: float) -> np.ndarray:
    return ((y - cy) ** 2 + (x - cx) ** 2) <= (r ** 2)


def _synthetic_two_island_image(shape: tuple[int, int] = (384, 384)) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    img = np.full(shape, 120, dtype=np.float32)

    # Two separated signal islands, each with internal structure so correlation
    # has enough texture to distinguish good and bad elastic corrections.
    islands = [
        ((120, 125), [(92, 100, 28, 21000), (132, 145, 30, 26000), (155, 98, 24, 18000)]),
        ((275, 265), [(245, 252, 28, 22000), (285, 290, 31, 27000), (310, 238, 24, 19000)]),
    ]
    for _center, circles in islands:
        for cy, cx, radius, value in circles:
            mask = _disk(yy, xx, cy, cx, radius)
            img[mask] = np.maximum(img[mask], float(value))

    # Add deterministic gradients/rings inside islands to make local warps measurable.
    for cy, cx in [(125, 120), (275, 265)]:
        island_mask = _disk(yy, xx, cy, cx, 72)
        ripple = 1800.0 * np.sin((xx - cx) / 13.0) + 1200.0 * np.cos((yy - cy) / 17.0)
        img[island_mask] = np.maximum(img[island_mask] + ripple[island_mask], 1000.0)

    return np.clip(img, 0, 65535).astype(np.uint16)


def _distort_moving_from_fixed(fixed: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    yy, xx = np.indices(fixed.shape, dtype=np.float32)
    # Global offset plus low-amplitude smooth local distortion.  Coordinates are
    # inverse-mapped so the generated moving image is shifted/distorted relative
    # to the fixed image.
    dy = 7.0 + 2.0 * np.sin(xx / 55.0)
    dx = -6.0 + 2.0 * np.cos(yy / 60.0)
    moving = map_coordinates(
        fixed.astype(np.float32),
        [yy - dy, xx - dx],
        order=1,
        mode="constant",
        cval=120.0,
    )
    return np.clip(moving, 0, 65535).astype(np.uint16)


def _write_single_channel_ome(path: Path, image: np.ndarray) -> None:
    stack = np.stack([image, np.zeros_like(image)], axis=0)
    tifffile.imwrite(
        str(path),
        stack,
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "Blank"]}},
    )


def _pearson_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float64, copy=False)
    bv = b[mask].astype(np.float64, copy=False)
    av -= float(av.mean())
    bv -= float(bv.mean())
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 0:
        return 0.0
    return float(np.sum(av * bv) / denom)


def _fast_elastic_kernel(
    fixed_crop: np.ndarray,
    moving_crop: np.ndarray,
    island_crop: np.ndarray,
    *,
    grid_spacing_px: int,
    max_iterations: int,
    max_step_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    from skimage.registration import phase_cross_correlation

    if int(island_crop.sum()) < 32:
        return None
    try:
        shift, _error, _phase = phase_cross_correlation(
            fixed_crop.astype(np.float32, copy=False),
            moving_crop.astype(np.float32, copy=False),
            upsample_factor=10,
        )
    except Exception:
        return None
    dy = float(np.clip(shift[0], -8.0, 8.0))
    dx = float(np.clip(shift[1], -8.0, 8.0))
    field_y = np.full(fixed_crop.shape, dy, dtype=np.float32)
    field_x = np.full(fixed_crop.shape, dx, dtype=np.float32)
    field_y[~island_crop.astype(bool)] = 0.0
    field_x[~island_crop.astype(bool)] = 0.0
    return field_y, field_x


def _run_registration(
    tmp: Path,
    fixed_path: Path,
    moving_path: Path,
    workers: int,
) -> tuple[float, list[dict], Path]:
    out_path = tmp / f"registered_workers_{workers}.ome.tiff"
    events: list[dict] = []
    merge_cycles_to_ome_tiff(
        [
            CycleInput(str(fixed_path), cycle=0, registration_marker="DAPI", channel_markers=["DAPI", "Blank"]),
            CycleInput(str(moving_path), cycle=1, registration_marker="DAPI", channel_markers=["DAPI", "Blank"]),
        ],
        str(out_path),
        reference_cycle=0,
        default_registration_marker="DAPI",
        registration_algorithm="tiled_rigid",
        downsample_for_registration=1,
        tiled_rigid_tile_size=96,
        tiled_rigid_search_factor=2.0,
        low_mem=True,
        strip_height=96,
        elastic_touchup=True,
        elastic_touchup_tile_size=96,
        elastic_touchup_skip_corr=0.99,
        elastic_touchup_bspline_spacing=32,
        elastic_touchup_max_iterations=2,
        elastic_touchup_large_island_px=3_000,
        elastic_touchup_workers=workers,
        pyramidal_output=False,
        progress_event_cb=events.append,
    )
    fixed_out = load_single_channel_tiff_native(str(out_path), 0)
    moving_out = load_single_channel_tiff_native(str(out_path), 2)
    mask = fixed_out > 1000
    return _pearson_corr(fixed_out, moving_out, mask), events, out_path


def _save_visual_artifacts(
    fixed_path: Path,
    moving_path: Path,
    registered_w1_path: Path,
    registered_w3_path: Path,
) -> None:
    VISUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixed_path, VISUAL_OUTPUT_DIR / "fixed_two_islands.ome.tiff")
    shutil.copy2(moving_path, VISUAL_OUTPUT_DIR / "moving_two_islands_distorted.ome.tiff")
    shutil.copy2(registered_w1_path, VISUAL_OUTPUT_DIR / "registered_workers_1.ome.tiff")
    shutil.copy2(registered_w3_path, VISUAL_OUTPUT_DIR / "registered_workers_3.ome.tiff")


class ElasticTouchupRegressionTest(unittest.TestCase):
    def _rigid_tile(
        self,
        y: int,
        x: int,
        *,
        island: int = 1,
        accepted: bool = False,
        dy: float = 0.0,
        dx: float = 0.0,
        gain: float = 0.1,
    ) -> _RigidTouchupTile:
        return _RigidTouchupTile(
            y0=y,
            y1=y + 10,
            x0=x,
            x1=x + 10,
            island_label=island,
            center_y=y + 5.0,
            center_x=x + 5.0,
            rigid_dy=dy,
            rigid_dx=dx,
            base_corr=0.1,
            candidate_corr=0.1 + gain,
            accepted=accepted,
        )

    def test_borrowed_rigid_touchup_propagates_across_failed_chain(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=20.0, dx=-6.0),
            self._rigid_tile(0, 10),
            self._rigid_tile(0, 20),
            self._rigid_tile(0, 30),
        ]

        counts = _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64, neighborhood_radius=1)

        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(counts["borrowed"], 3)
        self.assertEqual(tiles[-1].mode, "borrowed")
        self.assertAlmostEqual(tiles[-1].resolved_dy, 20.0, places=6)
        self.assertAlmostEqual(tiles[-1].resolved_dx, -6.0, places=6)

    def test_borrowed_rigid_touchup_uses_multiple_good_neighbors(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=10.0, dx=0.0, gain=0.2),
            self._rigid_tile(0, 20, accepted=True, dy=30.0, dx=0.0, gain=0.2),
            self._rigid_tile(0, 10),
        ]

        counts = _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64)

        self.assertEqual(counts["borrowed"], 1)
        self.assertEqual(tiles[2].mode, "borrowed")
        self.assertAlmostEqual(tiles[2].resolved_dy, 20.0, places=6)
        self.assertAlmostEqual(tiles[2].resolved_dx, 0.0, places=6)

    def test_borrowed_rigid_touchup_does_not_cross_islands(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, island=1, accepted=True, dy=14.0, dx=7.0),
            self._rigid_tile(0, 10, island=2),
        ]

        counts = _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64)

        self.assertEqual(counts["borrowed"], 0)
        self.assertEqual(counts["elastic_only"], 1)
        self.assertEqual(tiles[1].mode, "elastic_only")
        self.assertEqual(tiles[1].resolved_dy, 0.0)
        self.assertEqual(tiles[1].resolved_dx, 0.0)

    def test_borrowed_rigid_touchup_clamps_shift_magnitude(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=100.0, dx=0.0),
            self._rigid_tile(0, 10),
        ]

        _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=25)

        self.assertEqual(tiles[1].mode, "borrowed")
        self.assertLessEqual(abs(tiles[1].resolved_dy), 25.0)

    def test_high_correlation_tile_becomes_stable_zero_prior_anchor(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=24.0, dx=0.0),
            self._rigid_tile(0, 10),
        ]
        tiles[1].prior_anchor = True
        tiles[1].base_corr = 0.98
        tiles[1].candidate_corr = 0.98

        counts = _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64)

        self.assertEqual(counts["stable_zero"], 1)
        self.assertEqual(tiles[1].mode, "stable_zero")
        self.assertEqual(tiles[1].resolved_dy, 0.0)

    def test_smooth_rigid_prior_uses_stable_zero_anchor_to_reduce_jump(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=30.0, dx=0.0, gain=0.2),
            self._rigid_tile(0, 10),
        ]
        tiles[1].prior_anchor = True
        tiles[1].base_corr = 0.98
        tiles[1].candidate_corr = 0.98
        _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64)

        dy, dx = _smooth_rigid_prior_for_tile(tiles[1], tiles, stride_y=10, stride_x=10, max_shift=64)

        self.assertGreaterEqual(dy, 0.0)
        self.assertLess(dy, 30.0)
        self.assertAlmostEqual(dx, 0.0, places=6)

    def test_smooth_rigid_prior_blends_multiple_resolved_neighbors(self) -> None:
        tiles = [
            self._rigid_tile(0, 0, accepted=True, dy=10.0, dx=0.0, gain=0.2),
            self._rigid_tile(0, 20, accepted=True, dy=30.0, dx=0.0, gain=0.2),
            self._rigid_tile(0, 10),
        ]
        _resolve_borrowed_rigid_touchups(tiles, stride_y=10, stride_x=10, max_shift=64)

        dy, dx = _smooth_rigid_prior_for_tile(tiles[2], tiles, stride_y=10, stride_x=10, max_shift=64)

        self.assertAlmostEqual(dy, 20.0, places=6)
        self.assertAlmostEqual(dx, 0.0, places=6)

    def test_masked_rigid_touchup_accepts_improving_shift(self) -> None:
        fixed = _synthetic_two_island_image((160, 160)).astype(np.float32)
        moving = _apply_translation(fixed, -12.0, 8.0, order=1)
        mask = fixed > 1000

        dy, dx, base_corr, candidate_corr, accepted = _estimate_masked_rigid_touchup(
            fixed,
            moving,
            mask,
            max_shift=24.0,
            min_improvement=0.02,
        )

        self.assertTrue(accepted)
        self.assertGreater(candidate_corr, base_corr + 0.02)
        self.assertLessEqual(abs(dy), 24.0)
        self.assertLessEqual(abs(dx), 24.0)

    def test_masked_rigid_touchup_rejects_non_improving_shift(self) -> None:
        fixed = _synthetic_two_island_image((160, 160)).astype(np.float32)
        mask = fixed > 1000

        dy, dx, base_corr, candidate_corr, accepted = _estimate_masked_rigid_touchup(
            fixed,
            fixed.copy(),
            mask,
            max_shift=24.0,
            min_improvement=0.02,
        )

        self.assertFalse(accepted)
        self.assertEqual(dy, 0.0)
        self.assertEqual(dx, 0.0)
        self.assertAlmostEqual(candidate_corr, base_corr, places=6)

    def test_masked_rigid_touchup_respects_shift_bound(self) -> None:
        fixed = _synthetic_two_island_image((160, 160)).astype(np.float32)
        moving = _apply_translation(fixed, -30.0, 30.0, order=1)
        mask = fixed > 1000

        dy, dx, _base_corr, _candidate_corr, _accepted = _estimate_masked_rigid_touchup(
            fixed,
            moving,
            mask,
            max_shift=8.0,
            min_improvement=0.0,
        )

        self.assertLessEqual(abs(dy), 8.0)
        self.assertLessEqual(abs(dx), 8.0)

    def test_elastic_tile_trust_weight_suppresses_only_artificial_edges(self) -> None:
        weight = _elastic_tile_trust_weight(
            32,
            40,
            border=5,
            trim_top=True,
            trim_bottom=False,
            trim_left=True,
            trim_right=False,
        )

        self.assertEqual(weight.shape, (32, 40))
        self.assertTrue(np.all(weight[:5, :] == 0.0))
        self.assertTrue(np.all(weight[:, :5] == 0.0))
        self.assertTrue(np.any(weight[-5:, 8:] > 0.0))
        self.assertTrue(np.any(weight[8:, -5:] > 0.0))
        self.assertGreater(float(weight[16, 20]), 0.0)

    def test_elastic_tile_trust_weight_keeps_true_boundary_sides(self) -> None:
        weight = _elastic_tile_trust_weight(
            32,
            40,
            border=5,
            trim_top=False,
            trim_bottom=False,
            trim_left=False,
            trim_right=False,
        )

        self.assertGreater(float(weight[0, 20]), 0.0)
        self.assertGreater(float(weight[-1, 20]), 0.0)
        self.assertGreater(float(weight[16, 0]), 0.0)
        self.assertGreater(float(weight[16, -1]), 0.0)

    def test_strip_source_bounds_include_elastic_displacement(self) -> None:
        field_y = np.full((64, 48), 3.0, dtype=np.float32)
        field_y[:4, :] = 19.0

        src_y0, src_y1 = _strip_source_row_bounds_for_field(
            96,
            160,
            256,
            field_y,
            fallback_pad=7,
            safety_pad=4,
        )

        self.assertLessEqual(src_y0, 96 - 19 - 4)
        self.assertGreaterEqual(src_y1, 160 - 3 + 4)

    def test_elastix_touchup_does_not_write_transformix_scratch_to_cwd(self) -> None:
        yy, xx = np.indices((48, 48), dtype=np.float32)
        fixed = np.zeros((48, 48), dtype=np.float32)
        moving = np.zeros((48, 48), dtype=np.float32)
        fixed[_disk(yy, xx, 24, 24, 10)] = 1.0
        moving[_disk(yy, xx, 25, 23, 10)] = 1.0
        mask = fixed > 0

        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            scratch_cwd = Path(td)
            try:
                os.chdir(scratch_cwd)
                result = _run_elastix_bspline(
                    fixed,
                    moving,
                    mask,
                    grid_spacing_px=16,
                    max_iterations=1,
                    max_step_length=0.5,
                )
            finally:
                os.chdir(old_cwd)

            self.assertIsNotNone(result)
            self.assertFalse((scratch_cwd / "deformationField.nii").exists())

    def test_low_mem_tiled_elastic_registration_preserves_correlation(self) -> None:
        fixed = _synthetic_two_island_image()
        moving = _distort_moving_from_fixed(fixed)
        mask = fixed > 1000
        pre_corr = _pearson_corr(fixed, moving, mask)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixed_path = tmp / "fixed.ome.tiff"
            moving_path = tmp / "moving.ome.tiff"
            _write_single_channel_ome(fixed_path, fixed)
            _write_single_channel_ome(moving_path, moving)

            with patch(
                "cycif_seg.preprocess.organize_cycles._run_elastix_bspline",
                side_effect=_fast_elastic_kernel,
            ):
                corr_w1, events_w1, registered_w1_path = _run_registration(
                    tmp, fixed_path, moving_path, workers=1
                )
                corr_w3, events_w3, registered_w3_path = _run_registration(
                    tmp, fixed_path, moving_path, workers=3
                )
                _save_visual_artifacts(
                    fixed_path,
                    moving_path,
                    registered_w1_path,
                    registered_w3_path,
                )

        tile_events = [ev for ev in events_w3 if ev.get("phase") == "elastic_touchup_tile"]
        corr_scan_events = [
            ev for ev in tile_events
            if "scanning island correlation" in str(ev.get("msg", ""))
        ]
        submitted_events = [ev for ev in tile_events if "submitted tile" in str(ev.get("msg", ""))]
        completed_events = [
            ev for ev in tile_events
            if "tiles completed=" in str(ev.get("msg", "")) and "submitted tile" not in str(ev.get("msg", ""))
        ]
        category_events = [
            ev for ev in completed_events
            if "rigid+elastic=" in str(ev.get("msg", ""))
            and "elastic-only=" in str(ev.get("msg", ""))
            and "skip_corr=" in str(ev.get("msg", ""))
        ]

        self.assertGreaterEqual(fixed.shape[0] // 96, 4)
        self.assertGreaterEqual(len(corr_scan_events), 1)
        self.assertGreaterEqual(len(submitted_events), 6)
        self.assertGreaterEqual(len(completed_events), 1)
        self.assertGreaterEqual(len(category_events), 1)
        self.assertGreater(corr_w1, pre_corr + 0.05)
        self.assertGreater(corr_w3, pre_corr + 0.05)
        self.assertGreaterEqual(corr_w1, 0.90)
        self.assertGreaterEqual(corr_w3, 0.90)
        self.assertLessEqual(abs(corr_w1 - corr_w3), 0.03)


if __name__ == "__main__":
    unittest.main()
