from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

import numpy as np
import tifffile

from cycif_seg.io.ome_tiff import load_single_channel_tiff_native
from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff


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

        self.assertGreaterEqual(fixed.shape[0] // 96, 4)
        self.assertGreaterEqual(len(corr_scan_events), 1)
        self.assertGreaterEqual(len(submitted_events), 6)
        self.assertGreaterEqual(len(completed_events), 1)
        self.assertGreater(corr_w1, pre_corr + 0.05)
        self.assertGreater(corr_w3, pre_corr + 0.05)
        self.assertGreaterEqual(corr_w1, 0.90)
        self.assertGreaterEqual(corr_w3, 0.90)
        self.assertLessEqual(abs(corr_w1 - corr_w3), 0.03)


if __name__ == "__main__":
    unittest.main()
