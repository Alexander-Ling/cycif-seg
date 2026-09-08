from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from cycif_seg.cli.run_pipeline import _build_parser, main
from cycif_seg.preprocess.organize_cycles import (
    _RigidTouchupTile,
    _build_harmonic_tile_field,
    _elastic_touchup_island,
    _estimate_masked_rigid_touchup,
    _fill_missing_touchup_field,
    _harmonic_fill_tile_lattice,
    _masked_corr_score,
    _resolve_borrowed_rigid_touchups,
    _smooth_rigid_prior_for_tile,
    _write_island_map_debug_tiff,
)


def _tile(x: int, *, accepted: bool = False, dy: float = 0.0) -> _RigidTouchupTile:
    return _RigidTouchupTile(
        y0=0, y1=16, x0=x, x1=x + 16,
        island_label=1, center_y=7.5, center_x=x + 7.5,
        rigid_dy=dy, rigid_dx=0.0,
        base_corr=0.1, candidate_corr=0.3,
        accepted=accepted,
    )


def test_rigid_touchup_scores_the_full_field_not_only_foreground() -> None:
    fixed = np.zeros((16, 16), dtype=np.float32)
    fixed[:, 8:] = 10.0
    moving = fixed.copy()
    foreground = np.zeros_like(fixed, dtype=bool)
    foreground[:8, :8] = True

    _, _, base_corr, candidate_corr, accepted = _estimate_masked_rigid_touchup(
        fixed, moving, foreground, max_shift=1, min_improvement=2.0
    )

    assert _masked_corr_score(fixed, moving, foreground) == -1.0
    assert base_corr == pytest.approx(1.0)
    assert candidate_corr == pytest.approx(1.0)
    assert accepted is False


def test_unresolved_rigid_tile_is_missing_and_not_a_smoothing_vote() -> None:
    trusted = _tile(0, accepted=True, dy=12.0)
    unresolved = _tile(16)
    isolated = _tile(32)
    isolated.island_label = 2

    _resolve_borrowed_rigid_touchups(
        [trusted, unresolved, isolated], stride_y=16, stride_x=16, max_shift=64
    )

    assert unresolved.mode == "borrowed"
    assert unresolved.resolved_dy == pytest.approx(12.0)
    assert isolated.mode == "elastic_only"
    assert isolated.resolved_dy is None
    assert _smooth_rigid_prior_for_tile(
        isolated, [trusted, unresolved, isolated],
        stride_y=16, stride_x=16, max_shift=64,
    ) is None


def test_harmonic_fill_preserves_trusted_nodes_and_fills_hole() -> None:
    values_y = np.zeros((3, 3), dtype=np.float32)
    values_x = np.zeros((3, 3), dtype=np.float32)
    known = np.zeros((3, 3), dtype=bool)
    known[0, 0] = known[0, 2] = known[2, 0] = known[2, 2] = True
    values_y[known] = np.array([0.0, 2.0, 2.0, 4.0], dtype=np.float32)
    values_x[known] = np.array([4.0, 2.0, 2.0, 0.0], dtype=np.float32)

    field_y, field_x = _harmonic_fill_tile_lattice(values_y, values_x, known)

    np.testing.assert_array_equal(field_y[known], values_y[known])
    np.testing.assert_array_equal(field_x[known], values_x[known])
    assert field_y[1, 1] == pytest.approx(2.0, abs=0.02)
    assert field_x[1, 1] == pytest.approx(2.0, abs=0.02)


def test_harmonic_field_is_continuous_and_one_sample_is_constant() -> None:
    model = _build_harmonic_tile_field(
        [(15.5, 15.5, 7.0, -3.0)],
        bounds=(0, 64, 0, 64), stride_y=16, stride_x=16,
    )
    assert model is not None
    dy, dx = model.sample(0, 64, 0, 64)
    np.testing.assert_allclose(dy, 7.0, atol=1e-5)
    np.testing.assert_allclose(dx, -3.0, atol=1e-5)
    assert _build_harmonic_tile_field(
        [], bounds=(0, 64, 0, 64), stride_y=16, stride_x=16
    ) is None


def test_borrowed_field_fills_only_missing_same_island_support() -> None:
    model = _build_harmonic_tile_field(
        [(7.5, 7.5, 9.0, -4.0)],
        bounds=(0, 16, 0, 16), stride_y=8, stride_x=8,
    )
    field_y = np.full((16, 16), 3.0, dtype=np.float32)
    field_x = np.full((16, 16), 2.0, dtype=np.float32)
    direct = np.zeros((16, 16), dtype=bool)
    direct[:8, :] = True
    island = np.zeros((16, 16), dtype=bool)
    island[:, :12] = True

    filled = _fill_missing_touchup_field(
        field_y, field_x, direct, island, model, bounds=(0, 16, 0, 16)
    )

    assert filled == 8 * 12
    np.testing.assert_allclose(field_y[:8], 3.0)
    np.testing.assert_allclose(field_x[:8], 2.0)
    np.testing.assert_allclose(field_y[8:, :12], 9.0)
    np.testing.assert_allclose(field_x[8:, :12], -4.0)
    np.testing.assert_allclose(field_y[8:, 12:], 3.0)
    np.testing.assert_allclose(field_x[8:, 12:], 2.0)


def test_fully_failed_tiled_island_abstains_instead_of_returning_zero_weight() -> None:
    yy, xx = np.indices((96, 96), dtype=np.float32)
    fixed = np.sin(xx / 5.0) + np.cos(yy / 7.0)
    moving = np.roll(fixed, 3, axis=1)
    mask = np.ones((96, 96), dtype=bool)
    with (
        patch(
            "cycif_seg.preprocess.organize_cycles._estimate_masked_rigid_touchup",
            return_value=(0.0, 0.0, 0.0, 0.0, False),
        ),
        patch("cycif_seg.preprocess.organize_cycles._run_elastix_bspline", return_value=None),
    ):
        result = _elastic_touchup_island(
            fixed, moving, mask, (0.0, 0.0), mask,
            tile_size=48, skip_corr_threshold=0.999,
            min_fg_pixels=1, grid_spacing_px=16, max_iterations=1,
            large_island_px=1, rigid_max_shift=16,
        )
    assert result is None


def test_island_map_is_colored_deterministically_and_preserves_small_foreground(tmp_path: Path) -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0, 0] = 1
    labels[4:8, 4:8] = 2
    first = tmp_path / "first.tiff"
    second = tmp_path / "second.tiff"

    total_ds, _ = _write_island_map_debug_tiff(
        labels, out_path=first, cycle=3, source_downsample=4, target_max_dim=5
    )
    _write_island_map_debug_tiff(
        labels, out_path=second, cycle=3, source_downsample=4, target_max_dim=5
    )

    image = tifffile.imread(first)
    np.testing.assert_array_equal(image, tifffile.imread(second))
    assert image.shape == (5, 5, 3)
    assert np.any(image[0, 0] != 0)
    assert np.any(image[2, 2] != 0)
    assert not np.array_equal(image[0, 0], image[2, 2])
    assert total_ds == 8
    with tifffile.TiffFile(first) as tf:
        metadata = json.loads(tf.pages[0].description)
    assert metadata["cycle"] == 3
    assert metadata["island_count"] == 2
    assert metadata["total_downsample"] == 8


def test_cli_exposes_island_map_and_rejects_translation_mode() -> None:
    args = _build_parser().parse_args(["sample", "--debug-island-map"])
    assert args.debug_island_map is True
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--debug-island-map", "--registration-algorithm", "translation"])
    assert exc.value.code == 2
