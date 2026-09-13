from __future__ import annotations

from pathlib import Path

import pytest

from cycif_seg.cli.run_pipeline import (
    _build_parser,
    _select_sample_stitched_files,
    discover_cycles,
    main,
)


def _cycle(sample_dir: Path, name: str) -> Path:
    path = sample_dir / name
    path.mkdir()
    return path


def test_sample_wide_stitched_pattern_uses_first_complete_pattern(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    c2 = _cycle(tmp_path, "C2_sample_CD8")
    preferred = {
        c1: c1 / "C1_sample_cyseg-stitched.ome.tiff",
        c2: c2 / "C2_sample_cyseg-stitched.ome.tiff",
    }
    for path in preferred.values():
        path.touch()
    for folder in (c1, c2):
        (folder / f"{folder.name}_evos.ome.tiff").touch()

    pattern, selected = _select_sample_stitched_files(
        [c1, c2],
        stitched_patterns=["*_cyseg-stitched.ome.tiff", "*_evos.ome.tiff"],
    )

    assert pattern == "*_cyseg-stitched.ome.tiff"
    assert selected == preferred


def test_sample_wide_stitched_pattern_falls_back_as_a_whole(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    c2 = _cycle(tmp_path, "C2_sample_CD8")
    (c1 / "C1_sample_cyseg-stitched.ome.tiff").touch()
    fallback = {
        c1: c1 / "C1_sample_evos.ome.tiff",
        c2: c2 / "C2_sample_evos.ome.tiff",
    }
    for path in fallback.values():
        path.touch()

    pattern, selected = _select_sample_stitched_files(
        [c1, c2],
        stitched_patterns=["*_cyseg-stitched.ome.tiff", "*_evos.ome.tiff"],
    )

    assert pattern == "*_evos.ome.tiff"
    assert selected == fallback


def test_sample_wide_stitched_patterns_are_not_mixed(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    c2 = _cycle(tmp_path, "C2_sample_CD8")
    (c1 / "C1_sample_cyseg-stitched.ome.tiff").touch()
    (c2 / "C2_sample_evos.ome.tiff").touch()

    with pytest.raises(FileNotFoundError, match="No stitched pattern") as exc_info:
        _select_sample_stitched_files(
            [c1, c2],
            stitched_patterns=["*_cyseg-stitched.ome.tiff", "*_evos.ome.tiff"],
        )

    message = str(exc_info.value)
    assert "C2_sample_CD8" in message
    assert "C1_sample_CD3" in message


def test_ambiguous_match_is_an_error(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    (c1 / "first_evos.ome.tiff").touch()
    (c1 / "second_evos.ome.tiff").touch()

    with pytest.raises(ValueError, match="ambiguous"):
        _select_sample_stitched_files([c1], stitched_patterns=["*_evos.ome.tiff"])


def test_discovery_requires_one_complete_stitched_pattern(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    c2 = _cycle(tmp_path, "C2_sample_CD8")
    (c1 / "C1_sample_cyseg-stitched.ome.tiff").touch()
    (c2 / "C2_sample_evos.ome.tiff").touch()

    cycles, errors = discover_cycles(
        tmp_path,
        stitched_patterns=["*_cyseg-stitched.ome.tiff", "*_evos.ome.tiff"],
        require_stitched=True,
    )

    assert cycles == []
    assert len(errors) == 1
    assert "No stitched pattern" in errors[0]


def test_cli_preserves_stitched_pattern_preference_order() -> None:
    args = _build_parser().parse_args(
        [
            "sample",
            "--stitched-pattern",
            "*_cyseg-stitched.ome.tiff",
            "*_evos.ome.tiff",
        ]
    )

    assert args.stitched_pattern == [
        "*_cyseg-stitched.ome.tiff",
        "*_evos.ome.tiff",
    ]


def test_dry_run_logs_selected_sample_wide_pattern(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    c2 = _cycle(tmp_path, "C2_sample_CD8")
    (c1 / "C1_sample_evos.ome.tiff").touch()
    (c2 / "C2_sample_evos.ome.tiff").touch()

    status = main(
        [
            str(tmp_path),
            "--skip-stitch",
            "--stitched-pattern",
            "*_cyseg-stitched.ome.tiff",
            "*_evos.ome.tiff",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Selected stitched pattern for all 2 cycle(s): *_evos.ome.tiff" in captured.out


def test_incomplete_sample_exits_before_registration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    _cycle(tmp_path, "C2_sample_CD8")
    (c1 / "C1_sample_evos.ome.tiff").touch()

    status = main(
        [
            str(tmp_path),
            "--skip-stitch",
            "--stitched-pattern",
            "*_cyseg-stitched.ome.tiff",
            "*_evos.ome.tiff",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "No stitched pattern has exactly one matching file in all 2 cycle directories" in captured.err
    assert "C2_sample_CD8" in captured.err
    assert "Registering" not in captured.out


def test_valid_tiles_still_use_existing_coordinate_rules(tmp_path: Path) -> None:
    c1 = _cycle(tmp_path, "C1_sample_CD3")
    (c1 / "raw_Area_anything_12_34.ome.tiff").touch()

    cycles, errors = discover_cycles(tmp_path)

    assert errors == []
    assert len(cycles) == 1
    assert cycles[0]["n_tiles"] == 1
    assert cycles[0].get("pre_stitched") is None
