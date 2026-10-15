from pathlib import Path

import pytest

from cycif_seg.preprocess.batch_plan import (
    BatchSample,
    cycle_inputs_from_sample,
    letter_suffix,
    resolve_cycle_labels,
    validate_sample_cycle_numbers,
)


def test_unique_cycles_keep_numbers_and_get_string_labels():
    ints, labels = resolve_cycle_labels([2, 0, 1], ["S/C2_x/a.ome.tiff", "S/C0_x/b.ome.tiff", "S/C1_x/c.ome.tiff"])
    assert ints == [2, 0, 1]
    assert labels == ["2", "0", "1"]


def test_duplicates_in_different_folders_get_letter_labels_like_cycif_seg_run():
    paths = [
        "S/C1_redo/C1_redo_cyseg-stitched.ome.tiff",
        "S/C0_a/C0_a_cyseg-stitched.ome.tiff",
        "S/C1_first/C1_first_cyseg-stitched.ome.tiff",
        "S/C2_b/C2_b_cyseg-stitched.ome.tiff",
    ]
    ints, labels = resolve_cycle_labels([1, 0, 1, 2], paths)
    # Folder-name order within cycle 1: C1_first < C1_redo.
    assert labels == ["1b", "0", "1a", "2"]
    # Unique rank ints, lowest cycle first so it remains the reference.
    assert ints == [2, 0, 1, 3]


def test_duplicates_in_same_folder_are_rejected():
    with pytest.raises(ValueError, match="more than one file in folder 'C1_x'"):
        resolve_cycle_labels([1, 1], ["S/C1_x/C1_a.ome.tiff", "S/C1_x/C1_b.ome.tiff"])


def test_letter_suffix_wraps_past_z():
    assert [letter_suffix(i) for i in (0, 1, 25, 26, 27)] == ["a", "b", "z", "aa", "ab"]


def _sample(files, cycles, enabled=None):
    return BatchSample(
        name="S",
        input_dir=Path("S"),
        files=[Path(f) for f in files],
        cycles=cycles,
        cycles_enabled=enabled,
        registration_markers=["DAPI"] * len(files),
        channel_markers=[["DAPI"]] * len(files),
        channel_antibodies=[[""]] * len(files),
    )


def test_batch_sample_validation_and_inputs_follow_shared_rule():
    s = _sample(["S/C1_a/C1_a.ome.tiff", "S/C1_b/C1_b.ome.tiff", "S/C0_z/C0_z.ome.tiff"], [1, 1, 0])
    assert validate_sample_cycle_numbers(s) == (True, "")
    cyc = cycle_inputs_from_sample(s)
    assert [(c.cycle, c.label) for c in cyc] == [(1, "1a"), (2, "1b"), (0, "0")]

    same_folder = _sample(["S/C1_a/C1_x.ome.tiff", "S/C1_a/C1_y.ome.tiff"], [1, 1])
    ok, msg = validate_sample_cycle_numbers(same_folder)
    assert not ok and "C1_a" in msg

    # Disabling one of the two same-folder files resolves the ambiguity.
    same_folder.cycles_enabled = [True, False]
    assert validate_sample_cycle_numbers(same_folder) == (True, "")
    assert [c.label for c in cycle_inputs_from_sample(same_folder)] == ["1"]
