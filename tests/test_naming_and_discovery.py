from pathlib import Path

import pytest

from cycif_seg.cli.run_pipeline import _select_sample_stitched_files, discover_cycles
from cycif_seg.preprocess.batch_plan import (
    cycle_inputs_from_sample,
    scan_input_folder,
    scan_sample_dir,
    validate_sample_cycle_numbers,
)
from cycif_seg.preprocess.naming import cycle_number_from_name, detect_markers, markers_from_name


# ---------------------------------------------------------------------------
# Cycle numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, expected",
    [
        ("C3_Patient01_cyseg-stitched.ome.tiff", 3),
        ("c12_x", 12),
        ("C1.ome.tiff", 1),
        ("C2_[DAPI,CD3]", 2),
        ("Patient01_cycle4.ome.tiff", 4),
        ("Patient01_Cyc_5_stitched.ome.tif", 5),
        ("CD8_PDL1.ome.tiff", None),  # marker names are not cycle numbers
        ("PC12_line.ome.tiff", None),
        ("stitched.ome.tiff", None),
        ("cycle1_cycle2.ome.tiff", None),  # conflicting numbers -> unknown
    ],
)
def test_cycle_number_from_name(name, expected):
    assert cycle_number_from_name(name) == expected


# ---------------------------------------------------------------------------
# Marker lists
# ---------------------------------------------------------------------------
def test_marker_count_equal_to_channels_names_every_channel():
    res = markers_from_name("C1_P01_[DAPI,CD3,CD8,PDL1]", 4)
    assert res.markers == ["DAPI", "CD3", "CD8", "PDL1"]
    assert not res.dapi_assumed and not res.is_warning


def test_one_fewer_marker_assumes_dapi_and_says_so():
    res = markers_from_name("C1_P01_[CD3, CD8 ,PD-L1]", 4)
    assert res.markers == ["DAPI", "CD3", "CD8", "PD-L1"]
    assert res.dapi_assumed
    assert "assumed to be DAPI" in res.message


@pytest.mark.parametrize("field, n", [("[CD3,CD8]", 4), ("[A,B,C,D,E]", 4), ("[]", 2), ("[CD3,,CD8]", 3)])
def test_other_counts_assign_nothing_and_warn(field, n):
    res = markers_from_name(f"C1_P01_{field}", n)
    assert res.markers is None
    assert res.is_warning and res.message


def test_two_marker_lists_are_rejected():
    res = markers_from_name("C1_[DAPI,CD3]_[CD8]", 2)
    assert res.markers is None and res.is_warning


def test_file_name_list_takes_precedence_over_folder():
    res = detect_markers("C1_[DAPI,CD3].ome.tiff", "C1_P01_[DAPI,CD8]", 2)
    assert res.markers == ["DAPI", "CD3"]


def test_folder_list_used_when_file_has_none():
    res = detect_markers("C1_P01_cyseg-stitched.ome.tiff", "C1_P01_[CD8]", 2)
    assert res.markers == ["DAPI", "CD8"] and res.dapi_assumed


def test_invalid_file_list_is_not_overridden_by_folder():
    res = detect_markers("C1_[A,B,C].ome.tiff", "C1_P01_[DAPI,CD8]", 2)
    assert res.markers is None and res.is_warning


def test_legacy_folder_names_ignored_by_default_with_hint():
    res = detect_markers(None, "C1_BTC_MSK1_2_HSV1g_PDL1_CD8", 4)
    assert res.markers is None and res.is_warning
    assert "legacy names" in res.message


def test_legacy_names_opt_in():
    res = detect_markers(None, "C1_BTC_MSK1_2_HSV1g_PDL1_CD8", 4, legacy_names=True)
    assert res.markers == ["DAPI", "HSV1g", "PDL1", "CD8"]
    assert res.dapi_assumed and not res.is_warning and "legacy names" in res.message


def test_legacy_names_too_few_words_warns():
    res = detect_markers(None, "C1_CD3_CD8", 4, legacy_names=True)
    assert res.markers is None and res.is_warning


def test_legacy_names_bracket_list_still_wins():
    res = detect_markers(None, "C1_P01_[DAPI,CD3,CD8]", 3, legacy_names=True)
    assert res.markers == ["DAPI", "CD3", "CD8"] and not res.dapi_assumed


def test_legacy_names_flat_file_ignores_stitched_suffix():
    res = detect_markers("C1_P01_CD3_CD8_cyseg-stitched.ome.tiff", "P01", 3, legacy_names=True)
    assert res.markers == ["DAPI", "CD3", "CD8"]


def test_no_list_and_no_legacy_folder_warns():
    res = detect_markers("stitched.ome.tiff", "cycle_one", 3)
    assert res.markers is None and res.is_warning


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _standard_sample(root: Path, name: str, n_cycles: int = 2) -> list[Path]:
    out = []
    for c in range(n_cycles):
        folder = root / name / f"C{c}_{name}"
        _touch(folder / "area_001_0_0.ome.tiff")
        out.append(_touch(folder / f"C{c}_{name}_cyseg-stitched.ome.tiff"))
    return out


def test_experiment_folder_finds_each_sample(tmp_path):
    _standard_sample(tmp_path, "P01")
    _standard_sample(tmp_path, "P02")
    scan = scan_input_folder(tmp_path, tmp_path / "out")
    assert scan.mode == "experiment"
    assert [s.name for s in scan.samples] == ["P01", "P02"]
    assert scan.samples[0].cycles == [0, 1]
    assert scan.samples[0].output_path == tmp_path / "out" / "P01.ome.tiff"


def test_selecting_a_sample_folder_scans_it_as_one_sample(tmp_path):
    expected = _standard_sample(tmp_path, "P01", 3)
    scan = scan_input_folder(tmp_path / "P01", tmp_path / "out")
    assert scan.mode == "sample"
    assert len(scan.samples) == 1
    assert scan.samples[0].files == expected
    assert any("single sample" in n for n in scan.notes)


def test_flat_sample_with_images_directly_in_folder(tmp_path):
    sample = tmp_path / "P01"
    files = [_touch(sample / f"C{c}_P01.ome.tiff") for c in (1, 2)]
    scan = scan_input_folder(tmp_path, tmp_path / "out")
    assert len(scan.samples) == 1
    assert scan.samples[0].files == files and scan.samples[0].cycles == [1, 2]


def test_cycle_number_from_folder_when_file_name_lacks_it(tmp_path):
    folder = tmp_path / "P01" / "C3_P01"
    img = _touch(folder / "stitched_by_other_software.ome.tiff")
    entries, notes = scan_sample_dir(tmp_path / "P01")
    assert [(e.path, e.cycle, e.enabled) for e in entries] == [(img, 3, True)]


def test_preferred_image_chosen_and_others_disabled(tmp_path):
    folder = tmp_path / "P01" / "C1_P01"
    other = _touch(folder / "C1_P01_evos.ome.tiff")
    ours = _touch(folder / "C1_P01_cyseg-stitched.ome.tiff")
    entries, notes = scan_sample_dir(tmp_path / "P01")
    assert {e.path: e.enabled for e in entries} == {ours: True, other: False}
    assert any("2 images found" in n for n in notes)


def test_unnumbered_image_listed_disabled_with_blank_cycle(tmp_path):
    _touch(tmp_path / "P01" / "C0_P01" / "C0_P01.ome.tiff")
    odd = _touch(tmp_path / "P01" / "misc" / "something.ome.tiff")
    scan = scan_input_folder(tmp_path, tmp_path / "out")
    s = scan.samples[0]
    i = s.files.index(odd)
    assert s.cycles[i] is None and s.cycles_enabled[i] is False
    assert validate_sample_cycle_numbers(s)[0]  # disabled blank rows are fine

    s.cycles_enabled[i] = True  # enabling it without a number blocks the run
    ok, msg = validate_sample_cycle_numbers(s)
    assert not ok and "no cycle number" in msg
    with pytest.raises(ValueError, match="No cycle number"):
        cycle_inputs_from_sample(s)


def test_tile_only_sample_is_skipped_with_reason(tmp_path):
    _touch(tmp_path / "P01" / "C0_P01" / "area_001_0_0.ome.tiff")
    _standard_sample(tmp_path, "P02")
    scan = scan_input_folder(tmp_path, tmp_path / "out")
    assert [s.name for s in scan.samples] == ["P02"]
    assert any("P01" in n and "Step 0" in n for n in scan.notes)


def test_selecting_a_cycle_folder_warns(tmp_path):
    _standard_sample(tmp_path, "P01", 2)
    scan = scan_input_folder(tmp_path / "P01" / "C0_P01", tmp_path / "out")
    assert len(scan.samples) == 1
    assert any("Only one cycle" in n for n in scan.samples[0].scan_notes)


def test_pyramid_work_and_temp_files_are_ignored(tmp_path):
    files = _standard_sample(tmp_path, "P01")
    _touch(tmp_path / "P01" / "C0_P01" / ".C0_P01_cyseg-stitched.ome.tiff.flat_tmp.ome.tiff")
    _touch(tmp_path / "P01" / "cycif_pyramid_work_x" / "C9_junk.ome.tiff")
    entries, _ = scan_sample_dir(tmp_path / "P01")
    assert [e.path for e in entries] == files


# ---------------------------------------------------------------------------
# cycif-seg-run automatic fallback
# ---------------------------------------------------------------------------
def test_run_pipeline_falls_back_to_non_cyseg_files_when_no_pattern_given(tmp_path):
    c1 = tmp_path / "C1_s"
    c2 = tmp_path / "C2_s"
    a = _touch(c1 / "C1_s_cyseg-stitched.ome.tiff")
    b = _touch(c2 / "C2_s_evos.ome.tiff")
    pattern, selected = _select_sample_stitched_files([c1, c2])
    assert pattern.startswith("automatic")
    assert selected == {c1: a, c2: b}


def test_run_pipeline_uses_marker_list_from_folder(tmp_path, monkeypatch):
    import cycif_seg.cli.run_pipeline as rp

    monkeypatch.setattr(rp, "_get_tile_channel_count", lambda p: 3)
    _touch(tmp_path / "C1_s_[CD3,CD8]" / "C1_s_cyseg-stitched.ome.tiff")
    cycles, errors = discover_cycles(tmp_path, require_stitched=True)
    assert errors == []
    assert cycles[0]["channel_markers"] == ["DAPI", "CD3", "CD8"]


def test_run_pipeline_accepts_shared_cycle_folder_names(tmp_path, monkeypatch):
    import cycif_seg.cli.run_pipeline as rp

    monkeypatch.setattr(rp, "_get_tile_channel_count", lambda p: 2)
    for name in ("C1[DAPI,CD3]", "cycle2_s_[DAPI,CD8]", "C3", "notes"):
        _touch(tmp_path / name / "x_cyseg-stitched.ome.tiff")
    cycles, errors = discover_cycles(tmp_path, require_stitched=True)
    assert errors == []
    assert [(c["cycle_num"], c["folder"].name) for c in cycles] == [(1, "C1[DAPI,CD3]"), (2, "cycle2_s_[DAPI,CD8]"), (3, "C3")]


def test_run_pipeline_legacy_names_flag(tmp_path, monkeypatch):
    import cycif_seg.cli.run_pipeline as rp

    monkeypatch.setattr(rp, "_get_tile_channel_count", lambda p: 3)
    _touch(tmp_path / "C1_s_CD3_CD8" / "C1_s_CD3_CD8_cyseg-stitched.ome.tiff")
    off, _ = discover_cycles(tmp_path, require_stitched=True)
    on, _ = discover_cycles(tmp_path, require_stitched=True, legacy_names=True)
    assert off[0]["channel_markers"] is None
    assert on[0]["channel_markers"] == ["DAPI", "CD3", "CD8"]

    from cycif_seg.cli.run_pipeline import _build_parser

    assert _build_parser().parse_args(["s", "--legacy_names"]).legacy_names is True
    assert _build_parser().parse_args(["s"]).legacy_names is False


# ---------------------------------------------------------------------------
# Automatic cycle configuration
# ---------------------------------------------------------------------------
def _sample_with(tmp_path, names_by_file):
    from cycif_seg.preprocess.batch_plan import BatchSample

    files = [_touch(tmp_path / rel) for rel in names_by_file]
    s = BatchSample(name="S", input_dir=tmp_path, files=files, cycles=list(range(len(files))))
    reader = {f: names_by_file[rel] for f, rel in zip(files, names_by_file)}
    return s, (lambda p: reader[Path(p)])


def test_autofill_uses_marker_list_and_metadata(tmp_path):
    from cycif_seg.preprocess.batch_plan import autofill_cycle_config, sample_has_cycle_config

    s, reader = _sample_with(tmp_path, {
        "C0_s_[CD3,CD8,PDL1]/x.ome.tiff": ["DAPI", "Cy3", "Cy5", "Cy7"],   # N-1 list -> DAPI assumed
        "C1_s/y.ome.tiff": ["Hoechst-DAPI", "A", "B"],                      # no list -> metadata names
    })
    res = autofill_cycle_config(s, reader)
    assert s.channel_markers == [["DAPI", "CD3", "CD8", "PDL1"], ["Hoechst-DAPI", "A", "B"]]
    assert s.registration_markers == ["DAPI", "Hoechst-DAPI"]
    assert s.channel_antibodies == [[""] * 4, [""] * 3]
    assert res.used_marker_list
    assert any("assumed to be DAPI" in n for n in res.notes)
    assert sample_has_cycle_config(s)


def test_autofill_registration_found_via_marker_name(tmp_path):
    from cycif_seg.preprocess.batch_plan import autofill_cycle_config

    # Metadata has generic names; the marker list says channel 2 is DAPI.
    s, reader = _sample_with(tmp_path, {"C0_s_[CD3,DAPI]/x.ome.tiff": ["Channel 1", "Channel 2"]})
    autofill_cycle_config(s, reader)
    assert s.registration_markers == ["Channel 2"]  # metadata name, as Configure cycles requires


def test_autofill_missing_dapi_and_unreadable_files_are_noted(tmp_path):
    from cycif_seg.preprocess.batch_plan import autofill_cycle_config

    s, reader = _sample_with(tmp_path, {"C0_s/x.ome.tiff": ["A", "B"], "C1_s/y.ome.tiff": None})

    def flaky(p):
        if Path(p).name == "y.ome.tiff":
            raise OSError("bad file")
        return reader(p)

    res = autofill_cycle_config(s, flaky)
    assert s.registration_markers[0] == "A"
    assert any("no DAPI channel" in n for n in res.notes)
    assert any("could not read channel names" in n for n in res.notes)
    assert not res.used_marker_list


def test_autofill_legacy_names(tmp_path):
    from cycif_seg.preprocess.batch_plan import autofill_cycle_config

    s, reader = _sample_with(tmp_path, {"C0_s_CD3_CD8/x.ome.tiff": ["c1", "c2", "c3"]})
    autofill_cycle_config(s, reader)
    assert s.channel_markers == [["c1", "c2", "c3"]]
    autofill_cycle_config(s, reader, legacy_names=True)
    assert s.channel_markers == [["DAPI", "CD3", "CD8"]]
    assert s.registration_markers == ["c1"]  # found via marker name DAPI -> channel 1's metadata name
