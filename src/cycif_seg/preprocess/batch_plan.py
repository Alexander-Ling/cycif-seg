from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cycif_seg.preprocess.naming import STITCH_OUTPUT_SUFFIX, cycle_number_from_name, detect_markers, strip_ome_tiff_ext

PLAN_SCHEMA_VERSION = 1

default_tiled_rigid_tile_size: int = 2000
default_tiled_rigid_search_factor: float = 3.0
default_fast_large_island_refinement: bool = False
default_fast_large_island_sample_count: int = 5


@dataclass
class BatchSample:
    name: str
    input_dir: Path
    files: list[Path]
    tissue: str = ""
    species: str = ""
    output_path: Path | None = None
    enabled: bool = True

    # Per-file config (same order as files)
    cycles: list[int | None] | None = None  # None = cycle number unknown; must be set before running
    registration_markers: list[str] | None = None
    channel_markers: list[list[str]] | None = None
    channel_antibodies: list[list[str]] | None = None
    cycles_enabled: list[bool] | None = None
    registration_algorithm: str = "tiled_rigid"
    global_translation_only: bool = False
    tiled_rigid_allow_rotation: bool = False
    tiled_rigid_tile_size: int = default_tiled_rigid_tile_size
    tiled_rigid_search_factor: float = default_tiled_rigid_search_factor
    fast_large_island_refinement: bool = default_fast_large_island_refinement
    fast_large_island_sample_count: int = default_fast_large_island_sample_count
    pyramidal_output: bool = True
    low_mem: bool = True
    strip_height: int | None = None  # None = auto (canvas_H // 10) when low_mem=True
    elastic_touchup: bool = True
    elastic_touchup_tile_size: int = 2048
    elastic_touchup_skip_corr: float = 0.95
    elastic_touchup_bspline_spacing: int = 50
    elastic_touchup_max_iterations: int = 10
    elastic_touchup_large_island_px: int = 4_000_000
    elastic_touchup_workers: int = 0
    elastic_touchup_max_step_length: float = 1.0
    elastic_touchup_rigid_max_shift: float = 512.0
    debug_elastic_touchup: bool = False
    debug_dir: str | None = None
    # Messages from folder scanning, shown in the batch table (not saved in plans).
    scan_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------
_TILE_PREFIXES = ("area", "raw")
_SKIP_DIR_PREFIXES = (".", "cycif_pyramid_work")


def _list_dir(d: Path) -> list[Path]:
    try:
        return sorted(d.iterdir(), key=lambda p: p.name.lower())
    except Exception:
        return []


def _is_ome_tiff(p: Path) -> bool:
    return p.name.lower().endswith((".ome.tif", ".ome.tiff")) and p.is_file()


def _is_tile(p: Path) -> bool:
    return p.name.lower().startswith(_TILE_PREFIXES)


def _is_work_file(p: Path) -> bool:
    low = p.name.lower()
    return low.startswith(".") or "__pyramid_tmp__" in low or ".flat_tmp" in low


def _subdirs(d: Path) -> list[Path]:
    return [p for p in _list_dir(d) if p.is_dir() and not p.name.lower().startswith(_SKIP_DIR_PREFIXES)]


def _images_and_tiles(folder: Path) -> tuple[list[Path], int]:
    """Candidate cycle images (non-tile OME-TIFFs) in ``folder`` and the number of raw tiles."""
    files = [p for p in _list_dir(folder) if _is_ome_tiff(p)]
    tiles = [p for p in files if _is_tile(p)]
    images = [p for p in files if not _is_tile(p) and not _is_work_file(p)]
    return images, len(tiles)


def _image_preference(p: Path) -> tuple[int, str]:
    """Lower is preferred: cycif-seg stitched output, then files named C<number>_..., then anything else."""
    stem = strip_ome_tiff_ext(p.name).lower()
    if stem.endswith("_" + STITCH_OUTPUT_SUFFIX):
        tier = 0
    elif cycle_number_from_name(p.name) is not None:
        tier = 1
    else:
        tier = 2
    return tier, p.name.lower()


def choose_cycle_image(folder: Path) -> tuple[Path | None, list[Path]]:
    """The image to use for a cycle folder, plus the other candidates it was chosen over."""
    ranked = sorted(_images_and_tiles(folder)[0], key=_image_preference)
    return (ranked[0], ranked[1:]) if ranked else (None, [])


@dataclass
class CycleEntry:
    path: Path
    cycle: int | None
    enabled: bool = True


def scan_sample_dir(sample_dir: Path) -> tuple[list[CycleEntry], list[str]]:
    """Find cycle images for one sample.

    Accepts both layouts::

        sample/C1_.../<image>.ome.tiff     (one sub-folder per cycle)
        sample/C1_....ome.tiff             (cycle images directly in the sample folder)

    In a cycle folder, one image is chosen (cycif-seg stitched output first, then a file named
    ``C<number>_...``); any other images are listed but disabled. The cycle number comes from the
    file name, else the folder name. Images with no cycle number are listed, disabled, with a blank
    number for the user to fill in.
    """
    entries: list[CycleEntry] = []
    notes: list[str] = []
    tile_only: list[str] = []

    for sub in _subdirs(sample_dir):
        best, others = choose_cycle_image(sub)
        if best is None:
            if _images_and_tiles(sub)[1]:
                tile_only.append(sub.name)
            continue
        folder_cy = cycle_number_from_name(sub.name)
        ranked = [best] + others
        for k, p in enumerate(ranked):
            cy = cycle_number_from_name(p.name)
            if cy is None:
                cy = folder_cy
            entries.append(CycleEntry(p, cy, enabled=(k == 0 and cy is not None)))
        if len(ranked) > 1:
            notes.append(
                f"{sub.name}: {len(ranked)} images found; using {best.name}. "
                f"The others are listed but disabled. Switch them in Configure cycles if needed."
            )
        if entries[-len(ranked)].cycle is None:
            notes.append(
                f"{sub.name}: no cycle number in the folder or file name ({best.name}). "
                f"Left disabled. Set its cycle number in Configure cycles."
            )

    unnumbered_direct: list[str] = []
    for p in _images_and_tiles(sample_dir)[0]:
        cy = cycle_number_from_name(p.name)
        entries.append(CycleEntry(p, cy, enabled=cy is not None))
        if cy is None:
            unnumbered_direct.append(p.name)
    if unnumbered_direct and any(e.cycle is not None for e in entries):
        notes.append(
            f"{len(unnumbered_direct)} image(s) without a cycle number were left disabled: "
            + ", ".join(unnumbered_direct)
        )

    if tile_only:
        notes.append(
            f"{len(tile_only)} cycle folder(s) contain raw tiles but no stitched image "
            f"({', '.join(tile_only)}). Run Step 0 (stitching) first."
        )

    entries.sort(key=lambda e: (e.cycle is None, e.cycle if e.cycle is not None else 0, str(e.path).lower()))
    return entries, notes


def looks_like_sample_dir(d: Path) -> bool:
    """True if ``d`` holds cycle folders or numbered cycle images itself (i.e. it is a sample, not an experiment)."""
    for sub in _subdirs(d):
        if cycle_number_from_name(sub.name) is not None:
            images, n_tiles = _images_and_tiles(sub)
            if images or n_tiles:
                return True
    return any(cycle_number_from_name(p.name) is not None for p in _images_and_tiles(d)[0])


@dataclass
class ScanResult:
    samples: list[BatchSample]
    notes: list[str]
    mode: str  # "experiment" or "sample"


def scan_input_folder(
    root_dir: Path,
    output_dir: Path,
    tissue: str = "",
    species: str = "",
) -> ScanResult:
    """Scan the folder the user selected: an experiment folder (one sub-folder per sample) or a single sample folder."""
    root_dir = Path(root_dir)
    notes: list[str] = []
    if looks_like_sample_dir(root_dir):
        mode = "sample"
        candidates = [root_dir]
        notes.append(
            f"'{root_dir.name}' contains cycle folders or cycle images, so it was scanned as a single sample."
        )
    else:
        mode = "experiment"
        candidates = _subdirs(root_dir)

    samples: list[BatchSample] = []
    empty: list[str] = []
    for d in candidates:
        entries, sample_notes = scan_sample_dir(d)
        if not any(e.enabled for e in entries):
            if sample_notes or entries:
                reason = " ".join(sample_notes) or "no image has a cycle number"
                notes.append(f"Skipped '{d.name}': {reason}")
            else:
                empty.append(d.name)
            continue
        n_enabled = sum(1 for e in entries if e.enabled)
        if n_enabled == 1:
            sample_notes.append(
                "Only one cycle was found. If you selected a cycle folder, select the sample or experiment folder instead."
            )
        samples.append(
            BatchSample(
                name=d.name,
                input_dir=d,
                files=[e.path for e in entries],
                tissue=tissue,
                species=species,
                output_path=(Path(output_dir) / f"{d.name}.ome.tiff"),
                enabled=True,
                cycles=[e.cycle for e in entries],
                cycles_enabled=[bool(e.enabled) for e in entries],
                registration_algorithm="tiled_rigid",
                global_translation_only=False,
                scan_notes=sample_notes,
            )
        )
    if empty and mode == "experiment":
        notes.append(f"Skipped {len(empty)} folder(s) with no cycle images: {', '.join(empty)}")
    if not samples and mode == "experiment" and not candidates:
        notes.append(f"'{root_dir.name}' has no sub-folders. Select the experiment folder (one sub-folder per sample).")
    return ScanResult(samples=samples, notes=notes, mode=mode)


def sample_cycle_number(s: BatchSample, i: int) -> int | None:
    """Cycle number of file ``i`` (``None`` if unknown). Plans without a cycles list number files in order."""
    if s.cycles and i < len(s.cycles):
        return None if s.cycles[i] is None else int(s.cycles[i])
    return i


def letter_suffix(idx: int) -> str:
    """0 → 'a', 1 → 'b', …, 25 → 'z', 26 → 'aa', …"""
    result = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = chr(ord("a") + rem) + result
    return result


def resolve_cycle_labels(numbers: list[int], paths: list[str | Path]) -> tuple[list[int], list[str]]:
    """Make cycle numbers unique the same way ``cycif-seg-run`` does.

    The same cycle number in *different* cycle folders (e.g. a re-imaged cycle) is kept,
    labelled ``1a``, ``1b``, … in folder-name order. Two files with the same cycle number
    in the *same* folder are ambiguous and raise ``ValueError``.

    Returns ``(cycle_ints, labels)`` aligned with the inputs. Labels name the output channels
    (``<marker>_cy<label>``) and are always set, matching ``cycif-seg-run`` so resume
    fingerprints agree. Without duplicates the numbers are returned unchanged; with
    duplicates every cycle gets a unique rank-ordered int (lowest cycle number first, so it
    stays the reference cycle).
    """
    numbers = [int(n) for n in numbers]
    paths = [Path(str(p)) for p in paths]
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(numbers):
        groups.setdefault(n, []).append(i)
    if all(len(g) == 1 for g in groups.values()):
        return list(numbers), [str(n) for n in numbers]

    def _key(i: int) -> tuple:
        return (numbers[i], paths[i].parent.name.lower(), paths[i].name.lower())

    for n, g in groups.items():
        by_folder: dict[Path, list[str]] = {}
        for i in g:
            by_folder.setdefault(paths[i].parent, []).append(paths[i].name)
        for folder, names in by_folder.items():
            if len(names) > 1:
                raise ValueError(
                    f"Cycle {n} has more than one file in folder '{folder.name}': {', '.join(sorted(names))}. "
                    f"Disable all but one of them."
                )

    ints = [0] * len(numbers)
    labels = [""] * len(numbers)
    for rank, i in enumerate(sorted(range(len(numbers)), key=_key)):
        ints[i] = rank
        g = sorted(groups[numbers[i]], key=_key)
        labels[i] = f"{numbers[i]}{letter_suffix(g.index(i))}" if len(g) > 1 else str(numbers[i])
    return ints, labels


def enabled_cycle_entries(s: BatchSample) -> list[int]:
    """Indices into ``s.files`` of the enabled cycles."""
    return [
        i
        for i, _ in enumerate(s.files or [])
        if bool(s.cycles_enabled[i] if s.cycles_enabled and i < len(s.cycles_enabled) else True)
    ]


def _missing_cycle_numbers(s: BatchSample, idx: list[int]) -> list[str]:
    return [Path(str(s.files[i])).name for i in idx if sample_cycle_number(s, i) is None]


def validate_sample_cycle_numbers(s: BatchSample, *, allow_missing: bool = False) -> tuple[bool, str]:
    idx = enabled_cycle_entries(s)
    missing = _missing_cycle_numbers(s, idx)
    if missing and not allow_missing:
        return False, (
            f"Sample '{s.name}': no cycle number for {', '.join(missing)}. "
            f"Set it in Configure cycles, or disable that file."
        )
    idx = [i for i in idx if sample_cycle_number(s, i) is not None]
    try:
        resolve_cycle_labels([sample_cycle_number(s, i) for i in idx], [s.files[i] for i in idx])
    except ValueError as e:
        return False, f"Sample '{s.name}': {e} Reconfigure this sample before running batch preprocessing."
    return True, ""


def cycle_inputs_from_sample(s: BatchSample) -> list:
    """Build the enabled ``CycleInput`` list for a sample (raises ``ValueError`` on ambiguous cycles)."""
    from cycif_seg.preprocess.organize_cycles import CycleInput

    def _at(values, i):
        return values[i] if values and i < len(values) else None

    idx = enabled_cycle_entries(s)
    missing = _missing_cycle_numbers(s, idx)
    if missing:
        raise ValueError(f"No cycle number for {', '.join(missing)}. Set it in Configure cycles, or disable that file.")
    numbers = [int(sample_cycle_number(s, i)) for i in idx]
    ints, labels = resolve_cycle_labels(numbers, [s.files[i] for i in idx])
    return [
        CycleInput(
            path=str(s.files[i]),
            cycle=int(cy),
            label=label,
            tissue=(s.tissue or None),
            species=(s.species or None),
            registration_marker=_at(s.registration_markers, i),
            channel_markers=_at(s.channel_markers, i),
            channel_antibodies=_at(s.channel_antibodies, i),
        )
        for i, cy, label in zip(idx, ints, labels)
    ]


def _find_name(names: list[str], target: str) -> int | None:
    """Index of ``target`` in ``names``: exact (case-insensitive) match first, then substring."""
    t = target.strip().lower()
    for i, nm in enumerate(names):
        if (nm or "").strip().lower() == t:
            return i
    for i, nm in enumerate(names):
        if t and t in (nm or "").strip().lower():
            return i
    return None


@dataclass
class AutofillResult:
    notes: list[str]  # things the user should check, per sample
    used_marker_list: bool  # at least one cycle's names came from a [..] list (or legacy names)


def autofill_cycle_config(
    s: BatchSample,
    read_channel_names: Callable[[Path], list[str]],
    *,
    registration_marker: str = "DAPI",
    legacy_names: bool = False,
) -> AutofillResult:
    """Fill a sample's per-cycle config (what Configure cycles would default to).

    For each cycle image: channel names are read from the image metadata; marker names come
    from the ``[..]`` marker list in the file/folder name when it matches the channel count
    (or legacy names, if enabled), else the metadata names. The registration channel is the
    ``registration_marker`` (DAPI) channel, matched by metadata name, then by marker name;
    if neither has it, channel 1 is used and a note is added.
    """
    notes: list[str] = []
    used_list = False
    reg_markers: list[str] = []
    ch_markers: list[list[str]] = []
    antibodies: list[list[str]] = []
    for p in s.files:
        p = Path(p)
        try:
            names = [str(x) for x in (read_channel_names(p) or [])]
        except Exception as e:
            names = []
            notes.append(f"{p.name}: could not read channel names ({e}). Configure this sample manually.")
        res = detect_markers(p.name, p.parent.name, len(names), legacy_names=legacy_names) if names else None
        if res is not None and res.markers is not None:
            markers = list(res.markers)
            used_list = True
            if res.dapi_assumed or res.is_warning:
                notes.append(res.message)
        else:
            markers = list(names)
            if res is not None and res.source:  # a [..] list was present but unusable
                notes.append(res.message + " Using the image's own channel names.")
        idx = _find_name(names, registration_marker)
        if idx is None:
            idx = _find_name(markers, registration_marker)
        if idx is not None and idx < len(names):
            reg = names[idx]
        else:
            reg = names[0] if names else registration_marker
            if names:
                notes.append(
                    f"{p.name}: no {registration_marker} channel found; registering on channel 1 ({reg}). "
                    f"Change it in Configure cycles if that's wrong."
                )
        reg_markers.append(reg)
        ch_markers.append(markers)
        antibodies.append([""] * len(markers))
    s.registration_markers = reg_markers
    s.channel_markers = ch_markers
    s.channel_antibodies = antibodies
    if s.cycles is None:
        s.cycles = [None] * len(s.files)
    if s.cycles_enabled is None:
        s.cycles_enabled = [True] * len(s.files)
    return AutofillResult(notes=notes, used_marker_list=used_list)


def sample_has_cycle_config(s: BatchSample) -> bool:
    return bool(s.cycles and s.registration_markers and s.channel_markers and s.channel_antibodies)


def validate_samples_ready(samples: list[BatchSample]) -> tuple[bool, str]:
    """Check all enabled samples are fully configured and ready to run."""
    enabled = [s for s in samples if s.enabled]
    if not enabled:
        return False, "No samples enabled."
    for s in enabled:
        if not s.output_path:
            return False, f"Sample '{s.name}' missing output path."
        if not s.files:
            return False, f"Sample '{s.name}' has no input files."
        if not sample_has_cycle_config(s):
            return False, (
                f"Sample '{s.name}' missing per-cycle configuration "
                f"(cycles, registration markers, channel names)."
            )
        if s.cycles_enabled and not any(bool(x) for x in s.cycles_enabled):
            return False, f"Sample '{s.name}' has all cycles disabled. Enable at least one."
        ok, msg = validate_sample_cycle_numbers(s)
        if not ok:
            return False, msg
    return True, ""


def plan_to_dict(
    samples: list[BatchSample],
    root_dir: str | Path = "",
    output_dir: str | Path = "",
    default_tissue: str = "",
    default_species: str = "",
) -> dict:
    d: dict = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "defaults": {
            "registration_algorithm": "tiled_rigid",
            "global_translation_only": False,
            "tiled_rigid_tile_size": default_tiled_rigid_tile_size,
            "tiled_rigid_search_factor": default_tiled_rigid_search_factor,
            "fast_large_island_refinement": default_fast_large_island_refinement,
            "fast_large_island_sample_count": default_fast_large_island_sample_count,
            "tissue": default_tissue,
            "species": default_species,
        },
        "samples": [],
        "registration_algorithm": "tiled_rigid",
        "global_translation_only": False,
    }
    for s in samples:
        d["samples"].append({
            "name": s.name,
            "input_dir": str(s.input_dir),
            "files": [str(p) for p in s.files],
            "tissue": s.tissue,
            "species": s.species,
            "output_path": str(s.output_path or ""),
            "enabled": bool(s.enabled),
            "cycles": list(s.cycles or []),
            "registration_markers": list(s.registration_markers or []),
            "channel_markers": list(s.channel_markers or []),
            "channel_antibodies": list(s.channel_antibodies or []),
            "cycles_enabled": list(s.cycles_enabled or []),
            "registration_algorithm": str(getattr(s, "registration_algorithm", "tiled_rigid") or "tiled_rigid"),
            "global_translation_only": bool(getattr(s, "global_translation_only", False)),
            "tiled_rigid_allow_rotation": bool(getattr(s, "tiled_rigid_allow_rotation", False)),
            "tiled_rigid_tile_size": int(getattr(s, "tiled_rigid_tile_size", default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size),
            "tiled_rigid_search_factor": float(getattr(s, "tiled_rigid_search_factor", default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor),
            "fast_large_island_refinement": False,
            "fast_large_island_sample_count": int(getattr(s, "fast_large_island_sample_count", default_fast_large_island_sample_count) or default_fast_large_island_sample_count),
            "pyramidal_output": bool(getattr(s, "pyramidal_output", True)),
            "low_mem": bool(getattr(s, "low_mem", True)),
            "strip_height": int(getattr(s, "strip_height") or 0) if getattr(s, "strip_height", None) is not None else None,
            "elastic_touchup": bool(getattr(s, "elastic_touchup", False)),
            "elastic_touchup_tile_size": int(getattr(s, "elastic_touchup_tile_size", 2048) or 2048),
            "elastic_touchup_skip_corr": float(getattr(s, "elastic_touchup_skip_corr", 0.95) or 0.95),
            "elastic_touchup_bspline_spacing": int(getattr(s, "elastic_touchup_bspline_spacing", 50) or 50),
            "elastic_touchup_max_iterations": int(getattr(s, "elastic_touchup_max_iterations", 10) or 10),
            "elastic_touchup_large_island_px": int(getattr(s, "elastic_touchup_large_island_px", 4_000_000) or 4_000_000),
            "elastic_touchup_workers": int(getattr(s, "elastic_touchup_workers", 0) or 0),
            "elastic_touchup_max_step_length": float(getattr(s, "elastic_touchup_max_step_length", 1.0) or 1.0),
            "elastic_touchup_rigid_max_shift": float(getattr(s, "elastic_touchup_rigid_max_shift", 512.0) or 512.0),
        })
    return d


def plan_from_dict(d: dict) -> dict[str, Any]:
    """
    Deserialise a plan dict loaded from JSON.

    Returns ``{"root_dir": str, "output_dir": str, "defaults": dict, "samples": list[BatchSample]}``.
    Raises ``ValueError`` on unsupported schema version or duplicate cycle numbers.
    """
    if int(d.get("schema_version") or 0) != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported plan schema version: {d.get('schema_version')!r}")

    defs = d.get("defaults") or {}
    samples: list[BatchSample] = []

    for rec in d.get("samples") or []:
        s = BatchSample(
            name=str(rec.get("name") or ""),
            input_dir=Path(str(rec.get("input_dir") or "")).expanduser(),
            files=[Path(p).expanduser() for p in (rec.get("files") or [])],
            tissue=str(rec.get("tissue") or ""),
            species=str(rec.get("species") or ""),
            output_path=Path(str(rec.get("output_path"))).expanduser() if rec.get("output_path") else None,
            enabled=bool(rec.get("enabled", True)),
        )
        s.registration_algorithm = str(
            rec.get("registration_algorithm") or defs.get("registration_algorithm") or "tiled_rigid"
        )
        s.global_translation_only = bool(
            rec.get("global_translation_only", defs.get("global_translation_only", False))
        )
        if s.global_translation_only:
            s.registration_algorithm = "translation"
        s.tiled_rigid_allow_rotation = bool(rec.get("tiled_rigid_allow_rotation", False))
        s.tiled_rigid_tile_size = max(128, int(
            rec.get("tiled_rigid_tile_size", defs.get("tiled_rigid_tile_size", default_tiled_rigid_tile_size))
            or default_tiled_rigid_tile_size
        ))
        s.tiled_rigid_search_factor = max(1.0, float(
            rec.get("tiled_rigid_search_factor", defs.get("tiled_rigid_search_factor", default_tiled_rigid_search_factor))
            or default_tiled_rigid_search_factor
        ))
        s.fast_large_island_refinement = False
        s.fast_large_island_sample_count = max(1, int(
            rec.get("fast_large_island_sample_count", defs.get("fast_large_island_sample_count", default_fast_large_island_sample_count))
            or default_fast_large_island_sample_count
        ))
        s.pyramidal_output = bool(rec.get("pyramidal_output", True))
        s.low_mem = bool(rec.get("low_mem", True))
        _sh = rec.get("strip_height", None)
        s.strip_height = int(_sh) if _sh is not None and int(_sh) > 0 else None
        s.elastic_touchup = bool(rec.get("elastic_touchup", defs.get("elastic_touchup", True)))
        s.elastic_touchup_tile_size = max(64, int(rec.get("elastic_touchup_tile_size", defs.get("elastic_touchup_tile_size", 2048)) or 2048))
        s.elastic_touchup_skip_corr = float(rec.get("elastic_touchup_skip_corr", defs.get("elastic_touchup_skip_corr", 0.95)) or 0.95)
        s.elastic_touchup_bspline_spacing = max(4, int(rec.get("elastic_touchup_bspline_spacing", defs.get("elastic_touchup_bspline_spacing", 50)) or 50))
        s.elastic_touchup_max_iterations = max(1, int(rec.get("elastic_touchup_max_iterations", defs.get("elastic_touchup_max_iterations", 10)) or 10))
        s.elastic_touchup_large_island_px = max(1, int(rec.get("elastic_touchup_large_island_px", defs.get("elastic_touchup_large_island_px", 4_000_000)) or 4_000_000))
        s.elastic_touchup_workers = max(0, int(rec.get("elastic_touchup_workers", defs.get("elastic_touchup_workers", 0)) or 0))
        s.elastic_touchup_max_step_length = max(0.01, float(rec.get("elastic_touchup_max_step_length", defs.get("elastic_touchup_max_step_length", 1.0)) or 1.0))
        s.elastic_touchup_rigid_max_shift = max(1.0, float(rec.get("elastic_touchup_rigid_max_shift", defs.get("elastic_touchup_rigid_max_shift", 512.0)) or 512.0))
        s.debug_elastic_touchup = bool(rec.get("debug_elastic_touchup", defs.get("debug_elastic_touchup", False)))
        s.debug_dir = str(rec.get("debug_dir") or defs.get("debug_dir") or "") or None
        s.cycles = list(rec.get("cycles") or []) or None
        s.registration_markers = list(rec.get("registration_markers") or []) or None
        s.channel_markers = list(rec.get("channel_markers") or []) or None
        s.channel_antibodies = list(rec.get("channel_antibodies") or []) or None
        s.cycles_enabled = list(rec.get("cycles_enabled") or []) or None

        # Blank cycle numbers are allowed in a saved plan; they are caught before running.
        ok, msg = validate_sample_cycle_numbers(s, allow_missing=True)
        if not ok:
            raise ValueError(msg)
        samples.append(s)

    return {
        "root_dir": str(d.get("root_dir") or ""),
        "output_dir": str(d.get("output_dir") or ""),
        "defaults": defs,
        "samples": samples,
    }
