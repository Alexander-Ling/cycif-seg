from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = 1

default_tiled_rigid_tile_size: int = 2000
default_tiled_rigid_search_factor: float = 3.0
default_fast_large_island_refinement: bool = False
default_fast_large_island_sample_count: int = 5


def parse_cycle_number_from_filename(name: str) -> int | None:
    """Parse cycle number from stitched file names like ``C3_sample_...ome.tiff``."""
    try:
        m = re.match(r"^c(\d+)_", str(name or "").strip(), flags=re.IGNORECASE)
        if not m:
            return None
        return int(m.group(1))
    except Exception:
        return None


def find_stitched_cycle_files_in_sample_dir(sample_dir: Path) -> list[tuple[Path, int]]:
    """
    Discover stitched cycle OME-TIFFs using the expected layout::

        {input}/{sample_dir}/{cycle_dir}/C#_{stuff}.ome.tiff

    Rules:
    - only inspect immediate subdirectories of ``sample_dir`` as cycle dirs
    - ignore any OME-TIFF directly under ``sample_dir``
    - ignore tile files such as ``area_*.ome.tiff``
    - infer cycle number from the stitched filename prefix ``C#_``
    """
    out: list[tuple[Path, int]] = []
    try:
        cycle_dirs = [p for p in sorted(sample_dir.iterdir()) if p.is_dir()]
    except Exception:
        return out

    for cycle_dir in cycle_dirs:
        try:
            files = [p for p in sorted(cycle_dir.iterdir()) if p.is_file()]
        except Exception:
            continue
        for p in files:
            nm = p.name
            low = nm.lower()
            if ".ome." not in low:
                continue
            if not (low.endswith('.ome.tif') or low.endswith('.ome.tiff')):
                continue
            if low.startswith('area_'):
                continue
            cy = parse_cycle_number_from_filename(nm)
            if cy is None:
                continue
            out.append((p, int(cy)))

    out.sort(key=lambda t: (int(t[1]), str(t[0]).lower()))
    return out


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
    cycles: list[int] | None = None
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


def scan_root_for_samples(
    root_dir: Path,
    output_dir: Path,
    tissue: str = "",
    species: str = "",
) -> list[BatchSample]:
    """Scan ``root_dir`` for sample subdirectories, returning a list of BatchSample."""
    samples: list[BatchSample] = []
    try:
        subdirs = sorted(root_dir.iterdir())
    except Exception:
        return samples
    for sub in subdirs:
        if not sub.is_dir():
            continue
        stitched = find_stitched_cycle_files_in_sample_dir(sub)
        if not stitched:
            continue
        files = [p for (p, _cy) in stitched]
        cycles = [int(cy) for (_p, cy) in stitched]
        samples.append(
            BatchSample(
                name=sub.name,
                input_dir=sub,
                files=files,
                tissue=tissue,
                species=species,
                output_path=(output_dir / f"{sub.name}.ome.tiff"),
                enabled=True,
                cycles=cycles,
                registration_algorithm="tiled_rigid",
                global_translation_only=False,
            )
        )
    return samples


def enabled_cycle_numbers(s: BatchSample) -> list[int]:
    return [
        int(s.cycles[i] if s.cycles and i < len(s.cycles) else i)
        for i, _ in enumerate(s.files or [])
        if bool(s.cycles_enabled[i] if s.cycles_enabled and i < len(s.cycles_enabled) else True)
    ]


def duplicate_cycle_numbers(s: BatchSample) -> list[int]:
    seen: set[int] = set()
    dupes: set[int] = set()
    for cy in enabled_cycle_numbers(s):
        if cy in seen:
            dupes.add(cy)
        seen.add(cy)
    return sorted(dupes)


def validate_sample_cycle_numbers(s: BatchSample) -> tuple[bool, str]:
    dupes = duplicate_cycle_numbers(s)
    if dupes:
        return (
            False,
            f"Sample '{s.name}' has duplicate enabled cycle number(s): {dupes}. "
            f"Enabled cycle numbers are: {enabled_cycle_numbers(s)}. "
            f"Reconfigure this sample before running batch preprocessing.",
        )
    return True, ""


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
        s.cycles = list(rec.get("cycles") or []) or None
        s.registration_markers = list(rec.get("registration_markers") or []) or None
        s.channel_markers = list(rec.get("channel_markers") or []) or None
        s.channel_antibodies = list(rec.get("channel_antibodies") or []) or None
        s.cycles_enabled = list(rec.get("cycles_enabled") or []) or None

        ok, msg = validate_sample_cycle_numbers(s)
        if not ok:
            raise ValueError(msg)
        samples.append(s)

    return {
        "root_dir": str(d.get("root_dir") or ""),
        "output_dir": str(d.get("output_dir") or ""),
        "defaults": defs,
        "samples": samples,
    }
