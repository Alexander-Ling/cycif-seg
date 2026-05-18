"""
cycif-seg-run — stitch + register a single-sample CycIF tile directory.

Usage:
    cycif-seg-run SAMPLE_DIR [options]

Discovers cycle subdirectories (named C<N>_...) inside SAMPLE_DIR, stitches
tiles in each, then co-registers the stitched cycles into one merged OME-TIFF.

Channel names are parsed from the folder name: the last (n_channels - 1) tokens
(split on '_') are the non-DAPI channel markers; DAPI is always prepended as
channel 0.

Example:
    Folder: C1_BTC_MSK1_2_HSV1g_PDL1_CD8  (4-channel tiles)
    Channels: DAPI, HSV1g, PDL1, CD8
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from string import ascii_lowercase


_CYCLE_DIR_RE = re.compile(r"^C(\d+)_(.+)$", re.IGNORECASE)
_STITCH_SUFFIX = "cyseg-stitched"
_MERGE_SUFFIX = "cyseg-merged"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _get_tile_channel_count(tile_path: Path) -> int:
    from cycif_seg.io.ome_tiff import load_channel_names_only_fast
    try:
        names = load_channel_names_only_fast(str(tile_path))
        return max(1, len(names))
    except Exception:
        return 1


def _parse_channel_markers(name_suffix: str, n_channels: int) -> tuple[list[str], bool]:
    """Return (channel_markers, had_warning).

    Takes the last n_channels-1 underscore-split tokens from name_suffix and
    prepends "DAPI".  Returns had_warning=True if the suffix had too few tokens.
    """
    if n_channels <= 1:
        return ["DAPI"], False
    tokens = name_suffix.split("_")
    needed = n_channels - 1
    had_warning = len(tokens) < needed
    markers = tokens[-needed:] if len(tokens) >= needed else tokens
    # pad with empty strings if too few
    while len(markers) < needed:
        markers.append("")
    return ["DAPI"] + markers, had_warning


def _letter_suffix(idx: int) -> str:
    """0 → 'a', 1 → 'b', …, 25 → 'z', 26 → 'aa', …"""
    result = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = ascii_lowercase[rem] + result
    return result


def discover_cycles(
    sample_dir: Path,
    tile_regex: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Scan sample_dir for cycle subdirectories.

    Returns (cycle_infos, errors).

    Each cycle_info dict has keys:
        folder, cycle_num, cycle_int, label, name_suffix, channel_markers,
        n_channels, n_tiles
    """
    from cycif_seg.stitch import discover_cycle_tiles

    errors: list[str] = []
    warnings: list[str] = []

    # Collect all matching subdirectories
    raw: list[tuple[int, str, Path]] = []  # (cycle_num, name_suffix, folder)
    skipped: list[str] = []
    for entry in sorted(sample_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = _CYCLE_DIR_RE.match(entry.name)
        if not m:
            skipped.append(entry.name)
            continue
        raw.append((int(m.group(1)), m.group(2), entry))

    if skipped:
        print(
            f"  [SKIP] {len(skipped)} subdirector{'y' if len(skipped)==1 else 'ies'} "
            f"do not match the C<N>_... pattern and will be ignored: "
            + ", ".join(skipped),
            file=sys.stderr,
        )

    if not raw:
        return [], errors

    # Group by cycle_num to detect duplicates
    by_num: dict[int, list[tuple[str, Path]]] = {}
    for cycle_num, suffix, folder in raw:
        by_num.setdefault(cycle_num, []).append((suffix, folder))

    cycle_infos: list[dict] = []
    cycle_int_counter = 0  # monotonically increasing; guarantees uniqueness across all cycles

    for cycle_num in sorted(by_num):
        entries = by_num[cycle_num]
        is_dup = len(entries) > 1

        if is_dup:
            folders_str = ", ".join(f.name for _, f in entries)
            print(
                f"  [WARNING] Multiple directories share cycle number {cycle_num}: "
                f"{folders_str}. "
                f"Processing all of them with letter suffixes.",
                file=sys.stderr,
            )

        for idx, (suffix, folder) in enumerate(sorted(entries, key=lambda x: x[1].name)):
            label = f"{cycle_num}{_letter_suffix(idx)}" if is_dup else str(cycle_num)
            cycle_int = cycle_int_counter
            cycle_int_counter += 1

            # Count tiles
            tiles = discover_cycle_tiles(folder, tile_filename_regex=tile_regex)
            if not tiles:
                errors.append(
                    f"Cycle directory '{folder.name}' matches the C<N>_... pattern "
                    f"but contains no tile files."
                )
                continue

            # Get channel count from one tile
            sample_tile = next(iter(tiles.values()))
            n_channels = _get_tile_channel_count(sample_tile)
            channel_markers, had_warning = _parse_channel_markers(suffix, n_channels)

            if had_warning:
                print(
                    f"  [WARNING] '{folder.name}': folder name has fewer tokens than "
                    f"expected for {n_channels} channels. Missing channel names filled "
                    f"with empty strings.",
                    file=sys.stderr,
                )

            cycle_infos.append({
                "folder": folder,
                "cycle_num": cycle_num,
                "cycle_int": cycle_int,
                "label": label,
                "name_suffix": suffix,
                "channel_markers": channel_markers,
                "n_channels": n_channels,
                "n_tiles": len(tiles),
            })

    return cycle_infos, errors


# ---------------------------------------------------------------------------
# Stitch step
# ---------------------------------------------------------------------------

def _run_stitch(
    cycle_info: dict,
    *,
    stitch_channel: int,
    n_workers: int,
    pyramidal_output: bool,
    tile_regex: str | None,
    force: bool = False,
) -> str:
    from cycif_seg.stitch import stitch_cycle_tiles

    folder: Path = cycle_info["folder"]
    label: str = cycle_info["label"]
    expected = folder / f"{folder.name}_{_STITCH_SUFFIX}.ome.tiff"

    if not force and expected.exists():
        print(f"\n--- Cycle {label}: {folder.name} ---")
        print(f"  [skip] Stitched file already exists: {expected.name}")
        return str(expected)

    print(f"\n--- Stitching cycle {label}: {folder.name} ({cycle_info['n_tiles']} tiles) ---")

    result = stitch_cycle_tiles(
        folder,
        output_suffix=_STITCH_SUFFIX,
        stitch_channel=stitch_channel,
        n_workers=n_workers,
        pyramidal_output=pyramidal_output,
        tile_filename_regex=tile_regex or None,
        progress_cb=lambda msg: print(f"  {msg}"),
    )
    out = result["output_path"]
    shp = result["shape_yxc"]
    print(f"  Done. Output: {out}  ({shp[1]}x{shp[0]} px, {shp[2]} channels)")
    return out


def _find_stitched_file(cycle_info: dict) -> str:
    folder: Path = cycle_info["folder"]
    expected = folder / f"{folder.name}_{_STITCH_SUFFIX}.ome.tiff"
    if not expected.exists():
        raise FileNotFoundError(
            f"--skip-stitch: expected stitched file not found: {expected}"
        )
    return str(expected)


# ---------------------------------------------------------------------------
# Registration step
# ---------------------------------------------------------------------------

def _run_registration(
    cycle_infos: list[dict],
    stitched_paths: list[str],
    *,
    output_path: Path,
    registration_marker: str,
    registration_algorithm: str,
    strip_height: int | None,
    pyramidal_output: bool,
    force: bool = False,
) -> None:
    from cycif_seg.preprocess.organize_cycles import CycleInput, merge_cycles_to_ome_tiff

    if not force and output_path.exists():
        print(f"\n--- Registration ---")
        print(f"  [skip] Merged file already exists: {output_path}")
        print(f"  Pass --force-register to overwrite.")
        return

    cycles: list[CycleInput] = []
    for ci, spath in zip(cycle_infos, stitched_paths):
        cycles.append(CycleInput(
            path=spath,
            cycle=ci["cycle_int"],
            label=ci["label"],
            channel_markers=ci["channel_markers"],
            registration_marker=registration_marker,
        ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Registering {len(cycles)} cycle(s) → {output_path} ---")

    merge_cycles_to_ome_tiff(
        cycles=cycles,
        output_path=str(output_path),
        default_registration_marker=registration_marker,
        registration_algorithm=registration_algorithm,
        strip_height=strip_height,
        pyramidal_output=pyramidal_output,
        progress_cb=lambda msg: print(f"  {msg}"),
    )
    print(f"  Done. Merged output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cycif-seg-run",
        description=(
            "Stitch tile files in a CycIF sample directory, then co-register "
            "all cycles into a single merged OME-TIFF. Cycle subdirectories must "
            "be named C<N>_<rest> (e.g. C1_BTC_MSK1_2_HSV1g_PDL1_CD8). The first "
            "channel of each cycle is always DAPI; remaining channel names are taken "
            "from the last tokens of the folder name."
        ),
    )
    p.add_argument("sample_dir", help="Path to sample directory containing C<N>_... subdirectories")

    out_grp = p.add_argument_group("output")
    out_grp.add_argument("--output-dir", default=None, metavar="PATH",
                         help="Output directory for the merged TIFF (default: SAMPLE_DIR)")
    out_grp.add_argument("--output-name", default=None, metavar="NAME",
                         help="Merged output filename (default: <sample_dir_name>_cyseg-merged.ome.tiff)")

    stitch_grp = p.add_argument_group("stitching")
    stitch_grp.add_argument("--stitch-channel", type=int, default=0, metavar="INT",
                             help="Channel index used for tile alignment (default: 0 = DAPI)")
    stitch_grp.add_argument("--n-workers", type=int, default=1, metavar="INT",
                             help="Parallel worker threads for stitching (default: 1)")
    stitch_grp.add_argument("--tile-regex", default=None, metavar="REGEX",
                             help="Override default tile filename regex pattern")

    reg_grp = p.add_argument_group("registration")
    reg_grp.add_argument("--registration-marker", default="DAPI", metavar="STR",
                          help="Channel name used for inter-cycle registration (default: DAPI)")
    reg_grp.add_argument("--registration-algorithm", default="tiled_rigid",
                          choices=["tiled_rigid", "translation"],
                          help="Registration algorithm (default: tiled_rigid)")
    reg_grp.add_argument("--strip-height", type=int, default=None, metavar="INT",
                          help="Process registration in horizontal strips of this height (reduces RAM)")

    misc_grp = p.add_argument_group("misc")
    misc_grp.add_argument("--no-pyramidal", action="store_true",
                           help="Write flat OME-TIFF instead of pyramidal")

    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument("--stitch-only", action="store_true",
                           help="Stop after stitching; skip registration")
    mode_grp.add_argument("--skip-stitch", action="store_true",
                           help="Skip stitching; look for existing *_cyseg-stitched.ome.tiff files")
    misc_grp.add_argument("--force-stitch", action="store_true",
                           help="Re-stitch even if *_cyseg-stitched.ome.tiff already exists")
    misc_grp.add_argument("--force-register", action="store_true",
                           help="Re-register even if the merged output file already exists")
    misc_grp.add_argument("--dry-run", action="store_true",
                           help="Validate and print plan without executing")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not sample_dir.is_dir():
        print(f"Error: sample directory does not exist: {sample_dir}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else sample_dir
    )
    output_name = args.output_name or f"{sample_dir.name}_{_MERGE_SUFFIX}.ome.tiff"
    output_path = output_dir / output_name
    pyramidal = not args.no_pyramidal

    # ---- Discovery --------------------------------------------------------
    print(f"Scanning {sample_dir} for cycle directories...")
    cycle_infos, errors = discover_cycles(sample_dir, tile_regex=args.tile_regex)

    if errors:
        for e in errors:
            print(f"  [ERROR] {e}", file=sys.stderr)
        return 1

    if not cycle_infos:
        print(
            "Error: no valid cycle directories found.\n"
            "Expected subdirectories named C<N>_<rest> (e.g. C1_BTC_MSK1_2_HSV1g).",
            file=sys.stderr,
        )
        return 1

    cycle_infos = sorted(cycle_infos, key=lambda x: x["cycle_int"])

    print(f"\nFound {len(cycle_infos)} cycle(s):")
    for ci in cycle_infos:
        print(
            f"  Cycle {ci['label']:>4}  {ci['folder'].name}"
            f"  [{ci['n_tiles']} tiles, {ci['n_channels']} ch]"
            f"  channels: {', '.join(ci['channel_markers'])}"
        )

    if not args.stitch_only:
        print(f"\nMerged output will be written to: {output_path}")

    if args.dry_run:
        print("\n[dry-run] No files will be written.")
        return 0

    # ---- Stitch -----------------------------------------------------------
    stitched_paths: list[str] = []
    t0 = time.monotonic()

    for ci in cycle_infos:
        try:
            if args.skip_stitch:
                path = _find_stitched_file(ci)
                print(f"  [skip-stitch] Using existing: {path}")
            else:
                path = _run_stitch(
                    ci,
                    stitch_channel=args.stitch_channel,
                    n_workers=args.n_workers,
                    pyramidal_output=pyramidal,
                    tile_regex=args.tile_regex,
                    force=args.force_stitch,
                )
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(
                f"\n[ERROR] Stitching failed for cycle {ci['label']} "
                f"({ci['folder'].name}): {exc}",
                file=sys.stderr,
            )
            return 1
        stitched_paths.append(path)

    if args.stitch_only:
        elapsed = time.monotonic() - t0
        print(f"\nStitching complete ({elapsed:.1f}s). Skipping registration (--stitch-only).")
        return 0

    # ---- Register ---------------------------------------------------------
    try:
        _run_registration(
            cycle_infos,
            stitched_paths,
            output_path=output_path,
            registration_marker=args.registration_marker,
            registration_algorithm=args.registration_algorithm,
            strip_height=args.strip_height,
            pyramidal_output=pyramidal,
            force=args.force_register,
        )
    except Exception as exc:
        print(
            f"\n[ERROR] Registration failed: {exc}\n"
            f"Stitched files are preserved in their cycle directories.",
            file=sys.stderr,
        )
        return 1

    elapsed = time.monotonic() - t0
    print(f"\nPipeline complete ({elapsed:.1f}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
