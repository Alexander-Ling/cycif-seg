"""
cycif-seg-preprocess — headless CLI for Step 1 batch preprocessing.

Two subcommands:

  cycif-seg-preprocess plan  --root DIR --output DIR [options] PLAN.json
      Scans for samples, reads channel names from OME-TIFFs, and writes a
      plan JSON that can be reviewed/edited before running.

  cycif-seg-preprocess run   PLAN.json [--output-dir DIR] [--dry-run]
      Executes a plan JSON, printing progress to stdout.
"""
from __future__ import annotations

import argparse
import builtins
import json
import re
import sys
import time
import traceback
from pathlib import Path

from cycif_seg.preprocess.batch_plan import (
    BatchSample,
    default_tiled_rigid_tile_size,
    default_tiled_rigid_search_factor,
    enabled_cycle_numbers,
    find_stitched_cycle_files_in_sample_dir,
    plan_from_dict,
    plan_to_dict,
    sample_has_cycle_config,
    scan_root_for_samples,
    validate_samples_ready,
)
from cycif_seg.preprocess.organize_cycles import (
    CycleInput,
    inspect_registration_flat_resume_state,
    merge_cycles_to_ome_tiff,
)


def print(*args, **kwargs):  # type: ignore[override]
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


def _cycle_display_name(sample: BatchSample, path: str | Path, cycle: int) -> str:
    p = Path(path)
    sample_dir = Path(sample.input_dir).expanduser()
    try:
        sample_dir = sample_dir.resolve()
        parent_path = p.parent.resolve()
    except Exception:
        parent_path = p.parent
    parent = p.parent.name
    if parent and parent_path != sample_dir:
        return parent
    return p.stem or str(int(cycle))


def _cycle_display_map(sample: BatchSample, cycles_in: list[CycleInput]) -> dict[int, str]:
    return {
        int(ci.cycle): _cycle_display_name(sample, ci.path, int(ci.cycle))
        for ci in cycles_in
    }


def _format_cycle_display(cycle: object, cycle_names: dict[int, str]) -> str:
    cy = int(cycle)
    return cycle_names.get(cy) or str(cy)


def _rewrite_cycle_names(msg: str, cycle_names: dict[int, str]) -> str:
    out = str(msg)
    for cy, display in sorted(cycle_names.items(), key=lambda item: len(str(item[0])), reverse=True):
        if display == str(cy):
            continue
        pat = re.compile(rf"\b(Cycle|cycle)\s+{re.escape(str(cy))}\b")
        out = pat.sub(lambda m: f"{m.group(1)} {display}", out)
    return out


def _cmd_plan(args: argparse.Namespace) -> int:
    from cycif_seg.io.ome_tiff import load_channel_names_only_fast

    root_dir = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    plan_path = Path(args.plan_json).expanduser()
    reg_marker = (args.registration_marker or "DAPI").strip()
    pyramidal = not args.no_pyramidal

    if not root_dir.is_dir():
        print(f"Error: root directory does not exist: {root_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {root_dir} for samples...")
    samples = scan_root_for_samples(root_dir, output_dir, tissue=args.tissue, species=args.species)

    if not samples:
        print(
            "No samples found.\n"
            "Expected layout: {root}/{sample_dir}/{cycle_dir}/C#_*.ome.tiff\n"
            "Cycle files must be named with a C<number>_ prefix (e.g. C1_sample.ome.tiff).",
            file=sys.stderr,
        )
        return 1

    n_cycle_files = sum(len(s.files) for s in samples)
    print(f"Found {len(samples)} sample(s), {n_cycle_files} cycle file(s) total.")

    warnings: list[str] = []
    for si, s in enumerate(samples, 1):
        print(f"  [{si}/{len(samples)}] {s.name}: reading channel names from {len(s.files)} cycle file(s)...")
        registration_markers: list[str] = []
        channel_markers: list[list[str]] = []
        channel_antibodies: list[list[str]] = []

        for p in s.files:
            try:
                ch_names = list(load_channel_names_only_fast(str(p)) or [])
            except Exception as e:
                ch_names = []
                warnings.append(f"{s.name}/{p.name}: could not read channel names ({e})")

            reg = next(
                (nm for nm in ch_names if nm.strip().upper() == reg_marker.upper()),
                ch_names[0] if ch_names else reg_marker,
            )
            registration_markers.append(reg)
            channel_markers.append(ch_names)
            channel_antibodies.append([""] * len(ch_names))

        s.registration_markers = registration_markers
        s.channel_markers = channel_markers
        s.channel_antibodies = channel_antibodies
        s.tiled_rigid_tile_size = max(128, int(args.tile_size))
        s.tiled_rigid_search_factor = max(1.0, float(args.search_factor))
        s.pyramidal_output = pyramidal

    if warnings:
        print("\nWarnings (edit the plan JSON to fix marker names manually):")
        for w in warnings:
            print(f"  ! {w}")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    d = plan_to_dict(
        samples,
        root_dir=root_dir,
        output_dir=output_dir,
        default_tissue=args.tissue,
        default_species=args.species,
    )
    plan_path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print(f"\nPlan written to: {plan_path}")
    if warnings:
        print(
            "Some channel names could not be read automatically.\n"
            "Edit the plan JSON to set 'registration_markers' and 'channel_markers' manually."
        )
    else:
        print(
            "Review the plan JSON and adjust registration markers or channel names as needed,\n"
            "then run:"
        )
    print(f"  cycif-seg-preprocess run {plan_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan_json).expanduser()
    if not plan_path.is_file():
        print(f"Error: plan file not found: {plan_path}", file=sys.stderr)
        return 1

    try:
        d = json.loads(plan_path.read_text(encoding="utf-8"))
        result = plan_from_dict(d)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error loading plan: {e}", file=sys.stderr)
        return 1

    samples: list[BatchSample] = result["samples"]

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        for s in samples:
            name = s.output_path.name if s.output_path else f"{s.name}.ome.tiff"
            s.output_path = out_dir / name

    ok, msg = validate_samples_ready(samples)
    if not ok:
        print(f"Plan validation failed: {msg}", file=sys.stderr)
        print(
            "Tip: run 'cycif-seg-preprocess plan' to regenerate a plan with channel names,\n"
            "     or edit the JSON to add 'registration_markers' and 'channel_markers' fields.",
            file=sys.stderr,
        )
        return 1

    enabled = [s for s in samples if s.enabled]
    n_samp = len(enabled)

    if args.dry_run:
        print(f"Dry run: plan is valid. {n_samp} sample(s) would be processed:")
        for s in enabled:
            n_cy = len(enabled_cycle_numbers(s))
            print(f"  {s.name}: {n_cy} cycle(s) -> {s.output_path}")
        return 0

    print(f"Running batch preprocessing: {n_samp} sample(s)\n")
    failures: list[dict] = []
    reports: list[dict] = []
    start_total = time.monotonic()

    for si, s in enumerate(enabled, 1):
        print(f"[{si}/{n_samp}] {s.name}")
        out_path = Path(str(s.output_path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cycles_in: list[CycleInput] = []
        try:
            for i, p in enumerate(s.files):
                if s.cycles_enabled and i < len(s.cycles_enabled) and not bool(s.cycles_enabled[i]):
                    continue
                cycles_in.append(
                    CycleInput(
                        path=str(p),
                        cycle=int(s.cycles[i] if s.cycles and i < len(s.cycles) else i),
                        tissue=s.tissue or None,
                        species=s.species or None,
                        registration_marker=(
                            s.registration_markers[i]
                            if s.registration_markers and i < len(s.registration_markers)
                            else None
                        ),
                        channel_markers=(
                            s.channel_markers[i]
                            if s.channel_markers and i < len(s.channel_markers)
                            else None
                        ),
                        channel_antibodies=(
                            s.channel_antibodies[i]
                            if s.channel_antibodies and i < len(s.channel_antibodies)
                            else None
                        ),
                    )
                )
        except Exception as e:
            print(f"  ERROR building cycle inputs: {e}", file=sys.stderr)
            failures.append({"sample_name": s.name, "error": str(e)})
            continue

        _phase_labels = {
            "load_ref": "loading reference",
            "load_cycle": "loading",
            "global_registration": "global translation",
            "foreground_mask": "foreground mask",
            "identify_islands": "foreground islands",
            "region_refine": "region refinement",
            "bad_region_refine": "bad region refinement",
            "foreground_island_refine": "island refinement",
            "elastic_touchup": "elastic touch-up",
            "elastic_touchup_island": "elastic touch-up (island)",
            "write_cycle": "writing",
            "pyramid": "building pyramid",
        }
        cycle_names = _cycle_display_map(s, cycles_in)

        def _ev(ev: dict) -> None:
            phase = str(ev.get("phase") or "")
            if not phase:
                return
            cycle = ev.get("cycle")
            n = int(ev.get("n") or 0)
            idx = int(ev.get("idx") or 0)
            label = _phase_labels.get(phase, phase)
            cycle_txt = f" cycle {_format_cycle_display(cycle, cycle_names)}" if cycle is not None else ""
            progress_txt = f" ({idx}/{n})" if n > 0 else ""
            print(f"  {label}{cycle_txt}{progress_txt}", flush=True)

        start_s = time.monotonic()
        try:
            rep = merge_cycles_to_ome_tiff(
                cycles_in,
                str(out_path),
                default_registration_marker="DAPI",
                registration_algorithm=str(getattr(s, "registration_algorithm", "tiled_rigid") or "tiled_rigid"),
                global_translation_only=bool(getattr(s, "global_translation_only", False)),
                tiled_rigid_allow_rotation=bool(getattr(s, "tiled_rigid_allow_rotation", False)),
                tiled_rigid_tile_size=max(128, int(getattr(s, "tiled_rigid_tile_size", default_tiled_rigid_tile_size) or default_tiled_rigid_tile_size)),
                tiled_rigid_search_factor=max(1.0, float(getattr(s, "tiled_rigid_search_factor", default_tiled_rigid_search_factor) or default_tiled_rigid_search_factor)),
                fast_large_island_refinement=False,
                fast_large_island_sample_count=max(1, int(getattr(s, "fast_large_island_sample_count", 5) or 5)),
                elastic_touchup=(
                    bool(args.elastic_touchup) if args.elastic_touchup is not None
                    else bool(getattr(s, "elastic_touchup", True))
                ),
                elastic_touchup_tile_size=max(64, int(
                    args.elastic_touchup_tile_size if args.elastic_touchup_tile_size is not None
                    else (getattr(s, "elastic_touchup_tile_size", 2048) or 2048)
                )),
                elastic_touchup_skip_corr=float(
                    args.elastic_touchup_skip_corr if args.elastic_touchup_skip_corr is not None
                    else (getattr(s, "elastic_touchup_skip_corr", 0.95) or 0.95)
                ),
                elastic_touchup_bspline_spacing=max(4, int(
                    args.elastic_touchup_bspline_spacing if args.elastic_touchup_bspline_spacing is not None
                    else (getattr(s, "elastic_touchup_bspline_spacing", 50) or 50)
                )),
                elastic_touchup_max_iterations=max(1, int(
                    args.elastic_touchup_max_iterations if args.elastic_touchup_max_iterations is not None
                    else (getattr(s, "elastic_touchup_max_iterations", 10) or 10)
                )),
                elastic_touchup_large_island_px=max(1, int(getattr(s, "elastic_touchup_large_island_px", 4_000_000) or 4_000_000)),
                elastic_touchup_workers=max(0, int(
                    args.elastic_touchup_workers if args.elastic_touchup_workers is not None
                    else (getattr(s, "elastic_touchup_workers", 0) or 0)
                )),
                elastic_touchup_max_step_length=max(0.01, float(
                    args.elastic_touchup_max_step_length if args.elastic_touchup_max_step_length is not None
                    else (getattr(s, "elastic_touchup_max_step_length", 1.0) or 1.0)
                )),
                elastic_touchup_rigid_max_shift=max(1.0, float(
                    args.elastic_touchup_rigid_max_shift if args.elastic_touchup_rigid_max_shift is not None
                    else (getattr(s, "elastic_touchup_rigid_max_shift", 512.0) or 512.0)
                )),
                debug_elastic_touchup=bool(getattr(args, "debug_elastic_touchup", False)) or bool(getattr(s, "debug_elastic_touchup", False)),
                debug_dir=str(getattr(args, "debug_dir") or getattr(s, "debug_dir") or "") or None,
                pyramidal_output=bool(getattr(s, "pyramidal_output", False)),
                progress_event_cb=_ev,
                cancel_cb=None,
                low_mem=bool(getattr(s, "low_mem", True)),
                strip_height=(
                    args.strip_height if args.strip_height is not None
                    else getattr(s, "strip_height", None)
                ),
            )
            elapsed = time.monotonic() - start_s
            print(f"  done in {elapsed:.0f}s -> {out_path}")
            reports.append({
                "sample_name": s.name,
                "output_path": str(rep.get("output_path") or out_path),
                "pyramidal_output_path": rep.get("pyramidal_output_path"),
                "canvas_yx": list(rep.get("canvas_shape_yx") or []),
                "n_cycles": int(rep.get("n_cycles") or len(cycles_in)),
                "n_channels_total": int(rep.get("n_channels_total") or 0),
            })
        except Exception as e:
            elapsed = time.monotonic() - start_s
            print(f"  ERROR after {elapsed:.0f}s: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.stderr.flush()
            failures.append({"sample_name": s.name, "error": str(e)})

    total_elapsed = time.monotonic() - start_total
    print(f"\nDone in {total_elapsed:.0f}s: {len(reports)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed samples:")
        for f in failures:
            print(f"  {f['sample_name']}: {f['error']}")

    results = {
        "schema_version": 1,
        "plan": str(plan_path),
        "reports": reports,
        "failures": failures,
        "elapsed_seconds": round(total_elapsed, 1),
    }
    results_path = plan_path.with_name(plan_path.stem + "_results.json")
    try:
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to: {results_path}")
    except Exception as e:
        print(f"Warning: could not write results JSON: {e}", file=sys.stderr)

    return 1 if failures else 0


def _cmd_pyramid(args: argparse.Namespace) -> int:
    from cycif_seg.io.ome_tiff import (
        convert_flat_ome_to_pyramidal,
        inspect_tiff_pyramid,
        inspect_tiff_yxc,
    )

    input_path = Path(args.input_ome_tiff).expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: input OME-TIFF does not exist: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    replace_source = bool(args.replace_source)
    if replace_source == (output_path is not None):
        print("Error: specify exactly one of --replace-source or --output PATH.", file=sys.stderr)
        return 1
    if output_path is not None and output_path == input_path:
        print("Error: --output must differ from the input path. Use --replace-source for in-place conversion.", file=sys.stderr)
        return 1

    try:
        pyr = inspect_tiff_pyramid(str(input_path))
        if bool(pyr.get("is_pyramidal")):
            print(f"Error: input already appears to be pyramidal: {input_path}", file=sys.stderr)
            return 1
        info = inspect_tiff_yxc(str(input_path))
    except Exception as e:
        print(f"Error inspecting input OME-TIFF: {e}", file=sys.stderr)
        return 1

    y, x, c = info["shape_yxc"]
    dtype = info["dtype"]
    min_level_size = max(1, int(args.min_level_size))
    subifds = 0
    yy, xx = int(y), int(x)
    while yy > min_level_size and xx > min_level_size:
        yy = max(1, (yy + 1) // 2)
        xx = max(1, (xx + 1) // 2)
        subifds += 1
    if subifds < 1:
        print(
            f"Error: input is too small for a pyramid with --min-level-size {min_level_size}: "
            f"{int(y)}x{int(x)}",
            file=sys.stderr,
        )
        return 1

    target_path = input_path if replace_source else output_path
    tmp_target = target_path.with_name(f"{target_path.stem}.__pyramid_tmp__.ome.tiff")
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else None
    compression = None if str(args.compression).strip().lower() in {"", "none", "no", "false"} else args.compression

    print(f"Input:  {input_path}")
    print(f"Shape:  {int(y)} x {int(x)} x {int(c)} ({dtype})")
    print(f"Output: {target_path}")
    print(f"Mode:   {'replace source' if replace_source else 'write output'}")
    print(f"Levels: {subifds + 1} total ({subifds} downsampled)")
    if work_dir is not None:
        print(f"Work:   {work_dir}")
    if args.dry_run:
        print("\n[dry-run] No files will be written.")
        return 0

    start = time.monotonic()

    def _progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    try:
        tmp_out = convert_flat_ome_to_pyramidal(
            str(input_path),
            output_path=str(tmp_target),
            tile_size=max(16, int(args.tile_size)),
            compression=compression,
            min_level_size=min_level_size,
            out_chunk=max(1, int(args.out_chunk)),
            replace_source=False,
            temp_dir=str(tmp_target.parent),
            work_dir=str(work_dir) if work_dir is not None else None,
            resume=not bool(args.no_resume),
            keep_work_dir=bool(args.keep_work_dir),
            progress_cb=_progress,
            build_workers=max(1, int(args.workers)) if args.workers is not None else None,
        )
        final_info = inspect_tiff_pyramid(str(tmp_out))
        if not bool(final_info.get("is_pyramidal")):
            raise RuntimeError(f"Output failed pyramid validation: {tmp_out}")
        if replace_source:
            Path(tmp_out).replace(input_path)
            final_path = input_path
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_out).replace(output_path)
            final_path = output_path
    except Exception as e:
        try:
            tmp_target.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"Error building pyramidal OME-TIFF: {e}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s -> {final_path}")
    return 0


def _cycles_from_sample(sample: BatchSample) -> list[CycleInput]:
    cycles_in: list[CycleInput] = []
    for i, p in enumerate(sample.files):
        if sample.cycles_enabled and i < len(sample.cycles_enabled) and not bool(sample.cycles_enabled[i]):
            continue
        cy = int(sample.cycles[i] if sample.cycles and i < len(sample.cycles) else i)
        cycles_in.append(
            CycleInput(
                path=str(p),
                cycle=cy,
                label=str(cy),
                tissue=sample.tissue or None,
                species=sample.species or None,
                registration_marker=(
                    sample.registration_markers[i]
                    if sample.registration_markers and i < len(sample.registration_markers)
                    else None
                ),
                channel_markers=(
                    sample.channel_markers[i]
                    if sample.channel_markers and i < len(sample.channel_markers)
                    else None
                ),
                channel_antibodies=(
                    sample.channel_antibodies[i]
                    if sample.channel_antibodies and i < len(sample.channel_antibodies)
                    else None
                ),
            )
        )
    return cycles_in


def _sample_from_sample_dir(args: argparse.Namespace) -> BatchSample:
    from cycif_seg.io.ome_tiff import load_channel_names_only_fast

    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")
    stitched = find_stitched_cycle_files_in_sample_dir(sample_dir)
    if not stitched:
        raise ValueError(f"No stitched cycle OME-TIFFs found in sample directory: {sample_dir}")
    files = [p for p, _cy in stitched]
    cycles = [int(cy) for _p, cy in stitched]
    reg_marker = str(args.registration_marker or "DAPI").strip() or "DAPI"
    channel_markers: list[list[str]] = []
    registration_markers: list[str] = []
    for p in files:
        names = list(load_channel_names_only_fast(str(p)) or [])
        registration_markers.append(
            next((nm for nm in names if nm.strip().upper() == reg_marker.upper()), names[0] if names else reg_marker)
        )
        channel_markers.append(names)
    return BatchSample(
        name=sample_dir.name,
        input_dir=sample_dir,
        files=files,
        output_path=Path(args.output).expanduser().resolve() if args.output else None,
        cycles=cycles,
        registration_markers=registration_markers,
        channel_markers=channel_markers,
        channel_antibodies=[[""] * len(x) for x in channel_markers],
        cycles_enabled=[True] * len(files),
        registration_algorithm=str(args.registration_algorithm or "tiled_rigid"),
        global_translation_only=bool(str(args.registration_algorithm or "").strip().lower() == "translation"),
        tiled_rigid_tile_size=max(128, int(args.tile_size)),
        tiled_rigid_search_factor=max(1.0, float(args.search_factor)),
        pyramidal_output=False,
        low_mem=True,
        strip_height=int(args.strip_height) if args.strip_height and int(args.strip_height) > 0 else None,
        elastic_touchup=(bool(args.elastic_touchup) if args.elastic_touchup is not None else True),
        elastic_touchup_tile_size=max(64, int(getattr(args, "elastic_touchup_tile_size", None) or 2048)),
        elastic_touchup_skip_corr=float(getattr(args, "elastic_touchup_skip_corr", None) or 0.95),
        elastic_touchup_bspline_spacing=max(4, int(getattr(args, "elastic_touchup_bspline_spacing", None) or 50)),
        elastic_touchup_max_iterations=max(1, int(getattr(args, "elastic_touchup_max_iterations", None) or 10)),
        elastic_touchup_large_island_px=max(1, int(getattr(args, "elastic_touchup_large_island_px", None) or 4_000_000)),
        elastic_touchup_workers=max(0, int(getattr(args, "elastic_touchup_workers", None) or 0)),
        elastic_touchup_max_step_length=max(0.01, float(getattr(args, "elastic_touchup_max_step_length", None) or 1.0)),
        elastic_touchup_rigid_max_shift=max(1.0, float(getattr(args, "elastic_touchup_rigid_max_shift", None) or 512.0)),
        debug_elastic_touchup=bool(getattr(args, "debug_elastic_touchup", False)),
        debug_dir=str(getattr(args, "debug_dir") or "") or None,
    )


def _load_resume_sample(args: argparse.Namespace) -> BatchSample:
    if args.plan:
        plan_path = Path(args.plan).expanduser()
        d = json.loads(plan_path.read_text(encoding="utf-8"))
        result = plan_from_dict(d)
        samples = [s for s in result["samples"] if s.enabled]
        if args.sample:
            samples = [s for s in samples if s.name == args.sample]
        if not samples:
            raise ValueError("No enabled sample matched the resume-registration selection")
        if len(samples) > 1:
            names = ", ".join(s.name for s in samples)
            raise ValueError(f"Plan has multiple enabled samples; pass --sample. Enabled samples: {names}")
        s = samples[0]
        if args.output:
            s.output_path = Path(args.output).expanduser().resolve()
        return s
    return _sample_from_sample_dir(args)


def _cmd_resume_registration(args: argparse.Namespace) -> int:
    try:
        sample = _load_resume_sample(args)
    except Exception as e:
        print(f"Error loading resume inputs: {e}", file=sys.stderr)
        return 1
    if not sample.output_path:
        print("Error: output path is required (from plan or --output).", file=sys.stderr)
        return 1

    cycles_in = _cycles_from_sample(sample)
    if not cycles_in:
        print("Error: no enabled cycles to register.", file=sys.stderr)
        return 1
    cycle_names = _cycle_display_map(sample, cycles_in)

    out_path = Path(sample.output_path).expanduser().resolve()
    strip_height = (
        int(args.strip_height)
        if args.strip_height is not None and int(args.strip_height) > 0
        else getattr(sample, "strip_height", None)
    )
    registration_algorithm = str(getattr(sample, "registration_algorithm", "tiled_rigid") or "tiled_rigid")
    if args.registration_algorithm:
        registration_algorithm = str(args.registration_algorithm)
    global_translation_only = bool(getattr(sample, "global_translation_only", False))
    if registration_algorithm.strip().lower() == "translation":
        global_translation_only = True

    try:
        state = inspect_registration_flat_resume_state(
            cycles_in,
            out_path,
            registration_algorithm=registration_algorithm,
            global_translation_only=global_translation_only,
            tiled_rigid_tile_size=max(128, int(getattr(sample, "tiled_rigid_tile_size", default_tiled_rigid_tile_size))),
            tiled_rigid_search_factor=max(1.0, float(getattr(sample, "tiled_rigid_search_factor", default_tiled_rigid_search_factor))),
            low_mem=bool(getattr(sample, "low_mem", True)),
            strip_height=strip_height,
            elastic_touchup=(
                bool(args.elastic_touchup) if args.elastic_touchup is not None
                else bool(getattr(sample, "elastic_touchup", True))
            ),
            elastic_touchup_tile_size=max(64, int(getattr(args, "elastic_touchup_tile_size", None) or getattr(sample, "elastic_touchup_tile_size", 2048) or 2048)),
            elastic_touchup_skip_corr=float(getattr(args, "elastic_touchup_skip_corr", None) or getattr(sample, "elastic_touchup_skip_corr", 0.95) or 0.95),
            elastic_touchup_bspline_spacing=max(4, int(getattr(args, "elastic_touchup_bspline_spacing", None) or getattr(sample, "elastic_touchup_bspline_spacing", 50) or 50)),
            elastic_touchup_max_iterations=max(1, int(getattr(args, "elastic_touchup_max_iterations", None) or getattr(sample, "elastic_touchup_max_iterations", 10) or 10)),
            elastic_touchup_large_island_px=max(1, int(getattr(sample, "elastic_touchup_large_island_px", 4_000_000) or 4_000_000)),
            elastic_touchup_max_step_length=max(0.01, float(getattr(args, "elastic_touchup_max_step_length", None) or getattr(sample, "elastic_touchup_max_step_length", 1.0) or 1.0)),
            elastic_touchup_rigid_max_shift=max(1.0, float(getattr(args, "elastic_touchup_rigid_max_shift", None) or getattr(sample, "elastic_touchup_rigid_max_shift", 512.0) or 512.0)),
            completion=str(args.completion),
            force_from_cycle=args.force_from_cycle,
        )
    except Exception as e:
        print(f"Error inspecting partial registration output: {e}", file=sys.stderr)
        return 1

    completed = [int(c) for c in state["completed_cycles"]]
    first_incomplete = state["first_incomplete_cycle"]
    print(f"Sample: {sample.name}")
    print(f"Output: {out_path}")
    print(f"Detection: {state['source']}")
    print(f"Complete cycles: {completed if completed else 'none'}")
    if first_incomplete is None:
        print("First incomplete cycle: none (flat registration appears complete)")
    else:
        print(f"First incomplete cycle: {first_incomplete}")
    for msg in state.get("messages") or []:
        print(f"  {msg}")

    if args.dry_run:
        print("\n[dry-run] No files will be written.")
        return 0
    if first_incomplete is None and not args.pyramidal:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def _progress(msg: str) -> None:
        print(f"  {_rewrite_cycle_names(msg, cycle_names)}", flush=True)

    try:
        rep = merge_cycles_to_ome_tiff(
            cycles_in,
            str(out_path),
            default_registration_marker=str(args.registration_marker or "DAPI"),
            registration_algorithm=registration_algorithm,
            global_translation_only=global_translation_only,
            tiled_rigid_allow_rotation=bool(getattr(sample, "tiled_rigid_allow_rotation", False)),
            tiled_rigid_tile_size=max(128, int(getattr(sample, "tiled_rigid_tile_size", default_tiled_rigid_tile_size))),
            tiled_rigid_search_factor=max(1.0, float(getattr(sample, "tiled_rigid_search_factor", default_tiled_rigid_search_factor))),
            fast_large_island_refinement=False,
            fast_large_island_sample_count=max(1, int(getattr(sample, "fast_large_island_sample_count", 5) or 5)),
            elastic_touchup=(
                bool(args.elastic_touchup) if args.elastic_touchup is not None
                else bool(getattr(sample, "elastic_touchup", True))
            ),
            elastic_touchup_tile_size=max(64, int(getattr(args, "elastic_touchup_tile_size", None) or getattr(sample, "elastic_touchup_tile_size", 2048) or 2048)),
            elastic_touchup_skip_corr=float(getattr(args, "elastic_touchup_skip_corr", None) or getattr(sample, "elastic_touchup_skip_corr", 0.95) or 0.95),
            elastic_touchup_bspline_spacing=max(4, int(getattr(args, "elastic_touchup_bspline_spacing", None) or getattr(sample, "elastic_touchup_bspline_spacing", 50) or 50)),
            elastic_touchup_max_iterations=max(1, int(getattr(args, "elastic_touchup_max_iterations", None) or getattr(sample, "elastic_touchup_max_iterations", 10) or 10)),
            elastic_touchup_large_island_px=max(1, int(getattr(sample, "elastic_touchup_large_island_px", 4_000_000) or 4_000_000)),
            elastic_touchup_workers=max(0, int(getattr(args, "elastic_touchup_workers", None) or getattr(sample, "elastic_touchup_workers", 0) or 0)),
            elastic_touchup_max_step_length=max(0.01, float(getattr(args, "elastic_touchup_max_step_length", None) or getattr(sample, "elastic_touchup_max_step_length", 1.0) or 1.0)),
            elastic_touchup_rigid_max_shift=max(1.0, float(getattr(args, "elastic_touchup_rigid_max_shift", None) or getattr(sample, "elastic_touchup_rigid_max_shift", 512.0) or 512.0)),
            debug_elastic_touchup=bool(getattr(args, "debug_elastic_touchup", False)) or bool(getattr(sample, "debug_elastic_touchup", False)),
            debug_dir=str(getattr(args, "debug_dir") or getattr(sample, "debug_dir") or "") or None,
            pyramidal_output=bool(args.pyramidal),
            progress_cb=_progress,
            low_mem=bool(getattr(sample, "low_mem", True)),
            strip_height=strip_height,
            resume_flat_output=True,
            completed_cycles=completed,
            registration_progress_path=str(state["manifest_path"]),
            registration_fingerprint=state["fingerprint"],
        )
    except Exception as e:
        print(f"Error resuming registration: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        return 1

    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s -> {rep.get('output_path') or out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cycif-seg-preprocess",
        description="cycif-seg headless batch preprocessing (Step 1: merge/register cycles)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- plan subcommand ----
    p_plan = sub.add_parser(
        "plan",
        help="Scan a directory and write a batch plan JSON",
        description=(
            "Scan ROOT for sample subdirectories (each containing cycle subdirectories "
            "with C#_*.ome.tiff files), read channel names from the OME-TIFFs, and write "
            "a plan JSON. Review and edit the JSON before running."
        ),
    )
    p_plan.add_argument("--root", required=True, metavar="DIR",
                        help="Root input directory containing one subdirectory per sample")
    p_plan.add_argument("--output", required=True, metavar="DIR",
                        help="Output directory for merged OME-TIFFs")
    p_plan.add_argument("--tissue", default="", metavar="TEXT",
                        help="Default tissue label (applied to all samples)")
    p_plan.add_argument("--species", default="", metavar="TEXT",
                        help="Default species label")
    p_plan.add_argument("--registration-marker", default="DAPI", metavar="NAME",
                        help="Default registration channel name (default: DAPI)")
    p_plan.add_argument("--tile-size", type=int, default=default_tiled_rigid_tile_size,
                        metavar="N", help=f"Tiled-rigid tile size in pixels (default: {default_tiled_rigid_tile_size})")
    p_plan.add_argument("--search-factor", type=float, default=default_tiled_rigid_search_factor,
                        metavar="F", help=f"Registration search factor (default: {default_tiled_rigid_search_factor})")
    p_plan.add_argument("--no-pyramidal", action="store_true",
                        help="Disable pyramidal OME-TIFF output (default: pyramidal enabled)")
    p_plan.add_argument("plan_json", metavar="PLAN.json",
                        help="Output path for the plan JSON file")

    # ---- run subcommand ----
    p_run = sub.add_parser(
        "run",
        help="Execute a batch plan JSON",
        description=(
            "Load a plan JSON (created by 'plan' or edited manually) and run batch "
            "preprocessing, printing progress to stdout."
        ),
    )
    p_run.add_argument("plan_json", metavar="PLAN.json",
                       help="Path to the plan JSON file to execute")
    p_run.add_argument("--output-dir", metavar="DIR",
                       help="Override output directory for all samples (rebases output paths)")
    p_run.add_argument("--strip-height", type=int, default=None, metavar="N",
                       help=(
                           "Process the image in horizontal strips of N rows to reduce RAM. "
                           "low_mem=True (the default) auto-selects canvas_height/10 when not given. "
                           "Set to 0 to disable strip mode even when low_mem is active."
                       ))
    p_run.add_argument("--elastic-touchup", default=None, action=argparse.BooleanOptionalAction,
                       help="Enable/disable elastic touch-up (default: on; overrides plan setting)")
    p_run.add_argument("--elastic-touchup-tile-size", type=int, default=None, metavar="N",
                       help="Tile size (px) for large-island elastic tiling (overrides plan; default: 2048)")
    p_run.add_argument("--elastic-touchup-skip-corr", type=float, default=None, metavar="F",
                       help="Skip elastic if masked correlation already exceeds this value (overrides plan; default: 0.95)")
    p_run.add_argument("--elastic-touchup-bspline-spacing", type=int, default=None, metavar="N",
                       help="B-spline grid spacing in pixels at full resolution (overrides plan; default: 50)")
    p_run.add_argument("--elastic-touchup-max-iterations", type=int, default=None, metavar="N",
                       help="Maximum elastix iterations per resolution level (overrides plan; default: 10)")
    p_run.add_argument("--elastic-touchup-workers", type=int, default=None, metavar="N",
                       help="Worker threads for parallel tile processing (overrides plan; default: 0 = auto)")
    p_run.add_argument("--elastic-touchup-max-step-length", type=float, default=None, metavar="F",
                       help="Maximum optimizer step length per iteration in pixels (overrides plan; default: 1.0)")
    p_run.add_argument("--elastic-touchup-rigid-max-shift", type=float, default=None, metavar="PX",
                       help="Maximum pre-elastic local rigid shift in pixels (overrides plan; default: 512)")
    p_run.add_argument("--debug-elastic-touchup", action="store_true",
                       help="Save DAPI registration channel after rigid and after elastic touch-up for each cycle")
    p_run.add_argument("--debug-dir", type=str, default=None, metavar="DIR",
                       help="Directory for debug output images (default: <output_stem>_debug/ next to output file)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Validate the plan and print what would run, without processing")

    # ---- pyramid subcommand ----
    p_pyr = sub.add_parser(
        "pyramid",
        help="Build or resume a pyramidal OME-TIFF from an existing flat OME-TIFF",
        description=(
            "Convert a flat OME-TIFF into a pyramidal OME-TIFF. The command can reuse "
            "complete cycif_pyramid_work_*/levels/level_XX.dat files from an interrupted "
            "prior run, but partial final .__pyramid_tmp__.ome.tiff files are discarded."
        ),
    )
    p_pyr.add_argument("input_ome_tiff", metavar="INPUT.ome.tiff",
                       help="Flat OME-TIFF to convert")
    out_mode = p_pyr.add_mutually_exclusive_group(required=True)
    out_mode.add_argument("--replace-source", action="store_true",
                          help="Replace INPUT with the completed pyramidal OME-TIFF")
    out_mode.add_argument("--output", metavar="PATH",
                          help="Write pyramidal OME-TIFF to PATH")
    p_pyr.add_argument("--work-dir", metavar="DIR",
                       help="Specific cycif_pyramid_work directory to reuse or create")
    p_pyr.add_argument("--tile-size", type=int, default=512, metavar="N",
                       help="TIFF tile size in pixels (default: 512)")
    p_pyr.add_argument("--compression", default="zlib", metavar="NAME",
                       help="TIFF compression, or 'none' to disable (default: zlib)")
    p_pyr.add_argument("--min-level-size", type=int, default=128, metavar="N",
                       help="Stop adding downsampled levels once Y or X is <= N (default: 128)")
    p_pyr.add_argument("--out-chunk", type=int, default=1024, metavar="N",
                       help="Chunk size while building pyramid levels (default: 1024)")
    p_pyr.add_argument("--workers", type=int, default=None, metavar="N",
                       help="Worker threads for parallel chunk downsampling (default: os.cpu_count()-1)")
    p_pyr.add_argument("--no-resume", action="store_true",
                       help="Do not search for reusable level_XX.dat files")
    p_pyr.add_argument("--keep-work-dir", action="store_true",
                       help="Keep level_XX.dat files after successful conversion")
    p_pyr.add_argument("--dry-run", action="store_true",
                       help="Inspect input and print planned conversion without writing")

    # ---- resume-registration subcommand ----
    p_res = sub.add_parser(
        "resume-registration",
        help="Resume a partially completed flat registration OME-TIFF",
        description=(
            "Inspect an existing flat merged OME-TIFF, determine completed cycles from a "
            "registration progress sidecar or conservative pixel scan, and resume writing "
            "from the first incomplete cycle."
        ),
    )
    src_grp = p_res.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--plan", metavar="PLAN.json",
                         help="Preprocessing plan JSON containing the sample configuration")
    src_grp.add_argument("--sample-dir", metavar="DIR",
                         help="Sample directory containing stitched cycle OME-TIFFs")
    p_res.add_argument("--sample", metavar="NAME",
                       help="Sample name to use when --plan contains multiple enabled samples")
    p_res.add_argument("--output", metavar="PATH",
                       help="Output flat merged OME-TIFF path (required with --sample-dir; overrides plan output)")
    p_res.add_argument("--registration-marker", default="DAPI", metavar="NAME",
                       help="Registration channel name for --sample-dir mode or fallback (default: DAPI)")
    p_res.add_argument("--registration-algorithm", default=None,
                       choices=["tiled_rigid", "translation"],
                       help="Override registration algorithm")
    p_res.add_argument("--tile-size", type=int, default=default_tiled_rigid_tile_size, metavar="N",
                       help=f"Tiled-rigid tile size for --sample-dir mode (default: {default_tiled_rigid_tile_size})")
    p_res.add_argument("--search-factor", type=float, default=default_tiled_rigid_search_factor, metavar="F",
                       help=f"Tiled-rigid search factor for --sample-dir mode (default: {default_tiled_rigid_search_factor})")
    p_res.add_argument("--strip-height", type=int, default=None, metavar="N",
                       help="Process output in horizontal strips of N rows")
    p_res.add_argument("--completion", default="hybrid", choices=["hybrid", "manifest", "pixel-scan"],
                       help="How to determine completed cycles (default: hybrid)")
    p_res.add_argument("--force-from-cycle", type=int, default=None, metavar="N",
                       help="Trust cycles before N and resume by rewriting cycle N onward")
    p_res.add_argument("--pyramidal", action="store_true",
                       help="Build the pyramidal OME-TIFF after the flat registration completes")
    p_res.add_argument("--elastic-touchup", default=None, action=argparse.BooleanOptionalAction,
                       help="Enable/disable elastic touch-up (default: on; overrides plan setting)")
    p_res.add_argument("--elastic-touchup-tile-size", type=int, default=None, metavar="N",
                       help="Tile size (px) for large-island elastic tiling (overrides plan; default: 2048)")
    p_res.add_argument("--elastic-touchup-skip-corr", type=float, default=None, metavar="F",
                       help="Skip elastic if masked correlation already exceeds this value (overrides plan; default: 0.95)")
    p_res.add_argument("--elastic-touchup-bspline-spacing", type=int, default=None, metavar="N",
                       help="B-spline grid spacing in pixels at full resolution (overrides plan; default: 50)")
    p_res.add_argument("--elastic-touchup-max-iterations", type=int, default=None, metavar="N",
                       help="Maximum elastix iterations per resolution level (overrides plan; default: 10)")
    p_res.add_argument("--elastic-touchup-workers", type=int, default=None, metavar="N",
                       help="Worker threads for parallel tile processing (overrides plan; default: 0 = auto)")
    p_res.add_argument("--elastic-touchup-max-step-length", type=float, default=None, metavar="F",
                       help="Maximum optimizer step length per iteration in pixels (overrides plan; default: 1.0)")
    p_res.add_argument("--elastic-touchup-rigid-max-shift", type=float, default=None, metavar="PX",
                       help="Maximum pre-elastic local rigid shift in pixels (overrides plan; default: 512)")
    p_res.add_argument("--debug-elastic-touchup", action="store_true",
                       help="Save DAPI registration channel after rigid and after elastic touch-up for each cycle")
    p_res.add_argument("--debug-dir", type=str, default=None, metavar="DIR",
                       help="Directory for debug output images (default: <output_stem>_debug/ next to output file)")
    p_res.add_argument("--dry-run", action="store_true",
                       help="Inspect and print resume plan without writing")

    parsed = parser.parse_args()
    if parsed.command == "plan":
        sys.exit(_cmd_plan(parsed))
    elif parsed.command == "run":
        sys.exit(_cmd_run(parsed))
    elif parsed.command == "pyramid":
        sys.exit(_cmd_pyramid(parsed))
    elif parsed.command == "resume-registration":
        sys.exit(_cmd_resume_registration(parsed))


if __name__ == "__main__":
    main()
