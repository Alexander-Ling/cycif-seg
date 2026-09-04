# DFCI5_Resection_Specimen_4 Attempt1 registration diagnostic

Date inspected: 2026-09-04

## Run configuration

- Reference: Cycle 1
- Moving cycle present in this attempt: Cycle 14
- Cycle 15: not present in the discovered input set or output manifest
- Registration: `tiled_rigid`
- Low-memory strip mode: enabled
- Strip height: 3700 px
- Elastic touch-up: enabled
- Pre-elastic rigid maximum shift: 1024 px
- Elastic B-spline spacing: 50 px
- Elastic optimizer iterations: 10
- Elastic maximum step length: 1.0 px
- Elastic tile size: 2048 px
- Elastic skip-correlation threshold: 0.85

## Durable artifacts inspected

- Final merged OME-TIFF: `DFCI5_Resection_Specimen_4_cyseg-merged.ome.tiff`
- Rigid-only debug OME-TIFF: `DFCI5_Resection_Specimen_4_cyseg-merged_rigid_only.ome.tiff`
- Elastic field debug TIFF: `DFCI5_Resection_Specimen_4_cyseg-merged_elastic_field_cycle_1.tiff`
- Registration manifest: `DFCI5_Resection_Specimen_4_cyseg-merged.ome.tiff.cyseg-registration-progress.json`

## Findings

The final output remains poorly aligned. At pyramid level 6, a simple reference-foreground DAPI correlation between Cycle 1 and Cycle 14 was approximately 0.40. This is an output-level diagnostic, not the pipeline's per-tile acceptance metric.

The elastic field is substantial:

- Field size: 3951 x 2373, approximately 22x downsampled from full resolution.
- Approximately 20.1% of field pixels have nonzero displacement.
- Nonzero displacement magnitude: median approximately 61 px, 95th percentile approximately 224 px, maximum approximately 466 px.
- The horizontal component reaches approximately 440 px; the vertical component reaches approximately -183 to +73 px in the nonzero field values.
- Approximately 59% of pixels have nonzero accumulation weight.

These values show that the elastic stage is applying a large, spatially varying correction. Increasing the pre-elastic rigid bound therefore did not solve the failure; the issue is unlikely to be explained only by the previous 512 px rigid limit.

The rigid-only debug file is not currently a valid A/B image for direct comparison: its Cycle 1/reference channels are zero-valued while its Cycle 14 channels are populated. The reference side of that diagnostic artifact should be fixed before relying on it for visual or numeric rigid-versus-elastic comparisons.

## Most useful next diagnostic

Rerun with the updated code and `--debug-elastic-touchup`. The run will write:

- `*_elastic_diagnostics_cycle_<N>.jsonl`: one record per candidate tile, including island/bounding-box coordinates, baseline and rigid correlations, rigid decision, resolved prior, final status, timing, and usable-field result.
- `*_elastic_diagnostics_cycle_<N>_summary.json`: fixed candidate count, submitted/completed counts, parameters, and status totals.

The report should then group the JSONL records by island and compare:

1. baseline correlation before local rigid touch-up;
2. rigid candidate correlation and rejection rate;
3. elastic correlation improvement and failure/rejection rate;
4. accepted displacement magnitude and direction;
5. spatial distribution of empty/blank/failed tiles.

That grouping will distinguish a bad global translation, rigid candidates that are rejected despite useful shifts, elastic corrections that fail their acceptance gate, and a foreground/mask or image-content mismatch.
