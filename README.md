# cycif-seg
### Authors:
- Alexander Ling (alexander.l.ling@gmail.com; alling@bwh.harvard.edu)
-  E. Antonio Chiocca
### License: Creative Commons CC BY-NC-SA 4.0
### License Holders:
- Alexander Ling
- E. Antonio Chiocca
- Mass General Brigham
### Installation Requirements:
- Python 3.11
### Description:
cycif-seg is a simple tool for co-registration, segmentation, and marker quantification of CycIF images. It was made in a few days to accomplish a very specific task, so it isn't polished and likely has many bugs.
Still incomplete -- in active development.

### Installation Instructions

```
[Windows Powershell]:
#Please ensure you have installed python 3.11 (https://www.python.org/downloads/release/python-3110/) and git (https://git-scm.com/download/win) before running these commands

py -3.11 -m venv {path_to_save_environment_in}

{path_to_save_environment_in}\Scripts\activate

py -m pip install git+https://github.com/Alexander-Ling/cycif-seg.git

cycif-seg
```

### CLI Commands

The package installs three console commands:

- `cycif-seg`: launch the graphical napari app.
- `cycif-seg-preprocess`: headless batch preprocessing and recovery utilities.
- `cycif-seg-run`: single-sample stitch-and-register pipeline.

You can inspect the current options with:

```powershell
cycif-seg-preprocess --help
cycif-seg-run --help
```

### Batch preprocessing

### Input naming and automatic channel marker detection

The CLI uses two related but different naming rules, depending on which pipeline you run.

For `cycif-seg-run`, channel marker names are inferred from cycle directory names. Cycle directories must be immediate children of `SAMPLE_DIR` and must match:

```text
C<N>_<sample_or_description>_<marker1>_<marker2>_...
```

The cycle number comes from the leading `C<N>_`. The first image channel is always treated as `DAPI`. The remaining channel marker names are taken from the last `n_channels - 1` underscore-separated tokens in the directory name, where `n_channels` is read from a representative tile OME-TIFF.

Example for a 4-channel cycle:

```text
C1_Arbitrary_Information_Here_HSV1g_PDL1_CD8
```

is interpreted as:

```text
DAPI, HSV1g, PDL1, CD8
```

If the suffix has too few marker tokens, missing channel names are left blank and a warning is printed. Marker names should not contain underscores if you want automatic detection to work cleanly. Tile filenames inside each cycle directory must match the default tile-position pattern ending in underscore-separated integer coordinates, such as `area_*_<x>_<y>.ome.tiff` or `raw_*_<x>_<y>.ome.tiff`; use `--tile-regex` if your tile names differ.

For `cycif-seg-preprocess plan` and `resume-registration --sample-dir`, the expected layout is:

```text
ROOT/
  SAMPLE_NAME/
    C0_some_cycle_dir/
      C0_sample_or_cycle_name.ome.tiff
    C1_some_cycle_dir/
      C1_sample_or_cycle_name.ome.tiff
```

In this mode, the cycle number is inferred from the stitched OME-TIFF filename prefix `C<N>_`, and files such as `area_*.ome.tiff` are ignored. Channel marker names are read from OME-XML channel metadata in each OME-TIFF, not parsed from the filename or directory name. If channel names cannot be read, edit the generated plan JSON fields `registration_markers` and `channel_markers` before running.

Use `cycif-seg-preprocess plan` to scan a root directory, read cycle channel names, and write a plan JSON for review:

```powershell
cycif-seg-preprocess plan `
  --root INPUT_ROOT `
  --output OUTPUT_DIR `
  --registration-marker DAPI `
  PLAN.json
```

Useful options:

- `--tissue TEXT` and `--species TEXT`: add default metadata labels.
- `--tile-size N` and `--search-factor F`: tiled-rigid registration settings.
- `--no-pyramidal`: write flat merged OME-TIFFs instead of pyramidal outputs.

After reviewing or editing the plan JSON, run it:

```powershell
cycif-seg-preprocess run PLAN.json --strip-height 1000
```

Useful options:

- `--output-dir DIR`: rebase output paths for all samples in the plan.
- `--strip-height N`: process registration in horizontal strips to reduce RAM.
- `--dry-run`: validate and print the planned work without processing.

### Recovery utilities

Build or resume a pyramidal OME-TIFF from an existing flat OME-TIFF:

```powershell
cycif-seg-preprocess pyramid INPUT.ome.tiff --replace-source
```

or write to a separate file:

```powershell
cycif-seg-preprocess pyramid INPUT.ome.tiff --output OUTPUT.ome.tiff
```

This command reuses complete `cycif_pyramid_work_*/levels/level_XX.dat` files from interrupted runs and discards partial `.__pyramid_tmp__.ome.tiff` files. Useful options include `--work-dir DIR`, `--no-resume`, `--keep-work-dir`, `--tile-size N`, `--compression NAME`, `--min-level-size N`, `--out-chunk N`, and `--dry-run`.

Resume a partially completed flat registration OME-TIFF:

```powershell
cycif-seg-preprocess resume-registration --plan PLAN.json --sample SAMPLE_NAME
```

or use a sample directory directly:

```powershell
cycif-seg-preprocess resume-registration `
  --sample-dir SAMPLE_DIR `
  --output SAMPLE_cyseg-merged.ome.tiff `
  --registration-marker DAPI
```

This command uses a sidecar file named `<output>.cyseg-registration-progress.json` when available, otherwise it performs a conservative pixel scan to find the first incomplete cycle. Useful options include `--completion hybrid|manifest|pixel-scan`, `--force-from-cycle N`, `--registration-algorithm tiled_rigid|translation`, `--strip-height N`, `--pyramidal`, and `--dry-run`.

### Single-sample pipeline

Use `cycif-seg-run` when a sample directory contains cycle subdirectories named `C<N>_...`:

```powershell
cycif-seg-run SAMPLE_DIR --registration-marker DAPI --strip-height 1000
```

This command stitches tile files in each cycle directory, then registers cycles into one merged OME-TIFF. Useful options include:

- Output: `--output-dir PATH`, `--output-name NAME`.
- Stitching: `--stitch-channel INT`, `--n-workers INT`, `--tile-regex REGEX`.
- Registration: `--registration-marker STR`, `--registration-algorithm tiled_rigid|translation`, `--strip-height INT`.
- Modes: `--stitch-only`, `--skip-stitch`, `--force-stitch`, `--force-register`, `--no-pyramidal`, `--dry-run`.
