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
- Python 3.11 or newer (tested on 3.11 and 3.13)
- git
### Description:
cycif-seg is a simple tool for co-registration, segmentation, and marker quantification of CycIF images. It was made in a few days to accomplish a very specific task, so it isn't polished and likely has many bugs.
Still incomplete -- in active development.

### Installation Instructions

The first time cycif-seg opens, it adds a **cycif-seg** icon to your Desktop and app menu, so after installing you can start it without the command line. (pip can't run setup code during install, so the icon appears on the first launch at the end of these steps.)

#### Windows (10 or 11)

1. **Install PowerShell.** Open the Microsoft Store, search for **PowerShell**, and install it. (The built-in "Windows PowerShell" also works for every step below.) Open PowerShell from the Start menu.

2. **Install Python 3.13:**

   ```powershell
   winget install -e --id Python.Python.3.13
   ```

3. **Install git:**

   ```powershell
   winget install -e --id Git.Git
   ```

   **Close PowerShell and open a new window** so it picks up the newly installed programs.

4. **Install cycif-seg:**

   ```powershell
   py -3.13 -m pip install "git+https://github.com/Alexander-Ling/cycif-seg.git"
   ```

   This takes several minutes. A warning that some scripts are "installed in a directory which is not on PATH" is harmless.

5. **Launch cycif-seg once** (this also creates the Desktop and Start-menu icons):

   ```powershell
   py -3.13 -m cycif_seg
   ```

   From now on, double-click the **cycif-seg** icon on your Desktop or Start menu.

If `winget` isn't recognized, install **App Installer** from the Microsoft Store, or download the installers manually: [Python](https://www.python.org/downloads/) (3.11 or newer) and [git](https://git-scm.com/download/win). If `py` isn't recognized, use `python` in its place. If you installed a Python version other than 3.13, change `-3.13` to match (e.g. `py -3.11`).

#### Ubuntu (22.04 or newer)

1. **Install git, Python, and the system libraries the viewer window needs.** Open a Terminal and run:

   ```bash
   sudo apt update
   sudo apt install -y git python3 python3-venv python3-pip libgl1 libxkbcommon-x11-0 \
       libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
       libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 libxcb-cursor0
   python3 --version
   ```

   If `python3 --version` reports **3.10 or lower** (Ubuntu 22.04), install a newer Python and use `python3.13` in place of `python3` in step 2:

   ```bash
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install -y python3.13 python3.13-venv
   ```

2. **Install cycif-seg into its own environment** (Ubuntu doesn't allow pip to install into the system Python):

   ```bash
   python3 -m venv ~/cycif-seg-env
   ~/cycif-seg-env/bin/python -m pip install --upgrade pip
   ~/cycif-seg-env/bin/python -m pip install "git+https://github.com/Alexander-Ling/cycif-seg.git"
   ```

3. **Launch cycif-seg once** (this also adds it to the app menu and Desktop):

   ```bash
   ~/cycif-seg-env/bin/python -m cycif_seg
   ```

   From now on, open **cycif-seg** from the app menu or the Desktop icon. If the Desktop icon says it isn't trusted, right-click it and choose **Allow Launching**.

#### macOS

1. **Install the Xcode command line tools** (this provides git). Open Terminal and run:

   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (skip if `brew --version` already works), then Python 3.13:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.13
   ```

   Follow the "Next steps" Homebrew prints at the end of its install to add `brew` to your PATH, then open a new Terminal window.

3. **Install cycif-seg into its own environment:**

   ```bash
   python3.13 -m venv ~/cycif-seg-env
   ~/cycif-seg-env/bin/python -m pip install --upgrade pip
   ~/cycif-seg-env/bin/python -m pip install "git+https://github.com/Alexander-Ling/cycif-seg.git"
   ```

4. **Launch cycif-seg once** (this also creates `~/Applications/cycif-seg.app`):

   ```bash
   ~/cycif-seg-env/bin/python -m cycif_seg
   ```

   From now on, open **cycif-seg** from Launchpad or Spotlight, or drag it from `~/Applications` to the Dock.

#### Updating

**Close cycif-seg first** (Windows can't replace files that are in use), then re-run the install command with `--upgrade --force-reinstall --no-deps` added:

```powershell
# Windows
py -3.13 -m pip install --upgrade --force-reinstall --no-deps "git+https://github.com/Alexander-Ling/cycif-seg.git"
```

```bash
# Ubuntu / macOS
~/cycif-seg-env/bin/python -m pip install --upgrade --force-reinstall --no-deps "git+https://github.com/Alexander-Ling/cycif-seg.git"
```

Drop `--no-deps` if an update adds new dependencies.

#### Shortcuts and uninstalling

The icons point at the Python cycif-seg was installed into. If you move or reinstall Python, or delete an icon and want it back, recreate them with `--create-shortcut` (and remove them with `--remove-shortcut`):

```powershell
py -3.13 -m cycif_seg --create-shortcut                  # Windows
~/cycif-seg-env/bin/python -m cycif_seg --create-shortcut  # Ubuntu / macOS
```

To uninstall, remove the shortcuts, then `py -3.13 -m pip uninstall cycif-seg` on Windows, or delete the `~/cycif-seg-env` folder on Ubuntu/macOS.

### CLI Commands

The package installs these commands:

- `cycif-seg`: launch the graphical napari app (same as `python -m cycif_seg`). `--create-shortcut` / `--remove-shortcut` manage the Desktop and app-menu icons.
- `cycif-seg-gui`: launch the app without a console window (Windows).
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

When registering existing stitched files, the default filename pattern is `*_cyseg-stitched.ome.tiff`. Use `--stitched-pattern` to provide one or more sample-wide alternatives in preference order:

```powershell
cycif-seg-run SAMPLE_DIR --skip-stitch `
  --stitched-pattern '*_cyseg-stitched.ome.tiff' '*_evos.ome.tiff'
```

The first pattern having exactly one matching file in every cycle directory wins for the entire sample; files from different patterns are never mixed. If no pattern covers every cycle, registration does not start and the missing cycle directories are reported. The selected pattern is printed in the run log.

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

**Elastic touch-up** is enabled by default. After tiled-rigid island refinement, a B-spline elastic registration pass is run per cycle to correct sub-pixel residual misalignment. The default settings (tile size 1024 px, B-spline spacing 50 px, 100 iterations, max step length 1.0 px) work well in most cases. CLI flags let you override individual parameters without editing the plan JSON:

- `--no-elastic-touchup`: disable the elastic pass entirely.
- `--elastic-touchup-max-step-length F`: maximum optimizer step per iteration in pixels (default: 1.0). Lower values restrict how far pixels travel; raise to 2–4 if registration needs more correction.
- `--elastic-touchup-max-iterations N`: elastix iterations per resolution level (default: 100).
- `--elastic-touchup-bspline-spacing N`: B-spline control point spacing in full-resolution pixels (default: 50; lower = more local deformation).
- `--elastic-touchup-tile-size N`: tile size for the large-island tiled path (default: 1024).
- `--elastic-touchup-skip-corr F`: skip the elastic pass for a cycle when masked NCC already exceeds this threshold (default: 0.95).
- `--elastic-touchup-workers N`: parallel worker threads for tile processing (default: 0 = auto).

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

This command uses a sidecar file named `<output>.cyseg-registration-progress.json` when available, otherwise it performs a conservative pixel scan to find the first incomplete cycle. Useful options include `--completion hybrid|manifest|pixel-scan`, `--force-from-cycle N`, `--registration-algorithm tiled_rigid|translation`, `--strip-height N`, `--pyramidal`, and `--dry-run`. The same `--elastic-touchup` / `--no-elastic-touchup` and `--elastic-touchup-*` flags available for `run` also work here and override the plan or sample-directory defaults.

### Single-sample pipeline

Use `cycif-seg-run` when a sample directory contains cycle subdirectories named `C<N>_...`:

```powershell
cycif-seg-run SAMPLE_DIR --registration-marker DAPI --strip-height 1000
```

This command stitches tile files in each cycle directory, then registers cycles into one merged OME-TIFF. By default, it resumes interrupted work when possible: existing stitched files are reused, partial flat registration outputs are inspected with the registration progress sidecar and a conservative pixel scan, completed cycles are skipped, and interrupted pyramid builds reuse completed level files. If a complete flat merged output exists and pyramidal output is requested, the command continues directly into pyramid construction/resume instead of repeating registration. Use `--force-register` to discard resumable registration state and rebuild the merged output from scratch.

Useful options include:

- Output: `--output-dir PATH`, `--output-name NAME`.
- Stitching: `--stitch-channel INT`, `--n-workers INT`, `--tile-regex REGEX`.
- Registration: `--registration-marker STR`, `--registration-algorithm tiled_rigid|translation`, `--strip-height INT`.
- Modes: `--stitch-only`, `--skip-stitch`, `--force-stitch`, `--force-register`, `--no-pyramidal`, `--dry-run`.
