# Organizing and naming your files

cycif-seg finds samples, cycles, and channel names from your **folder and file names**. This page explains what it looks for, and what it does when names don't match.

## Recommended layout

```text
MyExperiment/                                  <- the folder you select as "Batch input folder"
├── Patient01/                                 <- one folder per sample (slide); any name
│   ├── C0_Patient01_[DAPI,CD3,CD8,PDL1]/      <- one folder per cycle: C<number>_..._[markers]
│   │   ├── area_001_0_0.ome.tiff              <- raw tiles from the microscope (Step 0 input)
│   │   ├── area_001_0_1.ome.tiff
│   │   ├── ...
│   │   └── C0_Patient01_[DAPI,CD3,CD8,PDL1]_cyseg-stitched.ome.tiff   <- written by Step 0
│   ├── C1_Patient01_[DAPI,CD4,FOXP3,KI67]/
│   │   └── ...
│   └── C2_Patient01_[DAPI,CD68,PANCK,SMA]/
│       └── ...
└── Patient02/
    └── ...
```

1. **Experiment folder:** one sub-folder per sample.
2. **Sample folders:** any name. The name becomes the sample name and the output file name.
3. **Cycle folders:** go inside a sample folder and start with `C`, the cycle number, and an underscore: `C0_…`, `C1_…`, `C12_…`.
4. **Marker list:** the channel names, **in square brackets and separated by commas**, anywhere in the cycle folder name (or the image file name), e.g. `[DAPI,CD3,CD8,PDL1]`. See [Channel names](#channel-names).
5. **Raw tiles** (Step 0 input) or **one stitched image** (Step 1 input) go inside each cycle folder.

### Other layouts that also work

The Step 1 batch scanner (and `cycif-seg-preprocess plan`) also accepts:

| You have | What to select | What happens |
|---|---|---|
| The layout above | The experiment folder | One sample per sub-folder. |
| One sample only | The **sample** folder | Detected automatically and scanned as a single sample. |
| Cycle images directly in each sample folder, no cycle sub-folders (`Patient01/C1_Patient01.ome.tiff`) | The experiment or sample folder | Each image is one cycle. |
| Images stitched by other software (`C1_Patient01/stitched.ome.tiff`) | Either | The cycle number is taken from the folder name. |
| Images named with the word *cycle* (`Patient01_cycle3.ome.tiff`) | Either | `cycle3` / `cyc3` / `cycle_3` is read as cycle 3. |

After **Scan samples**, a window lists anything it skipped or had to guess, and samples with notes show a warning icon (hover over the name for details).

### Automatic cycle configuration

Scanning also fills in each sample's cycle configuration, so samples are ready to run without opening **Configure cycles** for each one (`cycif-seg-preprocess plan` does the same):

- **Marker names:** from the `[..]` marker list when it matches the channel count (see [Channel names](#channel-names)), or the legacy-name rule if **Legacy names** is ticked; otherwise the channel names stored in the image.
- **Registration channel:** the channel named DAPI in the image metadata or, failing that, the one the marker list calls DAPI. If there is none, channel 1 is used and the scan results say so.
- **Antibodies:** left blank.

Review any sample with **Configure cycles**. Changes made there replace the automatic configuration for that sample, and **Copy cycle config to all samples** still copies one sample's settings to the rest. Ticking or unticking **Legacy names** takes effect on the next scan.

## Rules in detail

### Sample folders

- Only **direct** sub-folders of the selected folder are checked. Deeper folders aren't searched.
- If you select a folder that itself contains cycle folders or cycle images, it is treated as a single sample.
- A folder with no cycle images is skipped. The scan results list skipped folders and why.
- The sample folder's name becomes the output name:
  - Step 1 batch (app) and `cycif-seg-preprocess`: `<output folder>/<sample name>.ome.tiff`
  - `cycif-seg-run`: `<sample name>_cyseg-merged.ome.tiff`

### Cycle numbers

- A cycle number is read from the **file name** first, then the **cycle folder name**:
  - `C<number>` at the very start: `C3_…`, `c3_…`, `C3.ome.tiff`
  - or the word `cycle`/`cyc` followed by the number anywhere: `Patient01_cycle3`, `P01_cyc-3`
- Marker names such as `CD8` or `PDL1` are never read as cycle numbers.
- **The number comes from the name, not the sort order.** `C10_…` is cycle 10 even though it sorts before `C2_…`.
- **The lowest cycle number is the reference.** Every other cycle is aligned to it, so it should be your best-quality DAPI cycle (normally the first one).
- **No number found?** The image is still listed, but **disabled with a blank cycle number**. Open **Configure cycles**, type the number in the *Cycle* column and tick *Use*.
- **Same number twice?** If a cycle was re-imaged and both folders are kept (`C1_first/`, `C1_redo/`), both are registered and labelled `1a` and `1b` (in folder-name order). To use only one, untick the other in **Configure cycles**.

### Choosing the image in a cycle folder

Each cycle folder should hold one stitched image. If it holds several, cycif-seg picks one in this order and lists the rest as disabled:

1. The cycif-seg Step 0 output (`…_cyseg-stitched.ome.tiff`)
2. A file whose name starts with `C<number>_`
3. Any other `.ome.tif` / `.ome.tiff` (first alphabetically)

Raw tiles (`area…`, `raw…`) and temporary files are never chosen. To use a different image, untick the chosen one and tick the other in **Configure cycles**.

### Channel names

Every merged channel is named `<marker>_cy<cycle>`, e.g. `CD8_cy1`. Marker names come from one of two places.

**1. A marker list in the name.** Used by the app's **Auto-detect channel names** button and always by `cycif-seg-run`.

Put the markers **in square brackets, separated by commas**, in the image file name or its cycle folder name. The file name is checked first. Compare the number of markers listed with the number of channels in the image (N):

| Listed | Example (4-channel image) | Result |
|---|---|---|
| **N** markers | `C1_P01_[DAPI,CD3,CD8,PDL1]` | DAPI, CD3, CD8, PDL1: one name per channel, in order |
| **N − 1** markers | `C1_P01_[CD3,CD8,PDL1]` | DAPI, CD3, CD8, PDL1: **channel 1 is assumed to be DAPI**, and you are told so |
| Any other number | `C1_P01_[CD3,CD8]` | **No names assigned**, with a warning explaining the mismatch |

- Marker names can contain hyphens and spaces (`PD-L1`, `Ki-67`) but not commas or square brackets.
- Use one marker list per name.
- Anything outside the brackets (patient, date, slide ID) is ignored.

#### Legacy names (older datasets)

Older datasets often list markers as plain underscore-separated words with no brackets, e.g. `C1_Patient01_CD3_CD8_PDL1`. These are **ignored by default**, because the number of markers can't be checked. To read them, tick **Legacy names** next to **Auto-detect channel names** in **Configure cycles**, or pass `--legacy-names` to `cycif-seg-run`. When it's on, any image whose names have no `[..]` list is named like this:

- **Channel 1 is assumed to be DAPI.**
- The other channels are named after the **last N−1 underscore-separated words** of the cycle folder name (N = number of channels). Earlier words (patient, date, …) are ignored.

| Cycle folder (4-channel image) | Channel names with Legacy names on |
|---|---|
| `C1_Patient01_CD3_CD8_PDL1` | DAPI, CD3, CD8, PDL1 |
| `C1_BTC_MSK1_2_HSV1g_PDL1_CD8` | DAPI, HSV1g, PDL1, CD8 |
| `C1_CD3_CD8` (too few words) | none, with a warning |

Be careful: a missing or extra word silently shifts every name, a cycle whose channel 1 isn't DAPI is named wrongly, and marker names can't contain underscores (`PD_L1` reads as two markers). A `[..]` list, where present, always takes priority. Renaming to the bracket form removes the guesswork.

**2. The image's own metadata (the app's default).** The app reads the channel names stored inside each OME-TIFF by the microscope software. Review and edit them in the **Channel names** tab of **Configure cycles**.

When no marker names are assigned, the merged file uses the names from the image metadata.

### Tile files (Step 0 input)

- Tiles must be OME-TIFFs (`.ome.tif` or `.ome.tiff`) inside the cycle folder.
- **Name:** start with `area` or `raw` and end with `_<x>_<y>.ome.tiff`, where `x` and `y` are the tile's grid position. For example, `area_001_3_7.ome.tiff` is the tile at x=3, y=7.
- If your microscope names tiles differently, open **Advanced tile filename parsing** in the Batch stitch window (or pass `--tile-regex` to `cycif-seg-run`) and give a pattern whose two capture groups are x and y.
- Step 0 stitches tiles in any sub-folder of a sample, but Step 1 needs a cycle number, so name cycle folders `C<number>_…` from the start.
- Step 0 writes `<cycle folder name>_cyseg-stitched.ome.tiff` into each cycle folder.

### Single-slide merge (app)

**Single slide: Merge/Register cycles…** doesn't scan folders. You pick the files yourself:

- In the file picker, select **all** cycle files at once (Ctrl-click or Shift-click). The picker can only multi-select within one folder.
- Cycle numbers default to the order the file picker returns the files, starting at 1. **Check the Cycle column** in the **Registration** tab. The lowest number is the reference.
- **Auto-detect channel names** follows the marker-list rules above.

### `cycif-seg-run` (command line)

- Point it at **one sample folder**. Each sub-folder whose name contains a cycle number (same rules as the app: `C<number>…` at the start, or `cycle<number>`) is a cycle. The cycle number comes from the **folder** name; other sub-folders are listed as skipped.
- It stitches raw tiles if present. Otherwise it uses the stitched image chosen by the rules in [Choosing the image](#choosing-the-image-in-a-cycle-folder), unless you pass `--stitched-pattern` to choose by file-name pattern.
- Channel names always come from the marker list (see above). Add `--legacy-names` for older underscore-separated folder names.

## Common mistakes

| Symptom | Likely cause |
|---|---|
| "No samples were found" | The selected folder's sub-folders have no cycle images, or contain only raw tiles (run Step 0 first). The scan results say which. |
| A sample shows a warning icon | Hover over its name. Usually an extra image was found in a cycle folder, or an image had no cycle number. |
| A cycle is disabled with a blank number | Neither the file nor the folder name contains `C<number>` or `cycle<number>`. Type the number in **Configure cycles**. |
| Channels appear as `CD8_cy1a`, `CD8_cy1b` | Two cycle folders share the same number. Untick one if that wasn't intended. |
| "lists 4 marker(s) but the image has 3 channel(s)" | The marker list doesn't match the image. List one marker per channel, or one fewer if channel 1 is DAPI. |
| "channel 1 was assumed to be DAPI" | You listed one marker fewer than the number of channels. That's fine if channel 1 really is DAPI. Otherwise list all channels. |
| "No marker list found … turn on legacy names" | The names use the old underscore style. Tick **Legacy names** (or `--legacy-names`), or add a `[..]` list. |
| Registration is poor across all cycles | The lowest-numbered cycle (the reference) has weak or missing DAPI. |
| Step 0 finds no tiles | Tile names don't start with `area`/`raw`, or don't end with `_<x>_<y>.ome.tiff`. |
