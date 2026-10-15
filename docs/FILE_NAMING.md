# Organizing and naming your files

cycif-seg finds samples, cycles, and channel names from your **folder and file names**. If the names don't follow these rules, samples or cycles are silently skipped or numbered wrong. Read this before running Step 0 (stitching) or Step 1 (merge/register).

## The short version

Use this layout for every experiment:

```text
MyExperiment/                                  <- the folder you select as "Batch input folder"
├── Patient01_TumorA/                          <- one folder per sample (slide); any name
│   ├── C0_Patient01_CD3_CD8_PDL1/             <- one folder per cycle: C<number>_...
│   │   ├── area_001_0_0.ome.tiff              <- raw tiles from the microscope (Step 0 input)
│   │   ├── area_001_0_1.ome.tiff
│   │   ├── ...
│   │   └── C0_Patient01_CD3_CD8_PDL1_cyseg-stitched.ome.tiff   <- written by Step 0
│   ├── C1_Patient01_CD4_FOXP3_KI67/
│   │   └── ...
│   └── C2_Patient01_CD68_PANCK_SMA/
│       └── ...
└── Patient02_TumorB/
    └── ...
```

1. **Experiment folder** holds one sub-folder per sample.
2. **Sample folders** can have any name. The name becomes the sample name and the output file name.
3. **Cycle folders** go directly inside a sample folder and must start with `C`, the cycle number, and an underscore: `C0_…`, `C1_…`, `C12_…`.
4. **The end of each cycle folder name lists the markers** for channels 2, 3, 4, … separated by underscores. Channel 1 is always DAPI and is not listed. This is optional in the app (see [Channel names](#channel-names)) but required for `cycif-seg-run`.
5. **Raw tiles** go inside their cycle folder (see [Tile files](#tile-files-step-0-input)).

Step 0 then writes one stitched file into each cycle folder, and Step 1 finds those stitched files automatically.

## Rules in detail

### Sample folders

- Every **direct** sub-folder of the input folder is treated as a possible sample. Deeper folders aren't searched.
- A sample folder that contains no usable cycle folders is skipped without a warning. If a sample is missing after **Scan samples**, check its cycle folder names.
- The sample folder's name becomes:
  - Step 1 batch (app) and `cycif-seg-preprocess`: `<output folder>/<sample name>.ome.tiff`
  - `cycif-seg-run`: `<sample name>_cyseg-merged.ome.tiff`

### Cycle folders

- **Name:** `C<number>_<anything>`. Upper- or lower-case `c` both work. The number may be `0` or higher.
- **The cycle number comes from the name, not from the folder's sort order.** `C10_…` is cycle 10 even though it sorts before `C2_…`.
- **The lowest cycle number is the reference.** Every other cycle is aligned to it, so it should be your best-quality DAPI cycle (normally the first one).
- **Use each number once per sample.** If a cycle was re-imaged and you keep both folders (e.g. `C1_first/` and `C1_redo/`), both are registered and their channels are labelled `1a` and `1b` (in folder-name order) instead of `1`. To use only one of them, delete or rename the other folder, or untick it in **Configure cycles**.

> Step 0 in the app will stitch tiles in any sub-folder, whatever its name. Step 1 and `cycif-seg-run`, however, need the `C<number>_` prefix, so always name cycle folders this way from the start.

### Channel names

Every merged channel is named `<marker>_cy<cycle>`, e.g. `CD8_cy1`. The marker names come from one of two places:

**From the folder name** (`cycif-seg-run` always; the app when you click **Auto-detect channel names**):

- Channel 1 is always named `DAPI`.
- The remaining names are the **last N−1 underscore-separated words** of the cycle folder name, where N is the number of channels in the image.

| Cycle folder (4-channel images) | Channel names |
|---|---|
| `C1_Patient01_CD3_CD8_PDL1` | DAPI, CD3, CD8, PDL1 |
| `C1_BTC_MSK1_2_HSV1g_PDL1_CD8` | DAPI, HSV1g, PDL1, CD8 |
| `C1_CD3_CD8` (too few words) | DAPI, CD3, CD8, *(blank)*: add the missing marker |

- **Marker names must not contain underscores.** `PD_L1` would be read as two markers. Use `PDL1` or `PD-L1`.
- Anything before the markers (patient, date, slide ID) is ignored, so it can contain underscores.

**From the image metadata** (the app's default): the app reads channel names stored inside each OME-TIFF by the microscope software. You can review and edit them in the **Channel names** tab of the **Configure cycles** window.

### Tile files (Step 0 input)

- Tiles must be OME-TIFFs (`.ome.tif` or `.ome.tiff`) inside the cycle folder.
- **Name:** must start with `area` or `raw` and end with `_<x>_<y>.ome.tiff`, where `x` and `y` are the tile's integer grid position. For example `area_001_3_7.ome.tiff` is the tile at x=3, y=7.
- If your microscope names tiles differently, open **Advanced tile filename parsing** in the Batch stitch window (or pass `--tile-regex` to `cycif-seg-run`) and give a pattern whose two capture groups are x and y.

### Stitched files (Step 1 input)

- Step 0 writes `<cycle folder name>_cyseg-stitched.ome.tiff` into each cycle folder. Because the cycle folder starts with `C<number>_`, so does the stitched file.
- **Step 1 batch (app) and `cycif-seg-preprocess plan`:** in each cycle folder, every `.ome.tif`/`.ome.tiff` whose **file name** starts with `C<number>_` is used as that cycle's image. The cycle number comes from the **file name**. Tiles (`area_…`, `raw_…`) are ignored. Files placed directly in the sample folder are ignored.
- **`cycif-seg-run`:** the cycle number comes from the **folder name**, and the stitched file must match `*_cyseg-stitched.ome.tiff` (change this with `--stitched-pattern`).
- **Only one stitched file per cycle folder.** If a folder contains two (e.g. one from cycif-seg and one exported by the microscope software, both starting with `C1_`), Step 1 reports an error for that sample. Untick one in **Configure cycles**, or move it out of the folder.
- **Stitched elsewhere?** If you stitched with other software, rename each file to start with `C<number>_` and put it in its own cycle folder as above.

### Single-slide merge (app)

**Single slide: Merge/Register cycles…** doesn't scan folders. You pick the files yourself:

- In the file picker, select **all** cycle files at once (Ctrl-click or Shift-click). The picker can only multi-select within one folder.
- Cycle numbers default to the order the file picker returns the files (not necessarily the order you clicked), starting at 0. **Check the Cycle column** in the **Registration** tab and correct it; the lowest number becomes the reference.
- **Auto-detect channel names** works only if each file sits in a cycle folder named as described above.

## Common mistakes

| Symptom | Likely cause |
|---|---|
| A sample doesn't appear after **Scan samples** | Its cycle folders don't start with `C<number>_`, or the stitched files inside don't. |
| A cycle is missing from a sample | The stitched file's name doesn't start with `C<number>_`, or it sits directly in the sample folder instead of a cycle folder. |
| "Cycle N has more than one file in folder …" | Two stitched files in one cycle folder. Keep one. |
| Channels appear as `CD8_cy1a`, `CD8_cy1b` | Two cycle folders share the same number. Rename one if that wasn't intended. |
| Auto-detected channel names are shifted or blank | The folder lists too few or too many markers, or a marker name contains `_`. |
| Registration is poor across all cycles | The lowest-numbered cycle (the reference) has weak or missing DAPI. |
| Step 0 finds no tiles | Tile names don't start with `area`/`raw` or don't end with `_<x>_<y>.ome.tiff`. |
