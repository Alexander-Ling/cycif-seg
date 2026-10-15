from __future__ import annotations

from qtpy import QtCore, QtWidgets

NAMING_GUIDE_URL = "https://github.com/Alexander-Ling/cycif-seg/blob/main/docs/FILE_NAMING.md"

_LAYOUT_ROWS = [
    ("MyExperiment/", "batch input folder"),
    ("  Patient01/", "one folder per sample (any name)"),
    ("    C0_Patient01_[DAPI,CD3,CD8,PDL1]/", "one folder per cycle: C<number>_..._[markers]"),
    ("      area_001_0_0.ome.tiff", "raw tiles (Step 0 input)"),
    ("      C0_..._cyseg-stitched.ome.tiff", "Step 0 output / Step 1 input"),
    ("    C1_Patient01_[DAPI,CD4,FOXP3,KI67]/", ""),
    ("  Patient02/", ""),
]
_LAYOUT_WIDTH = max(len(path) for path, _ in _LAYOUT_ROWS) + 2


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_LAYOUT_HTML = "<pre>" + "\n".join(
    _esc(path.ljust(_LAYOUT_WIDTH) + "<- " + note if note else path) for path, note in _LAYOUT_ROWS
) + "</pre>"

_GUIDE_LINK = f'<p>Full guide: <a href="{NAMING_GUIDE_URL}">Organizing and naming your files</a></p>'

_MARKERS_HTML = (
    "<li><b>Marker names</b> (for <i>Auto-detect channel names</i>): list them in square brackets, separated by "
    "commas, in the cycle folder or file name, e.g. <code>[DAPI,CD3,CD8,PDL1]</code>. With N channels: "
    "N markers name every channel; N&minus;1 markers means channel 1 is assumed to be DAPI (you're told); "
    "any other count assigns no names and warns.</li>"
    "<li>Older datasets named without a [..] list (e.g. <code>C1_Patient01_CD3_CD8_PDL1</code>): tick "
    "<b>Legacy names</b> next to <i>Auto-detect channel names</i> (click its <b>?</b> for what it does).</li>"
)

LEGACY_NAMES_HELP_HTML = (
    "<p><b>Legacy names</b> reads marker names from older cycle folder names that have no "
    "<code>[..]</code> marker list.</p>"
    "<p><b>When ticked</b>, for each cycle image whose file and folder names have no <code>[..]</code> list:</p>"
    "<ul>"
    "<li><b>Channel 1 is assumed to be DAPI.</b></li>"
    "<li>The other channels are named after the <b>last N&minus;1 underscore-separated words</b> of the "
    "cycle folder name, where N is the number of channels in the image. Earlier words (patient, date, …) "
    "are ignored.</li>"
    "</ul>"
    "<pre>"
    "Folder:   C1_Patient01_CD3_CD8_PDL1   (4-channel image)\n"
    "Channels: DAPI, CD3, CD8, PDL1"
    "</pre>"
    "<p><b>Be careful:</b></p>"
    "<ul>"
    "<li>The number of markers can't be checked. A missing or extra word silently shifts every name, "
    "and a folder where channel 1 is not DAPI is named wrongly.</li>"
    "<li>Marker names must not contain underscores (<code>PD_L1</code> is read as two markers; use "
    "<code>PDL1</code> or <code>PD-L1</code>).</li>"
    "<li>If a name has a <code>[..]</code> list, the list is always used instead.</li>"
    "</ul>"
    "<p><b>When unticked</b> (the default), names without a <code>[..]</code> list get no marker names, "
    "and the image's own channel names are kept.</p>"
    "<p>Command line: <code>cycif-seg-run --legacy-names</code> does the same.</p>"
    + _GUIDE_LINK
)

STITCH_HELP = (
    "<p><b>Select the experiment folder</b> that contains one sub-folder per sample. "
    "Inside each sample folder, put one sub-folder per cycle containing that cycle's tiles.</p>"
    + _LAYOUT_HTML
    + "<ul>"
    "<li>Name cycle folders <b>C&lt;number&gt;_..._[markers]</b>, e.g. <code>C0_Patient01_[DAPI,CD3,CD8,PDL1]</code>. "
    "Step 0 doesn't need this, but Step 1 reads the cycle number and marker list from it.</li>"
    "<li>Tile names must start with <code>area</code> or <code>raw</code> and end with "
    "<code>_&lt;x&gt;_&lt;y&gt;.ome.tiff</code> (grid position). Otherwise use "
    "<i>Advanced tile filename parsing</i>.</li>"
    "<li>The stitched result is written into each cycle folder as "
    "<code>&lt;cycle folder&gt;_cyseg-stitched.ome.tiff</code>.</li>"
    "</ul>"
    + _GUIDE_LINK
)

BATCH_PREPROCESS_HELP = (
    "<p><b>Select the experiment folder</b> (one sub-folder per sample), usually the same folder used for "
    "Step 0. Selecting a single <b>sample</b> folder also works.</p>"
    + _LAYOUT_HTML
    + "<ul>"
    "<li><b>Cycle number</b>: read from the image file name, else its folder name: <code>C&lt;number&gt;_...</code> "
    "at the start, or <code>cycle&lt;number&gt;</code> anywhere. Images without one are listed disabled with a "
    "blank number. Fill it in under <i>Configure cycles</i>.</li>"
    "<li>The <b>lowest</b> cycle number is the reference all other cycles are aligned to.</li>"
    "<li>Cycle images can sit in their own cycle folders or directly in the sample folder. Files stitched by "
    "other software work. If a cycle folder holds several images, the cycif-seg stitched one is used and the "
    "rest are listed disabled.</li>"
    "<li>The same cycle number in two folders (a re-imaged cycle) is kept and labelled 1a, 1b.</li>"
    "<li>The sample folder name becomes the output file name: <code>&lt;sample&gt;.ome.tiff</code>.</li>"
    + _MARKERS_HTML
    + "</ul>"
    "<p><b>Scanning also fills in every sample's cycle configuration</b>: marker names from the "
    "<code>[..]</code> list (or legacy names, if ticked), otherwise from the image metadata; DAPI as the "
    "registration channel. Review or change it with <i>Configure cycles</i>. Anything skipped or guessed "
    "is listed after scanning, and affected samples show a warning icon.</p>"
    + _GUIDE_LINK
)

SINGLE_SLIDE_HELP = (
    "<p><b>Select all cycle images for one slide at once</b> (Ctrl-click or Shift-click). "
    "The file picker can only select multiple files from one folder.</p>"
    "<ul>"
    "<li>Cycle numbers default to the order the picker returns the files, starting at 1. "
    "<b>Check the Cycle column</b> in the Registration tab. The lowest number is the reference cycle.</li>"
    "<li>Channel names are read from the image metadata unless you click <i>Auto-detect channel names</i>.</li>"
    + _MARKERS_HTML
    + "<li>For many slides, or files in separate cycle folders, use <b>Batch slides…</b> instead.</li>"
    "</ul>"
    + _GUIDE_LINK
)


def make_help_button(parent: QtWidgets.QWidget, title: str, html: str) -> QtWidgets.QToolButton:
    """A small '?' button that shows ``html`` in a dialog with clickable links."""
    btn = QtWidgets.QToolButton(parent)
    btn.setText("?")
    btn.setToolTip(f"{title}: how files must be named and organized")

    def _show() -> None:
        box = QtWidgets.QMessageBox(parent)
        box.setWindowTitle(title)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setTextFormat(QtCore.Qt.RichText)
        box.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        box.setText(html)
        box.exec_()

    btn.clicked.connect(_show)
    return btn
