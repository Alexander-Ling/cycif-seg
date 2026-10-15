from __future__ import annotations

from qtpy import QtCore, QtWidgets

NAMING_GUIDE_URL = "https://github.com/Alexander-Ling/cycif-seg/blob/main/docs/FILE_NAMING.md"

_LAYOUT_ROWS = [
    ("MyExperiment/", "batch input folder"),
    ("  Patient01/", "one folder per sample (any name)"),
    ("    C0_Patient01_CD3_CD8_PDL1/", "one folder per cycle: C<number>_..."),
    ("      area_001_0_0.ome.tiff", "raw tiles (Step 0 input)"),
    ("      C0_..._cyseg-stitched.ome.tiff", "Step 0 output / Step 1 input"),
    ("    C1_Patient01_CD4_FOXP3_KI67/", ""),
    ("  Patient02/", ""),
]
_LAYOUT_WIDTH = max(len(path) for path, _ in _LAYOUT_ROWS) + 2


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_LAYOUT_HTML = "<pre>" + "\n".join(
    _esc(path.ljust(_LAYOUT_WIDTH) + "<- " + note if note else path) for path, note in _LAYOUT_ROWS
) + "</pre>"

_GUIDE_LINK = f'<p>Full guide: <a href="{NAMING_GUIDE_URL}">Organizing and naming your files</a></p>'

STITCH_HELP = (
    "<p><b>Select the experiment folder</b> that contains one sub-folder per sample. "
    "Inside each sample folder, put one sub-folder per cycle containing that cycle's tiles.</p>"
    + _LAYOUT_HTML
    + "<ul>"
    "<li>Name cycle folders <b>C&lt;number&gt;_...</b> (e.g. <code>C0_...</code>, <code>C1_...</code>). "
    "Step 0 doesn't require it, but Step 1 does.</li>"
    "<li>Tile names must start with <code>area</code> or <code>raw</code> and end with "
    "<code>_&lt;x&gt;_&lt;y&gt;.ome.tiff</code> (grid position). Otherwise use "
    "<i>Advanced tile filename parsing</i>.</li>"
    "<li>The stitched result is written into each cycle folder as "
    "<code>&lt;cycle folder&gt;_cyseg-stitched.ome.tiff</code>.</li>"
    "</ul>"
    + _GUIDE_LINK
)

BATCH_PREPROCESS_HELP = (
    "<p><b>Select the experiment folder</b> that contains one sub-folder per sample "
    "(usually the same folder used for Step 0).</p>"
    + _LAYOUT_HTML
    + "<ul>"
    "<li>Each cycle's image must be in its own cycle folder, and its <b>file name</b> must start with "
    "<b>C&lt;number&gt;_</b>. That number is the cycle number. Step 0 output already follows this.</li>"
    "<li>The <b>lowest</b> cycle number is the reference all other cycles are aligned to.</li>"
    "<li>Only one such file per cycle folder. The same number in two different folders "
    "(a re-imaged cycle) is kept and labelled 1a, 1b.</li>"
    "<li>The sample folder name becomes the output file name: <code>&lt;sample&gt;.ome.tiff</code>.</li>"
    "<li>Channel names are read from the image metadata. <i>Auto-detect channel names</i> instead reads them "
    "from the cycle folder name: DAPI, then the last N&minus;1 underscore-separated words.</li>"
    "</ul>"
    + _GUIDE_LINK
)

SINGLE_SLIDE_HELP = (
    "<p><b>Select all cycle images for one slide at once</b> (Ctrl-click or Shift-click). "
    "The file picker can only select multiple files from one folder.</p>"
    "<ul>"
    "<li>Cycle numbers default to the order the picker returns the files, starting at 0. "
    "<b>Check the Cycle column</b> in the Registration tab. The lowest number is the reference cycle.</li>"
    "<li>Channel names are read from the image metadata. <i>Auto-detect channel names</i> works only if each "
    "file sits in a cycle folder named <code>C&lt;number&gt;_..._&lt;marker2&gt;_&lt;marker3&gt;_...</code>.</li>"
    "<li>For many slides, or files in separate cycle folders, use <b>Batch slides…</b> instead.</li>"
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
