import napari
from qtpy import QtWidgets
from cycif_seg.ui.main_widget import CycIFMVPWidget


def main():
    viewer = napari.Viewer()
    w = CycIFMVPWidget(viewer)
    dock = viewer.window.add_dock_widget(w, area="right")
    # Allow the right-hand controls dock to be shrunk narrower than the
    # default QWidget size hint would normally permit.
    try:
        w.setMinimumWidth(0)
        w.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        dock.setMinimumWidth(120)
        dock.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
    except Exception:
        pass
    napari.run()
