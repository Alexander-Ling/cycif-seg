import napari
from cycif_seg.ui.main_widget import CycIFMVPWidget


def main():
    viewer = napari.Viewer()
    w = CycIFMVPWidget(viewer)
    viewer.window.add_dock_widget(w, area="right")
    napari.run()
