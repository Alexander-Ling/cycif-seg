import argparse

from cycif_seg import shortcut


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cycif-seg", description="Launch the cycif-seg GUI.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create-shortcut", action="store_true", help="Create desktop/app-menu shortcuts, then exit.")
    group.add_argument("--remove-shortcut", action="store_true", help="Remove desktop/app-menu shortcuts, then exit.")
    args = parser.parse_args(argv)

    if args.create_shortcut:
        for p in shortcut.create_shortcuts():
            print(f"Created: {p}")
        return
    if args.remove_shortcut:
        removed = shortcut.remove_shortcuts()
        print("\n".join(f"Removed: {p}" for p in removed) or "No shortcuts found.")
        return

    shortcut.ensure_shortcuts_once()

    import napari
    from qtpy import QtWidgets, QtCore
    from cycif_seg.ui.main_widget import CycIFMVPWidget

    viewer = napari.Viewer(title="cycif-seg (built on napari)")
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
    try:
        QtCore.QTimer.singleShot(
            0,
            lambda: viewer.window._qt_window.resizeDocks(
                [dock],
                [720], #Default width for right dock widget housing all of our tools
                QtCore.Qt.Horizontal,
            ),
        )
    except Exception:
        pass
    napari.run()
