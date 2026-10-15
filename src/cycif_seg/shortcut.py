"""Create/remove desktop and app-menu shortcuts that launch the cycif-seg GUI.

pip has no post-install hook, so ``ensure_shortcuts_once()`` is called on the first
GUI launch instead; ``cycif-seg --create-shortcut`` / ``--remove-shortcut`` do it on demand.
Shortcuts launch ``<this python> -m cycif_seg``, so they work from a venv without activating it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "cycif-seg"
_MARKER_NAME = "shortcuts_created"


def _config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME


def _napari_icon(ext: str) -> Path | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec("napari")  # locate without importing napari
        p = Path(spec.origin).parent / "resources" / f"icon.{ext}"
        return p if p.is_file() else None
    except Exception:
        return None


def _gui_python() -> str:
    """Interpreter for the shortcut; on Windows prefer pythonw.exe so no console window opens."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        w = exe.with_name("pythonw.exe")
        if w.is_file():
            return str(w)
    return str(exe)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _run_powershell(script: str) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        creationflags=flags,
        check=True,
    )
    return res.stdout


def _ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _windows_folders() -> list[Path]:
    # GetFolderPath honours OneDrive-redirected Desktops.
    out = _run_powershell(
        "[Environment]::GetFolderPath('Desktop'); [Environment]::GetFolderPath('Programs')"
    )
    return [Path(line.strip()) for line in out.splitlines() if line.strip()]


def _create_windows() -> list[Path]:
    icon = _napari_icon("ico")
    paths = [d / f"{APP_NAME}.lnk" for d in _windows_folders()]
    lines = ["$s = New-Object -ComObject WScript.Shell"]
    for p in paths:
        lines += [
            f"$l = $s.CreateShortcut({_ps_quote(p)})",
            f"$l.TargetPath = {_ps_quote(_gui_python())}",
            "$l.Arguments = '-m cycif_seg'",
            f"$l.WorkingDirectory = {_ps_quote(Path.home())}",
            "$l.Description = 'cycif-seg (built on napari)'",
        ]
        if icon is not None:
            lines.append(f"$l.IconLocation = {_ps_quote(str(icon) + ',0')}")
        lines.append("$l.Save()")
    _run_powershell("; ".join(lines))
    return paths


def _remove_windows() -> list[Path]:
    removed = []
    for d in _windows_folders():
        p = d / f"{APP_NAME}.lnk"
        if p.exists():
            p.unlink()
            removed.append(p)
    return removed


# ---------------------------------------------------------------------------
# Linux (freedesktop .desktop entries)
# ---------------------------------------------------------------------------
def _linux_desktop_dir() -> Path:
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path.home() / "Desktop"


def _linux_paths() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return [data_home / "applications" / f"{APP_NAME}.desktop", _linux_desktop_dir() / f"{APP_NAME}.desktop"]


def _linux_icon() -> str:
    src = _napari_icon("ico")
    if src is None:
        return ""
    dst = _config_dir() / "icon.png"
    try:
        from PIL import Image

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).save(dst)
        return str(dst)
    except Exception:
        return str(src)


def _create_linux() -> list[Path]:
    exec_cmd = f'"{_gui_python()}" -m cycif_seg'
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=cycif-seg (built on napari)\n"
        f"Exec={exec_cmd}\n"
        f"Icon={_linux_icon()}\n"
        "Terminal=false\n"
        "Categories=Science;Graphics;\n"
    )
    made = []
    for p in _linux_paths():
        if not p.parent.is_dir():
            if p.parent.name == "applications":
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                continue  # no Desktop folder (e.g. headless); the app-menu entry is enough
        p.write_text(entry, encoding="utf-8")
        p.chmod(0o755)
        # GNOME (Ubuntu) only launches desktop-folder launchers marked trusted.
        try:
            subprocess.run(["gio", "set", str(p), "metadata::trusted", "true"], capture_output=True, check=False)
        except Exception:
            pass
        made.append(p)
    return made


def _remove_linux() -> list[Path]:
    removed = []
    for p in _linux_paths():
        if p.exists():
            p.unlink()
            removed.append(p)
    return removed


# ---------------------------------------------------------------------------
# macOS (minimal .app bundle in ~/Applications)
# ---------------------------------------------------------------------------
def _mac_app() -> Path:
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def _create_mac() -> list[Path]:
    app = _mac_app()
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    res = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)
    icon = _napari_icon("icns")
    if icon is not None:
        shutil.copy2(icon, res / "icon.icns")
    (app / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f"  <key>CFBundleName</key><string>{APP_NAME}</string>\n"
        f"  <key>CFBundleDisplayName</key><string>{APP_NAME}</string>\n"
        "  <key>CFBundleIdentifier</key><string>org.cycif-seg.app</string>\n"
        f"  <key>CFBundleExecutable</key><string>{APP_NAME}</string>\n"
        "  <key>CFBundleIconFile</key><string>icon</string>\n"
        "  <key>CFBundlePackageType</key><string>APPL</string>\n"
        "  <key>NSHighResolutionCapable</key><true/>\n"
        "</dict></plist>\n",
        encoding="utf-8",
    )
    launcher = macos / APP_NAME
    launcher.write_text(f'#!/bin/sh\nexec "{_gui_python()}" -m cycif_seg "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    return [app]


def _remove_mac() -> list[Path]:
    app = _mac_app()
    if app.exists():
        shutil.rmtree(app)
        return [app]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_shortcuts() -> list[Path]:
    if sys.platform == "win32":
        made = _create_windows()
    elif sys.platform == "darwin":
        made = _create_mac()
    else:
        made = _create_linux()
    marker = _config_dir() / _MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(sys.executable, encoding="utf-8")
    return made


def remove_shortcuts() -> list[Path]:
    if sys.platform == "win32":
        removed = _remove_windows()
    elif sys.platform == "darwin":
        removed = _remove_mac()
    else:
        removed = _remove_linux()
    # Keep the marker so the next launch doesn't recreate what the user just removed.
    marker = _config_dir() / _MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("removed", encoding="utf-8")
    return removed


def ensure_shortcuts_once() -> None:
    """Create shortcuts on the first GUI launch only; never raises."""
    try:
        if (_config_dir() / _MARKER_NAME).exists():
            return
        made = create_shortcuts()
        if made:
            print("cycif-seg: created shortcut(s): " + ", ".join(str(p) for p in made), flush=True)
    except Exception as e:
        print(f"cycif-seg: could not create shortcuts ({e}); run `cycif-seg --create-shortcut` to retry.", flush=True)
        # Don't retry (and slow down) every launch; the explicit command still works.
        try:
            marker = _config_dir() / _MARKER_NAME
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("failed", encoding="utf-8")
        except Exception:
            pass
