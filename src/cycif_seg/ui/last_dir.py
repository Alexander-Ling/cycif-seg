from __future__ import annotations

import os
from pathlib import Path

# Folder of the most recent file/folder picked in any dialog during this session.
# Intentionally in-memory only: it resets when the app restarts.
_last_dir: str | None = None


def remember_dir(path: str | os.PathLike | None) -> None:
    """Record the folder of ``path`` (a picked file or folder) as the session's last-used folder."""
    global _last_dir
    if not path:
        return
    p = Path(str(path)).expanduser()
    d = p if p.is_dir() else p.parent
    if d.is_dir():
        _last_dir = str(d)


def last_dir(fallback: str | os.PathLike | None = None) -> str:
    """Return the session's last-used folder if it still exists, else ``fallback``, else the cwd."""
    if _last_dir and Path(_last_dir).is_dir():
        return _last_dir
    if fallback:
        return str(fallback)
    return os.getcwd()
