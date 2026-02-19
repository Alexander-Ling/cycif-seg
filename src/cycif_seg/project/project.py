from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import datetime
from typing import Any, Dict, Optional, Tuple


PROJECT_MANIFEST_NAME = "project.json"
PROJECT_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def is_project_dir(p: Path) -> bool:
    """Return True if *p* looks like a CycIF project directory."""
    try:
        return (p / PROJECT_MANIFEST_NAME).is_file()
    except Exception:
        return False


@dataclass
class CycIFProject:
    root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    dirty: bool = False

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        self.dirty = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def relpath(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(p)

    def abspath(self, rel: str) -> Path:
        rp = Path(rel)
        if rp.is_absolute():
            return rp
        return (self.root / rp).resolve()

    def ensure_dirs(self) -> None:
        for d in [self.data_dir, self.models_dir, self.exports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ---- convenience helpers for Step 1 outputs ----
    def set_merged_ome_path(
        self,
        out_path: Path,
        cycle_inputs: Optional[list[dict]] = None,
        *,
        tissue: Optional[str] = None,
        species: Optional[str] = None,
        canvas_yx: Optional[tuple] = None,
    ) -> None:
        self.manifest.setdefault("step1", {})
        self.manifest["step1"]["merged_ome_tiff"] = self.relpath(out_path)
        if cycle_inputs is not None:
            self.manifest["step1"]["cycle_inputs"] = cycle_inputs
        if tissue is not None:
            self.manifest["step1"]["tissue"] = str(tissue)
        if species is not None:
            self.manifest["step1"]["species"] = str(species)
        if canvas_yx:
            self.manifest["step1"]["canvas_yx"] = list(canvas_yx)
        self.mark_dirty()

    # ---- Step 2 / model tracking helpers ----
    def add_input(self, path: Path) -> None:
        rel = self.relpath(path)
        inputs = self.manifest.setdefault("inputs", [])
        if rel not in inputs:
            inputs.append(rel)
            self.mark_dirty()

    def add_model_record(self, *, stage: str, name: str, relpath: str, meta: Optional[dict] = None) -> None:
        """Record a model artifact in the manifest without writing to disk."""
        self.manifest.setdefault(stage, {})
        models = self.manifest[stage].setdefault("models", [])
        rec = {
            "name": str(name),
            "path": str(relpath),
            "meta": meta or {},
            "saved_utc": _utc_now_iso(),
        }
        models.append(rec)
        self.mark_dirty()


def create_project(root_dir: Path, *, name: Optional[str] = None) -> CycIFProject:
    """Create a new project directory with a manifest and standard subfolders.

    Parameters
    ----------
    root_dir:
        Folder to create/use as the project root. Must exist or be creatable.
    name:
        Optional human-friendly project name stored in the manifest.

    Returns
    -------
    CycIFProject
    """
    root_dir = Path(root_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root_dir / PROJECT_MANIFEST_NAME
    if manifest_path.exists():
        # Don't overwrite existing manifest; open instead.
        return open_project(root_dir)

    manifest: Dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "created_utc": _utc_now_iso(),
        "name": name or root_dir.name,
        "inputs": [],
        "step1": {},
        "step2": {},
        "step3": {},
        "step4": {},
        "exports": {},
    }

    prj = CycIFProject(root=root_dir, manifest_path=manifest_path, manifest=manifest)
    prj.ensure_dirs()
    prj.save()
    return prj


def open_project(path: Path) -> CycIFProject:
    """Open an existing project given a directory or a project.json file path."""
    path = Path(path).expanduser()
    if path.is_file() and path.name.lower() == PROJECT_MANIFEST_NAME:
        root = path.parent
        manifest_path = path
    else:
        root = path
        manifest_path = root / PROJECT_MANIFEST_NAME

    if not manifest_path.exists():
        raise FileNotFoundError(f"Not a CycIF project (missing {PROJECT_MANIFEST_NAME}): {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prj = CycIFProject(root=root.resolve(), manifest_path=manifest_path.resolve(), manifest=manifest)
    prj.ensure_dirs()
    return prj
