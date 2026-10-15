"""Naming rules shared by the GUI, ``cycif-seg-preprocess`` and ``cycif-seg-run``.

Cycle numbers
    ``C<number>`` at the start of a name (``C3_...``), or an explicit ``cycle<number>`` /
    ``cyc<number>`` word anywhere in it (``Patient01_cycle3``).

Marker lists
    An explicit field in square brackets, separated by commas:
    ``C1_Patient01_[DAPI,CD3,CD8,PDL1]``. With N image channels:

    * N markers listed     -> one name per channel, in order.
    * N - 1 markers listed -> channel 1 is assumed to be DAPI (and this is reported).
    * any other count      -> no names are assigned, with a warning.

    Legacy names (opt-in: ``legacy_names=True``, ``--legacy-names``, or the GUI's "Legacy names"
    checkbox): when a name has no bracket field, channel 1 is assumed to be DAPI and the other
    N - 1 names are the last N - 1 underscore-separated words of the cycle folder name
    (``C1_Patient01_CD3_CD8_PDL1`` -> DAPI, CD3, CD8, PDL1). Off by default because the number
    of markers in such names can't be checked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STITCH_OUTPUT_SUFFIX = "cyseg-stitched"  # Step 0 writes <cycle folder>_cyseg-stitched.ome.tiff

MARKER_FIELD_OPEN = "["
MARKER_FIELD_CLOSE = "]"
MARKER_DELIMITER = ","

_OME_TIFF_EXT_RE = re.compile(r"\.ome\.tiff?$", re.IGNORECASE)
_LEADING_CYCLE_RE = re.compile(r"^c(\d+)(?=[_.\-\s\[]|$)", re.IGNORECASE)
_CYCLE_WORD_RE = re.compile(r"(?:^|[_.\-\s])(?:cycle|cyc)[_\-\s]?(\d+)(?=[_.\-\s\[]|$)", re.IGNORECASE)
_MARKER_FIELD_RE = re.compile(re.escape(MARKER_FIELD_OPEN) + r"([^\[\]]*)" + re.escape(MARKER_FIELD_CLOSE))
_LEGACY_CYCLE_DIR_RE = re.compile(r"^C(\d+)_(.+)$", re.IGNORECASE)


def strip_ome_tiff_ext(name: str) -> str:
    return _OME_TIFF_EXT_RE.sub("", str(name))


def cycle_number_from_name(name: str) -> int | None:
    """Cycle number from a file or folder name, or ``None`` if it doesn't state one."""
    stem = strip_ome_tiff_ext(str(name or "").strip())
    m = _LEADING_CYCLE_RE.match(stem)
    if m:
        return int(m.group(1))
    hits = {int(x) for x in _CYCLE_WORD_RE.findall(stem)}
    if len(hits) == 1:
        return hits.pop()
    return None


@dataclass
class MarkerResult:
    """Outcome of reading marker names from a name for an image with ``n_channels`` channels."""

    markers: list[str] | None
    message: str = ""
    dapi_assumed: bool = False
    is_warning: bool = False
    source: str = ""  # the name the markers were read from
    listed: list[str] = field(default_factory=list)


def _marker_list_in(name: str) -> tuple[list[str] | None, str]:
    """Return (markers, error). ``markers`` is None when there is no bracket field."""
    fields = _MARKER_FIELD_RE.findall(strip_ome_tiff_ext(name))
    if not fields:
        return None, ""
    if len(fields) > 1:
        return None, f"it has {len(fields)} {MARKER_FIELD_OPEN}...{MARKER_FIELD_CLOSE} marker lists; use exactly one"
    items = [t.strip() for t in fields[0].split(MARKER_DELIMITER)]
    if not any(items):
        return None, "its marker list is empty"
    if any(not t for t in items):
        return None, f"its marker list has an empty entry (check for doubled or trailing '{MARKER_DELIMITER}')"
    return items, ""


def markers_from_name(name: str, n_channels: int) -> MarkerResult:
    """Apply the explicit marker-list rule to one name (file name or folder name)."""
    n = int(n_channels)
    items, err = _marker_list_in(name)
    if err:
        return MarkerResult(None, f"'{name}': {err}. No marker names assigned.", is_warning=True, source=name)
    if items is None:
        return MarkerResult(None, "", source="")
    k = len(items)
    if k == n:
        return MarkerResult(list(items), f"'{name}': {k} markers for {n} channels.", source=name, listed=items)
    if k == n - 1:
        return MarkerResult(
            ["DAPI"] + list(items),
            f"'{name}': {k} markers for {n} channels, so channel 1 was assumed to be DAPI.",
            dapi_assumed=True,
            source=name,
            listed=items,
        )
    return MarkerResult(
        None,
        f"'{name}': lists {k} marker(s) but the image has {n} channel(s). Expected {n} "
        f"(one per channel) or {n - 1} (channel 1 = DAPI). No marker names assigned.",
        is_warning=True,
        source=name,
        listed=items,
    )


LEGACY_NAMES_HELP = (
    "Legacy names: for datasets named before marker lists existed, e.g. a cycle folder "
    "C1_Patient01_CD3_CD8_PDL1 holding 4-channel images. When a name has no [..] marker list, "
    "channel 1 is assumed to be DAPI and the other channels are named after the LAST N-1 "
    "underscore-separated words of the cycle folder name (N = number of channels): "
    "DAPI, CD3, CD8, PDL1. Words before those (patient, date, ...) are ignored. "
    "Marker names must not contain underscores, and the number of markers can't be checked, "
    "so a missing or extra word silently shifts every name. A [..] marker list, when present, "
    "always takes priority."
)


def legacy_markers(file_name: str | None, folder_name: str | None, n_channels: int) -> MarkerResult:
    """Legacy rule: DAPI + last N-1 underscore words of a ``C<n>_...`` cycle folder name.

    Images placed directly in a sample folder use their own file name (minus the
    ``_cyseg-stitched`` suffix) instead.
    """
    n = int(n_channels)
    candidates = [folder_name]
    if file_name:
        stem = strip_ome_tiff_ext(file_name)
        if stem.lower().endswith("_" + STITCH_OUTPUT_SUFFIX):
            stem = stem[: -len(STITCH_OUTPUT_SUFFIX) - 1]
        candidates.append(stem)
    for name in candidates:
        m = _LEGACY_CYCLE_DIR_RE.match(str(name or ""))
        if not m:
            continue
        if n <= 1:
            return MarkerResult(["DAPI"], f"'{name}': legacy names: single channel assumed to be DAPI.", dapi_assumed=True, source=name)
        tokens = m.group(2).split("_")
        if len(tokens) < n - 1:
            return MarkerResult(
                None,
                f"'{name}': legacy names: needs at least {n - 1} underscore-separated marker words for "
                f"{n} channels but has {len(tokens)}. No marker names assigned.",
                is_warning=True,
                source=name,
            )
        return MarkerResult(
            ["DAPI"] + tokens[-(n - 1):],
            f"'{name}': legacy names: channel 1 assumed to be DAPI, others from the last {n - 1} words.",
            dapi_assumed=True,
            source=name,
        )
    return MarkerResult(None, "")


def detect_markers(
    file_name: str | None, folder_name: str | None, n_channels: int, *, legacy_names: bool = False
) -> MarkerResult:
    """Markers for one cycle image: the file name's [..] list, else the folder's, else (opt-in) legacy names."""
    for name in (file_name, folder_name):
        if not name:
            continue
        res = markers_from_name(name, n_channels)
        if res.source:  # the name had a marker list (valid or not): it decides
            return res
    if legacy_names:
        res = legacy_markers(file_name, folder_name, n_channels)
        if res.source:
            return res
    where = " or ".join(f"'{n}'" for n in (file_name, folder_name) if n)
    hint = "" if legacy_names else " For older datasets named like C1_Patient01_CD3_CD8_PDL1, turn on legacy names."
    return MarkerResult(
        None,
        f"No marker list found in {where}. Add one in square brackets, e.g. "
        f"C1_Patient01_{MARKER_FIELD_OPEN}DAPI{MARKER_DELIMITER}CD3{MARKER_DELIMITER}CD8{MARKER_FIELD_CLOSE}.{hint}",
        is_warning=True,
    )
