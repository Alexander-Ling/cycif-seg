from __future__ import annotations

import numpy as np
from napari.qt.threading import thread_worker

from cycif_seg.instance.watershed import cells_from_probs_boundary, nuclei_instances_from_probs


@thread_worker
def cells_from_probs_worker(p_nuc, p_nb, p_cyto, p_bg, params: dict):
    """Generate cells with coarse progress updates for the UI (boundary-aware nuclei)."""
    p_nuc = np.asarray(p_nuc, dtype=np.float32)
    p_nb = np.asarray(p_nb, dtype=np.float32)
    p_cyto = np.asarray(p_cyto, dtype=np.float32)
    p_bg = np.asarray(p_bg, dtype=np.float32)

    nuc_thresh = float(params.get("nuc_thresh", 0.35))
    nuc_core_thresh = float(params.get("nuc_core_thresh", max(0.05, nuc_thresh + 0.05)))
    boundary_thresh = float(params.get("boundary_thresh", 0.35))
    boundary_dilate = int(params.get("boundary_dilate", 2))

    cyto_thresh = float(params.get("cyto_thresh", 0.35))
    bg_thresh = float(params.get("bg_thresh", 0.5))
    max_grow_radius = int(params.get("max_grow_radius", 80))
    w_bg = float(params.get("w_bg", 0.5))

    min_nucleus_area = int(params.get("min_nucleus_area", 30))
    min_cell_area = int(params.get("min_cell_area", 200))
    peak_min_distance = int(params.get("peak_min_distance", 6))
    peak_footprint = int(params.get("peak_footprint", 9))

    total = 4

    yield ("stage", "Computing nuclei instances (boundary-aware)…", 1, total)
    nuclei, ndbg = nuclei_instances_from_probs(
        p_nuc,
        p_nb,
        nuc_thresh=nuc_thresh,
        nuc_core_thresh=nuc_core_thresh,
        boundary_thresh=boundary_thresh,
        boundary_dilate=boundary_dilate,
        min_nucleus_area=min_nucleus_area,
        peak_min_distance=max(1, peak_min_distance - 2),
        peak_footprint=max(3, peak_footprint - 2),
    )
    # Send to UI for optional overlay
    yield ("nuclei", nuclei)

    yield ("stage", "Running watershed (cells)…", 2, total)
    labels, debug = cells_from_probs_boundary(
        p_nuc,
        p_nb,
        p_cyto,
        p_bg,
        nuc_thresh=nuc_thresh,
        nuc_core_thresh=nuc_core_thresh,
        boundary_thresh=boundary_thresh,
        boundary_dilate=boundary_dilate,
        cyto_thresh=cyto_thresh,
        bg_thresh=bg_thresh,
        max_grow_radius=max_grow_radius,
        w_bg=w_bg,
        min_nucleus_area=min_nucleus_area,
        min_cell_area=min_cell_area,
        peak_min_distance=peak_min_distance,
        peak_footprint=peak_footprint,
    )

    # Also pass nuclei markers if UI wants the seeds/markers overlay (legacy checkbox)
    try:
        yield ("markers", debug.get("seeds") if isinstance(debug, dict) else nuclei)
    except Exception:
        pass

    yield ("stage", "Done.", 4, total)
    return labels, debug



@thread_worker
def nuclei_instances_from_probs_worker(p_nuc, p_nb, p_bg, params: dict, *, run_id: int = 0, is_cancelled=None):
    """Generate nuclei instances (boundary-aware) with coarse progress updates for the UI.

    Notes
    -----
    Cancellation is cooperative: we check `is_cancelled()` between coarse stages and
    suppress yields/returns if cancellation is requested.
    """
    p_nuc = np.asarray(p_nuc, dtype=np.float32)
    p_nb = np.asarray(p_nb, dtype=np.float32)
    p_bg = np.asarray(p_bg, dtype=np.float32)

    if callable(is_cancelled) and is_cancelled():
        return

    yield ("stage", run_id, "Computing nuclei instances (boundary-aware)…", 1, 2)

    # Optionally suppress nucleus probability where background is high
    bg_thresh = float(params.get("bg_thresh", 0.6))
    if bg_thresh is not None:
        p_nuc_eff = p_nuc.copy()
        p_nuc_eff[p_bg >= bg_thresh] = 0.0
    else:
        p_nuc_eff = p_nuc

    nuclei, debug = nuclei_instances_from_probs(
        p_nuc_eff,
        p_nb,
        nuc_thresh=float(params.get("nuc_thresh", 0.35)),
        nuc_core_thresh=float(params.get("nuc_core_thresh", 0.45)),
        boundary_thresh=float(params.get("boundary_thresh", 0.35)),
        boundary_dilate=int(params.get("boundary_dilate", 2)),
        min_nucleus_area=int(params.get("min_nucleus_area", 30)),
        peak_min_distance=int(params.get("peak_min_distance", 4)),
        peak_footprint=int(params.get("peak_footprint", 7)),
        smooth_sigma=float(params.get("smooth_sigma", 1.0)),
        dist_percentile=float(params.get("dist_percentile", 70.0)),
    )

    if callable(is_cancelled) and is_cancelled():
        return

    yield ("nuclei", run_id, nuclei)
    yield ("debug", run_id, debug)
    yield ("stage", run_id, "Done.", 2, 2)
