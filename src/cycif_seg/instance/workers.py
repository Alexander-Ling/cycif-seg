from __future__ import annotations

from napari.qt.threading import thread_worker
from cycif_seg.instance.watershed import cells_from_probs


@thread_worker
def cells_from_probs_worker(p_nuc, p_cyto, params: dict):
    labels, debug = cells_from_probs(p_nuc, p_cyto, **params)
    return labels, debug
