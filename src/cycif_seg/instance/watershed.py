from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label

from scipy.ndimage import gaussian_filter

def _relabel_instances(lbl: np.ndarray) -> np.ndarray:
    """
    Relabel an instance-labeled image to 1..N WITHOUT merging touching instances.
    (Do NOT use connected-components on a binary mask; that merges touching objects.)
    """
    lbl = lbl.astype(np.int32, copy=False)
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(lbl, dtype=np.int32)
    out = np.zeros_like(lbl, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
       out[lbl == int(old_id)] = int(new_id)
    return out

def nuclei_markers_from_prob(
    p_nuc: np.ndarray,
    *,
    nuc_thresh: float = 0.35,
    min_nucleus_area: int = 30,
    peak_min_distance: int = 4,
    peak_footprint: int = 7,
    smooth_sigma: float = 1.0,
    seed_percentile: float = 99.5,
    dist_percentile: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create nucleus instance markers from nucleus probability.
    This is designed to be robust even when probabilities are "soft" (e.g. max < 0.5).

    Returns:
      markers: int32 image (0 background; 1..N nuclei seeds)
      nuc_mask: boolean nucleus mask used to derive markers
    """
    if p_nuc.ndim != 2:
        raise ValueError("p_nuc must be 2D (Y,X).")

    p = p_nuc.astype(np.float32, copy=False)
    if smooth_sigma and smooth_sigma > 0:
        p = gaussian_filter(p, sigma=float(smooth_sigma))

    # Primary nucleus mask (can be relatively permissive)
    nuc_mask = p >= float(nuc_thresh)
    nuc_mask = binary_opening(nuc_mask, footprint=disk(1))
    nuc_mask = binary_closing(nuc_mask, footprint=disk(1))
    nuc_mask = remove_small_objects(nuc_mask, min_size=int(min_nucleus_area))

    if not np.any(nuc_mask):
        markers = np.zeros_like(p_nuc, dtype=np.int32)
        return markers, nuc_mask

    # Seeds from peaks of the distance transform inside the nucleus mask.
    # This is typically more robust than probability peaks for touching/overlapping nuclei.
    dist = ndi.distance_transform_edt(nuc_mask)
    if dist_percentile is not None:
        dp = float(np.percentile(dist[nuc_mask], float(dist_percentile)))
        dist = np.where(dist >= dp, dist, 0.0).astype(np.float32, copy=False)

    coords = peak_local_max(
        dist,
        labels=nuc_mask.astype(np.uint8),
        min_distance=int(peak_min_distance),
        footprint=np.ones((int(peak_footprint), int(peak_footprint)), dtype=bool),
        exclude_border=False,
    )

    if coords.size == 0:
        # Fallback: treat connected components as nuclei
        markers = cc_label(nuc_mask).astype(np.int32)
        return markers, nuc_mask

    seed_img = np.zeros_like(p, dtype=np.int32)
    seed_img[coords[:, 0], coords[:, 1]] = 1
    seed_markers = cc_label(seed_img > 0).astype(np.int32)

    # Split touching nuclei using watershed on distance inside nuc_mask
    nuclei_labels = watershed(-dist, markers=seed_markers, mask=nuc_mask).astype(np.int32)

    # Remove tiny nuclei after splitting and relabel
    nuclei_labels = nuclei_labels.astype(np.int32)
    if nuclei_labels.max() > 0 and min_nucleus_area and min_nucleus_area > 0:
        counts = np.bincount(nuclei_labels.ravel())
        keep = np.zeros(nuclei_labels.max() + 1, dtype=bool)
        keep[np.where(counts >= int(min_nucleus_area))[0]] = True
        keep[0] = False
        nuclei_labels = np.where(keep[nuclei_labels], nuclei_labels, 0).astype(np.int32)
        nuclei_labels = _relabel_instances(nuclei_labels)

    markers = nuclei_labels
    return markers, nuc_mask


def cells_from_probs(
    p_nuc: np.ndarray,
    p_cyto: np.ndarray,
    *,
    nuc_thresh: float = 0.35,
    cyto_thresh: float = 0.35,
    min_nucleus_area: int = 30,
    min_cell_area: int = 200,
    peak_min_distance: int = 6,
    peak_footprint: int = 9,
) -> tuple[np.ndarray, dict]:
    """
    Generate cell instance labels via nucleus-seeded watershed.

    Rules:
      - Exactly 1 nucleus seed per cell (as best-effort via peak seeds)
      - Cells expand ONLY into cytoplasm-allowed region.
      - Cytoplasm not assigned to any seed remains 0 (unassigned).

    Returns:
      cell_labels: int32 (0 unassigned; 1..N cells)
      debug: dict with intermediate images (markers, nuc_mask, cyto_mask)
    """
    if p_nuc.shape != p_cyto.shape:
        raise ValueError("p_nuc and p_cyto must have the same shape.")
    if p_nuc.ndim != 2:
        raise ValueError("probability maps must be 2D (Y,X).")

    markers, nuc_mask = nuclei_markers_from_prob(
        p_nuc,
        nuc_thresh=nuc_thresh,
        min_nucleus_area=min_nucleus_area,
        peak_min_distance=peak_min_distance,
        peak_footprint=peak_footprint,
    )

    print(f"markers.max() after nuclei_markers_from_prob = {markers.max()}")

    cyto_mask = p_cyto >= cyto_thresh

    # Ensure markers are inside the watershed mask.
    # We allow nuclei that are inside nuc_mask even if cyto_mask is weak there:
    ws_mask = cyto_mask | (markers > 0)

    if markers.max() == 0:
        # No nuclei => no cells
        cell_labels = np.zeros_like(markers, dtype=np.int32)
        return cell_labels, {"markers": markers, "nuc_mask": nuc_mask, "cyto_mask": cyto_mask}

    # Use a "basin" image where growth prefers higher cyto probability.
    # Watershed expects low values are basins; so negate cyto prob.
    elevation = -p_cyto.astype(np.float32)

    cell_labels = watershed(
        elevation,
        markers=markers,
        mask=ws_mask,
    ).astype(np.int32)

    print(f"cell_labels.max() after watershed: {cell_labels.max()}")

    # Remove tiny cells (often noise from tiny markers)
    if min_cell_area and min_cell_area > 0:
        # remove_small_objects expects boolean per label; easiest: relabel after filtering
        keep = np.zeros(cell_labels.max() + 1, dtype=bool)
        # 0 stays False
        counts = np.bincount(cell_labels.ravel())
        keep[np.where(counts >= int(min_cell_area))[0]] = True
        keep[0] = False
        filtered = np.where(keep[cell_labels], cell_labels, 0).astype(np.int32)
        # Relabel to make ids compact
        cell_labels = _relabel_instances(filtered)

    print(f"cell_labels.max() after filter and relabel: {cell_labels.max()}")

    debug = {
        "markers": markers,
        "nuc_mask": nuc_mask,
        "cyto_mask": cyto_mask,
        "ws_mask": ws_mask,
    }
    return cell_labels, debug
