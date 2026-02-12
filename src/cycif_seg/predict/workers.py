from __future__ import annotations

import numpy as np
from napari.qt.threading import thread_worker

from cycif_seg.model.rf_pixel import train_rf
from cycif_seg.predict.tiling import generate_tiles, sort_tiles_by_point


@thread_worker
def predict_rf_worker(
    img,
    use_channels,
    scribbles,
    build_features_fn,
    tile_size,
    center_yx,
):
    """
    Background worker:
    1) Train RF
    2) Predict in tiles prioritized by distance to center_yx
    3) Yield (y0, y1, x0, x1, P_tile)
    """
    X_full = build_features_fn(img, use_channels)

    train_mask = scribbles > 0
    X_train = X_full[train_mask].reshape(-1, X_full.shape[-1])
    y_train = (scribbles[train_mask] - 1).astype(np.uint8)

    rf = train_rf(X_train, y_train)

    H, W, F = X_full.shape
    tiles = list(generate_tiles(H, W, tile_size))
    tiles = sort_tiles_by_point(tiles, center_yx)

    for (y0, y1, x0, x1) in tiles:
        Xt = X_full[y0:y1, x0:x1, :].reshape(-1, F)
        Pt = rf.predict_proba(Xt).astype(np.float32)
        P_tile = Pt.reshape((y1 - y0, x1 - x0, 3))
        yield y0, y1, x0, x1, P_tile
