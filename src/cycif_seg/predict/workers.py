from __future__ import annotations
from time import time

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
    *,
    run_id: int = 0,
    is_cancelled=None,
    train_pad: int = 256,
    max_per_class: int = 200_000,
    rng_seed: int = 0,
):
    """
    Background worker:
    1) Train RF
    2) Predict in tiles prioritized by distance to center_yx
    3) Yield tagged tuples
    """
   
    if is_cancelled is None:
        is_cancelled = lambda: False  # noqa: E731

    H, W, C = img.shape

    # ---- Train on a crop around scribbles (fast) ----
    train_mask = scribbles > 0
    ys, xs = np.where(train_mask)
    if ys.size == 0:
        raise ValueError("No labeled pixels found in scribbles.")

    y0 = max(int(ys.min()) - int(train_pad), 0)
    y1 = min(int(ys.max()) + int(train_pad) + 1, H)
    x0 = max(int(xs.min()) - int(train_pad), 0)
    x1 = min(int(xs.max()) + int(train_pad) + 1, W)

    img_crop = img[y0:y1, x0:x1, :]
    scr_crop = scribbles[y0:y1, x0:x1]

    # Status: starting training (before any tiles appear)
    yield ("status", run_id, "Training: computing features on scribble region…")

    # Optionally cap training points per class to keep RF training fast
    rng = np.random.default_rng(int(rng_seed))
    yy_list = []
    xx_list = []
    for cls in (1, 2, 3):
        yy, xx = np.where(scr_crop == cls)
        n = yy.size
        if n == 0:
            continue
        if n > int(max_per_class):
            idx = rng.choice(n, size=int(max_per_class), replace=False)
            yy = yy[idx]
            xx = xx[idx]
        yy_list.append(yy)
        xx_list.append(xx)

    if not yy_list:
        raise ValueError("Scribbles contained no labeled pixels after cropping.")

    yy = np.concatenate(yy_list)
    xx = np.concatenate(xx_list)

    # Compute features only for the training crop
    t0 = time()
    X_crop = build_features_fn(img_crop, use_channels)
    F = X_crop.shape[-1]
    X_train = X_crop[yy, xx, :].reshape(-1, F)
    y_train = (scr_crop[yy, xx] - 1).astype(np.uint8)
    yield ("status", run_id, f"Training: fitting Random Forest… (F={F}, n={X_train.shape[0]})")


    if is_cancelled():
        return

    rf = train_rf(X_train, y_train)

    dt = time() - t0
    yield ("status", run_id, f"Training complete in {dt:0.1f}s. Predicting tiles…")

    tiles = list(generate_tiles(H, W, tile_size))
    tiles = sort_tiles_by_point(tiles, center_yx)
    n_tiles = len(tiles)
    yield ("progress_init", run_id, n_tiles)

    for i, (y0, y1, x0, x1) in enumerate(tiles, start=1):
        if is_cancelled():
            return

        # Build features for THIS TILE ONLY (memory + speed win)
        tile_img = img[y0:y1, x0:x1, :]
        X_tile = build_features_fn(tile_img, use_channels)
        Xt = X_tile.reshape(-1, X_tile.shape[-1])
        Pt = rf.predict_proba(Xt).astype(np.float32)
        P_tile = Pt.reshape((y1 - y0, x1 - x0, 3))
        # Include run_id so UI can ignore stale updates
        yield ("tile", run_id, y0, y1, x0, x1, P_tile)
        # Determinate progress update (cheap)
        yield ("progress", run_id, i, n_tiles)
