from __future__ import annotations
from time import time
import time as _time
import os
import concurrent.futures as cf

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
    progress_every: int = 1,
    batch_tiles: int = 1,
    feature_workers: int | None = None,
    prefetch_tiles: int = 0,
    rf_n_jobs: int | None = None,
):
    """
    Background worker:
    1) Train RF
    2) Predict in tiles prioritized by distance to center_yx
    3) Predict in tiles (optionally batched) and yield tagged tuples
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
    # ---- Timing instrumentation ----
    t_features_total = 0.0
    t_predict_total = 0.0
    n_feature_tiles = 0
    n_predict_tiles = 0

    progress_every = max(1, int(progress_every))
    batch_tiles = max(1, int(batch_tiles))
    if feature_workers is None:
        # Feature computation is often the bottleneck and is mostly numpy/scipy (releases GIL).
        # Use a small thread pool by default to precompute features while RF predicts.
        feature_workers = min(4, max(1, (os.cpu_count() or 4) // 4))
    feature_workers = max(1, int(feature_workers))
    if prefetch_tiles is None:
        prefetch_tiles = 0
    prefetch_tiles = int(prefetch_tiles)
    if prefetch_tiles <= 0:
        prefetch_tiles = max(batch_tiles * 2, 4)
    # Avoid oversubscription when feature extraction is threaded.
    if rf_n_jobs is None and feature_workers > 1:
        try:
            rf.n_jobs = 1
        except Exception:
            pass
    elif rf_n_jobs is not None:
        try:
            rf.n_jobs = int(rf_n_jobs)
        except Exception:
            pass

    i = 0

    def _compute_tile_features(y0, y1, x0, x1):
        tile_img = img[y0:y1, x0:x1, :]
        t0 = _time.perf_counter()
        X_tile = build_features_fn(tile_img, use_channels)
        dt = _time.perf_counter() - t0
        return (y0, y1, x0, x1, X_tile.reshape(-1, X_tile.shape[-1]), dt)

    tiles_iter = iter(tiles)
    pending: list[cf.Future] = []

    with cf.ThreadPoolExecutor(max_workers=feature_workers) as ex:
        # Prime the pipeline
        for _ in range(min(prefetch_tiles, n_tiles)):
            y0, y1, x0, x1 = next(tiles_iter)
            pending.append(ex.submit(_compute_tile_features, y0, y1, x0, x1))

        while pending:
            if is_cancelled():
                return

            batch_futs = pending[:batch_tiles]
            pending = pending[batch_tiles:]

            batch_items = []
            for f in batch_futs:
                y0, y1, x0, x1, Xt, dt_feat = f.result()
                t_features_total += dt_feat
                n_feature_tiles += 1
                batch_items.append((y0, y1, x0, x1, Xt))

            # Top up pipeline
            while len(pending) < prefetch_tiles:
                try:
                    y0, y1, x0, x1 = next(tiles_iter)
                except StopIteration:
                    break
                pending.append(ex.submit(_compute_tile_features, y0, y1, x0, x1))

            X_list = [Xt for (*_, Xt) in batch_items]
            Xt_all = np.concatenate(X_list, axis=0)
            t0 = _time.perf_counter()
            Pt_all = rf.predict_proba(Xt_all).astype(np.float32)
            dt_pred = _time.perf_counter() - t0
            t_predict_total += dt_pred
            n_predict_tiles += len(batch_items)

            offset = 0
            for (y0, y1, x0, x1, Xt) in batch_items:
                if is_cancelled():
                    return
                dy = int(y1 - y0)
                dx = int(x1 - x0)
                n = dy * dx
                P_tile = Pt_all[offset:offset + n, :].reshape((dy, dx, 3))
                offset += n
                i += 1
                yield ("tile", run_id, y0, y1, x0, x1, P_tile)
                if (i % progress_every) == 0 or i == n_tiles:
                    yield ("progress", run_id, i, n_tiles)

            # ---- Timing summary ----
            try:
                if n_feature_tiles > 0:
                    print(
                        f"[TIMING] Features: total={t_features_total:.2f}s "
                        f"avg_per_tile={t_features_total/n_feature_tiles:.4f}s"
                    )
                if n_predict_tiles > 0:
                    print(
                        f"[TIMING] Predict: total={t_predict_total:.2f}s "
                        f"avg_per_tile={t_predict_total/n_predict_tiles:.4f}s"
                    )
            except Exception:
                pass