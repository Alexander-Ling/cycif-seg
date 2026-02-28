from __future__ import annotations
from time import time
import time as _time
import os
import concurrent.futures as cf

import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
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

    # img can be a numpy array or a dask-backed lazy array. Ensure we hand
    # numpy arrays to feature builders.
    img_crop = np.asarray(img[y0:y1, x0:x1, :])
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

    # Provide the trained model back to the UI so it can be saved into a project.
    # Note: this is a reference to a sklearn model; callers should persist it via joblib/pickle.
    yield (
        "trained_model",
        run_id,
        rf,
        {
            "use_channels": list(use_channels),
            "train_crop_yxxy": [int(y0), int(y1), int(x0), int(x1)],
            "max_per_class": int(max_per_class),
            "rng_seed": int(rng_seed),
        },
    )

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
        tile_img = np.asarray(img[y0:y1, x0:x1, :])
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
            # If some classes are absent from scribbles, sklearn will omit them from
            # predict_proba output. Expand back to the full expected class set.
            # We currently expect 3 classes: nucleus, nuclear_boundary, background.
            k_expected = 3
            if Pt_all.shape[1] != k_expected:
                Pt_full = np.zeros((Pt_all.shape[0], k_expected), dtype=np.float32)
                for j, cls in enumerate(getattr(rf, 'classes_', [])):
                    cls_i = int(cls)
                    if 0 <= cls_i < k_expected:
                        Pt_full[:, cls_i] = Pt_all[:, j]
                Pt_all = Pt_full
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
                k = int(Pt_all.shape[1])
                P_tile = Pt_all[offset:offset + n, :].reshape((dy, dx, k))
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


@thread_worker
def propagate_nuclei_edits_worker(
    img,
    use_channels,
    base_labels,
    edited_labels,
    build_features_fn,
    tile_size,
    center_yx,
    run_id,
    *,
    batch_tiles=4,
    feature_workers=None,
    progress_every=10,
    training_budget=200_000,
    seed=0,
    train_dilate_px=16,
    is_cancelled=None,
):
    """Train an RF from *edited nuclei instance labels* and predict probabilities across the image.

    Strategy A (Step 2b):
      - derive sparse pixel labels from the *difference* between base_labels and edited_labels
      - train an RF on multi-scale features (nucleus / boundary / background)
      - predict in tiles (same streaming protocol as predict_rf_worker)

    Notes:
      - Scribbles are generated only near edited pixels (fast + focused).
      - Cancellation is controlled by the caller via is_cancelled().
    """
    if is_cancelled is None:
        is_cancelled = lambda: False  # noqa: E731

    H, W, _ = img.shape
    if base_labels.shape != (H, W):
        raise ValueError(f"base_labels has shape {getattr(base_labels, 'shape', None)}, expected {(H, W)}")
    if edited_labels.shape != (H, W):
        raise ValueError(f"edited_labels has shape {getattr(edited_labels, 'shape', None)}, expected {(H, W)}")

    # ---- Build sparse scribbles from edited instances (only near diffs) ----
    yield ("stage", run_id, "Building training labels from edits…", 1, 3)

    diff = (edited_labels != base_labels)
    if not np.any(diff):
        raise ValueError("No edits detected (edited labels match base labels).")

    if int(train_dilate_px) > 0:
        region = dilation(diff, disk(int(train_dilate_px)))
    else:
        region = diff

    # Labels derived from EDITED map inside region; everything else is unlabeled (=0)
    scribbles = np.zeros((H, W), dtype=np.uint8)
    ed = edited_labels
    nuc_mask = (ed > 0)
    boundary = find_boundaries(ed, mode="inner")  # bool

    # background / nucleus / boundary only within region
    r = region
    scribbles[r & (~nuc_mask)] = 3
    scribbles[r & nuc_mask] = 1
    scribbles[r & boundary] = 2  # override nucleus

    if is_cancelled():
        return

    # ---- Train RF (sample pixels; avoid dask boolean indexing) ----
    yield ("stage", run_id, "Training RF from edited nuclei…", 2, 3)

    # Collect coords by class for balanced sampling
    rng = np.random.default_rng(int(seed))
    coords = []
    classes = (1, 2, 3)
    max_total = int(training_budget) if training_budget is not None else None
    per_class = None
    if max_total is not None:
        per_class = max(1, max_total // len(classes))

    for c in classes:
        ys, xs = np.where(scribbles == c)
        n = ys.size
        if n == 0:
            continue
        if per_class is not None and n > per_class:
            sel = rng.choice(n, size=per_class, replace=False)
            ys = ys[sel]
            xs = xs[sel]
        coords.append((ys, xs, c))

    if not coords:
        raise ValueError("No training pixels found near edits. Try increasing train_dilate_px.")

    ys_all = np.concatenate([c[0] for c in coords], axis=0)
    xs_all = np.concatenate([c[1] for c in coords], axis=0)
    y_train = (scribbles[ys_all, xs_all] - 1).astype(np.uint8, copy=False)

    # Build features lazily, then gather sampled pixels via vindex (works with dask)
    X_full = build_features_fn(img, use_channels)

    try:
        X_train = X_full.vindex[ys_all, xs_all, :].compute()
    except Exception:
        # fallback for non-dask feature arrays
        X_train = np.asarray(X_full)[ys_all, xs_all, :]
    X_train = np.asarray(X_train, dtype=np.float32).reshape(-1, X_train.shape[-1])

    # If training_budget is set but class sampling left us under it, that's fine.

    # train_rf() owns RF hyperparameters (including n_jobs) in cycif_seg.model.rf_pixel
    rf = train_rf(X_train, y_train)

    yield (
        "trained_model",
        run_id,
        rf,
        {
            "kind": "rf_from_nuclei_edits",
            "use_channels": list(use_channels),
            "training_budget": int(training_budget) if training_budget is not None else None,
            "seed": int(seed),
            "train_dilate_px": int(train_dilate_px),
        },
    )

    if is_cancelled():
        return

    # ---- Predict in tiles (pipeline matches predict_rf_worker) ----
    yield ("stage", run_id, "Predicting tiles…", 3, 3)

    tiles = list(generate_tiles(H, W, tile_size))
    tiles = sort_tiles_by_point(tiles, center_yx)
    n_tiles = len(tiles)

    if feature_workers is None:
        feature_workers = max(1, min(os.cpu_count() or 1, 8))

    # Timing helpers
    t_features_total = 0.0
    t_predict_total = 0.0
    n_feature_tiles = 0
    n_predict_tiles = 0

    F = int(X_full.shape[-1])

    def build_one(y0, y1, x0, x1):
        t0 = time.time()
        # ensure numpy tile
        X_tile = np.asarray(X_full[y0:y1, x0:x1, :], dtype=np.float32).reshape(-1, F)
        dt = time.time() - t0
        return (y0, y1, x0, x1, X_tile, dt)

    def predict_many(items):
        nonlocal t_predict_total, n_predict_tiles
        t0 = time.time()
        Xt_all = np.concatenate([it[4] for it in items], axis=0)
        Pt_all = rf.predict_proba(Xt_all).astype(np.float32, copy=False)
        dt = time.time() - t0
        t_predict_total += dt
        n_predict_tiles += len(items)
        return Pt_all

    i = 0
    with cf.ThreadPoolExecutor(max_workers=int(feature_workers)) as ex:
        futures = [ex.submit(build_one, y0, y1, x0, x1) for (y0, y1, x0, x1) in tiles]

        batch_items = []
        for fut in cf.as_completed(futures):
            if is_cancelled():
                return
            y0, y1, x0, x1, Xt, dt = fut.result()
            t_features_total += float(dt)
            n_feature_tiles += 1
            batch_items.append((y0, y1, x0, x1, Xt))
            if len(batch_items) < int(batch_tiles):
                continue

            Pt_all = predict_many(batch_items)
            offset = 0
            for (yy0, yy1, xx0, xx1, _) in batch_items:
                dy = int(yy1 - yy0)
                dx = int(xx1 - xx0)
                n = dy * dx
                k = int(Pt_all.shape[1])
                P_tile = Pt_all[offset:offset + n, :].reshape((dy, dx, k))
                offset += n
                i += 1
                yield ("tile", run_id, yy0, yy1, xx0, xx1, P_tile)
                if (i % int(progress_every)) == 0 or i == n_tiles:
                    yield ("progress", run_id, i, n_tiles)

            batch_items = []

        if batch_items and not is_cancelled():
            Pt_all = predict_many(batch_items)
            offset = 0
            for (yy0, yy1, xx0, xx1, _) in batch_items:
                dy = int(yy1 - yy0)
                dx = int(xx1 - xx0)
                n = dy * dx
                k = int(Pt_all.shape[1])
                P_tile = Pt_all[offset:offset + n, :].reshape((dy, dx, k))
                offset += n
                i += 1
                yield ("tile", run_id, yy0, yy1, xx0, xx1, P_tile)
                if (i % int(progress_every)) == 0 or i == n_tiles:
                    yield ("progress", run_id, i, n_tiles)

    # Timing summary (optional)
    try:
        if n_feature_tiles > 0:
            print(
                f"[TIMING][edits] Features: total={t_features_total:.2f}s "
                f"avg_per_tile={t_features_total/n_feature_tiles:.4f}s"
            )
        if n_predict_tiles > 0:
            print(
                f"[TIMING][edits] Predict: total={t_predict_total:.2f}s "
                f"avg_per_batch={t_predict_total/n_predict_tiles:.4f}s"
            )
    except Exception:
        pass
