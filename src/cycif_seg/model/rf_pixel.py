from __future__ import annotations

import numpy as np
from typing import Callable, Tuple
from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=0,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    return rf


def sample_training_pixels(
    scribbles: np.ndarray,
    *,
    max_per_class: int = 200_000,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (ys, xs) coordinates for training pixels sampled from scribbles.
    scribbles: uint8 where 0=unlabeled, 1=nuc, 2=cyto, 3=bg
    We cap per-class points to keep training fast on huge images.
    """
    if scribbles.ndim != 2:
        raise ValueError("scribbles must be 2D (Y,X)")
    rng = np.random.default_rng(rng_seed)

    ys_all = []
    xs_all = []

    for cls in (1, 2, 3):
        ys, xs = np.where(scribbles == cls)
        n = ys.size
        if n == 0:
            continue
        if n > max_per_class:
            idx = rng.choice(n, size=max_per_class, replace=False)
            ys = ys[idx]
            xs = xs[idx]
        ys_all.append(ys)
        xs_all.append(xs)

    if not ys_all:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    ys = np.concatenate(ys_all).astype(np.int32, copy=False)
    xs = np.concatenate(xs_all).astype(np.int32, copy=False)
    return ys, xs


def fit_rf_from_scribbles(
    img_yxc: np.ndarray,
    use_channels: list[int],
    scribbles: np.ndarray,
    build_features_fn: Callable[[np.ndarray, list[int]], np.ndarray],
    *,
    max_per_class: int = 200_000,
    rng_seed: int = 0,
) -> RandomForestClassifier:
    """
    Train RF using features computed ONLY at labeled pixels (no full-image feature stack).
    build_features_fn must accept an image tile (Y,X,C) and return (Y,X,F).
    """
    ys, xs = sample_training_pixels(scribbles, max_per_class=max_per_class, rng_seed=rng_seed)
    if ys.size == 0:
        raise ValueError("No labeled pixels found in scribbles.")

    # Compute features on the full image ONCE, but only if needed to index arbitrary points.
    # NOTE: this still computes full features. We'll avoid that in the worker by doing
    # training on a bounding box crop (see predict/workers.py patch).
    X_full = build_features_fn(img_yxc, use_channels)
    X_train = X_full[ys, xs, :].reshape(-1, X_full.shape[-1])
    y_train = (scribbles[ys, xs] - 1).astype(np.uint8)
    return train_rf(X_train, y_train)


def iter_tiles(H: int, W: int, tile: int):
    for y0 in range(0, H, tile):
        y1 = min(H, y0 + tile)
        for x0 in range(0, W, tile):
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1


def predict_proba_tiled(
    rf: RandomForestClassifier,
    X_full: np.ndarray,
    *,
    tile: int = 512,
) -> np.ndarray:
    """
    Predict 3-class probabilities for a full (Y,X,F) feature volume in tiles.
    Returns P of shape (Y,X,3).
    """
    H, W, F = X_full.shape
    P = np.zeros((H, W, 3), dtype=np.float32)

    for y0, y1, x0, x1 in iter_tiles(H, W, tile):
        Xt = X_full[y0:y1, x0:x1, :].reshape(-1, F)
        Pt = rf.predict_proba(Xt).astype(np.float32)
        P[y0:y1, x0:x1, :] = Pt.reshape((y1 - y0, x1 - x0, 3))

    return P
