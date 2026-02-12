from __future__ import annotations

import numpy as np
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
