from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel


def build_features(
    img_yxc: np.ndarray,
    use_channels: list[int],
    *,
    sigmas=(0.0, 1.0, 2.0),
) -> np.ndarray:
    """
    Build per-pixel feature stack using only selected channels.
    Output shape: (Y, X, F)
    Features per channel:
      - raw (sigma=0)
      - gaussian blurred at sigma
      - sobel edge magnitude on raw
    """
    if len(use_channels) == 0:
        raise ValueError("No channels selected.")

    feats = []
    for c in use_channels:
        ch = img_yxc[..., c]
        for s in sigmas:
            if s == 0.0:
                feats.append(ch)
            else:
                feats.append(gaussian_filter(ch, sigma=s))
        feats.append(sobel(ch))

    X = np.stack(feats, axis=-1).astype(np.float32, copy=False)
    return X
