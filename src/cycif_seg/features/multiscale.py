from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace

from skimage.feature import structure_tensor, hessian_matrix

# hessian eigvals API can vary; we handle robustly.
try:
    from skimage.feature import hessian_matrix_eigvals as _hess_eigvals  # type: ignore
except Exception:  # pragma: no cover
    _hess_eigvals = None


def _structure_tensor_eigs(Axx: np.ndarray, Axy: np.ndarray, Ayy: np.ndarray):
    """
    Analytic eigenvalues of [[Axx, Axy],[Axy,Ayy]] per-pixel. Returns (l1, l2).
    Version-proof (doesn't depend on scikit-image eig helper names/signatures).
    """
    tr = Axx + Ayy
    disc = (Axx - Ayy) ** 2 + 4.0 * (Axy ** 2)
    # numerical safety
    root = np.sqrt(np.maximum(disc, 0.0))
    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)
    return l1, l2


def _hessian_eigs(H_elems):
    """
    Return (e1, e2) for 2D Hessian.
    """
    if _hess_eigvals is not None:
        e1, e2 = _hess_eigvals(H_elems)
        return e1, e2

    # Manual fallback for order="rc": Hrr, Hrc, Hcc
    Hrr, Hrc, Hcc = H_elems
    tr = Hrr + Hcc
    det = Hrr * Hcc - Hrc * Hrc
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    e1 = 0.5 * (tr + root)
    e2 = 0.5 * (tr - root)
    return e1, e2



def features_per_channel(
    *,
    sigmas=(2.0, 3.0, 4.0, 5.0, 6.0),
    use_intensity: bool = True,
    use_gaussian: bool = True,
    use_gradmag: bool = True,
    use_log: bool = True,
    use_dog: bool = True,
    use_structure_tensor: bool = True,
    use_hessian: bool = True,
) -> int:
    """Return number of feature maps produced per channel for the given config."""
    n = 0
    if use_intensity:
        n += 1
    sigmas = tuple(float(s) for s in sigmas)
    per_sigma = 0
    if use_gaussian:
        per_sigma += 1
    if use_gradmag:
        per_sigma += 1
    if use_log:
        per_sigma += 1
    if use_dog:
        per_sigma += 1
    if use_structure_tensor:
        per_sigma += 2
    if use_hessian:
        per_sigma += 2
    n += per_sigma * len(sigmas)
    return int(n)


def build_features(
    img_yxc: np.ndarray,
    use_channels: list[int],
    *,
    sigmas=(2.0, 3.0, 4.0, 5.0, 6.0),
    use_intensity: bool = True,
    use_gaussian: bool = True,
    use_gradmag: bool = True,
    use_log: bool = True,
    use_dog: bool = True,
    use_structure_tensor: bool = True,
    use_hessian: bool = True,
    dog_sigma_ratio: float = 1.6,
) -> np.ndarray:
    """
    ilastik-like feature stack for 2D multichannel images (Y,X,C).

    Per selected channel:
      - raw intensity (once)
      For each sigma in sigmas:
        - gaussian
        - gaussian gradient magnitude
        - laplacian of gaussian
        - difference of gaussians
        - structure tensor eigenvalues (2)
        - hessian eigenvalues (2)
    """
    if len(use_channels) == 0:
        raise ValueError("No channels selected.")
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")
    if img_yxc.shape[2] <= max(use_channels):
        raise ValueError("use_channels contains an index out of bounds.")

    img = img_yxc.astype(np.float32, copy=False)

    sigmas = tuple(float(s) for s in sigmas)
    if any(s <= 0 for s in sigmas):
        raise ValueError("All sigmas must be > 0 for ilastik-like features.")

    feats: list[np.ndarray] = []

    for c in use_channels:
        ch = img[..., c]

        if use_intensity:
            feats.append(ch)

        for s in sigmas:
            if use_gaussian:
                feats.append(gaussian_filter(ch, sigma=s))

            if use_gradmag:
                feats.append(gaussian_gradient_magnitude(ch, sigma=s))

            if use_log:
                feats.append(gaussian_laplace(ch, sigma=s))

            if use_dog:
                s2 = float(s) * float(dog_sigma_ratio)
                g1 = gaussian_filter(ch, sigma=s)
                g2 = gaussian_filter(ch, sigma=s2)
                feats.append(g1 - g2)

            if use_structure_tensor:
                Axx, Axy, Ayy = structure_tensor(ch, sigma=s)
                l1, l2 = _structure_tensor_eigs(Axx, Axy, Ayy)
                feats.append(l1.astype(np.float32, copy=False))
                feats.append(l2.astype(np.float32, copy=False))

            if use_hessian:
                H_elems = hessian_matrix(ch, sigma=s, order="rc")
                e1, e2 = _hessian_eigs(H_elems)
                feats.append(e1.astype(np.float32, copy=False))
                feats.append(e2.astype(np.float32, copy=False))

    X = np.stack(feats, axis=-1).astype(np.float32, copy=False)
    return X
