from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import os
import threading
import time as _time
import math
from typing import Callable, Iterable, Optional

import numpy as np

# Optional dependency: zarr + numcodecs
try:  # pragma: no cover
    import zarr  # type: ignore
    from numcodecs import Blosc  # type: ignore
except Exception:  # pragma: no cover
    zarr = None
    Blosc = None

from cycif_seg.features.multiscale import build_features, features_per_channel

# Debug logging for Step 2a Zarr feature cache.
# Enable with:
#   CYCIF_SEG_STEP2A_CACHE_DEBUG=1
_STEP2A_CACHE_DEBUG = str(os.environ.get("CYCIF_SEG_STEP2A_CACHE_DEBUG", "0")).strip().lower() not in {
    "0", "false", "no", "off", ""
}


def _dbg(msg: str) -> None:
    if _STEP2A_CACHE_DEBUG:
        try:
           print(f"[Step2aCache] {msg}")
        except Exception:
            pass


def _array_debug_summary(arr: np.ndarray) -> str:
    """Compact summary string for debug logging."""
    try:
        a = np.asarray(arr)
        if a.size == 0:
            return "shape=() empty"
        finite = np.isfinite(a)
        n_finite = int(finite.sum())
        n_total = int(a.size)
        if n_finite == 0:
            return f"shape={tuple(int(x) for x in a.shape)} finite=0/{n_total}"
        af = a[finite]
        amin = float(np.min(af))
        amax = float(np.max(af))
        amean = float(np.mean(af))
        return (
            f"shape={tuple(int(x) for x in a.shape)} "
            f"finite={n_finite}/{n_total} "
            f"min={amin:.6g} max={amax:.6g} mean={amean:.6g}"
        )
    except Exception as e:
        return f"summary_failed:{type(e).__name__}"


def _is_suspicious_array(arr: np.ndarray, *, nearly_constant_tol: float = 1e-6) -> tuple[bool, str]:
    """Return (is_suspicious, reason) for debug logging."""
    try:
        a = np.asarray(arr)
        if a.size == 0:
            return True, "empty"
        if not np.all(np.isfinite(a)):
            return True, "nonfinite"
        amin = float(np.min(a))
        amax = float(np.max(a))
        if amin == 0.0 and amax == 0.0:
            return True, "all_zero"
        if abs(amax - amin) <= float(nearly_constant_tol):
            return True, f"nearly_constant(range={amax - amin:.3g})"
        return False, ""
    except Exception as e:
        return True, f"summary_exception:{type(e).__name__}"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


@dataclass
class FeatureCacheConfig:
    sigmas: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)
    use_intensity: bool = True
    use_gaussian: bool = True
    use_gradmag: bool = True
    use_log: bool = True
    use_dog: bool = True
    use_structure_tensor: bool = True
    use_hessian: bool = True
    dog_sigma_ratio: float = 1.6
    feature_version: str = "multiscale_v1"

    # Intensity normalization (applied to tile image BEFORE feature computation)
    # Keeps features in a bounded range so they can be stored as float16 safely.
    norm_mode: str = "none"  # "none" or "p0.5_p99.5"
    norm_p_lo: float = 0.5
    norm_p_hi: float = 99.5
    norm_eps: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "sigmas": list(self.sigmas),
            "use_intensity": bool(self.use_intensity),
            "use_gaussian": bool(self.use_gaussian),
            "use_gradmag": bool(self.use_gradmag),
            "use_log": bool(self.use_log),
            "use_dog": bool(self.use_dog),
            "use_structure_tensor": bool(self.use_structure_tensor),
            "use_hessian": bool(self.use_hessian),
            "dog_sigma_ratio": float(self.dog_sigma_ratio),
            "feature_version": str(self.feature_version),
            "norm_mode": str(self.norm_mode),
            "norm_p_lo": float(self.norm_p_lo),
            "norm_p_hi": float(self.norm_p_hi),
            "norm_eps": float(self.norm_eps),
        }

    @staticmethod
    def from_dict(d: dict) -> "FeatureCacheConfig":
        d = dict(d or {})
        return FeatureCacheConfig(
            sigmas=tuple(float(x) for x in (d.get("sigmas") or (2.0, 3.0, 4.0, 5.0, 6.0))),
            use_intensity=bool(d.get("use_intensity", True)),
            use_gaussian=bool(d.get("use_gaussian", True)),
            use_gradmag=bool(d.get("use_gradmag", True)),
            use_log=bool(d.get("use_log", True)),
            use_dog=bool(d.get("use_dog", True)),
            use_structure_tensor=bool(d.get("use_structure_tensor", True)),
            use_hessian=bool(d.get("use_hessian", True)),
            dog_sigma_ratio=float(d.get("dog_sigma_ratio", 1.6)),
            feature_version=str(d.get("feature_version", "multiscale_v1")),
            norm_mode=str(d.get("norm_mode", "none")),
            norm_p_lo=float(d.get("norm_p_lo", 0.5)),
            norm_p_hi=float(d.get("norm_p_hi", 99.5)),
            norm_eps=float(d.get("norm_eps", 1e-6)),
        )

    def hash(self) -> str:
        return _sha1(json.dumps(self.to_dict(), sort_keys=True))


class ZarrTileFeatureCache:
    """Incremental per-channel feature cache stored in Zarr.

    Design goals
    ------------
    - Cache *only* channels the user selects (compute on demand).
    - Fill cache per-tile, matching the prediction tile_size, to avoid loading the full image.
    - Persist in the project folder to reuse across sessions/runs.
    - Keep cache keyed by (image fingerprint, feature config hash, tile size).

    Each cached channel is stored as:
      - chXX_features.zarr : (H, W, Fch) float16
      - chXX_mask.zarr     : (n_ty, n_tx) uint8, 1 if that tile has been computed

    Notes
    -----
    This cache reproduces the current behavior of *tile-local* feature computation
    (features computed on the tile image only). It does not require full-image features.
    """

    def __init__(
        self,
        cache_root: Path,
        *,
        image_fingerprint: str,
        image_shape_yxc: tuple[int, int, int],
        tile_size: int,
        cfg: FeatureCacheConfig | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.image_fingerprint = str(image_fingerprint)
        self.H, self.W, self.C = (int(image_shape_yxc[0]), int(image_shape_yxc[1]), int(image_shape_yxc[2]))
        self.tile_size = int(tile_size)
        self.cfg = cfg or FeatureCacheConfig()

        # Thread-safety: tile features may be requested from multiple worker threads.
        # - Channel stores are opened/created under a per-channel lock to avoid
        #   concurrent metadata writes on Windows (can raise WinError 5).
        # - Tile writes use a separate per-channel lock to avoid two writers
        #   filling the same tile simultaneously.
        self._global_lock = threading.Lock()
        self._chan_locks: dict[int, threading.Lock] = {}
        self._chan_arrays: dict[int, tuple[object, object]] = {}
        self._tile_write_locks: dict[int, threading.Lock] = {}

        # Lightweight instrumentation to verify cache hit/miss behavior.
        self._stats_lock = threading.Lock()
        self._stats = {
            "tile_requests": 0,
            "tile_full_hit": 0,
            "tile_any_miss": 0,
            "channels_missing": 0,
            "t_compute_s": 0.0,
            "t_write_s": 0.0,
            "t_read_s": 0.0,
            "bytes_read": 0,
            "bytes_written": 0,
        }

        # Derived
        self.Fch = int(
            features_per_channel(
                sigmas=self.cfg.sigmas,
                use_intensity=self.cfg.use_intensity,
                use_gaussian=self.cfg.use_gaussian,
                use_gradmag=self.cfg.use_gradmag,
                use_log=self.cfg.use_log,
                use_dog=self.cfg.use_dog,
                use_structure_tensor=self.cfg.use_structure_tensor,
                use_hessian=self.cfg.use_hessian,
            )
        )
        self.n_ty = int((self.H + self.tile_size - 1) // self.tile_size)
        self.n_tx = int((self.W + self.tile_size - 1) // self.tile_size)

        # Write config on init (helpful for debugging)
        try:
            (self.cache_root / "config.json").write_text(
                json.dumps(
                    {
                        "image_fingerprint": self.image_fingerprint,
                        "image_shape_yxc": [self.H, self.W, self.C],
                        "tile_size": self.tile_size,
                        "features_per_channel": self.Fch,
                        "cfg": self.cfg.to_dict(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


    def _norm_stats_path(self) -> Path:
        return self.cache_root / "norm_stats.json"

    def _load_norm_stats(self) -> dict[str, dict[str, float]]:
        """Load per-channel normalization stats (lo/hi) from disk."""
        try:
            p = self._norm_stats_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # stored as {"<channel>": {"lo": float, "hi": float}}
                    out: dict[str, dict[str, float]] = {}
                    for k, v in data.items():
                        if isinstance(v, dict) and "lo" in v and "hi" in v:
                            out[str(k)] = {"lo": float(v["lo"]), "hi": float(v["hi"])}
                    return out
        except Exception:
            pass
        return {}

    def _save_norm_stats(self, stats: dict[str, dict[str, float]]) -> None:
        try:
            self._norm_stats_path().write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    def _compute_channel_quantiles(self, img, c: int) -> tuple[float, float]:
        """Compute robust (lo, hi) for one channel using a deterministic subsample."""
        p_lo = float(getattr(self.cfg, "norm_p_lo", 0.5))
        p_hi = float(getattr(self.cfg, "norm_p_hi", 99.5))

        # img is (H,W,C) numpy or dask-backed.
        H = int(self.H); W = int(self.W)
        target = 200_000  # ~200k samples per channel is plenty for robust percentiles
        step = int(max(1, math.sqrt((H * W) / float(target))))
        sy = step
        sx = step

        try:
            chan = img[::sy, ::sx, int(c)]
            if hasattr(chan, "compute"):
                chan = chan.compute()
            arr = np.asarray(chan, dtype=np.float32).ravel()
        except Exception:
            # Fallback: use a coarser stride
            sy = max(1, int(step * 2))
            sx = max(1, int(step * 2))
            chan = img[::sy, ::sx, int(c)]
            if hasattr(chan, "compute"):
                chan = chan.compute()
            arr = np.asarray(chan, dtype=np.float32).ravel()

        if arr.size == 0:
            return (0.0, 1.0)

        # Remove NaNs/Infs just in case
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (0.0, 1.0)

        lo = float(np.percentile(arr, p_lo))
        hi = float(np.percentile(arr, p_hi))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = lo + 1.0
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    def _ensure_norm_stats(self, channels: list[int], img) -> dict[int, tuple[float, float]]:
        """Ensure (lo,hi) exists for each channel, computing + persisting if missing."""
        mode = str(getattr(self.cfg, "norm_mode", "none") or "none").lower()
        if mode == "none":
            return {int(c): (0.0, 1.0) for c in channels}

        # Only one supported robust mode for now.
        with self._global_lock:
            stats = self._load_norm_stats()
            changed = False
            out: dict[int, tuple[float, float]] = {}
            for c in channels:
                key = str(int(c))
                if key not in stats:
                    lo, hi = self._compute_channel_quantiles(img, int(c))
                    stats[key] = {"lo": float(lo), "hi": float(hi)}
                    changed = True
                out[int(c)] = (float(stats[key]["lo"]), float(stats[key]["hi"]))
            if changed:
                self._save_norm_stats(stats)
            return out

    def _normalize_tile(self, tile_img_yxc: np.ndarray, use_channels: list[int], img_for_stats) -> np.ndarray:
        """Normalize selected channels of a tile into [0,1] float32."""
        mode = str(getattr(self.cfg, "norm_mode", "none") or "none").lower()
        if mode == "none":
            return np.asarray(tile_img_yxc, dtype=np.float32)

        lohi = self._ensure_norm_stats(use_channels, img_for_stats)
        eps = float(getattr(self.cfg, "norm_eps", 1e-6))

        x = np.asarray(tile_img_yxc, dtype=np.float32, copy=True)
        for c in use_channels:
            lo, hi = lohi[int(c)]
            denom = float(hi - lo) + eps
            xc = (x[..., int(c)] - float(lo)) / denom
            # Clip to [0, 1] so all downstream features remain bounded
            x[..., int(c)] = np.clip(xc, 0.0, 1.0)
        return x


    @staticmethod
    def available() -> bool:
        return zarr is not None and Blosc is not None

    @staticmethod
    def compute_image_fingerprint(path: str | None, image_shape_yxc: tuple[int, int, int]) -> str:
        """Compute a stable-ish fingerprint for the current image.

        Uses (abs path, size, mtime) when available plus image shape, to
        detect updates if the file changes.
        """
        p = None
        try:
            if path:
                p = Path(path)
        except Exception:
            p = None

        parts = {
            "path": str(p.resolve()) if p and p.exists() else str(path or ""),
            "shape": [int(image_shape_yxc[0]), int(image_shape_yxc[1]), int(image_shape_yxc[2])],
            "size": 0,
            "mtime": 0,
        }
        try:
            if p and p.exists():
                st = p.stat()
                parts["size"] = int(st.st_size)
                parts["mtime"] = int(st.st_mtime)
        except Exception:
            pass
        return _sha1(json.dumps(parts, sort_keys=True))

    def _channel_dir(self, c: int) -> Path:
        return self.cache_root / f"ch{int(c):03d}"

    def _get_channel_lock(self, c: int) -> threading.Lock:
        c = int(c)
        with self._global_lock:
            lk = self._chan_locks.get(c)
            if lk is None:
                lk = threading.Lock()
                self._chan_locks[c] = lk
            return lk

    def _get_tile_write_lock(self, c: int) -> threading.Lock:
        c = int(c)
        with self._global_lock:
            lk = self._tile_write_locks.get(c)
            if lk is None:
                lk = threading.Lock()
                self._tile_write_locks[c] = lk
            return lk

    def prepare_channels(self, channels: Iterable[int]) -> None:
        """Ensure on-disk stores exist for the given channels (serial, thread-safe)."""
        for c in channels:
            self._open_channel_arrays(int(c))

    def stats_snapshot(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def stats_summary(self) -> str:
        s = self.stats_snapshot()
        n = max(1, int(s.get("tile_requests", 0)))
        hit = int(s.get("tile_full_hit", 0))
        miss = int(s.get("tile_any_miss", 0))
        cm = int(s.get("channels_missing", 0))
        t_c = float(s.get("t_compute_s", 0.0))
        t_w = float(s.get("t_write_s", 0.0))
        t_r = float(s.get("t_read_s", 0.0))
        br = int(s.get("bytes_read", 0))
        bw = int(s.get("bytes_written", 0))
        return (
            "[FeatureCache] "
            f"tiles={n} full_hit={hit} any_miss={miss} missing_ch_total={cm} | "
            f"t_compute={t_c:.2f}s t_write={t_w:.2f}s t_read={t_r:.2f}s | "
            f"read={br/1e6:.1f}MB written={bw/1e6:.1f}MB"
        )

    def _reset_channel_cache(
        self,
        zf,
        zm,
        *,
        c: int,
        compressor,
        reason: str = "",
    ) -> None:
        """Rebuild both feature and mask datasets for one channel.

        If the feature array is recreated, the tile-validity mask must also be
        recreated. Otherwise tiles may remain marked as cached while their feature
        values are blank/newly initialized.
        """
        _dbg(
            "reset_channel_cache "
            f"channel={int(c)} reason={reason or 'unspecified'} "
            f"shape=({self.H},{self.W},{self.Fch}) tiles=({self.n_ty},{self.n_tx})"
        )

        try:
            if "data" in zf:
                del zf["data"]
        except Exception as e:
            raise RuntimeError(f"Unable to reset feature cache for channel {c}: {e}")

        zf.create_dataset(
            "data",
            shape=(self.H, self.W, self.Fch),
            chunks=(min(self.tile_size, self.H), min(self.tile_size, self.W), self.Fch),
            dtype="float16",
            compressor=compressor,
            overwrite=False,
        )
        zf.attrs.update(
            {
                "channel_index": int(c),
                "features_per_channel": int(self.Fch),
                "tile_size": int(self.tile_size),
                "feature_config_hash": str(self.cfg.hash()),
                "image_fingerprint": str(self.image_fingerprint),
            }
        )

        try:
            if "mask" in zm:
                del zm["mask"]
        except Exception as e:
            raise RuntimeError(f"Unable to reset mask cache for channel {c}: {e}")

        zm.create_dataset(
            "mask",
            shape=(self.n_ty, self.n_tx),
            chunks=(min(64, self.n_ty), min(64, self.n_tx)),
            dtype="uint8",
            compressor=compressor,
            overwrite=False,
        )
        zm.attrs.update(
            {
                "channel_index": int(c),
                "tile_size": int(self.tile_size),
                "feature_config_hash": str(self.cfg.hash()),
                "image_fingerprint": str(self.image_fingerprint),
            }
        )

        _dbg(
            "reset_channel_cache_done "
            f"channel={int(c)} dtype=float16 "
            f"feat_shape={(self.H, self.W, self.Fch)} "
            f"mask_shape={(self.n_ty, self.n_tx)}"
        )

    def _open_channel_arrays(self, c: int):
        """Open (features_array, mask_array) for channel c, creating if needed.

        This may be called from multiple threads (tile feature workers). We guard
        store initialization with a per-channel lock to avoid concurrent metadata
        writes (notably on Windows), which can raise PermissionError/WinError 5.
        """
        if zarr is None:
            raise RuntimeError("zarr is not available")

        c = int(c)

        # Fast path: already opened in this process.
        existing = self._chan_arrays.get(c)
        if existing is not None:
            return existing

        lk = self._get_channel_lock(c)
        with lk:
            existing = self._chan_arrays.get(c)
            if existing is not None:
                return existing

            ch_dir = self._channel_dir(c)
            ch_dir.mkdir(parents=True, exist_ok=True)

            # Use fast compression.
            compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)

            feat_path = ch_dir / "features.zarr"
            mask_path = ch_dir / "mask.zarr"

            # Keep both stores in Zarr v2 for compatibility with numcodecs codecs.
            zf = zarr.open_group(str(feat_path), mode="a", zarr_format=2)
            zm = zarr.open_group(str(mask_path), mode="a", zarr_format=2)

            created_feat = False
            created_mask = False
            if "data" not in zf:
                zf.create_dataset(
                    "data",
                    shape=(self.H, self.W, self.Fch),
                    chunks=(min(self.tile_size, self.H), min(self.tile_size, self.W), self.Fch),
                    dtype="float16",
                    compressor=compressor,
                    overwrite=False,
                )
                # Only set attrs at creation time to avoid repeated metadata rewrites.
                zf.attrs.update(
                    {
                        "channel_index": int(c),
                        "features_per_channel": int(self.Fch),
                        "tile_size": int(self.tile_size),
                        "feature_config_hash": str(self.cfg.hash()),
                        "image_fingerprint": str(self.image_fingerprint),
                    }
                )
                created_feat = True
            if "mask" not in zm:
                zm.create_dataset(
                    "mask",
                    shape=(self.n_ty, self.n_tx),
                    chunks=(min(64, self.n_ty), min(64, self.n_tx)),
                    dtype="uint8",
                    compressor=compressor,
                    overwrite=False,
                )
                zm.attrs.update(
                    {
                        "channel_index": int(c),
                        "tile_size": int(self.tile_size),
                        "feature_config_hash": str(self.cfg.hash()),
                        "image_fingerprint": str(self.image_fingerprint),
                    }
                )

                created_mask = True

            # Validate existing cache contents.
            #
            # Current design intentionally stores normalized features as float16.
            # If an older/incompatible cache is found, rebuild BOTH the feature
            # data and the tile mask so they cannot get out of sync.
            try:
                arr0 = zf["data"]
                mask0 = zm["mask"]

                arr_dtype = str(arr0.dtype)
                arr_shape = tuple(int(x) for x in arr0.shape)
                mask_shape = tuple(int(x) for x in mask0.shape)

                zf_fch = _safe_int(zf.attrs.get("features_per_channel", -1))
                zf_tile = _safe_int(zf.attrs.get("tile_size", -1))
                zm_tile = _safe_int(zm.attrs.get("tile_size", -1))
                zf_hash = str(zf.attrs.get("feature_config_hash", ""))
                zm_hash = str(zm.attrs.get("feature_config_hash", ""))
                zf_fp = str(zf.attrs.get("image_fingerprint", ""))
                zm_fp = str(zm.attrs.get("image_fingerprint", ""))

                reset_reasons: list[str] = []
                if arr_dtype != "float16":
                    reset_reasons.append(f"dtype={arr_dtype}")
                if arr_shape != (self.H, self.W, self.Fch):
                    reset_reasons.append(f"arr_shape={arr_shape}")
                if mask_shape != (self.n_ty, self.n_tx):
                    reset_reasons.append(f"mask_shape={mask_shape}")
                if zf_fch != int(self.Fch):
                    reset_reasons.append(f"features_per_channel={zf_fch}")
                if zf_tile != int(self.tile_size):
                    reset_reasons.append(f"zf_tile_size={zf_tile}")
                if zm_tile != int(self.tile_size):
                    reset_reasons.append(f"zm_tile_size={zm_tile}")
                if zf_hash != str(self.cfg.hash()):
                    reset_reasons.append("zf_feature_config_hash_mismatch")
                if zm_hash != str(self.cfg.hash()):
                    reset_reasons.append("zm_feature_config_hash_mismatch")
                if zf_fp != str(self.image_fingerprint):
                    reset_reasons.append("zf_image_fingerprint_mismatch")
                if zm_fp != str(self.image_fingerprint):
                    reset_reasons.append("zm_image_fingerprint_mismatch")

                if _STEP2A_CACHE_DEBUG and (created_feat or created_mask):
                    _dbg(
                        "open_channel_created "
                        f"channel={int(c)} created_feat={created_feat} created_mask={created_mask} "
                        f"dtype={arr_dtype} arr_shape={arr_shape} mask_shape={mask_shape}"
                    )

                if reset_reasons:
                    self._reset_channel_cache(
                        zf,
                        zm,
                        c=int(c),
                        compressor=compressor,
                        reason=";".join(reset_reasons),
                    )
            except Exception as e:
                _dbg(f"open_channel_validation_exception channel={int(c)} exc={e!r}")
                # If validation fails for any reason, rebuild both stores to restore
                # consistency rather than risk using a partially invalid cache.
                self._reset_channel_cache(
                    zf,
                    zm,
                    c=int(c),
                    compressor=compressor,
                    reason=f"validation_exception:{type(e).__name__}",
                )

            arr = zf["data"]
            m = zm["mask"]
            self._chan_arrays[c] = (arr, m)
            return arr, m

    def get_point_features(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        use_channels: list[int],
        img,
    ) -> np.ndarray:
        """Get features for a set of pixel coordinates using the tile cache.

        Parameters
        ----------
        ys, xs:
            1D arrays of global pixel coordinates (same length).
        img:
            Full image array (numpy or dask-backed). Only required tiles will be loaded.

        Returns
        -------
        X:
            (N, F_total) float32 feature matrix.
        """
        ys = np.asarray(ys, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        if ys.shape != xs.shape:
            raise ValueError("ys and xs must have the same shape")
        if ys.ndim != 1:
            ys = ys.reshape(-1)
            xs = xs.reshape(-1)

        use_channels = [int(c) for c in (use_channels or [])]
        if not use_channels:
            raise ValueError("No channels selected.")

        n = int(ys.size)
        F_total = int(self.Fch) * int(len(use_channels))
        X_out = np.empty((n, F_total), dtype=np.float32)

        # Group points by tile index to minimize tile reads/computes.
        ty = (ys // int(self.tile_size)).astype(np.int64, copy=False)
        tx = (xs // int(self.tile_size)).astype(np.int64, copy=False)
        key = ty * int(self.n_tx) + tx

        uniq, inv = np.unique(key, return_inverse=True)

        for u_i, k in enumerate(uniq):
            idxs = np.where(inv == u_i)[0]
            if idxs.size == 0:
                continue
            t_y = int(k // int(self.n_tx))
            t_x = int(k % int(self.n_tx))

            y0 = int(t_y * int(self.tile_size))
            x0 = int(t_x * int(self.tile_size))
            y1 = min(y0 + int(self.tile_size), int(self.H))
            x1 = min(x0 + int(self.tile_size), int(self.W))

            tile_img = np.asarray(img[y0:y1, x0:x1, :])
            X_tile = self.get_tile_features(y0, y1, x0, x1, use_channels, tile_img, img=img)

            ly = (ys[idxs] - y0).astype(np.int64, copy=False)
            lx = (xs[idxs] - x0).astype(np.int64, copy=False)

            X_out[idxs, :] = X_tile[ly, lx, :]

        return X_out


    def get_tile_features(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        use_channels: list[int],
        tile_img_yxc: np.ndarray,
        img=None,
    ) -> np.ndarray:
        """Return features for the requested tile, computing/storing if needed.

        Parameters
        ----------
        tile_img_yxc:
            The (dy,dx,C) tile slice from the image.
        """
        if not self.available():
            # should never be called when unavailable, but be defensive
            tile_for_feat = self._normalize_tile(tile_img_yxc, use_channels, img_for_stats=(img if img is not None else tile_img_yxc))
            return build_features(tile_for_feat, use_channels)

        y0 = int(y0); y1 = int(y1); x0 = int(x0); x1 = int(x1)
        if y1 <= y0 or x1 <= x0:
            raise ValueError("Empty tile requested.")
        use_channels = [int(c) for c in use_channels]
        if not use_channels:
            raise ValueError("No channels selected.")

        ty = int(y0 // self.tile_size)
        tx = int(x0 // self.tile_size)

        # Determine which channels are missing for this tile
        missing: list[int] = []
        arrays = {}
        masks = {}
        for c in use_channels:
            arr, m = self._open_channel_arrays(c)
            arrays[c] = arr
            masks[c] = m
            try:
                if int(m[ty, tx]) == 0:
                    missing.append(c)
            except Exception:
                missing.append(c)

        if _STEP2A_CACHE_DEBUG and missing:
            _dbg(
                "tile_cache_miss "
                f"tile=({ty},{tx}) yx=({y0}:{y1},{x0}:{x1}) "
                f"use_channels={list(use_channels)} "
                f"missing={list(missing)}"
            )

        dy = int(y1 - y0)
        dx = int(x1 - x0)
        bytes_per_channel = dy * dx * int(self.Fch) * 2  # float16 stored (features bounded by normalization)

        with self._stats_lock:
            self._stats["tile_requests"] += 1
            self._stats["channels_missing"] += int(len(missing))
            if len(missing) == 0:
                self._stats["tile_full_hit"] += 1
            else:
                self._stats["tile_any_miss"] += 1

        # Compute missing channels in one shot (build_features over all use_channels)
        if missing:
            # We compute for all *use_channels* (cheap vs multiple passes) and write only missing.
            t0 = _time.perf_counter()
            tile_for_feat = self._normalize_tile(tile_img_yxc, use_channels, img_for_stats=(img if img is not None else tile_img_yxc))
            X_all = build_features(
                tile_for_feat,
                use_channels,
                sigmas=self.cfg.sigmas,
                use_intensity=self.cfg.use_intensity,
                use_gaussian=self.cfg.use_gaussian,
                use_gradmag=self.cfg.use_gradmag,
                use_log=self.cfg.use_log,
                use_dog=self.cfg.use_dog,
                use_structure_tensor=self.cfg.use_structure_tensor,
                use_hessian=self.cfg.use_hessian,
                dog_sigma_ratio=self.cfg.dog_sigma_ratio,
            )
            if _STEP2A_CACHE_DEBUG:
                suspicious_all, suspicious_reason_all = _is_suspicious_array(X_all)
                if suspicious_all:
                    _dbg(
                        "tile_compute_suspicious "
                        f"tile=({ty},{tx}) reason={suspicious_reason_all} "
                        f"X_all={_array_debug_summary(X_all)} "
                        f"expected_lastdim={int(self.Fch) * int(len(use_channels))}"
                    )

            dt_c = _time.perf_counter() - t0
            with self._stats_lock:
                self._stats["t_compute_s"] += float(dt_c)
            if X_all.shape[-1] != self.Fch * len(use_channels):
                # Should not happen unless feature code changes
                raise RuntimeError("Unexpected feature dimension; cannot split by channel.")

            for j, c in enumerate(use_channels):
                if c not in missing:
                    continue
                lock = self._get_tile_write_lock(c)
                t1 = _time.perf_counter()
                with lock:
                    # re-check mask in case another thread filled it
                    try:
                        if int(masks[c][ty, tx]) != 0:
                            continue
                    except Exception:
                        pass
                    Xc = X_all[..., j * self.Fch : (j + 1) * self.Fch].astype(np.float16, copy=False)
                    arrays[c][y0:y1, x0:x1, :] = Xc
                    try:
                        masks[c][ty, tx] = 1
                    except Exception:
                        pass
                    if _STEP2A_CACHE_DEBUG:
                        suspicious_c, suspicious_reason_c = _is_suspicious_array(Xc)
                        if suspicious_c:
                            _dbg(
                                "tile_write_suspicious "
                                f"tile=({ty},{tx}) channel={int(c)} reason={suspicious_reason_c} "
                                f"Xc={_array_debug_summary(Xc)}"
                            )
                dt_w = _time.perf_counter() - t1
                with self._stats_lock:
                    self._stats["t_write_s"] += float(dt_w)
                    self._stats["bytes_written"] += int(bytes_per_channel)

        t2 = _time.perf_counter()

        # Read all requested channels from cache and concatenate
        feats = []
        for c in use_channels:
            fc = np.asarray(arrays[c][y0:y1, x0:x1, :], dtype=np.float32)
            feats.append(fc)
            with self._stats_lock:
                self._stats["bytes_read"] += int(bytes_per_channel)
        dt_r = _time.perf_counter() - t2
        with self._stats_lock:
            self._stats["t_read_s"] += float(dt_r)
        out = np.concatenate(feats, axis=-1)
        if _STEP2A_CACHE_DEBUG:
            suspicious_out, suspicious_reason_out = _is_suspicious_array(out)
            if suspicious_out:
                ch_summaries = []
                for j, c in enumerate(use_channels):
                    try:
                        ch_summaries.append(
                            f"c{int(c)}:{_array_debug_summary(feats[j])}"
                        )
                    except Exception:
                        ch_summaries.append(f"c{int(c)}:summary_failed")
                _dbg(
                    "tile_read_suspicious "
                    f"tile=({ty},{tx}) reason={suspicious_reason_out} "
                    f"concat={_array_debug_summary(out)} "
                    f"per_channel=[{'; '.join(ch_summaries)}]"
                )
        return out
