from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from skimage.registration import phase_cross_correlation

try:
    # SciPy is faster than skimage.warp for pure translations.
    from scipy.ndimage import shift as ndi_shift  # type: ignore
except Exception:  # pragma: no cover
    ndi_shift = None

import tifffile

from cycif_seg.io.ome_tiff import load_multichannel_tiff, load_multichannel_tiff_native, save_ome_tiff_yxc


@dataclass(frozen=True)
class CycleInput:
    """One imaging cycle file + metadata used for merging."""

    path: str
    cycle: int
    tissue: str | None = None
    species: str | None = None
    # Optional: the nuclear marker used to register this cycle.
    # If None, we'll attempt to use a marker named "DAPI" (case-insensitive).
    registration_marker: str | None = None

    # Optional: user-edited per-channel marker + antibody annotations.
    # If provided, each list should have length == n_channels in the file.
    channel_markers: list[str] | None = None
    channel_antibodies: list[str] | None = None


def _find_channel_index(ch_names: list[str], marker: str) -> int | None:
    m = (marker or "").strip().lower()
    if not m:
        return None
    for i, nm in enumerate(ch_names):
        if (nm or "").strip().lower() == m:
            return i
    # weak match: contains
    for i, nm in enumerate(ch_names):
        if m in (nm or "").strip().lower():
            return i
    return None


def _pad_to_shape(img_yx: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Zero-pad (or center-crop) a 2D image to shape_yx."""
    if img_yx.ndim != 2:
        raise ValueError(f"Expected 2D array. Got shape={img_yx.shape}")
    y, x = int(img_yx.shape[0]), int(img_yx.shape[1])
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    out = np.zeros((Y, X), dtype=np.float32)

    # center placement
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs] = img_yx[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs].astype(
        np.float32, copy=False
    )
    return out


def _pad_channels_to_canvas(img_yxc: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Zero-pad (or center-crop) a (Y,X,C) image to a common canvas (Yc,Xc,C)."""
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")
    y, x, c = img_yxc.shape
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    out = np.zeros((Y, X, c), dtype=np.float32)
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs, :] = img_yxc[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs, :].astype(
        np.float32, copy=False
    )
    return out


def _pad_plane_to_canvas(plane_yx: np.ndarray, canvas_yx: tuple[int, int], *, out_dtype=np.float32) -> np.ndarray:
    """Pad/crop a single (Y,X) plane to a common canvas."""
    if plane_yx.ndim != 2:
        raise ValueError(f"Expected (Y,X). Got shape={plane_yx.shape}")
    y, x = int(plane_yx.shape[0]), int(plane_yx.shape[1])
    Y, X = int(canvas_yx[0]), int(canvas_yx[1])
    out = np.zeros((Y, X), dtype=out_dtype)
    y0 = max((Y - y) // 2, 0)
    x0 = max((X - x) // 2, 0)
    ys = min(y, Y)
    xs = min(x, X)
    in_y0 = max((y - Y) // 2, 0)
    in_x0 = max((x - X) // 2, 0)
    out[y0 : y0 + ys, x0 : x0 + xs] = plane_yx[in_y0 : in_y0 + ys, in_x0 : in_x0 + xs].astype(
        out_dtype, copy=False
    )
    return out


def _tiff_info_yxc(path: str) -> tuple[tuple[int, int, int], np.dtype]:
    """Return ((Y,X,C), dtype) without reading full pixel data."""
    with tifffile.TiffFile(path) as tf:
        series0 = tf.series[0]
        shp = tuple(int(x) for x in series0.shape)
        dtype = np.dtype(series0.dtype)

    if len(shp) != 3:
        raise ValueError(f"Expected 3D TIFF series. Got shape={shp} for {path}")

    # Same heuristic as _normalize_to_yxc
    if shp[0] <= 32 and shp[1] > 64 and shp[2] > 64:
        # (C,Y,X)
        y, x, c = shp[1], shp[2], shp[0]
    else:
        # assume (Y,X,C)
        y, x, c = shp[0], shp[1], shp[2]

    return (int(y), int(x), int(c)), dtype


def _cast_preserve_dtype(plane_f32: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast float32 plane back to dtype while preserving range for integer types."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return plane_f32.astype(dt, copy=False)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return np.clip(plane_f32, info.min, info.max).astype(dt, copy=False)
    return plane_f32.astype(dt, copy=False)


def estimate_translation(
    ref_yx: np.ndarray,
    mov_yx: np.ndarray,
    *,
    downsample: int = 4,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """Return (dy, dx) translation to apply to mov to align to ref."""
    if ref_yx.ndim != 2 or mov_yx.ndim != 2:
        raise ValueError("estimate_translation expects 2D arrays")

    # phase_cross_correlation requires same shape. Pad/crop to a common canvas.
    Y = int(max(ref_yx.shape[0], mov_yx.shape[0]))
    X = int(max(ref_yx.shape[1], mov_yx.shape[1]))
    ref2 = _pad_to_shape(ref_yx, (Y, X))
    mov2 = _pad_to_shape(mov_yx, (Y, X))

    if downsample and downsample > 1:
        ref_ds = ref2[::downsample, ::downsample]
        mov_ds = mov2[::downsample, ::downsample]
    else:
        ref_ds, mov_ds = ref2, mov2

    # phase_cross_correlation returns shift to apply to mov to match ref
    shift_ds, _, _ = phase_cross_correlation(
        ref_ds,
        mov_ds,
        upsample_factor=int(upsample_factor),
        normalization=None,
    )
    dy = float(shift_ds[0]) * float(downsample)
    dx = float(shift_ds[1]) * float(downsample)
    return dy, dx


def apply_translation_yxc(
    img_yxc: np.ndarray,
    dy: float,
    dx: float,
    *,
    order: int = 1,
    preserve_range: bool = True,
) -> np.ndarray:
    """Translate (Y,X,C) image by (dy,dx)."""
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C). Got shape={img_yxc.shape}")
    out = np.empty_like(img_yxc, dtype=np.float32)

    if ndi_shift is not None:
        # SciPy: shift takes (dy, dx)
        for c in range(img_yxc.shape[2]):
            out[..., c] = ndi_shift(
                img_yxc[..., c],
                shift=(float(dy), float(dx)),
                order=int(order),
                mode="constant",
                cval=0.0,
                prefilter=(int(order) > 1),
            ).astype(np.float32, copy=False)
        return out

    # Fallback: skimage
    from skimage.transform import AffineTransform, warp

    tform = AffineTransform(translation=(dx, dy))
    for c in range(img_yxc.shape[2]):
        warped = warp(
            img_yxc[..., c],
            inverse_map=tform.inverse,
            order=int(order),
            mode="constant",
            cval=0.0,
            preserve_range=bool(preserve_range),
        )
        out[..., c] = warped.astype(np.float32, copy=False)
    return out


def merge_cycles_to_ome_tiff(
    cycles: Iterable[CycleInput],
    output_path: str,
    *,
    reference_cycle: int | None = None,
    default_registration_marker: str = "DAPI",
    downsample_for_registration: int = 4,
    upsample_factor: int = 10,
    low_mem: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    progress_event_cb: Optional[Callable[[dict], None]] = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    """
    Load multiple cycle OME-TIFFs, optionally rigid-translate register each cycle to a reference,
    and save one merged OME-TIFF with channels renamed as "cy{cycle}:{channel}".

    Current implementation is intentionally minimal (translation-only) as an MVP for Step (1).
    We can extend to full affine/nonlinear registration and richer metadata later.

    Returns a small report dict (useful for logging/UI).
    """
    cycles = list(cycles)
    if not cycles:
        raise ValueError("No cycles provided")

    def _check_cancel() -> None:
        """Raise RuntimeError('Cancelled') if cancel_cb signals cancellation."""
        try:
            if cancel_cb is not None and bool(cancel_cb()):
                raise RuntimeError("Cancelled")
        except RuntimeError:
            raise
        except Exception:
            # Do not allow a buggy cancel callback to crash preprocessing.
            return

    # Validate cycle numbering: must be non-negative and unique.
    cy_vals = [int(ci.cycle) for ci in cycles]
    if any(cy < 0 for cy in cy_vals):
        raise ValueError(f"Cycle numbers must be >= 0. Got: {cy_vals}")
    if len(set(cy_vals)) != len(cy_vals):
        raise ValueError(f"Cycle numbers must be unique. Got: {cy_vals}")

    # Sort by cycle index for stable output ordering.
    cycles = sorted(cycles, key=lambda x: int(x.cycle))

    if reference_cycle is None:
        reference_cycle = int(cycles[0].cycle)

    if bool(low_mem):
        # Low-memory merge: keep only the current cycle + reference reg channel + final merged output in RAM.
        # This avoids holding all cycles in memory at once.
        _check_cancel()

        # First pass: gather shapes/dtypes without loading all pixels.
        infos: list[tuple[CycleInput, tuple[int, int, int], np.dtype]] = []
        canvas_y = 0
        canvas_x = 0
        total_ch = 0
        base_dtype: np.dtype | None = None
        for ci in cycles:
            shp_yxc, dt = _tiff_info_yxc(ci.path)
            if base_dtype is None:
                base_dtype = dt
            elif np.dtype(dt) != np.dtype(base_dtype):
                raise ValueError(
                    f"All cycles must have the same dtype for low_mem merge. Got {base_dtype} vs {dt} ({ci.path})"
                )
            canvas_y = max(canvas_y, int(shp_yxc[0]))
            canvas_x = max(canvas_x, int(shp_yxc[1]))
            total_ch += int(shp_yxc[2])
            infos.append((ci, shp_yxc, dt))

        # For progress callbacks in low-memory mode.
        n_cycles = int(len(infos))

        assert base_dtype is not None
        canvas_yx = (int(canvas_y), int(canvas_x))

        # Identify reference cycle
        ref_ci: CycleInput | None = None
        for ci, _, _ in infos:
            if int(ci.cycle) == int(reference_cycle):
                ref_ci = ci
                break
        if ref_ci is None:
            raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

        # Allocate final merged output once.
        merged = np.zeros((int(canvas_yx[0]), int(canvas_yx[1]), int(total_ch)), dtype=base_dtype)

        # Build merged channel names as we go.
        merged_names: list[str] = []
        seen: dict[str, int] = {}

        # Load reference registration channel once (float32), but do not keep whole reference image.
        ref_marker = ref_ci.registration_marker or default_registration_marker
        if progress_cb:
            progress_cb(f"Loading reference cycle {int(ref_ci.cycle)}")
        if progress_event_cb:
            progress_event_cb(
                {
                    "phase": "load",
                    "cycle": int(ref_ci.cycle),
                    "idx": 1,
                    "n": int(n_cycles),
                    "msg": f"Loading reference cycle {int(ref_ci.cycle)}",
                }
            )
        ref_img, ref_names = load_multichannel_tiff_native(ref_ci.path)
        ref_ch_idx = _find_channel_index(ref_names, ref_marker)
        ref_yx: np.ndarray | None
        if ref_ch_idx is None:
            ref_yx = None
        else:
            ref_yx = _pad_plane_to_canvas(ref_img[..., ref_ch_idx], canvas_yx, out_dtype=np.float32)

        shifts: dict[int, tuple[float, float]] = {}

        # Helper to name output channels
        def _append_channel_name(ci: CycleInput, orig_name: str, j: int) -> None:
            cy = int(ci.cycle)
            markers = ci.channel_markers or []
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (orig_name or "").strip() or "Channel"
            out_nm = f"{base}_cy{cy}"
            k = int(seen.get(out_nm, 0))
            out_nm2 = f"{out_nm}_{k+1}" if k else out_nm
            seen[out_nm] = k + 1
            merged_names.append(out_nm2)

        # Write reference cycle first (unregistered)
        out_c = 0
        for ci in cycles:
            _check_cancel()

            if int(ci.cycle) == int(ref_ci.cycle):
                # Use already-loaded ref_img/ref_names
                img = ref_img
                ch_names = ref_names
                dy = 0.0
                dx = 0.0
            else:
                if progress_cb:
                    progress_cb(f"Registering cycle {int(ci.cycle)}")
                if progress_event_cb:
                    # idx as position in sorted cycles
                    idx = int([c.cycle for c in cycles].index(ci.cycle) + 1)
                    progress_event_cb(
                        {
                            "phase": "load",
                            "cycle": int(ci.cycle),
                            "idx": int(idx),
                            "n": int(n_cycles),
                            "msg": f"Registering cycle {int(ci.cycle)}",
                        }
                    )
                img, ch_names = load_multichannel_tiff_native(ci.path)

                # Determine shift for this cycle
                if ref_yx is None:
                    dy, dx = (0.0, 0.0)
                else:
                    marker = ci.registration_marker or default_registration_marker
                    ch_idx = _find_channel_index(ch_names, marker)
                    if ch_idx is None:
                        dy, dx = (0.0, 0.0)
                    else:
                        mov_yx = _pad_plane_to_canvas(img[..., ch_idx], canvas_yx, out_dtype=np.float32)
                        dy, dx = estimate_translation(
                            ref_yx,
                            mov_yx,
                            downsample=int(downsample_for_registration),
                            upsample_factor=int(upsample_factor),
                        )

            shifts[int(ci.cycle)] = (float(dy), float(dx))

            # Register/write each channel plane directly into the final output.
            for j in range(int(img.shape[2])):
                plane = _pad_plane_to_canvas(img[..., j], canvas_yx, out_dtype=np.float32)
                if dy != 0.0 or dx != 0.0:
                    # Use the existing translation implementation on a 1-channel stack to reuse code.
                    plane_yxc = plane[..., None]
                    plane_reg = apply_translation_yxc(plane_yxc, dy=dy, dx=dx, order=1)[..., 0]
                else:
                    plane_reg = plane
                merged[..., int(out_c)] = _cast_preserve_dtype(plane_reg, base_dtype)
                _append_channel_name(ci, ch_names[j], j)
                out_c += 1

            # Release non-reference cycle arrays ASAP.
            if int(ci.cycle) != int(ref_ci.cycle):
                try:
                    del img
                except Exception:
                    pass

        if progress_cb:
            progress_cb("Writing OME-TIFF")
        if progress_event_cb:
            progress_event_cb(
                {
                    "phase": "write",
                    "idx": int(n_cycles),
                    "n": int(n_cycles),
                    "msg": f"Writing output: {output_path}",
                }
            )

        save_ome_tiff_yxc(output_path, merged, merged_names)

        return {
            "output_path": output_path,
            "reference_cycle": int(reference_cycle),
            "default_registration_marker": default_registration_marker,
            "shifts_yx": shifts,
            "canvas_yx": canvas_yx,
            "inputs": [{"cycle": int(ci.cycle), "path": ci.path} for ci in cycles],
            "n_channels_out": int(merged.shape[2]),
            "shape_yxc": tuple(int(x) for x in merged.shape),
            "low_mem": True,
        }

    imgs: list[np.ndarray] = []
    names: list[list[str]] = []
    paths: list[str] = []
    n_cycles = int(len(cycles))
    for k, ci in enumerate(cycles, start=1):
        _check_cancel()
        if progress_cb:
            progress_cb(f"Registering cycle {int(ci.cycle)}")
        if progress_event_cb:
            progress_event_cb(
                {
                    "phase": "load",
                    "cycle": int(ci.cycle),
                    "idx": int(k),
                    "n": int(n_cycles),
                    "msg": f"Registering cycle {int(ci.cycle)}",
                }
            )
        img, ch = load_multichannel_tiff(ci.path)
        imgs.append(img)
        names.append(ch)
        paths.append(ci.path)

    # Choose a common canvas so we can merge/register cycles of different pixel dimensions.
    canvas_y = int(max(im.shape[0] for im in imgs))
    canvas_x = int(max(im.shape[1] for im in imgs))
    canvas_yx = (canvas_y, canvas_x)
    imgs = [_pad_channels_to_canvas(im, canvas_yx) for im in imgs]

    # Reference index
    ref_idx = None
    for i, ci in enumerate(cycles):
        if int(ci.cycle) == int(reference_cycle):
            ref_idx = i
            break
    if ref_idx is None:
        raise ValueError(f"reference_cycle={reference_cycle} not found in inputs")

    ref_img = imgs[ref_idx]
    ref_names = names[ref_idx]

    # Determine the channel used for registration in the reference cycle.
    ref_marker = cycles[ref_idx].registration_marker or default_registration_marker
    ref_ch_idx = _find_channel_index(ref_names, ref_marker)
    if ref_ch_idx is None:
        # If we can't find it, we skip registration entirely.
        ref_yx = None
    else:
        ref_yx = ref_img[..., ref_ch_idx]

    shifts: dict[int, tuple[float, float]] = {}
    aligned_imgs: list[np.ndarray] = []

    for i, ci in enumerate(cycles):
        _check_cancel()
        img_i = imgs[i]

        # Report progress as "cycles processed" (includes the reference cycle).
        # This maps cleanly to a determinate per-sample progress bar in the UI.
        def _emit_register_progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            if progress_event_cb:
                progress_event_cb(
                    {
                        "phase": "register",
                        "cycle": int(ci.cycle),
                        "idx": int(i + 1),
                        "n": int(n_cycles),
                        "msg": msg,
                    }
                )

        if i == ref_idx or ref_yx is None:
            aligned_imgs.append(img_i.astype(np.float32, copy=False))
            shifts[int(ci.cycle)] = (0.0, 0.0)
            _emit_register_progress(f"Registered cycle {int(ci.cycle)} (reference)")
            continue

        marker = ci.registration_marker or default_registration_marker
        ch_idx = _find_channel_index(names[i], marker)
        if ch_idx is None:
            # Can't register this cycle; leave as-is.
            aligned_imgs.append(img_i.astype(np.float32, copy=False))
            shifts[int(ci.cycle)] = (0.0, 0.0)
            _emit_register_progress(
                f"Registered cycle {int(ci.cycle)} (skipped: marker '{marker}' not found)"
            )
            continue

        _emit_register_progress(f"Registering cycle {int(ci.cycle)}")

        _check_cancel()

        mov_yx = img_i[..., ch_idx]
        dy, dx = estimate_translation(
            ref_yx,
            mov_yx,
            downsample=int(downsample_for_registration),
            upsample_factor=int(upsample_factor),
        )
        aligned = apply_translation_yxc(img_i, dy=dy, dx=dx, order=1)
        aligned_imgs.append(aligned)
        shifts[int(ci.cycle)] = (dy, dx)

        _emit_register_progress(f"Registered cycle {int(ci.cycle)}")

    if progress_cb:
        progress_cb("Merging channels")
    if progress_event_cb:
        progress_event_cb(
            {
                "phase": "merge",
                "idx": int(n_cycles),
                "n": int(n_cycles),
                "msg": "Merging channels",
            }
        )

    merged = np.concatenate(aligned_imgs, axis=2)

    merged_names: list[str] = []
    seen: dict[str, int] = {}
    for ci, ch in zip(cycles, names):
        cy = int(ci.cycle)
        markers = ci.channel_markers or []
        for j, orig in enumerate(ch):
            marker = (markers[j] if j < len(markers) else "").strip()
            base = marker or (orig or "").strip() or "Channel"
            out_nm = f"{base}_cy{cy}"
            k = int(seen.get(out_nm, 0))
            if k:
                out_nm2 = f"{out_nm}_{k+1}"
            else:
                out_nm2 = out_nm
            seen[out_nm] = k + 1
            merged_names.append(out_nm2)

    save_ome_tiff_yxc(output_path, merged, merged_names)

    if progress_event_cb:
        progress_event_cb(
            {
                "phase": "write",
                "idx": int(n_cycles),
                "n": int(n_cycles),
                "msg": f"Wrote output: {output_path}",
            }
        )

    return {
        "output_path": output_path,
        "reference_cycle": int(reference_cycle),
        "default_registration_marker": default_registration_marker,
        "shifts_yx": shifts,
        "canvas_yx": canvas_yx,
        "inputs": [{"cycle": int(ci.cycle), "path": ci.path} for ci in cycles],
        "n_channels_out": int(merged.shape[2]),
        "shape_yxc": tuple(int(x) for x in merged.shape),
    }
