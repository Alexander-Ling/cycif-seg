from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from cycif_seg.io.ome_tiff import (
    IncrementalZarrRegisteredWriter,
    convert_registered_zarr_to_pyramidal,
    inspect_tiff_pyramid,
    load_single_channel_tiff_native,
)
from cycif_seg.preprocess.organize_cycles import (
    CycleInput,
    merge_cycles_to_ome_tiff,
    registration_progress_sidecar_path,
    registration_zarr_store_path,
)


def _synthetic_plane(shape: tuple[int, int], seed: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    img = (
        500.0
        + 300.0 * np.sin((yy + seed * 5) / 6.0)
        + 300.0 * np.cos((xx + seed * 3) / 7.0)
    )
    cy, cx = shape[0] * 0.4, shape[1] * 0.6
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (min(shape) / 5.0) ** 2
    img[blob] += 20_000.0
    return np.clip(img, 0, 65535).astype(np.uint16)


def _write_two_channel_ome(path: Path, planes: np.ndarray, names: list[str]) -> None:
    tifffile.imwrite(
        str(path),
        np.asarray(planes),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": list(names)}},
    )


class RegisteredZarrPyramidConversionTest(unittest.TestCase):
    def test_writer_nests_chunks_and_converts_to_pyramidal_ome_tiff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            zarr_path = td_path / "registered.zarr"
            out_path = td_path / "registered.ome.tif"
            shape_yx = (40, 50)
            planes = [_synthetic_plane(shape_yx, 0), _synthetic_plane(shape_yx, 1)]

            writer = IncrementalZarrRegisteredWriter(
                str(zarr_path),
                (shape_yx[0], shape_yx[1], 2),
                np.dtype("uint16"),
                ["MarkerA", "MarkerB"],
                chunk_yx=(16, 16),
            )
            try:
                for ch, plane in enumerate(planes):
                    writer.write_channel(ch, plane)
            finally:
                writer.close()

            # The EMLINK fix nests chunk files under per-axis directories
            # ("/" separator) instead of dumping every chunk as a flat
            # dotted-name file ("." separator, the zarr v2 default) into a
            # single directory -- which is what exhausted directory-entry
            # limits on cluster filesystems for large canvases.
            self.assertTrue((zarr_path / "0" / "0" / "0").is_file())
            self.assertTrue((zarr_path / "1" / "0" / "0").is_file())
            self.assertFalse((zarr_path / "0.0.0").exists())
            self.assertFalse((zarr_path / "1.0.0").exists())

            convert_registered_zarr_to_pyramidal(
                str(zarr_path),
                str(out_path),
                tile_size=16,
                min_level_size=8,
                out_chunk=16,
                build_workers=1,
            )

            pyramid_info = inspect_tiff_pyramid(str(out_path))
            self.assertTrue(bool(pyramid_info["is_pyramidal"]))
            for ch, expected in enumerate(planes):
                np.testing.assert_array_equal(
                    load_single_channel_tiff_native(str(out_path), ch), expected
                )


class RegistrationResumeTest(unittest.TestCase):
    def test_resume_after_interruption_keeps_completed_cycle_intact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            shape_yx = (48, 64)
            # Both cycles share an identical DAPI (registration-marker) plane so
            # registration converges on a trivial zero shift -- keeping the run
            # small, fast, and deterministic -- while each cycle's second
            # channel carries distinct content to merge into the output.
            shared_dapi = _synthetic_plane(shape_yx, 0)
            cyc1_planes = np.stack([shared_dapi, _synthetic_plane(shape_yx, 1)])
            cyc2_planes = np.stack([shared_dapi, _synthetic_plane(shape_yx, 2)])

            cyc1_path = tmp / "cycle1.ome.tiff"
            cyc2_path = tmp / "cycle2.ome.tiff"
            _write_two_channel_ome(cyc1_path, cyc1_planes, ["DAPI", "MarkerA"])
            _write_two_channel_ome(cyc2_path, cyc2_planes, ["DAPI", "MarkerB"])

            cycles = [
                CycleInput(str(cyc1_path), cycle=1, registration_marker="DAPI", channel_markers=["DAPI", "MarkerA"]),
                CycleInput(str(cyc2_path), cycle=2, registration_marker="DAPI", channel_markers=["DAPI", "MarkerB"]),
            ]
            out_path = tmp / "merged.ome.tiff"
            sidecar_path = registration_progress_sidecar_path(str(out_path))
            zarr_store_path = registration_zarr_store_path(str(out_path))

            run_kwargs = dict(
                reference_cycle=1,
                default_registration_marker="DAPI",
                registration_algorithm="translation",
                downsample_for_registration=1,
                low_mem=True,
                strip_height=16,
                pyramidal_output=True,
                pyramidal_tile_size=16,
                pyramidal_min_level_size=8,
            )

            def _cancel_once_reference_cycle_completes() -> bool:
                if not sidecar_path.is_file():
                    return False
                try:
                    manifest = json.loads(sidecar_path.read_text())
                except Exception:
                    return False
                rec = manifest.get("cycles", {}).get("1")
                return bool(rec and rec.get("status") == "complete")

            # Interrupt deterministically: the manifest only flips Cycle 1 to
            # "complete" right before the moving-cycle loop's first
            # cancellation check, so this fires at exactly that point.
            with self.assertRaisesRegex(RuntimeError, "Cancelled"):
                merge_cycles_to_ome_tiff(
                    cycles,
                    str(out_path),
                    cancel_cb=_cancel_once_reference_cycle_completes,
                    **run_kwargs,
                )

            # The intermediate Zarr store survives the interruption, but the
            # final merged pyramid (built only at the very end) does not exist yet.
            self.assertTrue(zarr_store_path.is_dir())
            self.assertFalse(out_path.exists())

            manifest = json.loads(sidecar_path.read_text())
            completed_cycles = sorted(
                int(cy) for cy, rec in manifest["cycles"].items() if rec.get("status") == "complete"
            )
            self.assertEqual(completed_cycles, [1])

            merge_cycles_to_ome_tiff(
                cycles,
                str(out_path),
                resume_flat_output=True,
                completed_cycles=completed_cycles,
                registration_fingerprint=manifest["fingerprint"],
                **run_kwargs,
            )

            self.assertTrue(out_path.is_file())
            self.assertFalse(zarr_store_path.exists())

            # Regression check: Cycle 1 completed *before* the interruption and
            # must come through the resumed run byte-for-byte unchanged. The bug
            # being guarded against here reopened the resumed run against the
            # wrong path, so `open_existing` was False, the intermediate store
            # got wiped and recreated empty, and "skipped" cycles ended up blank.
            for local_ch, expected in enumerate(cyc1_planes):
                np.testing.assert_array_equal(
                    load_single_channel_tiff_native(str(out_path), local_ch), expected
                )

            # Cycle 2 is registered fresh after resuming and must contain real,
            # non-blank data that still resembles its source.
            for local_ch in range(cyc2_planes.shape[0]):
                roundtrip = load_single_channel_tiff_native(str(out_path), 2 + local_ch)
                self.assertGreater(float(np.std(roundtrip)), 0.0)
                corr = float(np.corrcoef(
                    roundtrip.astype(np.float64).ravel(),
                    cyc2_planes[local_ch].astype(np.float64).ravel(),
                )[0, 1])
                self.assertGreater(corr, 0.8)


if __name__ == "__main__":
    unittest.main()
