from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from cycif_seg.io import ome_tiff
from cycif_seg.io.ome_tiff import (
    CycleTiffChannelReader,
    IncrementalZarrRegisteredWriter,
    convert_registered_zarr_to_pyramidal,
    inspect_tiff_pyramid,
    load_channel_strip,
    resolve_registered_zarr_chunk_yx,
)


@unittest.skipIf(ome_tiff.zarr is None, "zarr is required")
class RegisteredZarrChunkingTest(unittest.TestCase):
    def test_auto_chunking_matches_strip_height_and_byte_cap(self) -> None:
        chunk_yx, policy = resolve_registered_zarr_chunk_yx(
            (4000, 20_000, 6),
            np.dtype("uint16"),
            strip_height=3700,
            fallback_yx=(512, 512),
            target_mb=64,
        )

        self.assertEqual(chunk_yx[0], 3700)
        self.assertGreater(chunk_yx[1], 512)
        self.assertLessEqual(policy["chunk_bytes"], 64 * 1024 * 1024)
        self.assertEqual(policy["y_source"], "strip_height")
        self.assertEqual(policy["x_source"], "target_mb")

    def test_explicit_chunk_overrides_are_clamped_to_canvas(self) -> None:
        chunk_yx, policy = resolve_registered_zarr_chunk_yx(
            (100, 200, 2),
            np.dtype("float32"),
            strip_height=32,
            explicit_y=1000,
            explicit_x=1000,
            target_mb=1,
        )

        self.assertEqual(chunk_yx, (100, 200))
        self.assertEqual(policy["y_source"], "explicit")
        self.assertEqual(policy["x_source"], "explicit")

    def test_writer_preserves_boundary_crossing_strip_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registered.zarr"
            writer = IncrementalZarrRegisteredWriter(
                str(path),
                (8, 10, 2),
                np.dtype("uint16"),
                ["a", "b"],
                chunk_yx=(3, 4),
                write_workers=2,
                max_pending_writes=1,
            )
            try:
                strip = np.arange(5 * 10, dtype=np.uint16).reshape(5, 10)
                writer.write_channel_strip(1, strip, 2)
                arr = writer.array

                self.assertEqual(tuple(int(v) for v in arr.chunks), (1, 3, 4))
                self.assertEqual(tuple(arr.attrs["registered_zarr_chunk_yx"]), (3, 4))
                np.testing.assert_array_equal(np.asarray(arr[1, 2:7, :]), strip)
                self.assertEqual(int(np.asarray(arr[0]).sum()), 0)
            finally:
                writer.close()

    def test_writer_checks_cancellation_before_queuing_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registered.zarr"
            writer = IncrementalZarrRegisteredWriter(
                str(path),
                (8, 10, 1),
                np.dtype("uint16"),
                ["a"],
                chunk_yx=(3, 4),
                write_workers=2,
                cancel_cb=lambda: True,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "Cancelled"):
                    writer.write_channel_strip(0, np.ones((3, 10), dtype=np.uint16), 0)
            finally:
                writer.abort()

    def test_queue_channel_strip_preserves_values_after_wait(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registered.zarr"
            writer = IncrementalZarrRegisteredWriter(
                str(path),
                (9, 11, 1),
                np.dtype("uint16"),
                ["a"],
                chunk_yx=(3, 4),
                write_workers=2,
                max_pending_writes=8,
            )
            try:
                strip0 = np.arange(4 * 11, dtype=np.uint16).reshape(4, 11)
                strip1 = (1000 + np.arange(5 * 11, dtype=np.uint16)).reshape(5, 11)
                writer.queue_channel_strip(0, strip0, 0)
                writer.queue_channel_strip(0, strip1, 4)
                writer.wait_pending_writes()

                expected = np.vstack([strip0, strip1])
                np.testing.assert_array_equal(np.asarray(writer.array[0, :, :]), expected)
                timing = writer.timing_snapshot()
                self.assertGreaterEqual(int(timing["zarr_write_calls"]), 1)
            finally:
                writer.close()

    def test_registered_zarr_to_pyramidal_smoke_preserves_base(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            zarr_path = td_path / "registered.zarr"
            out_path = td_path / "registered.ome.tif"
            data = np.arange(2 * 144 * 160, dtype=np.uint16).reshape(2, 144, 160)
            writer = IncrementalZarrRegisteredWriter(
                str(zarr_path),
                (144, 160, 2),
                np.dtype("uint16"),
                ["a", "b"],
                chunk_yx=(48, 64),
                write_workers=2,
            )
            try:
                writer.queue_channel_strip(0, data[0, :72, :], 0)
                writer.queue_channel_strip(0, data[0, 72:, :], 72)
                writer.queue_channel_strip(1, data[1, :, :], 0)
                writer.wait_pending_writes()
            finally:
                writer.close()

            convert_registered_zarr_to_pyramidal(
                str(zarr_path),
                str(out_path),
                tile_size=64,
                min_level_size=32,
                out_chunk=64,
                build_workers=1,
            )

            with tifffile.TiffFile(out_path) as tf:
                base = np.asarray(tf.series[0].levels[0].asarray())
            pyramid_info = inspect_tiff_pyramid(str(out_path))
            self.assertTrue(bool(pyramid_info["is_pyramidal"]))
            np.testing.assert_array_equal(base, data)


class CycleTiffChannelReaderTest(unittest.TestCase):
    def test_reader_matches_load_channel_strip_for_cyx_and_yxc(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cyx_path = td_path / "cyx.ome.tif"
            yxc_path = td_path / "yxc.ome.tif"
            cyx = np.arange(3 * 72 * 80, dtype=np.uint16).reshape(3, 72, 80)
            yxc = np.moveaxis(cyx, 0, -1)

            tifffile.imwrite(cyx_path, cyx, metadata={"axes": "CYX"}, ome=True, bigtiff=True)
            tifffile.imwrite(yxc_path, yxc, metadata={"axes": "YXC"}, ome=True, bigtiff=True)

            for path in (cyx_path, yxc_path):
                with CycleTiffChannelReader(str(path)) as reader:
                    for ch in range(3):
                        cached = reader.load_channel_strip(ch, 7, 31)
                        baseline = load_channel_strip(str(path), ch, 7, 31)
                        np.testing.assert_array_equal(cached, baseline)


if __name__ == "__main__":
    unittest.main()
