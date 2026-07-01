from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from cycif_seg.ui import image_controller
from cycif_seg.ui.image_controller import ImageController, ViewerTileRuntime


class Step2aViewerLoadingTest(unittest.TestCase):
    def test_pyramidal_multiscale_probe_skips_full_resolution_plane(self) -> None:
        with (
            mock.patch.object(ImageController, "_dask_available", return_value=True),
            mock.patch.object(
                image_controller,
                "inspect_tiff_pyramid",
                return_value={"is_pyramidal": True},
            ),
            mock.patch.object(
                image_controller,
                "load_single_channel_tiff_native",
                side_effect=AssertionError("full plane load should be skipped"),
            ),
        ):
            used_lazy, first_plane = ImageController._load_initial_display_probe(
                "large.ome.tif",
                want_multiscale=True,
            )

        self.assertTrue(used_lazy)
        self.assertIsNone(first_plane)

    def test_flat_or_non_multiscale_probe_preserves_full_plane_fallback(self) -> None:
        expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
        with (
            mock.patch.object(ImageController, "_dask_available", return_value=True),
            mock.patch.object(
                image_controller,
                "inspect_tiff_pyramid",
                return_value={"is_pyramidal": False},
            ),
            mock.patch.object(
                image_controller,
                "load_single_channel_tiff_native",
                return_value=expected,
            ) as full_loader,
        ):
            used_lazy, first_plane = ImageController._load_initial_display_probe(
                "flat.ome.tif",
                want_multiscale=True,
            )

        self.assertFalse(used_lazy)
        full_loader.assert_called_once_with("flat.ome.tif", 0)
        np.testing.assert_array_equal(first_plane, expected)

    def test_generation_counters_invalidate_stale_loads_and_channel_updates(self) -> None:
        runtime = ViewerTileRuntime(max_workers=1)

        load_a = runtime.next_load_generation()
        channel_a = runtime.next_channel_generation()
        self.assertTrue(runtime.is_current_load(load_a))
        self.assertTrue(runtime.is_current_channel(channel_a))

        load_b = runtime.next_load_generation()
        self.assertFalse(runtime.is_current_load(load_a))
        self.assertTrue(runtime.is_current_load(load_b))
        self.assertFalse(runtime.is_current_channel(channel_a))

        channel_b = runtime.next_channel_generation()
        self.assertFalse(runtime.is_current_channel(channel_a))
        self.assertTrue(runtime.is_current_channel(channel_b))

        runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
