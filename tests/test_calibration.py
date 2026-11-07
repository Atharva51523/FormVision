from __future__ import annotations

import unittest

from src.calibration import (
    DEFAULT_REFERENCE_DIAMETER_M,
    build_calibration,
    pixel_displacement,
    pixels_to_metres,
)


class CalibrationTests(unittest.TestCase):
    def test_confirmed_same_plane_plate_yields_pixels_per_metre(self) -> None:
        result = build_calibration(
            90.0,
            DEFAULT_REFERENCE_DIAMETER_M,
            confirmed=True,
            same_plane=True,
        )

        self.assertTrue(result.reliable)
        self.assertAlmostEqual(result.pixels_per_metre, 90.0 / 0.450)
        self.assertEqual(result.units_velocity, "m/s")
        self.assertEqual(result.units_displacement, "m")
        self.assertIsNone(result.warning)
        self.assertAlmostEqual(pixels_to_metres(90.0, result), 0.450)

    def test_unconfirmed_default_stays_in_pixels(self) -> None:
        result = build_calibration(90.0, confirmed=False, same_plane=True)

        self.assertFalse(result.reliable)
        self.assertIsNone(result.pixels_per_metre)
        self.assertEqual(result.units_velocity, "px/s")
        self.assertIsNone(pixels_to_metres(90.0, result))
        self.assertIn("not confirmed", result.warning or "")

    def test_missing_pixel_diameter_does_not_report_metres(self) -> None:
        result = build_calibration(None, confirmed=True, same_plane=True)

        self.assertFalse(result.reliable)
        self.assertIsNone(result.pixels_per_metre)
        self.assertEqual(result.units_displacement, "px")

    def test_off_plane_reference_is_unreliable(self) -> None:
        result = build_calibration(100.0, 0.450, confirmed=True, same_plane=False)

        self.assertFalse(result.reliable)
        self.assertIsNone(result.pixels_per_metre)
        self.assertIn("same plane", result.warning or "")

    def test_pixel_displacement_is_absolute(self) -> None:
        self.assertEqual(pixel_displacement(120.0, 80.0), 40.0)
        self.assertEqual(pixel_displacement(80.0, 120.0), 40.0)

    def test_rejects_non_positive_metre_diameter(self) -> None:
        with self.assertRaises(ValueError):
            build_calibration(80.0, 0.0, confirmed=True)


if __name__ == "__main__":
    unittest.main()
