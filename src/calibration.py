"""Pixel-to-metre calibration from a visible reference plate."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_REFERENCE_DIAMETER_M = 0.450
MIN_PIXEL_DIAMETER = 8.0


@dataclass(frozen=True)
class CalibrationResult:
    """Result of converting a plate measurement into a scale factor."""

    pixels_per_metre: float | None
    reference_diameter_m: float
    reference_diameter_px: float | None
    confirmed: bool
    same_plane: bool
    reliable: bool
    units_displacement: str
    units_velocity: str
    warning: str | None


def build_calibration(
    reference_diameter_px: float | None,
    reference_diameter_m: float = DEFAULT_REFERENCE_DIAMETER_M,
    *,
    confirmed: bool,
    same_plane: bool = True,
) -> CalibrationResult:
    """Return a scale factor only when the user confirmed a valid plate size.

    Metres and metres/second are reported only when calibration is reliable.
    Unconfirmed defaults, missing pixel measurements, and off-plane plates
    fall back to pixel units.
    """
    if not math.isfinite(reference_diameter_m) or reference_diameter_m <= 0.0:
        raise ValueError("Reference diameter in metres must be a positive finite value.")

    pixel_ok = (
        reference_diameter_px is not None
        and math.isfinite(reference_diameter_px)
        and reference_diameter_px >= MIN_PIXEL_DIAMETER
    )
    pixels_per_metre = (
        float(reference_diameter_px) / float(reference_diameter_m) if pixel_ok else None
    )

    warnings: list[str] = []
    if not confirmed:
        warnings.append(
            "Plate diameter was not confirmed. Displacement is shown in pixels and "
            "velocity in pixels/second."
        )
    if not pixel_ok:
        warnings.append(
            "No usable plate diameter in pixels was provided or detected, so a "
            "metre scale cannot be calculated."
        )
    if not same_plane:
        warnings.append(
            "The reference plate is not in approximately the same plane as the bar "
            "path. Perspective will distort a metre scale, so pixel units are used."
        )

    reliable = bool(confirmed and pixel_ok and same_plane and pixels_per_metre)
    if reliable:
        units_displacement = "m"
        units_velocity = "m/s"
        warning = None
    else:
        units_displacement = "px"
        units_velocity = "px/s"
        warning = " ".join(warnings) if warnings else None

    return CalibrationResult(
        pixels_per_metre=pixels_per_metre if reliable else None,
        reference_diameter_m=float(reference_diameter_m),
        reference_diameter_px=float(reference_diameter_px) if pixel_ok else None,
        confirmed=confirmed,
        same_plane=same_plane,
        reliable=reliable,
        units_displacement=units_displacement,
        units_velocity=units_velocity,
        warning=warning,
    )


def pixels_to_metres(
    displacement_px: float, calibration: CalibrationResult | None
) -> float | None:
    """Convert a pixel length to metres when calibration is reliable."""
    if calibration is None or not calibration.reliable or calibration.pixels_per_metre is None:
        return None
    if not math.isfinite(displacement_px):
        return None
    return float(displacement_px) / float(calibration.pixels_per_metre)


def pixel_displacement(first_y: float, second_y: float) -> float:
    """Absolute vertical distance in pixels between two image-space samples."""
    if not math.isfinite(first_y) or not math.isfinite(second_y):
        raise ValueError("Pixel coordinates must be finite.")
    return abs(float(second_y) - float(first_y))
