"""Concentric barbell velocity from tracked positions and repetition windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from src.barbell_tracker import BarSample
from src.calibration import CalibrationResult, pixels_to_metres
from src.rep_detector import RepetitionSegment

# Image y grows downward; a negative dy is an upward (concentric) bar move.
UPWARD = "upward"
DOWNWARD = "downward"
STATIONARY = "stationary"


@dataclass(frozen=True)
class RepetitionVelocity:
    """Measured concentric kinematics for one repetition."""

    rep_number: int
    start_index: int
    bottom_index: int
    concentric_start: int
    concentric_end: int
    duration_s: float | None
    concentric_duration_s: float | None
    mean_concentric_velocity: float | None
    peak_concentric_velocity: float | None
    vertical_displacement: float | None
    tracking_confidence: float
    units_displacement: str
    units_velocity: str
    valid: bool
    warning: str | None


def timestamps_to_seconds(timestamp_ms: Sequence[int | float]) -> np.ndarray:
    """Convert millisecond timestamps to seconds without assuming constant FPS."""
    values = np.asarray(timestamp_ms, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Timestamps must be finite and non-negative.")
    return values / 1000.0


def frame_intervals_s(
    timestamp_ms: Sequence[int | float],
    *,
    fps: float | None = None,
) -> np.ndarray:
    """Return dt between consecutive samples.

    Missing or non-increasing timestamps fall back to ``1 / fps`` when FPS is
    valid. Intervals that still cannot be determined are NaN so callers can
    skip them instead of inventing a speed.
    """
    times = timestamps_to_seconds(timestamp_ms)
    dt = np.diff(times)
    fallback = None
    if fps is not None and math.isfinite(fps) and fps > 0:
        fallback = 1.0 / float(fps)
    result = np.empty(dt.shape, dtype=float)
    for i, interval in enumerate(dt):
        if math.isfinite(interval) and interval > 0:
            result[i] = interval
        elif fallback is not None:
            result[i] = fallback
        else:
            result[i] = float("nan")
    return result


def smooth_vertical_positions(
    y_values: Sequence[float | None],
    *,
    window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth contiguous tracked samples. Gaps stay NaN and are not filled."""
    series = _to_nan_array(y_values)
    if window < 3 or window % 2 == 0:
        raise ValueError("Smoothing window must be an odd integer >= 3.")
    if polyorder < 1 or polyorder >= window:
        raise ValueError("polyorder must be positive and smaller than the window.")
    smoothed = series.copy()
    for start, end in _finite_spans(series):
        span = series[start:end]
        if span.size < window:
            smoothed[start:end] = span
            continue
        smoothed[start:end] = savgol_filter(span, window_length=window, polyorder=polyorder)
    return smoothed


def vertical_pixel_velocity(
    y_values: Sequence[float | None],
    timestamp_ms: Sequence[int | float],
    *,
    fps: float | None = None,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Instantaneous upward velocity in pixels/second.

    Positive values are upward. Velocity is not computed across tracking gaps
    or missing frames.
    """
    y = smooth_vertical_positions(y_values, window=smoothing_window)
    dt = frame_intervals_s(timestamp_ms, fps=fps)
    velocity = np.full(y.shape, np.nan, dtype=float)
    for i, interval in enumerate(dt):
        if not math.isfinite(interval) or interval <= 0:
            continue
        if not math.isfinite(y[i]) or not math.isfinite(y[i + 1]):
            continue
        # Upward motion decreases image-space y.
        velocity[i + 1] = -(y[i + 1] - y[i]) / interval
    return velocity


def movement_direction(velocity_px_s: float, *, rest_threshold: float = 5.0) -> str:
    """Classify a signed vertical velocity as upward, downward, or stationary."""
    if not math.isfinite(velocity_px_s):
        raise ValueError("Velocity must be finite.")
    if velocity_px_s > rest_threshold:
        return UPWARD
    if velocity_px_s < -rest_threshold:
        return DOWNWARD
    return STATIONARY


def concentric_end_from_bar(
    y_values: Sequence[float | None],
    bottom_index: int,
    limit_index: int,
) -> int:
    """Advance from a squat bottom until the bar finishes rising.

    Tracking gaps are skipped. The search never invents a coordinate; it only
    inspects observed samples. If the bar never rises, ``bottom_index`` is
    returned.
    """
    y = _to_nan_array(y_values)
    if bottom_index < 0 or bottom_index >= y.size:
        raise ValueError("bottom_index is outside the trajectory.")
    if limit_index <= bottom_index:
        return bottom_index
    limit_index = min(limit_index, y.size - 1)
    last_y = y[bottom_index]
    last_index = bottom_index
    rose = False
    for index in range(bottom_index + 1, limit_index + 1):
        if not math.isfinite(y[index]):
            continue
        if not math.isfinite(last_y):
            last_y = y[index]
            last_index = index
            continue
        if y[index] < last_y - 0.5:
            rose = True
            last_y = y[index]
            last_index = index
            continue
        if rose and y[index] > last_y + 1.0:
            return last_index
        if y[index] <= last_y:
            last_y = y[index]
            last_index = index
    return last_index if rose else bottom_index


def calculate_repetition_velocities(
    samples: Sequence[BarSample],
    segments: Sequence[RepetitionSegment],
    *,
    fps: float | None = None,
    calibration: CalibrationResult | None = None,
    smoothing_window: int = 5,
) -> list[RepetitionVelocity]:
    """Compute per-repetition concentric metrics from tracked bar samples."""
    units_disp, units_vel = _units(calibration)
    if not samples:
        return []
    y_raw = [sample.y if sample.is_tracked else None for sample in samples]
    times_ms = [sample.timestamp_ms for sample in samples]
    try:
        y_smooth = smooth_vertical_positions(y_raw, window=smoothing_window)
        velocity = vertical_pixel_velocity(
            y_raw, times_ms, fps=fps, smoothing_window=smoothing_window
        )
    except ValueError:
        y_smooth = _to_nan_array(y_raw)
        velocity = np.full(y_smooth.shape, np.nan, dtype=float)

    results: list[RepetitionVelocity] = []
    for number, segment in enumerate(segments, start=1):
        results.append(
            _velocity_for_segment(
                number,
                segment,
                samples,
                y_smooth,
                velocity,
                times_ms,
                fps=fps,
                calibration=calibration,
                units_disp=units_disp,
                units_vel=units_vel,
            )
        )
    return results


def _velocity_for_segment(
    number: int,
    segment: RepetitionSegment,
    samples: Sequence[BarSample],
    y_smooth: np.ndarray,
    velocity: np.ndarray,
    times_ms: Sequence[int],
    *,
    fps: float | None,
    calibration: CalibrationResult | None,
    units_disp: str,
    units_vel: str,
) -> RepetitionVelocity:
    n = len(samples)
    start = int(np.clip(segment.start_index, 0, n - 1))
    bottom = int(np.clip(segment.bottom_index, 0, n - 1))
    concentric_start = int(np.clip(segment.concentric_start, 0, n - 1))
    search_limit = int(np.clip(segment.concentric_end, 0, n - 1))
    bar_end = concentric_end_from_bar(y_smooth, concentric_start, search_limit)
    concentric_end = bar_end

    duration_s = _duration_s(times_ms, start, concentric_end, fps)
    concentric_duration_s = _duration_s(times_ms, concentric_start, concentric_end, fps)

    window = samples[concentric_start : concentric_end + 1]
    tracked_conf = [sample.confidence for sample in window if sample.is_tracked]
    tracking_confidence = float(np.mean(tracked_conf)) if tracked_conf else 0.0
    tracked_fraction = len(tracked_conf) / max(1, len(window))

    y_start = y_smooth[concentric_start]
    y_end = y_smooth[concentric_end]
    displacement_px = None
    if math.isfinite(y_start) and math.isfinite(y_end):
        # Upward concentric displacement is a decrease in image y.
        displacement_px = float(y_start - y_end)

    mean_px_s = None
    if (
        displacement_px is not None
        and concentric_duration_s is not None
        and concentric_duration_s > 0
        and displacement_px >= 0
    ):
        mean_px_s = displacement_px / concentric_duration_s

    peak_px_s = None
    if concentric_end > concentric_start:
        peak_slice = velocity[concentric_start + 1 : concentric_end + 1]
        finite_peak = peak_slice[np.isfinite(peak_slice)]
        upward = finite_peak[finite_peak > 0]
        if upward.size:
            peak_px_s = float(np.max(upward))

    warning = None
    valid = True
    if tracked_fraction < 0.5 or tracking_confidence <= 0:
        valid = False
        warning = "Tracking was lost for most of this repetition; velocity is not reported."
        mean_px_s = None
        peak_px_s = None
        displacement_px = None
    elif concentric_duration_s is None or displacement_px is None or displacement_px < 0:
        valid = False
        warning = "The concentric phase could not be measured from the bar trajectory."
        mean_px_s = None
        peak_px_s = None
        displacement_px = None if displacement_px is not None and displacement_px < 0 else displacement_px

    displacement = _scale_displacement(displacement_px, calibration)
    mean_velocity = _scale_velocity(mean_px_s, calibration)
    peak_velocity = _scale_velocity(peak_px_s, calibration)

    return RepetitionVelocity(
        rep_number=number,
        start_index=start,
        bottom_index=bottom,
        concentric_start=concentric_start,
        concentric_end=concentric_end,
        duration_s=duration_s,
        concentric_duration_s=concentric_duration_s,
        mean_concentric_velocity=mean_velocity if valid else None,
        peak_concentric_velocity=peak_velocity if valid else None,
        vertical_displacement=displacement if valid else None,
        tracking_confidence=tracking_confidence,
        units_displacement=units_disp,
        units_velocity=units_vel,
        valid=valid,
        warning=warning,
    )


def _duration_s(
    times_ms: Sequence[int],
    start: int,
    end: int,
    fps: float | None,
) -> float | None:
    if end <= start:
        return None
    start_ms = float(times_ms[start])
    end_ms = float(times_ms[end])
    if math.isfinite(start_ms) and math.isfinite(end_ms) and end_ms > start_ms:
        return (end_ms - start_ms) / 1000.0
    if fps is not None and math.isfinite(fps) and fps > 0:
        return (end - start) / float(fps)
    return None


def _scale_displacement(
    displacement_px: float | None, calibration: CalibrationResult | None
) -> float | None:
    if displacement_px is None:
        return None
    metres = pixels_to_metres(displacement_px, calibration)
    return metres if metres is not None else float(displacement_px)


def _scale_velocity(
    velocity_px_s: float | None, calibration: CalibrationResult | None
) -> float | None:
    if velocity_px_s is None:
        return None
    metres = pixels_to_metres(velocity_px_s, calibration)
    return metres if metres is not None else float(velocity_px_s)


def _units(calibration: CalibrationResult | None) -> tuple[str, str]:
    if calibration is not None and calibration.reliable:
        return calibration.units_displacement, calibration.units_velocity
    return "px", "px/s"


def _to_nan_array(values: Sequence[float | None]) -> np.ndarray:
    series = np.empty(len(values), dtype=float)
    for i, value in enumerate(values):
        series[i] = float(value) if value is not None and math.isfinite(value) else float("nan")
    return series


def _finite_spans(series: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(series):
        if math.isfinite(value):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, series.size))
    return spans
