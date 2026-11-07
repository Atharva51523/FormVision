"""Offline squat repetition detection from knee-angle time series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class RepetitionSegment:
    """Indices that bound one squat repetition.

    ``concentric_start`` is the bottom. ``concentric_end`` is the following
    standing position when it can be located from the knee-angle series.
    """

    start_index: int
    bottom_index: int
    concentric_start: int
    concentric_end: int
    end_index: int


def detect_repetitions(
    knee_angles: Sequence[float],
    *,
    minimum_prominence: float = 20.0,
    minimum_distance: int = 8,
) -> list[int]:
    """Return sample indices of squat bottoms, represented by knee-angle minima."""
    if minimum_prominence <= 0:
        raise ValueError("minimum_prominence must be positive.")
    if minimum_distance < 1:
        raise ValueError("minimum_distance must be at least one sample.")
    values = np.asarray(knee_angles, dtype=float)
    if values.ndim != 1:
        raise ValueError("knee_angles must be a one-dimensional sequence.")
    finite = np.isfinite(values)
    if int(finite.sum()) < 3:
        return []

    # Interpolate short missing spans only for peak search; never emit a peak
    # whose original angle was missing.
    sample_indices = np.arange(values.size)
    interpolated = np.interp(sample_indices, sample_indices[finite], values[finite])
    peaks, _ = find_peaks(
        -interpolated,
        prominence=minimum_prominence,
        distance=minimum_distance,
    )
    return [int(index) for index in peaks if finite[index]]


def segment_repetitions(
    knee_angles: Sequence[float],
    *,
    minimum_prominence: float = 20.0,
    minimum_distance: int = 8,
) -> list[RepetitionSegment]:
    """Return eccentric-to-concentric windows around detected squat bottoms."""
    values = np.asarray(knee_angles, dtype=float)
    if values.ndim != 1:
        raise ValueError("knee_angles must be a one-dimensional sequence.")
    bottoms = detect_repetitions(
        values,
        minimum_prominence=minimum_prominence,
        minimum_distance=minimum_distance,
    )
    if not bottoms:
        return []

    finite = np.isfinite(values)
    sample_indices = np.arange(values.size)
    interpolated = np.interp(sample_indices, sample_indices[finite], values[finite])
    standings, _ = find_peaks(
        interpolated,
        prominence=minimum_prominence,
        distance=minimum_distance,
    )
    standings = [int(index) for index in standings if finite[index]]

    segments: list[RepetitionSegment] = []
    last_end = 0
    for i, bottom in enumerate(bottoms):
        next_bottom = bottoms[i + 1] if i + 1 < len(bottoms) else values.size - 1
        previous_stands = [index for index in standings if last_end <= index < bottom]
        next_stands = [index for index in standings if bottom < index <= next_bottom]
        start_index = previous_stands[-1] if previous_stands else last_end
        concentric_end = next_stands[0] if next_stands else next_bottom
        end_index = concentric_end
        last_end = end_index
        segments.append(
            RepetitionSegment(
                start_index=int(start_index),
                bottom_index=int(bottom),
                concentric_start=int(bottom),
                concentric_end=int(concentric_end),
                end_index=int(end_index),
            )
        )
    return segments
