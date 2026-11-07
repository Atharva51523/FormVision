from __future__ import annotations

import unittest

from src.barbell_tracker import STATUS_LOST, STATUS_TRACKED, BarSample
from src.calibration import build_calibration
from src.rep_detector import RepetitionSegment
from src.velocity_calculator import (
    DOWNWARD,
    UPWARD,
    calculate_repetition_velocities,
    concentric_end_from_bar,
    frame_intervals_s,
    movement_direction,
    smooth_vertical_positions,
    timestamps_to_seconds,
    vertical_pixel_velocity,
)


def _sample(
    index: int,
    y: float | None,
    *,
    fps: float = 30.0,
    x: float = 100.0,
    confidence: float = 0.9,
) -> BarSample:
    tracked = y is not None
    return BarSample(
        frame_index=index,
        timestamp_ms=int(round(index * 1000.0 / fps)),
        x=x if tracked else None,
        y=y,
        confidence=confidence if tracked else 0.0,
        status=STATUS_TRACKED if tracked else STATUS_LOST,
    )


def _v_rep_y(frames_per_phase: int = 10, top: float = 80.0, bottom: float = 180.0) -> list[float]:
    down = [
        top + (bottom - top) * i / (frames_per_phase - 1)
        for i in range(frames_per_phase)
    ]
    up = [
        bottom + (top - bottom) * (i + 1) / (frames_per_phase - 1)
        for i in range(frames_per_phase - 1)
    ]
    return down + up


class VelocityCalculatorTests(unittest.TestCase):
    def test_timestamps_use_milliseconds_not_frame_count(self) -> None:
        seconds = timestamps_to_seconds([0, 40, 80])
        self.assertEqual(list(seconds), [0.0, 0.04, 0.08])

    def test_missing_or_repeated_timestamps_fall_back_to_fps(self) -> None:
        intervals = frame_intervals_s([0, 0, 80], fps=25.0)
        self.assertAlmostEqual(intervals[0], 1.0 / 25.0)
        self.assertAlmostEqual(intervals[1], 0.08)

    def test_missing_timestamps_without_fps_are_nan(self) -> None:
        intervals = frame_intervals_s([10, 10, 10], fps=None)
        self.assertTrue(all(interval != interval for interval in intervals))

    def test_smoothing_does_not_fill_tracking_gaps(self) -> None:
        values = [10.0, 12.0, 11.0, None, None, 40.0, 41.0, 39.0]
        smoothed = smooth_vertical_positions(values, window=3)
        self.assertTrue(smoothed[3] != smoothed[3])  # NaN
        self.assertTrue(smoothed[4] != smoothed[4])
        self.assertTrue(smoothed[0] == smoothed[0])
        self.assertTrue(smoothed[-1] == smoothed[-1])

    def test_pixel_velocity_skips_lost_frames(self) -> None:
        y = [100.0, 90.0, None, 70.0, 60.0]
        timestamps = [0, 100, 200, 300, 400]
        velocity = vertical_pixel_velocity(y, timestamps, smoothing_window=3)
        self.assertTrue(velocity[2] != velocity[2])
        self.assertTrue(velocity[3] != velocity[3])
        self.assertGreater(velocity[1], 0.0)
        self.assertGreater(velocity[4], 0.0)

    def test_direction_distinguishes_up_and_down(self) -> None:
        self.assertEqual(movement_direction(40.0), UPWARD)
        self.assertEqual(movement_direction(-40.0), DOWNWARD)

    def test_synthetic_squat_mean_and_peak_velocity(self) -> None:
        y = _v_rep_y(10, top=80.0, bottom=180.0)
        samples = [_sample(i, value) for i, value in enumerate(y)]
        bottom = int(max(range(len(y)), key=lambda index: y[index]))
        segments = [
            RepetitionSegment(
                start_index=0,
                bottom_index=bottom,
                concentric_start=bottom,
                concentric_end=len(y) - 1,
                end_index=len(y) - 1,
            )
        ]
        rows = calculate_repetition_velocities(samples, segments, fps=30.0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row.valid)
        self.assertEqual(row.units_velocity, "px/s")
        self.assertGreater(row.vertical_displacement or 0.0, 80.0)
        self.assertGreater(row.mean_concentric_velocity or 0.0, 0.0)
        self.assertGreaterEqual(
            row.peak_concentric_velocity or 0.0,
            row.mean_concentric_velocity or 0.0,
        )
        self.assertGreater(row.concentric_duration_s or 0.0, 0.2)

    def test_calibrated_velocity_uses_metres(self) -> None:
        y = _v_rep_y(10, top=80.0, bottom=180.0)
        samples = [_sample(i, value) for i, value in enumerate(y)]
        bottom = int(max(range(len(y)), key=lambda index: y[index]))
        segments = [
            RepetitionSegment(0, bottom, bottom, len(y) - 1, len(y) - 1)
        ]
        calibration = build_calibration(100.0, 0.450, confirmed=True, same_plane=True)
        rows = calculate_repetition_velocities(
            samples, segments, fps=30.0, calibration=calibration
        )
        self.assertEqual(rows[0].units_velocity, "m/s")
        self.assertEqual(rows[0].units_displacement, "m")
        self.assertLess(rows[0].vertical_displacement or 99.0, 1.0)

    def test_tracking_loss_during_concentric_does_not_invent_velocity(self) -> None:
        y: list[float | None] = [80.0] * 5 + [180.0] + [None] * 8 + [80.0]
        samples = [_sample(i, value) for i, value in enumerate(y)]
        segments = [RepetitionSegment(0, 5, 5, 14, 14)]
        rows = calculate_repetition_velocities(samples, segments, fps=30.0)
        self.assertFalse(rows[0].valid)
        self.assertIsNone(rows[0].mean_concentric_velocity)
        self.assertIsNone(rows[0].peak_concentric_velocity)

    def test_concentric_end_follows_upward_bar(self) -> None:
        y = [100.0, 140.0, 180.0, 150.0, 120.0, 90.0, 95.0]
        self.assertEqual(concentric_end_from_bar(y, 2, 6), 5)


if __name__ == "__main__":
    unittest.main()
