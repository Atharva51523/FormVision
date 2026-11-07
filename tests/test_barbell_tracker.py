from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.barbell_tracker import (
    STATUS_LOST,
    STATUS_SEARCHING,
    STATUS_TRACKED,
    BarbellTracker,
    BarSample,
    detect_barbell,
    draw_barbell_overlay,
    summarize_tracking,
)


def _bar_frame(
    *,
    width: int = 320,
    height: int = 240,
    y: int = 80,
    x0: int = 40,
    x1: int = 280,
    radius: int = 22,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv_y = int(y)
    frame[max(0, cv_y - 2) : min(height, cv_y + 3), x0:x1] = (210, 210, 210)
    cv2.circle(frame, (x0 + 8, cv_y), radius, (40, 40, 40), -1)
    cv2.circle(frame, (x0 + 8, cv_y), radius, (180, 180, 180), 2)
    cv2.circle(frame, (x1 - 8, cv_y), radius, (40, 40, 40), -1)
    cv2.circle(frame, (x1 - 8, cv_y), radius, (180, 180, 180), 2)
    return frame


class BarbellTrackerTests(unittest.TestCase):
    def test_detects_synthetic_shaft_near_true_centre(self) -> None:
        frame = _bar_frame(y=90)
        detection = detect_barbell(frame)
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertAlmostEqual(detection.y, 90.0, delta=12.0)
        self.assertGreater(detection.confidence, 0.5)

    def test_tracks_moving_bar_and_reports_loss_without_inventing(self) -> None:
        tracker = BarbellTracker(max_jump_fraction=0.3, smoothing_alpha=1.0)
        tracker.begin_video()
        samples = []
        for index, y in enumerate(range(70, 130, 4)):
            sample = tracker.update(
                _bar_frame(y=y), frame_index=index, timestamp_ms=index * 40
            )
            samples.append(sample)
        tracked = [sample for sample in samples if sample.is_tracked]
        self.assertGreaterEqual(len(tracked), 8)
        ys = [sample.y for sample in tracked if sample.y is not None]
        self.assertGreater(ys[-1], ys[0])

        lost = tracker.update(
            np.zeros((240, 320, 3), dtype=np.uint8),
            frame_index=len(samples),
            timestamp_ms=len(samples) * 40,
        )
        self.assertIsNone(lost.x)
        self.assertIsNone(lost.y)
        self.assertEqual(lost.confidence, 0.0)
        self.assertIn(lost.status, {STATUS_LOST, STATUS_SEARCHING})

    def test_overlay_draws_marker_only_at_tracked_position(self) -> None:
        lost_frame = np.zeros((80, 80, 3), dtype=np.uint8)
        lost = BarSample(0, 0, None, None, 0.0, STATUS_LOST)
        draw_barbell_overlay(lost_frame, lost, [(10.0, 10.0)])
        self.assertEqual(int(lost_frame[40, 40].sum()), 0)

        tracked_frame = np.zeros((80, 80, 3), dtype=np.uint8)
        tracked = BarSample(1, 40, 40.0, 40.0, 0.9, STATUS_TRACKED)
        draw_barbell_overlay(tracked_frame, tracked, [(40.0, 40.0)], show_coordinates=True)
        self.assertGreater(int(tracked_frame[40, 40].sum()), 0)

    def test_summarize_tracking_marks_unreliable_when_mostly_lost(self) -> None:
        samples = [
            BarSample(0, 0, None, None, 0.0, STATUS_LOST),
            BarSample(1, 40, 10.0, 10.0, 0.4, STATUS_TRACKED),
            BarSample(2, 80, None, None, 0.0, STATUS_LOST),
        ]
        quality = summarize_tracking(samples)
        self.assertFalse(quality.reliable)
        self.assertIsNotNone(quality.warning)


if __name__ == "__main__":
    unittest.main()
