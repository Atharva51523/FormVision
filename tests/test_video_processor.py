from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.barbell_tracker import STATUS_TRACKED, BarSample
from src.video_processor import VideoProcessingError, process_video


class FakeCapture:
    def __init__(
        self, frames: list[np.ndarray | None], *, opened: bool = True, expected: int = 0
    ) -> None:
        self.frames = iter(frames)
        self.opened = opened
        self.expected = expected
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        import cv2

        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 16.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 12.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self.expected),
        }[prop]

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frame = next(self.frames)
        except StopIteration:
            return False, None
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class FakeEstimator:
    def __init__(self, detection: bool = False) -> None:
        self.detection = detection
        self.timestamps: list[int] = []
        self.started = False

    def begin_video(self) -> None:
        self.started = True

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> tuple | None:
        self.timestamps.append(timestamp_ms)
        return (object(),) if self.detection else None

    @staticmethod
    def draw_pose(frame: np.ndarray, landmarks: tuple | None) -> np.ndarray:
        return frame


class FakeTracker:
    def __init__(self) -> None:
        self.started = False
        self.calls = 0

    def begin_video(self) -> None:
        self.started = True

    def update(self, frame, *, frame_index: int, timestamp_ms: int) -> BarSample:
        self.calls += 1
        return BarSample(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=5.0,
            y=6.0,
            confidence=0.9,
            status=STATUS_TRACKED,
        )


class VideoProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.mp4"
        self.input_path.write_bytes(b"video")
        self.output_path = self.root / "nested" / "output.mp4"
        self.frame = np.zeros((12, 16, 3), dtype=np.uint8)

    def test_processes_frames_preserves_metadata_and_releases_resources(self) -> None:
        capture = FakeCapture([self.frame.copy(), self.frame.copy()], expected=2)
        writer = FakeWriter()
        estimator = FakeEstimator(detection=True)
        with (
            patch("src.video_processor.cv2.VideoCapture", return_value=capture),
            patch("src.video_processor.cv2.VideoWriter", return_value=writer),
        ):
            result = process_video(self.input_path, self.output_path, estimator)

        self.assertEqual(result.frame_count, 2)
        self.assertEqual(result.detected_frame_count, 2)
        self.assertEqual((result.fps, result.width, result.height), (25.0, 16, 12))
        self.assertEqual(estimator.timestamps, [0, 40])
        self.assertTrue(estimator.started)
        self.assertEqual(len(writer.frames), 2)
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)
        self.assertTrue(self.output_path.parent.is_dir())

    def test_bar_tracker_samples_are_returned_without_using_pose(self) -> None:
        capture = FakeCapture([self.frame.copy(), self.frame.copy()], expected=2)
        writer = FakeWriter()
        tracker = FakeTracker()
        with (
            patch("src.video_processor.cv2.VideoCapture", return_value=capture),
            patch("src.video_processor.cv2.VideoWriter", return_value=writer),
        ):
            result = process_video(
                self.input_path,
                self.output_path,
                FakeEstimator(),
                barbell_tracker=tracker,
            )

        self.assertTrue(tracker.started)
        self.assertEqual(tracker.calls, 2)
        self.assertEqual(len(result.bar_samples), 2)
        self.assertEqual(result.bar_samples[0].x, 5.0)
        self.assertEqual(result.bar_samples[0].y, 6.0)
        self.assertGreater(int(writer.frames[0].sum()), 0)

    def test_missing_source_is_reported_before_opening_capture(self) -> None:
        with self.assertRaises(FileNotFoundError):
            process_video(self.root / "missing.mp4", self.output_path, FakeEstimator())

    def test_unsupported_output_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(VideoProcessingError, "Unsupported output extension"):
            process_video(self.input_path, self.root / "output.webm", FakeEstimator())

    def test_unopenable_input_releases_capture(self) -> None:
        capture = FakeCapture([], opened=False)
        with patch("src.video_processor.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(VideoProcessingError, "could not open"):
                process_video(self.input_path, self.output_path, FakeEstimator())
        self.assertTrue(capture.released)

    def test_premature_decode_failure_is_reported_and_writer_is_released(self) -> None:
        capture = FakeCapture([self.frame.copy()], expected=3)
        writer = FakeWriter()
        with (
            patch("src.video_processor.cv2.VideoCapture", return_value=capture),
            patch("src.video_processor.cv2.VideoWriter", return_value=writer),
        ):
            with self.assertRaisesRegex(VideoProcessingError, "decoding stopped"):
                process_video(self.input_path, self.output_path, FakeEstimator())
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)

    def test_empty_video_fails_clearly(self) -> None:
        capture = FakeCapture([], expected=0)
        writer = FakeWriter()
        with (
            patch("src.video_processor.cv2.VideoCapture", return_value=capture),
            patch("src.video_processor.cv2.VideoWriter", return_value=writer),
        ):
            with self.assertRaisesRegex(VideoProcessingError, "No frames"):
                process_video(self.input_path, self.output_path, FakeEstimator())
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)


if __name__ == "__main__":
    unittest.main()
