from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.pose_estimator import PoseEstimator, PoseLandmark, extract_landmarks


class FakeLandmarker:
    def __init__(self, result: object) -> None:
        self.result = result
        self.timestamps: list[int] = []
        self.close_count = 0

    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        self.close_count += 1


class PoseEstimatorTests(unittest.TestCase):
    def test_extracts_landmarks_and_optional_confidence(self) -> None:
        result = SimpleNamespace(
            pose_landmarks=[
                [
                    SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.9, presence=0.8),
                    SimpleNamespace(x=0.75, y=0.3, z=0.2),
                ]
            ]
        )

        landmarks = extract_landmarks(result)

        self.assertEqual(len(landmarks or ()), 2)
        self.assertEqual(
            landmarks[0],
            PoseLandmark(0.25, 0.5, -0.1, visibility=0.9, presence=0.8),
        )
        self.assertIsNone(landmarks[1].visibility)
        self.assertIsNone(landmarks[1].presence)

    def test_no_detection_returns_none(self) -> None:
        self.assertIsNone(extract_landmarks(SimpleNamespace(pose_landmarks=[])))
        self.assertIsNone(extract_landmarks(SimpleNamespace()))

    def test_draw_pose_keeps_missing_pose_frame_unchanged(self) -> None:
        import numpy as np

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        original = frame.copy()

        result = PoseEstimator.draw_pose(frame, None)

        self.assertIs(result, frame)
        np.testing.assert_array_equal(frame, original)

    def test_draw_pose_renders_detected_joints(self) -> None:
        import numpy as np

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        landmarks = tuple(
            PoseLandmark(0.5, 0.5, 0.0, visibility=0.9) for _ in range(33)
        )

        PoseEstimator.draw_pose(frame, landmarks)

        self.assertGreater(int(frame.sum()), 0)

    def test_inference_uses_timestamp_and_preserves_no_detection(self) -> None:
        import numpy as np

        landmarker = FakeLandmarker(SimpleNamespace(pose_landmarks=[]))
        estimator = PoseEstimator(landmarker=landmarker)
        try:
            self.assertIsNone(estimator.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), 0))
            self.assertEqual(landmarker.timestamps, [0])
        finally:
            estimator.close()
        self.assertEqual(landmarker.close_count, 0)

    def test_rejects_non_monotonic_timestamps(self) -> None:
        import numpy as np

        landmarker = FakeLandmarker(SimpleNamespace(pose_landmarks=[]))
        estimator = PoseEstimator(landmarker=landmarker)
        try:
            estimator.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), 20)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                estimator.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), 20)
        finally:
            estimator.close()


if __name__ == "__main__":
    unittest.main()
