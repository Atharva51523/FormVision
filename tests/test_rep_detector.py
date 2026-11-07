from __future__ import annotations

import unittest

from src.rep_detector import detect_repetitions, segment_repetitions


def _squat_angles(reps: int = 2) -> list[float]:
    series: list[float] = [170.0] * 6
    for _ in range(reps):
        series.extend(list(range(165, 79, -8)))
        series.append(75.0)
        series.extend(list(range(83, 171, 8)))
        series.extend([170.0] * 6)
    return [float(value) for value in series]


class RepDetectorTests(unittest.TestCase):
    def test_detects_bottoms_of_synthetic_squats(self) -> None:
        angles = _squat_angles(2)
        bottoms = detect_repetitions(angles, minimum_prominence=20.0, minimum_distance=4)
        self.assertEqual(len(bottoms), 2)
        for index in bottoms:
            self.assertLess(angles[index], 90.0)

    def test_missing_angles_are_not_emitted_as_bottoms(self) -> None:
        angles: list[float] = [170.0] * 8 + [float("nan")] * 5 + [170.0] * 8
        self.assertEqual(detect_repetitions(angles, minimum_prominence=10.0), [])

    def test_segments_concentric_from_bottom_to_stand(self) -> None:
        angles = _squat_angles(1)
        segments = segment_repetitions(angles, minimum_prominence=20.0, minimum_distance=4)
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.concentric_start, segment.bottom_index)
        self.assertGreater(segment.concentric_end, segment.bottom_index)
        self.assertGreaterEqual(angles[segment.concentric_end], angles[segment.bottom_index])
        self.assertLessEqual(segment.start_index, segment.bottom_index)


if __name__ == "__main__":
    unittest.main()
