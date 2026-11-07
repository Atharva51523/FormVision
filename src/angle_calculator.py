"""Geometry helpers for joint-angle measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.pose_estimator import PoseLandmark


def joint_angle(
    first: Sequence[float], vertex: Sequence[float], last: Sequence[float]
) -> float | None:
    """Return the angle at ``vertex`` in degrees, or ``None`` for degenerate points."""
    a = np.asarray(first, dtype=float) - np.asarray(vertex, dtype=float)
    b = np.asarray(last, dtype=float) - np.asarray(vertex, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size not in (2, 3):
        raise ValueError("Joint points must be matching 2D or 3D coordinates.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or not math.isfinite(denominator):
        return None
    cosine = float(np.dot(a, b) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def squat_knee_angles(
    landmarks: Sequence[PoseLandmark],
) -> dict[str, float | None]:
    """Calculate left and right hip-knee-ankle angles from a pose."""
    if len(landmarks) < 29:
        raise ValueError("A complete 33-point MediaPipe pose is required.")
    return {
        "left_knee": joint_angle(
            (landmarks[23].x, landmarks[23].y),
            (landmarks[25].x, landmarks[25].y),
            (landmarks[27].x, landmarks[27].y),
        ),
        "right_knee": joint_angle(
            (landmarks[24].x, landmarks[24].y),
            (landmarks[26].x, landmarks[26].y),
            (landmarks[28].x, landmarks[28].y),
        ),
    }
