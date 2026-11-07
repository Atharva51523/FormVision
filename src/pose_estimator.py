"""MediaPipe Pose Landmarker integration and skeleton drawing."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# MediaPipe Pose's 33-landmark topology, expressed as (start, end) indices.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


class LandmarkerProtocol(Protocol):
    """The subset of the MediaPipe task API used by this application."""

    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        """Return a result for one frame."""

    def close(self) -> None:
        """Release native task resources."""


@dataclass(frozen=True)
class PoseLandmark:
    """A normalized image-space pose landmark and optional confidence values."""

    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


def extract_landmarks(result: object) -> tuple[PoseLandmark, ...] | None:
    """Return the first detected pose, or ``None`` when no pose is present."""
    poses = getattr(result, "pose_landmarks", None)
    if not poses:
        return None

    extracted: list[PoseLandmark] = []
    for landmark in poses[0]:
        extracted.append(
            PoseLandmark(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=_optional_float(landmark, "visibility"),
                presence=_optional_float(landmark, "presence"),
            )
        )
    return tuple(extracted) or None


def _optional_float(landmark: object, attribute: str) -> float | None:
    value = getattr(landmark, attribute, None)
    return float(value) if value is not None else None


class PoseEstimator:
    """Run a local MediaPipe Pose Landmarker model on sequential video frames.

    The task and model are created once per estimator. A new task is created at
    the start of each video so its video timestamps can start at zero.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        landmarker: LandmarkerProtocol | None = None,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self._landmarker = landmarker
        self._owns_landmarker = landmarker is None
        self._last_timestamp_ms: int | None = None

        if self._landmarker is None:
            self._landmarker = self._create_landmarker()

    @staticmethod
    def _resolve_model_path(model_path: str | Path | None) -> Path:
        configured_path = model_path or os.environ.get("FORMVISION_MODEL_PATH")
        if configured_path:
            return Path(configured_path).expanduser().resolve()
        return (Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task")

    def _create_landmarker(self) -> LandmarkerProtocol:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Pose Landmarker model not found at '{self.model_path}'. "
                "Download pose_landmarker_lite.task and set FORMVISION_MODEL_PATH "
                "or pass model_path explicitly."
            )

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not initialize MediaPipe Pose Landmarker from "
                f"'{self.model_path}': {exc}"
            ) from exc

        LOGGER.info("Initialized Pose Landmarker using model %s", self.model_path)
        return landmarker

    def begin_video(self) -> None:
        """Reset MediaPipe's tracking state before processing another clip."""
        if self._owns_landmarker:
            self.close()
            self._landmarker = self._create_landmarker()
        elif self._landmarker is not None:
            reset = getattr(self._landmarker, "reset_video", None)
            if callable(reset):
                reset()
        self._last_timestamp_ms = None

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> tuple[PoseLandmark, ...] | None:
        """Estimate one frame; return ``None`` if no person was detected."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Cannot estimate pose on an empty video frame.")
        if timestamp_ms < 0:
            raise ValueError("Frame timestamp must be non-negative.")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError("Video frame timestamps must be strictly increasing.")
        if self._landmarker is None:
            raise RuntimeError("Pose Landmarker is closed; initialize it before use.")

        try:
            import mediapipe as mp

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except (ImportError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Pose inference failed at {timestamp_ms} ms: {exc}") from exc

        self._last_timestamp_ms = timestamp_ms
        landmarks = extract_landmarks(result)
        if landmarks is None:
            LOGGER.debug("No person detected at %d ms", timestamp_ms)
        return landmarks

    @staticmethod
    def draw_pose(
        frame_bgr: np.ndarray,
        landmarks: tuple[PoseLandmark, ...] | None,
        *,
        visibility_threshold: float = 0.3,
    ) -> np.ndarray:
        """Draw visible pose connections and joints onto a frame in place."""
        if landmarks is None:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        points: list[tuple[int, int] | None] = []
        for landmark in landmarks:
            confident = (
                landmark.visibility is None
                or landmark.visibility >= visibility_threshold
            )
            valid = all(math.isfinite(value) for value in (landmark.x, landmark.y))
            if not confident or not valid:
                points.append(None)
                continue
            x = int(round(landmark.x * width))
            y = int(round(landmark.y * height))
            points.append((x, y) if 0 <= x < width and 0 <= y < height else None)

        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                first, second = points[start], points[end]
                if first is not None and second is not None:
                    cv2.line(
                        frame_bgr, first, second, (50, 220, 50), 3, cv2.LINE_AA
                    )

        for point in points:
            if point is not None:
                cv2.circle(frame_bgr, point, 5, (30, 80, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, point, 7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def close(self) -> None:
        """Release the MediaPipe task; safe to call more than once."""
        if self._landmarker is not None and self._owns_landmarker:
            self._landmarker.close()
            self._landmarker = None
        self._last_timestamp_ms = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
