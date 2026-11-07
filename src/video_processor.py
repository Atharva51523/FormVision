"""Sequential OpenCV video processing and annotated video output."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.angle_calculator import squat_knee_angles
from src.barbell_tracker import (
    BarbellTracker,
    BarSample,
    TrackingQuality,
    draw_barbell_overlay,
    median_plate_diameter_px,
    summarize_tracking,
)
from src.calibration import CalibrationResult, build_calibration
from src.pose_estimator import PoseLandmark
from src.rep_detector import RepetitionSegment, segment_repetitions
from src.velocity_calculator import RepetitionVelocity, calculate_repetition_velocities

LOGGER = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised when a source video cannot be decoded or output cannot be saved."""


class PoseEstimatorProtocol(Protocol):
    def begin_video(self) -> None:
        """Reset estimator tracking state before a new clip."""

    def process_frame(
        self, frame_bgr: object, timestamp_ms: int
    ) -> tuple[PoseLandmark, ...] | None:
        """Estimate the pose for a single frame."""

    @staticmethod
    def draw_pose(
        frame_bgr: object, landmarks: tuple[PoseLandmark, ...] | None
    ) -> object:
        """Draw landmarks onto a frame."""


@dataclass(frozen=True)
class ProcessingResult:
    """Summary of a completed annotated-video export."""

    output_path: Path
    frame_count: int
    detected_frame_count: int
    fps: float
    width: int
    height: int
    bar_samples: tuple[BarSample, ...] = ()
    repetition_segments: tuple[RepetitionSegment, ...] = ()
    repetition_velocities: tuple[RepetitionVelocity, ...] = ()
    tracking_quality: TrackingQuality | None = None
    calibration: CalibrationResult | None = None
    knee_angles: tuple[float, ...] = ()


@dataclass(frozen=True)
class CalibrationSettings:
    """User-supplied plate calibration controls."""

    reference_diameter_m: float
    confirmed: bool
    same_plane: bool = True
    reference_diameter_px: float | None = None


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    estimator: PoseEstimatorProtocol,
    *,
    barbell_tracker: BarbellTracker | None = None,
    enable_bar_tracking: bool = True,
    calibration_settings: CalibrationSettings | None = None,
    show_bar_coordinates: bool = True,
) -> ProcessingResult:
    """Annotate a video while preserving its reported frame rate and dimensions."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: '{source}'.")
    if source == destination:
        raise ValueError("Input and output video paths must be different.")
    if destination.suffix.lower() not in {".mp4", ".avi"}:
        raise VideoProcessingError(
            "Unsupported output extension. Use an .mp4 or .avi output file."
        )

    capture = cv2.VideoCapture(str(source))
    writer = None
    tracker = barbell_tracker
    if enable_bar_tracking and tracker is None:
        tracker = BarbellTracker()
    try:
        if not capture.isOpened():
            raise VideoProcessingError(
                f"OpenCV could not open '{source}'. Check that its codec is supported."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise VideoProcessingError(
                f"Video metadata is invalid (fps={fps}, width={width}, height={height})."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(
            *("mp4v" if destination.suffix.lower() == ".mp4" else "XVID")
        )
        writer = cv2.VideoWriter(
            str(destination), fourcc, fps, (width, height)
        )
        if not writer.isOpened():
            raise VideoProcessingError(
                f"OpenCV could not create output video '{destination}'."
            )

        estimator.begin_video()
        if tracker is not None:
            tracker.begin_video()
        frame_count = 0
        detected_frame_count = 0
        bar_samples: list[BarSample] = []
        knee_angles: list[float] = []
        trajectory: list[tuple[float, float] | None] = []
        while True:
            success, frame = capture.read()
            if not success:
                if frame_count == 0:
                    raise VideoProcessingError(
                        f"No frames could be decoded from '{source}'."
                    )
                if expected_frames > 0 and frame_count < expected_frames:
                    raise VideoProcessingError(
                        f"Video decoding stopped at frame {frame_count} of "
                        f"{expected_frames} in '{source}'."
                    )
                break
            if frame is None or frame.size == 0:
                raise VideoProcessingError(
                    f"OpenCV returned an empty frame at index {frame_count}."
                )
            if frame.shape[:2] != (height, width):
                raise VideoProcessingError(
                    f"Frame {frame_count} has dimensions {frame.shape[1]}x"
                    f"{frame.shape[0]}, expected {width}x{height}."
                )

            timestamp_ms = int(round(frame_count * 1000.0 / fps))
            landmarks = estimator.process_frame(frame, timestamp_ms)
            if landmarks is not None:
                detected_frame_count += 1
            knee_angles.append(_mean_knee_angle(landmarks))
            sample = None
            if tracker is not None:
                sample = tracker.update(
                    frame, frame_index=frame_count, timestamp_ms=timestamp_ms
                )
                bar_samples.append(sample)
                if sample.is_tracked:
                    trajectory.append((float(sample.x), float(sample.y)))
                else:
                    trajectory.append(None)
            annotated = estimator.draw_pose(frame, landmarks)
            if sample is not None:
                draw_barbell_overlay(
                    annotated,
                    sample,
                    trajectory,
                    show_coordinates=show_bar_coordinates,
                )
            writer.write(annotated)
            frame_count += 1

        if frame_count == 0:
            raise VideoProcessingError(f"No frames were processed from '{source}'.")

        calibration = _resolve_calibration(bar_samples, calibration_settings)
        segments = tuple(segment_repetitions(knee_angles)) if knee_angles else ()
        velocities = tuple(
            calculate_repetition_velocities(
                bar_samples,
                segments,
                fps=fps,
                calibration=calibration,
            )
        )
        quality = summarize_tracking(bar_samples) if bar_samples else None
        LOGGER.info(
            "Wrote %d frames (%d with a detected pose) to %s",
            frame_count,
            detected_frame_count,
            destination,
        )
        return ProcessingResult(
            output_path=destination,
            frame_count=frame_count,
            detected_frame_count=detected_frame_count,
            fps=fps,
            width=width,
            height=height,
            bar_samples=tuple(bar_samples),
            repetition_segments=segments,
            repetition_velocities=velocities,
            tracking_quality=quality,
            calibration=calibration,
            knee_angles=tuple(knee_angles),
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _mean_knee_angle(landmarks: tuple[PoseLandmark, ...] | None) -> float:
    if landmarks is None or len(landmarks) < 29:
        return float("nan")
    try:
        angles = squat_knee_angles(landmarks)
    except (TypeError, ValueError, AttributeError):
        return float("nan")
    values = [value for value in angles.values() if value is not None and math.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _resolve_calibration(
    samples: list[BarSample],
    settings: CalibrationSettings | None,
) -> CalibrationResult | None:
    if settings is None:
        return None
    pixel_diameter = settings.reference_diameter_px
    if pixel_diameter is None:
        pixel_diameter = median_plate_diameter_px(samples)
    return build_calibration(
        pixel_diameter,
        settings.reference_diameter_m,
        confirmed=settings.confirmed,
        same_plane=settings.same_plane,
    )
