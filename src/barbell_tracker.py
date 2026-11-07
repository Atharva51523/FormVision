"""OpenCV barbell shaft tracking with explicit failure states.

The tracker never reports a guessed coordinate. When detection and temporal
tracking both fail, ``x`` and ``y`` are ``None`` and confidence is zero.
Pose landmarks are not used as a substitute for the bar position.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

STATUS_SEARCHING = "searching"
STATUS_TRACKED = "tracked"
STATUS_LOST = "lost"

# High-contrast overlay colours (BGR) that remain visible on typical gym floors.
MARKER_COLOR = (0, 255, 255)
PATH_COLOR = (255, 0, 255)
STATUS_COLOR = (255, 255, 255)
STATUS_SHADOW = (0, 0, 0)


@dataclass(frozen=True)
class BarSample:
    """One frame of barbell tracking output."""

    frame_index: int
    timestamp_ms: int
    x: float | None
    y: float | None
    confidence: float
    status: str
    plate_diameter_px: float | None = None

    @property
    def is_tracked(self) -> bool:
        return (
            self.status == STATUS_TRACKED
            and self.x is not None
            and self.y is not None
            and self.confidence > 0.0
        )


@dataclass(frozen=True)
class TrackingQuality:
    """Clip-level summary of barbell tracking reliability."""

    tracked_fraction: float
    mean_confidence: float
    lost_frame_count: int
    tracked_frame_count: int
    reliable: bool
    warning: str | None


@dataclass
class _Detection:
    x: float
    y: float
    bbox: tuple[float, float, float, float]
    confidence: float
    plate_diameter_px: float | None = None


class BarbellTracker:
    """Detect a near-horizontal barbell shaft and track it across frames.

    Detection uses Hough line (shaft) and circle (plate) evidence. A CSRT
    tracker provides temporal continuity. CSRT predictions are accepted only
    when they stay consistent with image evidence or a bounded motion model.
    """

    def __init__(
        self,
        *,
        max_jump_fraction: float = 0.18,
        lost_before_search: int = 8,
        smoothing_alpha: float = 0.35,
    ) -> None:
        if not 0.0 < max_jump_fraction <= 1.0:
            raise ValueError("max_jump_fraction must be in (0, 1].")
        if lost_before_search < 1:
            raise ValueError("lost_before_search must be at least 1.")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")
        self.max_jump_fraction = max_jump_fraction
        self.lost_before_search = lost_before_search
        self.smoothing_alpha = smoothing_alpha
        self._tracker: cv2.Tracker | None = None
        self._last_valid: tuple[float, float] | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._lost_streak = 0
        self._plate_diameters: list[float] = []
        self._frame_size: tuple[int, int] | None = None

    def begin_video(self) -> None:
        """Reset tracker state before a new clip."""
        self._tracker = None
        self._last_valid = None
        self._last_bbox = None
        self._lost_streak = 0
        self._plate_diameters = []
        self._frame_size = None

    def median_plate_diameter_px(self) -> float | None:
        """Return the median detected bumper/plate diameter, if any."""
        if not self._plate_diameters:
            return None
        return float(np.median(np.asarray(self._plate_diameters, dtype=float)))

    def update(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> BarSample:
        """Track the barbell on one BGR frame and return an observation."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Cannot track a barbell on an empty video frame.")
        if timestamp_ms < 0:
            raise ValueError("Frame timestamp must be non-negative.")
        height, width = frame_bgr.shape[:2]
        self._frame_size = (width, height)

        detection = detect_barbell(frame_bgr, search_bbox=self._search_region(width, height))
        tracker_obs = self._update_csrt(frame_bgr)
        fused = self._fuse(detection, tracker_obs, width, height)

        if fused is None:
            self._lost_streak += 1
            if self._lost_streak >= self.lost_before_search:
                self._tracker = None
                self._last_bbox = None
            return BarSample(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                x=None,
                y=None,
                confidence=0.0,
                status=STATUS_SEARCHING if self._last_valid is None else STATUS_LOST,
                plate_diameter_px=None,
            )

        x, y = self._smooth(fused.x, fused.y)
        self._last_valid = (x, y)
        self._last_bbox = fused.bbox
        self._lost_streak = 0
        if fused.plate_diameter_px is not None and fused.plate_diameter_px > 0:
            self._plate_diameters.append(float(fused.plate_diameter_px))
        if detection is not None:
            self._reinit_tracker(frame_bgr, fused.bbox)
        elif self._tracker is None:
            self._reinit_tracker(frame_bgr, fused.bbox)

        return BarSample(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=x,
            y=y,
            confidence=float(min(1.0, max(0.0, fused.confidence))),
            status=STATUS_TRACKED,
            plate_diameter_px=fused.plate_diameter_px,
        )

    def _search_region(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        if self._last_bbox is None:
            return None
        x, y, w, h = self._last_bbox
        pad_x = max(w * 0.75, width * 0.08)
        pad_y = max(h * 3.0, height * 0.12)
        x0 = int(max(0, x - pad_x))
        y0 = int(max(0, y - pad_y))
        x1 = int(min(width, x + w + pad_x))
        y1 = int(min(height, y + h + pad_y))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _update_csrt(self, frame_bgr: np.ndarray) -> _Detection | None:
        if self._tracker is None:
            return None
        try:
            ok, bbox = self._tracker.update(frame_bgr)
        except cv2.error:
            self._tracker = None
            return None
        if not ok:
            self._tracker = None
            return None
        x, y, w, h = (float(v) for v in bbox)
        if w <= 1 or h <= 1:
            return None
        support = _shaft_support(frame_bgr, (x, y, w, h))
        if support < 0.01:
            return None
        return _Detection(
            x=x + w / 2.0,
            y=y + h / 2.0,
            bbox=(x, y, w, h),
            confidence=min(0.62, 0.35 + 4.0 * support),
        )

    def _fuse(
        self,
        detection: _Detection | None,
        tracker_obs: _Detection | None,
        width: int,
        height: int,
    ) -> _Detection | None:
        max_jump = self.max_jump_fraction * float(min(width, height))
        candidates: list[_Detection] = []
        for item in (detection, tracker_obs):
            if item is None:
                continue
            if not _in_frame(item.x, item.y, width, height):
                continue
            if self._last_valid is not None:
                jump = math.hypot(item.x - self._last_valid[0], item.y - self._last_valid[1])
                if jump > max_jump:
                    # Large jumps are rejected unless independent shaft evidence agrees.
                    if detection is None or item is tracker_obs:
                        continue
                    detect_jump = math.hypot(
                        detection.x - self._last_valid[0], detection.y - self._last_valid[1]
                    )
                    if detect_jump > max_jump and tracker_obs is None:
                        continue
            candidates.append(item)

        if detection is not None and tracker_obs is not None:
            agreement = math.hypot(detection.x - tracker_obs.x, detection.y - tracker_obs.y)
            if agreement <= max(8.0, 0.04 * min(width, height)):
                plate = detection.plate_diameter_px
                fused_conf = min(1.0, 0.5 * detection.confidence + 0.5 * tracker_obs.confidence + 0.2)
                mid = _Detection(
                    x=0.6 * detection.x + 0.4 * tracker_obs.x,
                    y=0.6 * detection.y + 0.4 * tracker_obs.y,
                    bbox=detection.bbox,
                    confidence=fused_conf,
                    plate_diameter_px=plate,
                )
                if _in_frame(mid.x, mid.y, width, height):
                    if self._last_valid is None or math.hypot(
                        mid.x - self._last_valid[0], mid.y - self._last_valid[1]
                    ) <= max_jump * 1.25:
                        return mid

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _smooth(self, x: float, y: float) -> tuple[float, float]:
        if self._last_valid is None:
            return x, y
        alpha = self.smoothing_alpha
        prev_x, prev_y = self._last_valid
        return (alpha * x + (1.0 - alpha) * prev_x, alpha * y + (1.0 - alpha) * prev_y)

    def _reinit_tracker(
        self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        clipped = _clip_bbox(bbox, frame_bgr.shape[1], frame_bgr.shape[0])
        if clipped is None:
            return
        tracker = _create_csrt_tracker()
        if tracker is None:
            self._tracker = None
            return
        try:
            if not tracker.init(frame_bgr, clipped):
                self._tracker = None
                return
        except cv2.error:
            self._tracker = None
            return
        self._tracker = tracker
        self._last_bbox = tuple(float(v) for v in clipped)


def detect_barbell(
    frame_bgr: np.ndarray,
    *,
    search_bbox: tuple[int, int, int, int] | None = None,
) -> _Detection | None:
    """Detect a barbell shaft and optional plates in a BGR image."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    height, width = frame_bgr.shape[:2]
    roi, offset = _crop_search(frame_bgr, search_bbox)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    shaft = _detect_shaft(gray)
    plates = _detect_plates(gray)

    ox, oy = offset
    plate_diameter = None
    if plates:
        plate_diameter = float(np.median([2.0 * circle[2] for circle in plates]))

    if shaft is not None and plates:
        shaft_x, shaft_y, bbox = shaft
        plate_x, plate_y = _plates_midpoint(plates)
        if math.hypot(shaft_x - plate_x, shaft_y - plate_y) <= 0.12 * max(width, height):
            x = 0.5 * shaft_x + 0.5 * plate_x + ox
            y = 0.5 * shaft_y + 0.5 * plate_y + oy
            conf = 0.92
        else:
            x, y = shaft_x + ox, shaft_y + oy
            conf = 0.78
        abs_bbox = (bbox[0] + ox, bbox[1] + oy, bbox[2], bbox[3])
        return _Detection(x, y, abs_bbox, conf, plate_diameter)

    if shaft is not None:
        shaft_x, shaft_y, bbox = shaft
        abs_bbox = (bbox[0] + ox, bbox[1] + oy, bbox[2], bbox[3])
        return _Detection(
            shaft_x + ox,
            shaft_y + oy,
            abs_bbox,
            0.72,
            plate_diameter,
        )

    if len(plates) >= 2:
        x, y = _plates_midpoint(plates)
        radius = float(np.median([circle[2] for circle in plates]))
        bbox = (x - 4 * radius + ox, y - radius + oy, 8 * radius, 2 * radius)
        return _Detection(x + ox, y + oy, bbox, 0.7, plate_diameter)

    return None


def _detect_shaft(
    gray: np.ndarray,
) -> tuple[float, float, tuple[float, float, float, float]] | None:
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 140)
    min_length = max(24, int(0.22 * width))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, int(0.08 * width)),
        minLineLength=min_length,
        maxLineGap=max(8, int(0.04 * width)),
    )
    if lines is None:
        return None

    scored: list[tuple[float, np.ndarray]] = []
    for line in lines:
        x1, y1, x2, y2 = (float(v) for v in line[0])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        # Allow a modest tilt so the collar/shaft remains visible in 3/4 views.
        if angle > 28.0:
            continue
        horizontality = 1.0 - (angle / 28.0)
        scored.append((length * (0.55 + 0.45 * horizontality), np.array([x1, y1, x2, y2])))

    if not scored:
        return None
    _score, best = max(scored, key=lambda item: item[0])
    x1, y1, x2, y2 = (float(v) for v in best)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    length = math.hypot(x2 - x1, y2 - y1)
    box_w = max(length, 16.0)
    box_h = max(12.0, 0.06 * height)
    bbox = (cx - box_w / 2.0, cy - box_h / 2.0, box_w, box_h)
    return cx, cy, bbox


def _detect_plates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    height, width = gray.shape[:2]
    min_dim = min(height, width)
    if min_dim < 40:
        return []
    min_radius = max(6, int(0.03 * min_dim))
    max_radius = max(min_radius + 1, int(0.22 * min_dim))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(0.12 * width)),
        param1=90,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    found = [(float(x), float(y), float(r)) for x, y, r in circles[0]]
    if len(found) < 2:
        return found
    # Keep plate-like pairs that share a similar height and radius.
    best_pair: tuple[float, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None
    for i, first in enumerate(found):
        for second in found[i + 1 :]:
            radius_ratio = min(first[2], second[2]) / max(first[2], second[2])
            if radius_ratio < 0.55:
                continue
            if abs(first[1] - second[1]) > 0.12 * height:
                continue
            separation = abs(first[0] - second[0])
            if separation < 3.0 * max(first[2], second[2]):
                continue
            score = radius_ratio * separation
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, (first, second))
    if best_pair is None:
        return found
    return [best_pair[1][0], best_pair[1][1]]


def _plates_midpoint(plates: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    xs = [plate[0] for plate in plates]
    ys = [plate[1] for plate in plates]
    return float(np.mean(xs)), float(np.mean(ys))


def _crop_search(
    frame_bgr: np.ndarray, search_bbox: tuple[int, int, int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    if search_bbox is None:
        return frame_bgr, (0, 0)
    x, y, w, h = (int(v) for v in search_bbox)
    height, width = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame_bgr, (0, 0)
    return frame_bgr[y0:y1, x0:x1], (x0, y0)


def _create_csrt_tracker() -> cv2.Tracker | None:
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if callable(factory):
        try:
            return factory()
        except cv2.error:
            LOGGER.warning("cv2.TrackerCSRT_create failed; trying legacy CSRT.")
    legacy = getattr(cv2, "legacy", None)
    legacy_factory = getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None
    if callable(legacy_factory):
        try:
            return legacy_factory()
        except cv2.error:
            LOGGER.warning("legacy CSRT tracker is unavailable.")
    LOGGER.warning("No CSRT tracker is available; using detection only.")
    return None


def _in_frame(x: float, y: float, width: int, height: int) -> bool:
    return math.isfinite(x) and math.isfinite(y) and 0.0 <= x < width and 0.0 <= y < height


def _shaft_support(
    frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]
) -> float:
    """Fraction of strong edges inside a tracker box; empty crops score ~0."""
    height, width = frame_bgr.shape[:2]
    clipped = _clip_bbox(bbox, width, height)
    if clipped is None:
        return 0.0
    x, y, w, h = clipped
    crop = frame_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    x0 = int(max(0, math.floor(x)))
    y0 = int(max(0, math.floor(y)))
    x1 = int(min(width, math.ceil(x + w)))
    y1 = int(min(height, math.ceil(y + h)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def draw_barbell_overlay(
    frame_bgr: np.ndarray,
    sample: BarSample,
    trajectory: Sequence[tuple[float, float] | None],
    *,
    show_coordinates: bool = True,
) -> np.ndarray:
    """Draw the tracked marker, recent path, and status onto ``frame_bgr``."""
    segments = _trajectory_polylines(trajectory)
    for segment in segments:
        if len(segment) >= 2:
            points = np.round(np.asarray(segment, dtype=np.float32)).astype(np.int32)
            cv2.polylines(frame_bgr, [points], False, PATH_COLOR, 2, cv2.LINE_AA)

    if sample.is_tracked:
        point = (int(round(sample.x or 0.0)), int(round(sample.y or 0.0)))
        cv2.drawMarker(
            frame_bgr,
            point,
            MARKER_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(frame_bgr, point, 10, MARKER_COLOR, 2, cv2.LINE_AA)

    status_text = f"Bar: {sample.status}  conf={sample.confidence:.2f}"
    if show_coordinates and sample.is_tracked:
        status_text += f"  ({sample.x:.0f}, {sample.y:.0f})"
    _put_outlined_text(frame_bgr, status_text, (12, 28))
    return frame_bgr


def _trajectory_polylines(
    trajectory: Sequence[tuple[float, float] | None],
) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in trajectory:
        if point is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        current.append(point)
    if len(current) >= 2:
        segments.append(current)
    return segments


def _put_outlined_text(frame_bgr: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(
        frame_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_SHADOW, 3, cv2.LINE_AA
    )
    cv2.putText(
        frame_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_COLOR, 1, cv2.LINE_AA
    )


def summarize_tracking(
    samples: Sequence[BarSample],
    *,
    min_tracked_fraction: float = 0.65,
    min_mean_confidence: float = 0.5,
) -> TrackingQuality:
    """Summarize tracking reliability without inventing missing samples."""
    if not samples:
        return TrackingQuality(
            tracked_fraction=0.0,
            mean_confidence=0.0,
            lost_frame_count=0,
            tracked_frame_count=0,
            reliable=False,
            warning="No barbell tracking samples were produced.",
        )
    tracked = [sample for sample in samples if sample.is_tracked]
    tracked_fraction = len(tracked) / len(samples)
    mean_confidence = (
        float(np.mean([sample.confidence for sample in tracked])) if tracked else 0.0
    )
    lost_frame_count = len(samples) - len(tracked)
    reliable = (
        tracked_fraction >= min_tracked_fraction and mean_confidence >= min_mean_confidence
    )
    warning = None
    if not tracked:
        warning = (
            "The barbell could not be tracked. Velocity is unavailable. "
            "Film a side or 45-degree view with the full bar and plates visible."
        )
    elif not reliable:
        warning = (
            "Barbell tracking is unreliable. Treat velocity numbers as incomplete. "
            f"Tracked {tracked_fraction:.0%} of frames with mean confidence "
            f"{mean_confidence:.2f}."
        )
    return TrackingQuality(
        tracked_fraction=tracked_fraction,
        mean_confidence=mean_confidence,
        lost_frame_count=lost_frame_count,
        tracked_frame_count=len(tracked),
        reliable=reliable,
        warning=warning,
    )


def median_plate_diameter_px(samples: Sequence[BarSample]) -> float | None:
    """Median plate diameter from samples that actually observed plates."""
    diameters = [
        sample.plate_diameter_px
        for sample in samples
        if sample.plate_diameter_px is not None and sample.plate_diameter_px > 0
    ]
    if not diameters:
        return None
    return float(np.median(np.asarray(diameters, dtype=float)))
