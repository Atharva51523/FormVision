"""Streamlit interface for creating an annotated pose-overlay video."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.calibration import DEFAULT_REFERENCE_DIAMETER_M
from src.pose_estimator import PoseEstimator
from src.video_processor import CalibrationSettings, VideoProcessingError, process_video

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading the local pose model...")
def get_pose_estimator(model_path: str) -> PoseEstimator:
    """Reuse the initialized model across Streamlit reruns."""
    return PoseEstimator(model_path)


st.set_page_config(page_title="FormVision", page_icon="🏋️", layout="wide")
st.title("FormVision — AI Exercise Technique Analyzer")
st.write(
    "Upload a squat video to create an annotated video with a MediaPipe pose "
    "skeleton and barbell tracking. Videos are processed locally."
)

model_default = os.environ.get("FORMVISION_MODEL_PATH", str(DEFAULT_MODEL))
model_path = st.text_input("Pose Landmarker model file", value=model_default)
uploaded_video = st.file_uploader(
    "Choose a video", type=["mp4", "mov", "avi", "mkv", "webm"]
)

st.subheader("Plate calibration")
st.caption(
    "Enter the true diameter of a visible plate that sits in approximately the "
    "same plane as the bar path. The 0.450 m default is a standard bumper "
    "plate and is used only after you confirm it."
)
reference_diameter_m = st.number_input(
    "Reference plate diameter (metres)",
    min_value=0.05,
    max_value=0.60,
    value=float(DEFAULT_REFERENCE_DIAMETER_M),
    step=0.001,
    format="%.3f",
)
confirmed = st.checkbox("I confirm this plate diameter is correct", value=False)
same_plane = st.checkbox(
    "The reference plate is in approximately the same plane as the bar movement",
    value=True,
)
reference_diameter_px = st.number_input(
    "Reference plate diameter in pixels (0 = auto-detect from the video)",
    min_value=0.0,
    value=0.0,
    step=1.0,
)
show_bar_coordinates = st.checkbox("Show bar coordinates on the annotated video", value=True)


def _render_barbell_section(result) -> None:
    st.header("Barbell Tracking and Velocity")
    quality = result.tracking_quality
    calibration = result.calibration

    if quality is not None and quality.warning:
        st.warning(quality.warning)
    if calibration is not None and calibration.warning:
        st.info(calibration.warning)

    col_track, col_cal = st.columns(2)
    with col_track:
        st.subheader("Tracking quality")
        if quality is None:
            st.write("Barbell tracking did not run.")
        else:
            st.metric("Tracked frames", f"{quality.tracked_fraction:.0%}")
            st.metric("Mean confidence", f"{quality.mean_confidence:.2f}")
            st.write(
                f"{quality.tracked_frame_count} tracked / "
                f"{quality.lost_frame_count} untracked frames. "
                f"Status: {'reliable' if quality.reliable else 'unreliable'}."
            )
    with col_cal:
        st.subheader("Calibration status")
        if calibration is None:
            st.write("Calibration was not configured.")
        elif calibration.reliable:
            st.success(
                f"Reliable metre scale: {calibration.pixels_per_metre:.1f} px/m "
                f"from a {calibration.reference_diameter_m:.3f} m plate "
                f"({calibration.reference_diameter_px:.1f} px)."
            )
        else:
            st.warning(
                "Reporting pixels and pixels/second. Metres are not shown because "
                "calibration is unavailable or unreliable."
            )

    st.subheader("Bar path")
    path_figure = _bar_path_figure(result)
    if path_figure is None:
        st.write("No tracked bar positions were available to plot.")
    else:
        st.plotly_chart(path_figure, use_container_width=True)

    st.subheader("Rep-by-rep velocity")
    table = _velocity_table(result)
    if table.empty:
        st.write("No squat repetitions were detected, so velocity rows are empty.")
    else:
        st.dataframe(table, use_container_width=True)
        valid = [row for row in result.repetition_velocities if row.valid]
        if valid:
            units = valid[0].units_velocity
            mean_of_means = sum(
                row.mean_concentric_velocity or 0.0 for row in valid
            ) / len(valid)
            peak = max(row.peak_concentric_velocity or 0.0 for row in valid)
            metric_cols = st.columns(2)
            metric_cols[0].metric(
                "Mean concentric velocity", f"{mean_of_means:.3f} {units}"
            )
            metric_cols[1].metric("Peak concentric velocity", f"{peak:.3f} {units}")
        else:
            st.warning(
                "Repetitions were found but none had enough barbell tracking to "
                "report velocity."
            )


def _bar_path_figure(result):
    xs = [sample.x for sample in result.bar_samples if sample.is_tracked]
    ys = [sample.y for sample in result.bar_samples if sample.is_tracked]
    if not xs:
        return None
    figure = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": "#ff00ff", "width": 2},
            marker={"size": 4, "color": "#ffe600"},
            name="Bar path",
        )
    )
    figure.update_yaxes(autorange="reversed", title="Image y (px)")
    figure.update_xaxes(title="Image x (px)")
    figure.update_layout(
        height=360,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        title="Observed bar trajectory (untracked frames omitted)",
    )
    return figure


def _velocity_table(result) -> pd.DataFrame:
    rows = []
    for item in result.repetition_velocities:
        rows.append(
            {
                "Rep": item.rep_number,
                "Duration (s)": _fmt(item.duration_s),
                "Concentric duration (s)": _fmt(item.concentric_duration_s),
                f"Mean concentric velocity ({item.units_velocity})": _fmt(
                    item.mean_concentric_velocity
                ),
                f"Peak concentric velocity ({item.units_velocity})": _fmt(
                    item.peak_concentric_velocity
                ),
                f"Vertical displacement ({item.units_displacement})": _fmt(
                    item.vertical_displacement
                ),
                "Tracking confidence": _fmt(item.tracking_confidence),
                "Valid": item.valid,
                "Notes": item.warning or "",
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


if uploaded_video is not None and st.button("Analyze video", type="primary"):
    try:
        estimator = get_pose_estimator(str(Path(model_path).expanduser().resolve()))
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(uploaded_video.name).stem)
        output_path = OUTPUT_DIR / f"{safe_stem or 'video'}_annotated.mp4"
        pixel_override = (
            float(reference_diameter_px) if reference_diameter_px > 0 else None
        )
        settings = CalibrationSettings(
            reference_diameter_m=float(reference_diameter_m),
            confirmed=bool(confirmed),
            same_plane=bool(same_plane),
            reference_diameter_px=pixel_override,
        )
        with tempfile.TemporaryDirectory(prefix="formvision-") as work_dir:
            input_path = Path(work_dir) / Path(uploaded_video.name).name
            input_path.write_bytes(uploaded_video.getvalue())
            with st.spinner("Processing video frames..."):
                result = process_video(
                    input_path,
                    output_path,
                    estimator,
                    calibration_settings=settings,
                    show_bar_coordinates=show_bar_coordinates,
                )
        st.success(
            f"Processed {result.frame_count} frames; pose detected in "
            f"{result.detected_frame_count}."
        )
        st.caption(
            "Yellow cross: tracked bar centre. Magenta: observed path. "
            "Lost frames do not reuse a previous coordinate."
        )
        st.video(str(result.output_path))
        st.download_button(
            "Download annotated video",
            data=result.output_path.read_bytes(),
            file_name=result.output_path.name,
            mime="video/mp4",
        )
        _render_barbell_section(result)
    except (FileNotFoundError, ValueError, RuntimeError, VideoProcessingError) as exc:
        LOGGER.exception("Video analysis failed")
        st.error(str(exc))
