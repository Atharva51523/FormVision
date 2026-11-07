"""Summaries of technique consistency across repetition measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def analyze_consistency(
    repetitions: Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    """Summarize repetition features with mean, standard deviation, and CV percent."""
    if not repetitions:
        return pd.DataFrame(columns=["feature", "mean", "std", "cv_percent"])

    values = pd.DataFrame(repetitions, dtype=float)
    rows: list[dict[str, float | str]] = []
    for feature in values.columns:
        series = values[feature].replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(series.mean()) if not series.empty else float("nan")
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        cv_percent = (
            float(std / abs(mean) * 100.0)
            if len(series) > 1 and mean != 0 and np.isfinite(mean)
            else 0.0
        )
        rows.append(
            {"feature": str(feature), "mean": mean, "std": std, "cv_percent": cv_percent}
        )
    return pd.DataFrame(rows, columns=["feature", "mean", "std", "cv_percent"])
