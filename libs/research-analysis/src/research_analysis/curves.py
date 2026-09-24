"""Analytical aggregation utilities for RL training curves.

Operates on the tidy ``condition_name | seed | step | metric | value`` DataFrame
produced by :func:`research_analysis.dataset.loading.load_experiment_dataset`.

All public functions:

- Accept a tidy polars DataFrame and return a tidy polars DataFrame.
- Require uniform step indices across seeds within a condition and metric.
- Operate on one metric at a time (pass the metric name as a string).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from research_analysis.bootstrap import bootstrap_ci
from research_analysis.duplicates import duplicate_seed_error
from research_analysis.non_finite import require_finite
from research_analysis.statistics import (
    pointwise_tolerance_interval,
    select_quantile_run_index,
    tolerance_interval_order_indices,
)

_MEAN_CURVES_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "step": pl.Int64,
    "mean": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
}
_MEDIAN_CURVE_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "step": pl.Int64,
    "value": pl.Float64,
}
_TOLERANCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "step": pl.Int64,
    "low": pl.Float64,
    "high": pl.Float64,
}
_EVENT_RESPONSE_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "event_step": pl.Int64,
    "pre_event_value": pl.Float64,
    "drop": pl.Float64,
    "recovery_slope": pl.Float64,
    "steps_to_recovery": pl.Int64,
}
_FLATLINE_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "flatlined": pl.Boolean,
    "tail_range": pl.Float64,
    "whole_range": pl.Float64,
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _require_one_run_per_seed(frame: pl.DataFrame, metric: str) -> None:
    keys = ["condition_name", "seed"]
    if "run_id" in frame.columns:
        shared = frame.group_by(keys).agg(pl.col("run_id").unique().sort()).filter(pl.col("run_id").list.len() > 1)
        run_ids = shared["run_id"][0].to_list() if not shared.is_empty() else []
    else:
        shared = frame.filter(pl.struct([*keys, "step"]).is_duplicated())
        run_ids = []
    if shared.is_empty():
        return
    first = shared.sort(keys).row(0, named=True)
    raise duplicate_seed_error(f"metric {metric!r} for condition {first['condition_name']!r}", first["seed"], run_ids)


def _condition_curve_array(
    df: pl.DataFrame, metric: str, condition: str
) -> tuple[list[int], NDArray[np.float64]]:
    """Return (sorted_steps, array) where array has shape (n_seeds, n_steps)."""
    subset = df.filter(
        (pl.col("metric") == metric) & (pl.col("condition_name") == condition)
    ).sort(["seed", "step"])
    require_finite(subset, f"metric {metric!r} for condition {condition!r}")
    _require_one_run_per_seed(subset, metric)

    seed_frames = subset.partition_by("seed", maintain_order=True)
    steps = seed_frames[0]["step"].to_list()
    for seed_frame in seed_frames[1:]:
        seed_steps = seed_frame["step"].to_list()
        if seed_steps == steps:
            continue
        divergent_seed = seed_frame["seed"][0]
        raise ValueError(
            f"metric {metric!r} has non-uniform steps across seeds for condition "
            f"{condition!r}; first divergent seed is {divergent_seed}"
        )
    rows = [seed_frame["value"].to_numpy() for seed_frame in seed_frames]
    return steps, np.array(rows, dtype=np.float64)


def _conditions_with_metric(df: pl.DataFrame, metric: str) -> list[str]:
    return df.filter(pl.col("metric") == metric)["condition_name"].unique().sort().to_list()


# ── Public API ────────────────────────────────────────────────────────────────


def mean_curves(
    df: pl.DataFrame,
    metric: str,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 0,
) -> pl.DataFrame:
    """Pointwise mean learning curve with bootstrap confidence interval bands per condition.

    Returns a DataFrame with columns: ``condition_name | step | mean | ci_low | ci_high``.

    Raises:
        ValueError: If any condition has fewer than 2 seeds (required for bootstrap).
    """
    rng = np.random.default_rng(random_seed)
    frames: list[pl.DataFrame] = []
    for condition in _conditions_with_metric(df, metric):
        steps, array = _condition_curve_array(df, metric, condition)
        ci = bootstrap_ci(
            array,
            confidence=confidence,
            n_resamples=n_resamples,
            rng=rng,
        )
        frames.append(
            pl.DataFrame(
                {
                    "condition_name": [condition] * len(steps),
                    "step": steps,
                    "mean": np.mean(array, axis=0).tolist(),
                    "ci_low": ci.ci_low.tolist(),
                    "ci_high": ci.ci_high.tolist(),
                }
            )
        )

    if not frames:
        return pl.DataFrame(schema=_MEAN_CURVES_SCHEMA)
    return pl.concat(frames).sort(["condition_name", "step"])


def median_run_curve(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Learning curve of the seed closest to the median performance, per condition.

    Returns a DataFrame with columns: ``condition_name | step | value``.
    """
    return quantile_run_curve(df, metric, quantile=0.5)


def quantile_run_curve(
    df: pl.DataFrame,
    metric: str,
    *,
    quantile: float,
) -> pl.DataFrame:
    """Learning curve of the seed closest to a performance quantile, per condition."""
    frames: list[pl.DataFrame] = []
    for condition in _conditions_with_metric(df, metric):
        steps, array = _condition_curve_array(df, metric, condition)
        idx = select_quantile_run_index(array, quantile=quantile)
        frames.append(
            pl.DataFrame(
                {
                    "condition_name": [condition] * len(steps),
                    "step": steps,
                    "value": array[idx].tolist(),
                }
            )
        )

    if not frames:
        return pl.DataFrame(schema=_MEDIAN_CURVE_SCHEMA)
    return pl.concat(frames).sort(["condition_name", "step"])


def tolerance_bands(
    df: pl.DataFrame,
    metric: str,
    *,
    confidence: float = 0.95,
    coverage: float = 0.90,
) -> pl.DataFrame:
    """Non-parametric tolerance interval bounds per condition.

    Returns a DataFrame with columns: ``condition_name | step | low | high``.

    Raises:
        ValueError: If any condition has insufficient seeds for the requested
            confidence and coverage levels, listing all failing conditions.
    """
    conditions = _conditions_with_metric(df, metric)

    # Validate all conditions before computing (fail fast, report all failures at once).
    insufficient = []
    for condition in conditions:
        _, array = _condition_curve_array(df, metric, condition)
        try:
            tolerance_interval_order_indices(len(array), confidence=confidence, coverage=coverage)
        except ValueError:
            insufficient.append(f"{condition!r} ({len(array)} seeds)")

    if insufficient:
        raise ValueError(
            f"Insufficient seeds for {confidence:.0%} confidence / {coverage:.0%} coverage "
            f"tolerance interval: {', '.join(insufficient)}."
        )

    frames: list[pl.DataFrame] = []
    for condition in conditions:
        steps, array = _condition_curve_array(df, metric, condition)
        ti = pointwise_tolerance_interval(array, confidence=confidence, coverage=coverage)
        frames.append(
            pl.DataFrame(
                {
                    "condition_name": [condition] * len(steps),
                    "step": steps,
                    "low": ti.ci_low.tolist(),
                    "high": ti.ci_high.tolist(),
                }
            )
        )

    if not frames:
        return pl.DataFrame(schema=_TOLERANCE_SCHEMA)
    return pl.concat(frames).sort(["condition_name", "step"])


def event_response(
    df: pl.DataFrame,
    event_metric: str,
    outcome_metric: str,
    *,
    event_threshold: float = 0.5,
    recovery_fraction: float = 0.90,
) -> pl.DataFrame:
    """Detect the first event crossing per condition and measure outcome drop and recovery.

    The event step is the first step where the mean ``event_metric`` crosses
    ``event_threshold``. Conditions with no crossing are omitted.

    Args:
        df: Tidy DataFrame (``condition_name | seed | step | metric | value``).
        event_metric: Metric signalling the event (e.g. ``"mask_active"``).
        outcome_metric: Metric to measure impact on (e.g. ``"returned_episode_returns"``).
        event_threshold: Threshold above which ``event_metric`` signals the event.
        recovery_fraction: Fraction of the drop to consider recovered. Default 0.90.

    Returns:
        DataFrame with columns:
        ``condition_name | event_step | pre_event_value | drop | recovery_slope | steps_to_recovery``.

        ``steps_to_recovery`` (from the post-event minimum) and ``recovery_slope`` (value per
        step) are in units of the ``step`` column, so subsampled logging does not rescale them.
        ``steps_to_recovery`` is ``null`` if the curve did not reach the recovery target
        within the observed window.

        Conditions with no event crossing are omitted from the result.
    """
    event_conditions = set(_conditions_with_metric(df, event_metric))
    outcome_conditions = set(_conditions_with_metric(df, outcome_metric))
    conditions = sorted(event_conditions & outcome_conditions)

    rows: list[dict[str, object]] = []
    for condition in conditions:
        event_steps, event_array = _condition_curve_array(df, event_metric, condition)
        event_mean = np.mean(event_array, axis=0)

        crossing_indices = [i for i, v in enumerate(event_mean) if v > event_threshold]
        if not crossing_indices:
            continue
        event_idx = crossing_indices[0]
        event_step = event_steps[event_idx]

        outcome_steps, outcome_array = _condition_curve_array(df, outcome_metric, condition)
        outcome_mean = np.mean(outcome_array, axis=0)

        # Align event index to outcome step index (steps may differ in length or offset).
        step_to_outcome_idx = {step: idx for idx, step in enumerate(outcome_steps)}
        if event_step not in step_to_outcome_idx:
            continue
        outcome_event_idx = step_to_outcome_idx[event_step]

        pre_event_value = float(outcome_mean[outcome_event_idx - 1]) if outcome_event_idx > 0 else float(outcome_mean[0])
        post_curve = outcome_mean[outcome_event_idx:]
        if len(post_curve) == 0:
            continue

        post_min = float(np.min(post_curve))
        post_min_offset = int(np.argmin(post_curve))
        drop = max(0.0, pre_event_value - post_min)

        if drop > 0.0:
            recovery_target = post_min + recovery_fraction * drop
            remaining = post_curve[post_min_offset:]
            recovery_offset = next(
                (i for i, v in enumerate(remaining) if float(v) >= recovery_target),
                None,
            )
            post_steps = outcome_steps[outcome_event_idx + post_min_offset :]
            steps_to_recovery: int | None = (
                None if recovery_offset is None else post_steps[recovery_offset] - post_steps[0]
            )
            span = post_steps[-1] - post_steps[0]
            recovery_slope = float(post_curve[-1] - post_min) / span if span > 0 else 0.0
        else:
            steps_to_recovery = 0
            recovery_slope = 0.0

        rows.append(
            {
                "condition_name": condition,
                "event_step": event_step,
                "pre_event_value": pre_event_value,
                "drop": drop,
                "recovery_slope": recovery_slope,
                "steps_to_recovery": steps_to_recovery,
            }
        )

    return pl.DataFrame(rows or None, schema=_EVENT_RESPONSE_SCHEMA)


def flatline_flags(
    df: pl.DataFrame,
    metric: str,
    *,
    tail_fraction: float = 0.5,
    relative_tolerance: float = 0.01,
) -> pl.DataFrame:
    """Flag conditions whose mean learning curve stops moving well before the run ends.

    Compares the last ``tail_fraction`` of the mean curve's own range against the
    whole curve's range. A tail range below ``relative_tolerance`` of the whole
    range means the tail carries almost no additional information relative to the
    rest of the run -- consistent with either early convergence or a stalled run.
    This check cannot tell those two apart; it only flags that the tail is worth a
    look, same as a caller would get from eyeballing the curve. A metric logged at
    fewer than two steps is a scalar, not a curve, and is never flagged.

    Returns a DataFrame with columns: ``condition_name | flatlined | tail_range |
    whole_range``.
    """
    rows: list[dict[str, object]] = []
    for condition in _conditions_with_metric(df, metric):
        steps, array = _condition_curve_array(df, metric, condition)
        curve = np.mean(array, axis=0)
        whole_range = float(np.max(curve) - np.min(curve))
        cutoff = int(len(curve) * (1.0 - tail_fraction))
        tail = curve[cutoff:]
        tail_range = float(np.max(tail) - np.min(tail)) if len(tail) else 0.0
        flatlined = len(steps) >= 2 and (whole_range == 0.0 or (tail_range / whole_range) < relative_tolerance)
        rows.append(
            {
                "condition_name": condition,
                "flatlined": flatlined,
                "tail_range": tail_range,
                "whole_range": whole_range,
            }
        )

    return pl.DataFrame(rows or None, schema=_FLATLINE_SCHEMA).sort("condition_name")
