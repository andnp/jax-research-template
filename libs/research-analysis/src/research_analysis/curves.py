"""Analytical aggregation utilities for RL training curves.

Operates on the tidy ``condition_name | seed | step | metric | value`` DataFrame
produced by :func:`research_analysis.experiment.load_experiment_metrics`.

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
from research_analysis.statistics import (
    pointwise_tolerance_interval,
    select_median_run_index,
    tolerance_interval_order_indices,
)

_FINAL_VALUES_SCHEMA: dict[str, type[pl.DataType]] = {
    "condition_name": pl.String,
    "seed": pl.Int64,
    "value": pl.Float64,
}
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


# ── Internal helpers ──────────────────────────────────────────────────────────


def _condition_curve_array(
    df: pl.DataFrame, metric: str, condition: str
) -> tuple[list[int], NDArray[np.float64]]:
    """Return (sorted_steps, array) where array has shape (n_seeds, n_steps)."""
    subset = df.filter(
        (pl.col("metric") == metric) & (pl.col("condition_name") == condition)
    ).sort(["seed", "step"])

    seeds = subset["seed"].unique().sort().to_list()
    steps = subset.filter(pl.col("seed") == seeds[0]).sort("step")["step"].to_list()

    rows = [
        subset.filter(pl.col("seed") == seed).sort("step")["value"].to_numpy()
        for seed in seeds
    ]
    return steps, np.array(rows, dtype=np.float64)


def _conditions_with_metric(df: pl.DataFrame, metric: str) -> list[str]:
    return df.filter(pl.col("metric") == metric)["condition_name"].unique().sort().to_list()


# ── Public API ────────────────────────────────────────────────────────────────


def final_values(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Last recorded value per (condition_name, seed) for a given metric.

    Returns a DataFrame with columns: ``condition_name | seed | value``.
    """
    return (
        df.filter(pl.col("metric") == metric)
        .group_by(["condition_name", "seed"])
        .agg(pl.col("value").last())
        .sort(["condition_name", "seed"])
    )


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
    frames: list[pl.DataFrame] = []
    for condition in _conditions_with_metric(df, metric):
        steps, array = _condition_curve_array(df, metric, condition)
        ci = bootstrap_ci(
            array,
            confidence=confidence,
            n_resamples=n_resamples,
            rng=np.random.default_rng(random_seed),
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
    frames: list[pl.DataFrame] = []
    for condition in _conditions_with_metric(df, metric):
        steps, array = _condition_curve_array(df, metric, condition)
        idx = select_median_run_index(array)
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
            steps_to_recovery: int | None = None if recovery_offset is None else recovery_offset
            denom = len(post_curve) - post_min_offset - 1
            recovery_slope = float(post_curve[-1] - post_min) / denom if denom > 0 else 0.0
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
