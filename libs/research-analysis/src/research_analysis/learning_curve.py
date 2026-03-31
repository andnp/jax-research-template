"""Learning-curve primitives for step-weighted RL analysis.

This module implements the copy-forward interpolation described in the
science guide: episodic returns are converted into a per-step curve by
assigning each episode's return to every step until the next episode ends.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars.exceptions import InvalidOperationError


def _as_step_array(cumulative_steps: NDArray[np.integer]) -> NDArray[np.int64]:
    steps = np.asarray(cumulative_steps, dtype=np.int64)
    if steps.ndim != 1:
        raise ValueError("cumulative_steps must be a 1-D array")
    if steps.size == 0:
        raise ValueError("cumulative_steps must not be empty")
    if np.any(steps <= 0):
        raise ValueError("cumulative_steps must be strictly positive")

    step_differences = np.diff(steps)
    if np.any(step_differences == 0):
        raise ValueError("cumulative_steps must not contain duplicates")
    if np.any(step_differences < 0):
        raise ValueError("cumulative_steps must be strictly increasing")

    return steps


def _as_return_array(episodic_returns: NDArray[np.floating]) -> NDArray[np.float64]:
    returns = np.asarray(episodic_returns, dtype=np.float64)
    if returns.ndim != 1:
        raise ValueError("episodic_returns must be a 1-D array")
    if returns.size == 0:
        raise ValueError("episodic_returns must not be empty")
    return returns


def _require_dataframe(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    return frame


def _require_column(frame: pl.DataFrame, column_name: str) -> pl.Series:
    if column_name not in frame.columns:
        raise ValueError(f"frame must contain column {column_name!r}")
    return frame.get_column(column_name)


def _require_non_null(series: pl.Series, *, column_name: str) -> pl.Series:
    if series.null_count() > 0:
        raise ValueError(f"column {column_name!r} must not contain nulls")
    return series


def _require_integer_steps(series: pl.Series, *, column_name: str) -> NDArray[np.int64]:
    if not series.dtype.is_integer():
        raise TypeError(f"column {column_name!r} must have an integer dtype")
    return series.to_numpy()


def _require_float_returns(series: pl.Series, *, column_name: str) -> NDArray[np.float64]:
    try:
        return series.cast(pl.Float64, strict=True).to_numpy()
    except InvalidOperationError as error:
        raise TypeError(f"column {column_name!r} must be numeric or castable to float64") from error


def step_weighted_returns(
    cumulative_steps: NDArray[np.integer],
    episodic_returns: NDArray[np.floating],
    *,
    end_step: int | None = None,
) -> NDArray[np.float64]:
    """Expand episodic returns into a step-weighted copy-forward curve.

    Args:
        cumulative_steps: Monotonically increasing cumulative episode end
            steps. Each value marks the exclusive upper bound of an episode's
            copy-forward segment.
        episodic_returns: Episodic returns aligned with ``cumulative_steps``.
        end_step: Optional total output length. When provided, the final return
            is copied forward through ``end_step``.

    Returns:
        A ``float64`` array of length ``end_step`` (or the last cumulative
        step when ``end_step`` is omitted) containing per-step returns.

    Raises:
        ValueError: If inputs are empty, misaligned, not strictly positive,
            not strictly increasing, or if ``end_step`` is smaller than the
            last cumulative step.
    """
    steps = _as_step_array(cumulative_steps)
    returns = _as_return_array(episodic_returns)

    if steps.shape[0] != returns.shape[0]:
        raise ValueError("cumulative_steps and episodic_returns must have the same length")

    final_step = int(steps[-1])
    if end_step is None:
        output_length = final_step
    else:
        output_length = int(end_step)
        if output_length < final_step:
            raise ValueError("end_step must be at least the last cumulative step")

    curve = np.empty(output_length, dtype=np.float64)
    start = 0
    for stop, episode_return in zip(steps, returns, strict=True):
        curve[start:stop] = episode_return
        start = int(stop)

    if output_length > final_step:
        curve[final_step:output_length] = returns[-1]

    return curve


def step_weighted_returns_from_dataframe(
    frame: pl.DataFrame,
    *,
    cumulative_steps_column: str,
    episodic_returns_column: str,
    end_step: int | None = None,
) -> NDArray[np.float64]:
    """Validate a Polars frame and delegate to ``step_weighted_returns``.

    Args:
        frame: Source frame containing cumulative step and episodic return data.
        cumulative_steps_column: Column containing cumulative episode end steps.
        episodic_returns_column: Column containing episodic returns.
        end_step: Optional total output length. When provided, the final return
            is copied forward through ``end_step``.

    Raises:
        TypeError: If ``frame`` is not a Polars DataFrame, the step column is
            not integer-typed, or the return column cannot be cast to float64.
        ValueError: If either column is missing or contains null values.
    """
    validated_frame = _require_dataframe(frame)
    step_series = _require_non_null(
        _require_column(validated_frame, cumulative_steps_column),
        column_name=cumulative_steps_column,
    )
    return_series = _require_non_null(
        _require_column(validated_frame, episodic_returns_column),
        column_name=episodic_returns_column,
    )

    return step_weighted_returns(
        _require_integer_steps(step_series, column_name=cumulative_steps_column),
        _require_float_returns(return_series, column_name=episodic_returns_column),
        end_step=end_step,
    )
