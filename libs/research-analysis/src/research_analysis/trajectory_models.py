"""Generic numerical models for tidy trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from research_analysis.trajectory import TrajectorySchema, require_valid_trajectories

FITTED_MODEL_FIELDS: Final[tuple[tuple[str, type[pl.DataType]], ...]] = (
    ("model", pl.String),
    ("beta_0", pl.Float64),
    ("m_1", pl.Float64),
    ("tau", pl.Int64),
    ("y_at_tau", pl.Float64),
    ("m_2", pl.Float64),
    ("sigma_sq", pl.Float64),
    ("rss", pl.Float64),
    ("n_points", pl.Int64),
    ("n_parameters", pl.Int64),
    ("bic", pl.Float64),
)
ZERO_RSS_BIC: Final[float] = -float(np.finfo(np.float64).max)


def _bic(n_points: int, rss: float, n_parameters: int, *, value_scale: float = 1.0) -> float:
    zero_tolerance = 100.0 * np.finfo(np.float64).eps * max(1.0, value_scale)
    if rss <= zero_tolerance:
        return ZERO_RSS_BIC
    return n_points * math.log(rss / n_points) + n_parameters * math.log(n_points)


def _rss_is_lower(candidate: float, current: float) -> bool:
    tolerance = 1e-12 * max(1.0, abs(candidate), abs(current))
    return candidate < current and not np.isclose(candidate, current, rtol=1e-12, atol=tolerance)


def _fit_linear(step: NDArray[np.int64], value: NDArray[np.float64]) -> tuple[float, float, float]:
    time = np.asarray(step, dtype=np.float64)
    center = (float(np.min(time)) + float(np.max(time))) / 2.0
    scale = float(np.max(np.abs(time - center)))
    scaled_time = (time - center) / scale
    design = np.column_stack((np.ones(time.size), scaled_time))
    coefficients, _, rank, _ = np.linalg.lstsq(design, value, rcond=None)
    if rank < 2:
        raise ValueError("linear trajectory design is rank deficient")
    intercept_scaled, slope_scaled = coefficients
    slope = float(slope_scaled / scale)
    intercept = float(intercept_scaled - slope_scaled * center / scale)
    fitted = intercept + slope * time
    rss = float(np.dot(value - fitted, value - fitted))
    return intercept, slope, max(0.0, rss)


def _fitted_schema(
    frame: pl.DataFrame,
    identity_columns: tuple[str, ...],
) -> dict[str, type[pl.DataType] | pl.DataType]:
    return {
        **{column: frame.schema[column] for column in identity_columns},
        **dict(FITTED_MODEL_FIELDS),
    }


def _require_residual_degrees_of_freedom(n_points: int, n_parameters: int, model: str) -> None:
    if n_points <= n_parameters:
        raise ValueError(
            f"{model} fit requires more observations than {n_parameters} parameters "
            f"(got {n_points} observations)"
        )


def fit_null_linear_trajectories(
    frame: pl.DataFrame,
    metric: str,
    *,
    schema: TrajectorySchema,
) -> pl.DataFrame:
    """Fit straight-line unsegmented null models per trajectory."""
    report = require_valid_trajectories(frame, metric, schema=schema)
    identity_columns = report.identity_columns
    selected = frame.filter(pl.col(schema.metric[0]) == metric)
    rows: list[dict[str, object]] = []
    for group in selected.partition_by(list(identity_columns), maintain_order=True):
        step = group.get_column(schema.step[0]).to_numpy()
        value = group.get_column(schema.value[0]).to_numpy()
        if len(step) < 3:
            raise ValueError("null linear fit requires at least 3 observations per trajectory")
        beta_0, m_1, rss = _fit_linear(step, value)
        n_points = len(step)
        identity_values = group.select(list(identity_columns)).row(0)
        row = dict(zip(identity_columns, identity_values, strict=True))
        row.update(
            {
                "model": "null_linear",
                "beta_0": beta_0,
                "m_1": m_1,
                "tau": None,
                "y_at_tau": None,
                "m_2": None,
                "sigma_sq": rss / (n_points - 2),
                "rss": rss,
                "n_points": n_points,
                "n_parameters": 2,
                "bic": _bic(n_points, rss, 2, value_scale=float(np.dot(value, value))),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_fitted_schema(frame, identity_columns))


def _candidate_knots(
    step: NDArray[np.int64],
    *,
    min_margin: float,
    min_points_per_segment: int,
) -> NDArray[np.int64]:
    if not 0.0 <= min_margin < 0.5:
        raise ValueError("min_margin must be in [0.0, 0.5)")
    if min_points_per_segment < 1:
        raise ValueError("min_points_per_segment must be positive")

    time = np.asarray(step, dtype=np.float64)
    lower = float(np.min(time))
    upper = float(np.max(time))
    margin = min_margin * (upper - lower)
    lower_bound = lower + margin
    upper_bound = upper - margin
    boundary_tolerance = 8.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(lower_bound),
        abs(upper_bound),
        abs(upper - lower),
    )
    candidates = step[
        (time >= lower_bound - boundary_tolerance)
        & (time <= upper_bound + boundary_tolerance)
    ]
    valid = [
        int(knot)
        for knot in candidates
        if np.count_nonzero(time < knot) >= min_points_per_segment
        and np.count_nonzero(time > knot) >= min_points_per_segment
    ]
    if not valid:
        raise ValueError("trajectory has no knot candidates with sufficient segment points")
    return np.asarray(valid, dtype=np.int64)


@dataclass(frozen=True)
class _ProfileMoments:
    step: NDArray[np.int64]
    center: float
    scale: float
    prefix_time: NDArray[np.float64]
    prefix_time_sq: NDArray[np.float64]
    prefix_value: NDArray[np.float64]
    prefix_time_value: NDArray[np.float64]
    prefix_value_sq: NDArray[np.float64]


@dataclass(frozen=True)
class PiecewiseKnotFit:
    """Numerical fit for one candidate knot in a continuous piecewise model."""

    beta_0: float
    m_1: float
    m_2: float
    rss: float
    y_at_tau: float


def _profile_moments(step: NDArray[np.int64], value: NDArray[np.float64]) -> _ProfileMoments:
    time = np.asarray(step, dtype=np.float64)
    center = (float(np.min(time)) + float(np.max(time))) / 2.0
    scale = float(np.max(np.abs(time - center)))
    scaled_time = (time - center) / scale

    def prefix(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate((np.array([0.0]), np.cumsum(values, dtype=np.float64)))

    return _ProfileMoments(
        step=step,
        center=center,
        scale=scale,
        prefix_time=prefix(scaled_time),
        prefix_time_sq=prefix(scaled_time * scaled_time),
        prefix_value=prefix(value),
        prefix_time_value=prefix(scaled_time * value),
        prefix_value_sq=prefix(value * value),
    )


def _solve_profile(
    gram: NDArray[np.float64],
    response: NDArray[np.float64],
    model: str,
) -> NDArray[np.float64]:
    if np.linalg.matrix_rank(gram) < response.size:
        raise ValueError(f"{model} trajectory design is rank deficient")
    try:
        return np.linalg.solve(gram, response)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{model} trajectory design solve failed") from error


def _fit_piecewise_at_knot(moments: _ProfileMoments, tau: int) -> PiecewiseKnotFit:
    n_points = moments.step.size
    right_start = int(np.searchsorted(moments.step, tau, side="right"))
    right_count = n_points - right_start
    total_time = moments.prefix_time[-1]
    total_time_sq = moments.prefix_time_sq[-1]
    total_value = moments.prefix_value[-1]
    total_time_value = moments.prefix_time_value[-1]
    right_time = total_time - moments.prefix_time[right_start]
    right_time_sq = total_time_sq - moments.prefix_time_sq[right_start]
    right_value = total_value - moments.prefix_value[right_start]
    right_time_value = total_time_value - moments.prefix_time_value[right_start]
    scaled_tau = (tau - moments.center) / moments.scale
    sum_hinge = right_time - scaled_tau * right_count
    sum_time_hinge = right_time_sq - scaled_tau * right_time
    sum_hinge_sq = right_time_sq - 2.0 * scaled_tau * right_time + scaled_tau**2 * right_count
    sum_value_hinge = right_time_value - scaled_tau * right_value
    gram = np.array(
        [
            [n_points, total_time, sum_hinge],
            [total_time, total_time_sq, sum_time_hinge],
            [sum_hinge, sum_time_hinge, sum_hinge_sq],
        ],
        dtype=np.float64,
    )
    response = np.array([total_value, total_time_value, sum_value_hinge], dtype=np.float64)
    coefficients = _solve_profile(gram, response, "piecewise")
    intercept_scaled, slope_scaled, delta_scaled = coefficients
    m_1 = float(slope_scaled / moments.scale)
    delta = float(delta_scaled / moments.scale)
    m_2 = m_1 + delta
    beta_0 = float(intercept_scaled - slope_scaled * moments.center / moments.scale)
    rss = float(moments.prefix_value_sq[-1] - np.dot(coefficients, response))
    return PiecewiseKnotFit(
        beta_0=beta_0,
        m_1=m_1,
        m_2=m_2,
        rss=max(0.0, rss),
        y_at_tau=beta_0 + m_1 * tau,
    )


def profile_piecewise_fits(
    step: NDArray[np.int64],
    value: NDArray[np.float64],
    *,
    min_margin: float = 0.05,
    min_points_per_segment: int = 3,
) -> tuple[NDArray[np.int64], tuple[PiecewiseKnotFit, ...]]:
    """Evaluate continuous piecewise fits over the valid candidate knot grid."""
    moments = _profile_moments(step, value)
    knots = _candidate_knots(
        step,
        min_margin=min_margin,
        min_points_per_segment=min_points_per_segment,
    )
    fits = tuple(_fit_piecewise_at_knot(moments, int(knot)) for knot in knots)
    return knots, fits


def fit_piecewise_linear_trajectories(
    frame: pl.DataFrame,
    metric: str,
    *,
    schema: TrajectorySchema,
    min_margin: float = 0.05,
    min_points_per_segment: int = 3,
) -> pl.DataFrame:
    """Fit unconstrained single-knot piecewise-linear models per trajectory."""
    report = require_valid_trajectories(frame, metric, schema=schema)
    identity_columns = report.identity_columns
    selected = frame.filter(pl.col(schema.metric[0]) == metric)
    rows: list[dict[str, object]] = []
    for group in selected.partition_by(list(identity_columns), maintain_order=True):
        step = group.get_column(schema.step[0]).to_numpy()
        value = group.get_column(schema.value[0]).to_numpy()
        n_points = len(step)
        _require_residual_degrees_of_freedom(n_points, 4, "piecewise linear")
        knots, fits = profile_piecewise_fits(
            step,
            value,
            min_margin=min_margin,
            min_points_per_segment=min_points_per_segment,
        )
        best_index = 0
        for index, fit in enumerate(fits[1:], start=1):
            if _rss_is_lower(fit.rss, fits[best_index].rss):
                best_index = index
        tau = int(knots[best_index])
        fit = fits[best_index]
        identity_values = group.select(list(identity_columns)).row(0)
        row = dict(zip(identity_columns, identity_values, strict=True))
        row.update(
            {
                "model": "piecewise_linear",
                "beta_0": fit.beta_0,
                "m_1": fit.m_1,
                "tau": tau,
                "y_at_tau": fit.y_at_tau,
                "m_2": fit.m_2,
                "sigma_sq": fit.rss / (n_points - 4),
                "rss": fit.rss,
                "n_points": n_points,
                "n_parameters": 4,
                "bic": _bic(n_points, fit.rss, 4, value_scale=float(np.dot(value, value))),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_fitted_schema(frame, identity_columns))


def _fit_zero_slope_at_knot(
    moments: _ProfileMoments,
    tau: int,
) -> tuple[float, float, float, float]:
    left_end = int(np.searchsorted(moments.step, tau, side="right"))
    right_count = moments.step.size - left_end
    scaled_tau = (tau - moments.center) / moments.scale
    total_value = moments.prefix_value[-1]
    left_time = moments.prefix_time[left_end]
    left_time_sq = moments.prefix_time_sq[left_end]
    left_value = moments.prefix_value[left_end]
    left_time_value = moments.prefix_time_value[left_end]
    right_value = total_value - left_value
    sum_plateau = left_time + scaled_tau * right_count
    sum_plateau_sq = left_time_sq + scaled_tau**2 * right_count
    sum_value_plateau = left_time_value + scaled_tau * right_value
    gram = np.array(
        [
            [moments.step.size, sum_plateau],
            [sum_plateau, sum_plateau_sq],
        ],
        dtype=np.float64,
    )
    response = np.array([total_value, sum_value_plateau], dtype=np.float64)
    coefficients = _solve_profile(gram, response, "zero-slope")
    intercept_scaled, slope_scaled = coefficients
    m_1 = float(slope_scaled / moments.scale)
    beta_0 = float(intercept_scaled - m_1 * moments.center)
    rss = float(moments.prefix_value_sq[-1] - np.dot(coefficients, response))
    return beta_0, m_1, max(0.0, rss), beta_0 + m_1 * tau


def fit_zero_slope_trajectories(
    frame: pl.DataFrame,
    metric: str,
    *,
    schema: TrajectorySchema,
    min_margin: float = 0.05,
    min_points_per_segment: int = 3,
) -> pl.DataFrame:
    """Fit zero-slope asymptotic plateau models per trajectory."""
    report = require_valid_trajectories(frame, metric, schema=schema)
    identity_columns = report.identity_columns
    selected = frame.filter(pl.col(schema.metric[0]) == metric)
    rows: list[dict[str, object]] = []
    for group in selected.partition_by(list(identity_columns), maintain_order=True):
        step = group.get_column(schema.step[0]).to_numpy()
        value = group.get_column(schema.value[0]).to_numpy()
        n_points = len(step)
        _require_residual_degrees_of_freedom(n_points, 3, "zero-slope")
        moments = _profile_moments(step, value)
        knots = _candidate_knots(
            step,
            min_margin=min_margin,
            min_points_per_segment=min_points_per_segment,
        )
        fits = [_fit_zero_slope_at_knot(moments, int(knot)) for knot in knots]
        best_index = 0
        for index, fit in enumerate(fits[1:], start=1):
            if _rss_is_lower(fit[2], fits[best_index][2]):
                best_index = index
        tau = int(knots[best_index])
        beta_0, m_1, rss, y_at_tau = fits[best_index]
        identity_values = group.select(list(identity_columns)).row(0)
        row = dict(zip(identity_columns, identity_values, strict=True))
        row.update(
            {
                "model": "zero_slope",
                "beta_0": beta_0,
                "m_1": m_1,
                "tau": tau,
                "y_at_tau": y_at_tau,
                "m_2": 0.0,
                "sigma_sq": rss / (n_points - 3),
                "rss": rss,
                "n_points": n_points,
                "n_parameters": 3,
                "bic": _bic(n_points, rss, 3, value_scale=float(np.dot(value, value))),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_fitted_schema(frame, identity_columns))


def predict_trajectory(
    steps: NDArray[np.int64],
    *,
    beta_0: float,
    m_1: float,
    tau: float | None = None,
    m_2: float | None = None,
) -> NDArray[np.float64]:
    """Evaluate a fitted linear or continuous piecewise-linear trajectory."""
    if tau is None:
        if m_2 is not None:
            raise ValueError("m_2 requires a piecewise-linear knot")
        return np.asarray(beta_0 + m_1 * np.asarray(steps, dtype=np.float64), dtype=np.float64)

    if m_2 is None:
        raise ValueError("a piecewise-linear knot requires m_2")
    if isinstance(tau, bool) or not math.isfinite(tau):
        raise ValueError("tau must be finite")

    time = np.asarray(steps, dtype=np.float64)
    hinge = np.maximum(time - tau, 0.0)
    return np.asarray(beta_0 + m_1 * time + (m_2 - m_1) * hinge, dtype=np.float64)
