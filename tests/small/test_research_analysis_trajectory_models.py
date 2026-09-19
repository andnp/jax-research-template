"""Behavior tests for generic structural trajectory models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from research_analysis.trajectory import TrajectorySchema, validate_trajectories
from research_analysis.trajectory_models import (
    FITTED_MODEL_FIELDS,
    ZERO_RSS_BIC,
    PiecewiseKnotFit,
    fit_null_linear_trajectories,
    fit_piecewise_linear_trajectories,
    fit_zero_slope_trajectories,
    predict_trajectory,
    profile_piecewise_fits,
)

_SCHEMA = TrajectorySchema(
    identity=(("scenario", pl.String), ("replicate", pl.Int64)),
    step=("time", pl.Int64),
    metric=("signal", pl.String),
    value=("measurement", pl.Float64),
)


def _trajectory_frame(
    steps: list[int],
    values: list[float],
    *,
    scenario: str = "baseline",
    replicate: int = 3,
    batch: str = "a",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scenario": [scenario] * len(steps),
            "replicate": [replicate] * len(steps),
            "batch": [batch] * len(steps),
            "time": steps,
            "signal": ["reward"] * len(steps),
            "measurement": values,
        }
    )


class TestPredictTrajectory:
    def test_evaluates_null_linear_state(self) -> None:
        """
        Verify a missing knot and tail slope produce a linear trajectory.
        """
        steps = np.arange(4, dtype=np.int64)

        result = predict_trajectory(steps, beta_0=1.0, m_1=2.0)

        assert result.dtype == np.float64
        assert result.shape == steps.shape
        assert np.allclose(result, [1.0, 3.0, 5.0, 7.0])

    def test_evaluates_continuous_piecewise_state(self) -> None:
        """
        Verify both slopes join continuously at the supplied knot.
        """
        steps = np.arange(5, dtype=np.int64)

        result = predict_trajectory(steps, beta_0=2.0, m_1=1.0, tau=2, m_2=-1.0)

        assert np.allclose(result, [2.0, 3.0, 4.0, 3.0, 2.0])

    def test_evaluates_zero_slope_state(self) -> None:
        """
        Verify an explicit zero tail slope produces a stationary plateau.
        """
        steps = np.arange(5, dtype=np.int64)

        result = predict_trajectory(steps, beta_0=2.0, m_1=1.0, tau=2, m_2=0.0)

        assert np.allclose(result, [2.0, 3.0, 4.0, 4.0, 4.0])

    def test_rejects_incomplete_piecewise_states(self) -> None:
        """
        Verify knot and tail-slope parameters must be supplied together.
        """
        steps = np.arange(3, dtype=np.int64)

        with pytest.raises(ValueError, match="requires a piecewise-linear knot"):
            predict_trajectory(steps, beta_0=0.0, m_1=1.0, m_2=0.0)
        with pytest.raises(ValueError, match="requires m_2"):
            predict_trajectory(steps, beta_0=0.0, m_1=1.0, tau=1)


class TestNullLinearFit:
    def test_recovers_known_line(self) -> None:
        """
        Verify the null fit recovers a deterministic intercept and slope.
        """
        steps = [0, 2, 4, 6]
        frame = _trajectory_frame(steps, [3.0 + 2.0 * step for step in steps])

        result = fit_null_linear_trajectories(frame, "reward", schema=_SCHEMA)
        row = result.row(0, named=True)

        assert row["model"] == "null_linear"
        assert row["beta_0"] == pytest.approx(3.0)
        assert row["m_1"] == pytest.approx(2.0)
        assert row["rss"] == pytest.approx(0.0)
        assert row["n_points"] == 4
        assert row["n_parameters"] == 2

    def test_preserves_identity_columns_and_state_fields(self) -> None:
        """
        Verify dynamic identities survive fitting and null fields stay absent.
        """
        frame = _trajectory_frame([0, 1, 2], [1.0, 2.0, 3.0], replicate=7, batch="b")

        result = fit_null_linear_trajectories(frame, "reward", schema=_SCHEMA)
        row = result.row(0, named=True)

        assert result.columns == [
            "scenario",
            "replicate",
            "batch",
            *[column for column, _ in FITTED_MODEL_FIELDS],
        ]
        assert row["replicate"] == 7
        assert row["batch"] == "b"
        assert row["tau"] is None
        assert row["y_at_tau"] is None
        assert row["m_2"] is None
        assert result.schema["tau"] == pl.Int64

    def test_uses_finite_limiting_bic_for_zero_rss(self) -> None:
        """
        Verify exact fits use a finite BIC limit instead of NaN or infinity.
        """
        frame = _trajectory_frame([0, 1, 2], [4.0, 4.0, 4.0])

        result = fit_null_linear_trajectories(frame, "reward", schema=_SCHEMA)
        bic = result["bic"][0]

        assert bic == ZERO_RSS_BIC
        assert np.isfinite(bic)

    def test_rejects_invalid_and_underdetermined_trajectories(self) -> None:
        """
        Verify validation rejects duplicate steps and fitting rejects zero residual degrees.
        """
        invalid = _trajectory_frame([0, 1, 1], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="unique"):
            fit_null_linear_trajectories(invalid, "reward", schema=_SCHEMA)

        short = _trajectory_frame([0, 1], [1.0, 2.0])
        assert validate_trajectories(short, "reward", schema=_SCHEMA).is_valid
        with pytest.raises(ValueError, match="at least 3 observations"):
            fit_null_linear_trajectories(short, "reward", schema=_SCHEMA)


class TestPiecewiseLinearFit:
    def test_recovers_exact_grid_knot(self) -> None:
        """
        Verify profile least squares recovers a known continuous knot and slopes.
        """
        steps = np.arange(10, dtype=np.int64)
        values = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=4, m_2=-0.1)

        result = fit_piecewise_linear_trajectories(
            _trajectory_frame(steps.tolist(), values.tolist()),
            "reward",
            schema=_SCHEMA,
        )
        row = result.row(0, named=True)

        assert row["model"] == "piecewise_linear"
        assert row["beta_0"] == pytest.approx(2.0, abs=1e-10)
        assert row["m_1"] == pytest.approx(0.5, abs=1e-10)
        assert row["tau"] == 4
        assert row["y_at_tau"] == pytest.approx(4.0, abs=1e-10)
        assert row["m_2"] == pytest.approx(-0.1, abs=1e-10)
        assert row["n_parameters"] == 4
        assert row["rss"] == pytest.approx(0.0, abs=1e-12)

    def test_preserves_dynamic_identity_columns(self) -> None:
        """
        Verify one fitted row is returned for each complete trajectory identity.
        """
        first = _trajectory_frame(list(range(10)), [float(step) for step in range(10)])
        second = _trajectory_frame(
            list(range(10)),
            [float(2 * step) for step in range(10)],
            scenario="candidate",
            replicate=4,
            batch="b",
        )

        result = fit_piecewise_linear_trajectories(
            pl.concat([first, second]),
            "reward",
            schema=_SCHEMA,
        )

        assert result.height == 2
        assert set(result["scenario"].to_list()) == {"baseline", "candidate"}
        assert set(result["replicate"].to_list()) == {3, 4}

    def test_profile_returns_named_candidate_results(self) -> None:
        """
        Verify the public profile helper exposes valid knots and named fit values.
        """
        steps = np.arange(10, dtype=np.int64)
        values = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=4, m_2=-0.1)

        knots, fits = profile_piecewise_fits(steps, values)

        assert knots.tolist() == list(range(3, 7))
        assert isinstance(fits[1], PiecewiseKnotFit)
        assert fits[1].rss >= 0.0
        assert fits[1].y_at_tau == pytest.approx(fits[1].beta_0 + fits[1].m_1 * knots[1])

    def test_excludes_knots_outside_time_margin(self) -> None:
        """
        Verify min_margin uses the observed time range rather than checkpoint count.
        """
        steps = np.arange(10, dtype=np.int64)
        values = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=1, m_2=-0.1)

        result = fit_piecewise_linear_trajectories(
            _trajectory_frame(steps.tolist(), values.tolist()),
            "reward",
            schema=_SCHEMA,
            min_margin=0.2,
        )

        assert result["tau"][0] in range(2, 8)

    def test_resolves_equal_profiles_to_earliest_knot(self) -> None:
        """
        Verify a flat trajectory uses the earliest eligible profile knot.
        """
        frame = _trajectory_frame(list(range(10)), [4.0] * 10)

        result = fit_piecewise_linear_trajectories(frame, "reward", schema=_SCHEMA)

        assert result["tau"][0] == 3

    def test_rejects_invalid_candidate_settings_and_degrees_of_freedom(self) -> None:
        """
        Verify candidate settings and residual degrees of freedom fail explicitly.
        """
        frame = _trajectory_frame(list(range(5)), [float(step) for step in range(5)])
        with pytest.raises(ValueError, match="min_margin"):
            fit_piecewise_linear_trajectories(frame, "reward", schema=_SCHEMA, min_margin=0.5)
        with pytest.raises(ValueError, match="more observations than 4 parameters"):
            fit_piecewise_linear_trajectories(
                _trajectory_frame(list(range(4)), [1.0, 2.0, 2.0, 2.0]),
                "reward",
                schema=_SCHEMA,
                min_points_per_segment=1,
            )

    def test_accepts_one_point_per_segment_with_positive_degrees_of_freedom(self) -> None:
        """
        Verify one observation per segment remains valid above four parameters.
        """
        steps = np.arange(5, dtype=np.int64)
        values = predict_trajectory(steps, beta_0=1.0, m_1=0.5, tau=2, m_2=-0.1)

        result = fit_piecewise_linear_trajectories(
            _trajectory_frame(steps.tolist(), values.tolist()),
            "reward",
            schema=_SCHEMA,
            min_points_per_segment=1,
        )

        assert np.isfinite(result["sigma_sq"][0])


class TestZeroSlopeFit:
    def test_resolves_equal_profiles_to_earliest_knot(self) -> None:
        """
        Verify a flat trajectory uses the earliest eligible constrained knot.
        """
        frame = _trajectory_frame(list(range(10)), [4.0] * 10)

        result = fit_zero_slope_trajectories(frame, "reward", schema=_SCHEMA)

        assert result["tau"][0] == 3

    def test_recovers_exact_plateau(self) -> None:
        """
        Verify the constrained profile fit recovers a stationary tail.
        """
        steps = np.arange(10, dtype=np.int64)
        values = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=4, m_2=0.0)

        result = fit_zero_slope_trajectories(
            _trajectory_frame(steps.tolist(), values.tolist()),
            "reward",
            schema=_SCHEMA,
        )
        row = result.row(0, named=True)

        assert row["model"] == "zero_slope"
        assert row["beta_0"] == pytest.approx(2.0, abs=1e-10)
        assert row["m_1"] == pytest.approx(0.5, abs=1e-10)
        assert row["tau"] == 4
        assert row["y_at_tau"] == pytest.approx(4.0, abs=1e-10)
        assert row["m_2"] == 0.0
        assert row["n_parameters"] == 3
        assert row["rss"] == pytest.approx(0.0, abs=1e-12)

    def test_rejects_missing_candidates_and_degrees_of_freedom(self) -> None:
        """
        Verify the constrained fit rejects grids without valid candidates or residual degrees.
        """
        with pytest.raises(ValueError, match="sufficient segment points"):
            fit_zero_slope_trajectories(
                _trajectory_frame(list(range(5)), [float(step) for step in range(5)]),
                "reward",
                schema=_SCHEMA,
            )
        with pytest.raises(ValueError, match="more observations than 3 parameters"):
            fit_zero_slope_trajectories(
                _trajectory_frame([0, 1, 2], [1.0, 2.0, 2.0]),
                "reward",
                schema=_SCHEMA,
                min_points_per_segment=1,
            )


class TestStructuralFitCalibration:
    def test_recovers_seeded_noisy_models(self) -> None:
        """
        Verify deterministic noise still recovers known piecewise and plateau parameters.
        """
        steps = np.arange(20, dtype=np.int64)
        piecewise_truth = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=8, m_2=-0.1)
        plateau_truth = predict_trajectory(steps, beta_0=2.0, m_1=0.5, tau=8, m_2=0.0)

        piecewise_values = piecewise_truth + np.random.default_rng(42).normal(0.0, 0.02, size=steps.size)
        plateau_values = plateau_truth + np.random.default_rng(17).normal(0.0, 0.02, size=steps.size)
        piecewise = fit_piecewise_linear_trajectories(
            _trajectory_frame(steps.tolist(), piecewise_values.tolist()),
            "reward",
            schema=_SCHEMA,
        ).row(0, named=True)
        plateau = fit_zero_slope_trajectories(
            _trajectory_frame(steps.tolist(), plateau_values.tolist()),
            "reward",
            schema=_SCHEMA,
        ).row(0, named=True)

        assert piecewise["tau"] == 8
        assert piecewise["m_1"] == pytest.approx(0.5, abs=0.05)
        assert piecewise["m_2"] == pytest.approx(-0.1, abs=0.05)
        assert plateau["tau"] == 8
        assert plateau["m_1"] == pytest.approx(0.5, abs=0.02)
        assert plateau["m_2"] == 0.0

    @pytest.mark.parametrize(
        "fit",
        [fit_null_linear_trajectories, fit_piecewise_linear_trajectories, fit_zero_slope_trajectories],
    )
    def test_zero_rss_bic_is_finite_for_each_model(
        self,
        fit: Callable[..., pl.DataFrame],
    ) -> None:
        """
        Verify every structural fit uses the same finite zero-RSS BIC limit.
        """
        frame = _trajectory_frame(list(range(10)), [4.0] * 10)

        result = fit(frame, "reward", schema=_SCHEMA)
        bic = result["bic"][0]

        assert bic == ZERO_RSS_BIC
        assert np.isfinite(bic)

    def test_rejects_irregular_grid_input(self) -> None:
        """
        Verify structural fitting preserves the uniform-step input contract.
        """
        frame = _trajectory_frame([0, 1, 3, 4, 5, 6, 7, 8], [0.0] * 8)

        with pytest.raises(ValueError, match="uniformly spaced"):
            fit_piecewise_linear_trajectories(frame, "reward", schema=_SCHEMA)
