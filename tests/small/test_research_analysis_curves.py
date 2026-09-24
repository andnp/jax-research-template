"""Small tests for research_analysis.curves."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from research_analysis.curves import (
    event_response,
    flatline_flags,
    mean_curves,
    median_run_curve,
    quantile_run_curve,
    tolerance_bands,
)
from research_analysis.non_finite import NonFiniteValuesError


def _make_df(
    conditions: list[str],
    seeds: list[int],
    steps: list[int],
    metric: str,
    value_fn: Callable[[str, int, int], float],
) -> pl.DataFrame:
    """Build a tidy metrics DataFrame from a value function f(condition, seed, step)."""
    rows = [
        {
            "condition_name": cond,
            "seed": seed,
            "step": step,
            "metric": metric,
            "value": value_fn(cond, seed, step),
        }
        for cond in conditions
        for seed in seeds
        for step in steps
    ]
    return pl.DataFrame(rows)


class TestNonFiniteCurves:
    def test_mean_curves_reject_a_diverged_seed(self) -> None:
        """A NaN seed must fail the curve instead of being averaged into NaN bands."""
        df = _make_df(["A"], [0, 1, 2], [0, 1], "r", lambda c, s, t: float("nan") if (s, t) == (1, 1) else 1.0)

        with pytest.raises(NonFiniteValuesError) as error:
            mean_curves(df, "r", n_resamples=10)

        assert error.value.offenders["seed"].to_list() == [1]


class TestInterleavedRuns:
    def test_rejects_two_runs_sharing_a_condition_and_seed(self) -> None:
        """Runs from different environments or configs must not be interleaved into one seed's curve."""
        df = _make_df(["A"], [0, 1], [0, 1], "r", lambda c, s, t: float(t)).with_columns(
            run_id=pl.col("seed") + 1
        )
        other_env = df.filter(pl.col("seed") == 0).with_columns(run_id=pl.lit(9, dtype=pl.Int64))

        with pytest.raises(ValueError, match=r"more than one run for seed 0 \(runs \['1', '9'\]\)"):
            mean_curves(pl.concat([df, other_env]), "r", n_resamples=10)

    def test_rejects_repeated_steps_without_run_identity(self) -> None:
        """Without run ids, a repeated (seed, step) pair still reveals pooled runs."""
        df = _make_df(["A"], [0, 1], [0, 1], "r", lambda c, s, t: float(t))

        with pytest.raises(ValueError, match="more than one run for seed 0"):
            mean_curves(pl.concat([df, df.filter(pl.col("seed") == 0)]), "r", n_resamples=10)


class TestMeanCurves:
    def test_returns_correct_schema(self) -> None:
        df = _make_df(["A"], [0, 1, 2], [0, 1, 2], "r", lambda c, s, t: float(t))
        result = mean_curves(df, "r")
        assert set(result.columns) == {"condition_name", "step", "mean", "ci_low", "ci_high"}

    def test_mean_matches_arithmetic_mean(self) -> None:
        # seed 0: t, seed 1: t+10 — mean should be t+5
        df = _make_df(["A"], [0, 1], [0, 1, 2], "r", lambda c, s, t: t + s * 10)
        result = mean_curves(df, "r", n_resamples=100, random_seed=42)
        means = result.sort("step")["mean"].to_list()
        assert means == pytest.approx([5.0, 6.0, 7.0], abs=1e-9)

    def test_rejects_nonuniform_seed_curve_lengths(self) -> None:
        """Reject jagged per-seed curves with an explicit alignment error."""
        df = pl.DataFrame(
            [
                {"condition_name": "A", "seed": 0, "step": 0, "metric": "r", "value": 1.0},
                {"condition_name": "A", "seed": 0, "step": 1, "metric": "r", "value": 2.0},
                {"condition_name": "A", "seed": 1, "step": 0, "metric": "r", "value": 3.0},
            ]
        )

        with pytest.raises(ValueError, match="non-uniform steps across seeds.*seed is 1"):
            mean_curves(df, "r", n_resamples=100, random_seed=0)

    def test_ci_bounds_bracket_mean(self) -> None:
        df = _make_df(["A"], [0, 1, 2, 3], [0, 1, 2], "r", lambda c, s, t: t + s * 5)
        result = mean_curves(df, "r", n_resamples=200, random_seed=0)
        assert (result["ci_low"] <= result["mean"]).all()
        assert (result["mean"] <= result["ci_high"]).all()

    def test_raises_with_single_seed(self) -> None:
        df = _make_df(["A"], [0], [0, 1], "r", lambda c, s, t: float(t))
        with pytest.raises(ValueError, match="at least 2 seeds"):
            mean_curves(df, "r")

    def test_multiple_conditions_independent(self) -> None:
        df = _make_df(["A", "B"], [0, 1], [0, 1], "r", lambda c, s, t: 100.0 if c == "A" else 200.0)
        result = mean_curves(df, "r", n_resamples=50, random_seed=0)
        a_mean = result.filter(pl.col("condition_name") == "A")["mean"].mean()
        b_mean = result.filter(pl.col("condition_name") == "B")["mean"].mean()
        assert a_mean == pytest.approx(100.0)
        assert b_mean == pytest.approx(200.0)

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        """Same random_seed must reproduce identical CI bands across full calls."""
        df = _make_df(["A", "B"], [0, 1, 2, 3], [0, 1], "r", lambda c, s, t: t + s * 5)
        first = mean_curves(df, "r", n_resamples=200, random_seed=7)
        second = mean_curves(df, "r", n_resamples=200, random_seed=7)
        assert first.equals(second)

    def test_conditions_draw_distinct_bootstrap_streams(self) -> None:
        """Two conditions with identical values must not get identical CI bands from
        reusing the same RNG state, which would mask each condition's own resampling."""
        df = _make_df(["A", "B"], [0, 1, 2, 3], [0, 1], "r", lambda c, s, t: t + s * 5)
        result = mean_curves(df, "r", n_resamples=200, random_seed=7)
        a_bands = result.filter(pl.col("condition_name") == "A").sort("step")[["ci_low", "ci_high"]]
        b_bands = result.filter(pl.col("condition_name") == "B").sort("step")[["ci_low", "ci_high"]]
        assert not a_bands.equals(b_bands)

    def test_empty_df_returns_empty_frame(self) -> None:
        df = pl.DataFrame(schema={"condition_name": pl.String, "seed": pl.Int64, "step": pl.Int64, "metric": pl.String, "value": pl.Float64})
        result = mean_curves(df, "r")
        assert result.is_empty()


class TestMedianRunCurve:
    def test_returns_correct_schema(self) -> None:
        df = _make_df(["A"], [0, 1, 2], [0, 1], "r", lambda c, s, t: float(s))
        result = median_run_curve(df, "r")
        assert set(result.columns) == {"condition_name", "step", "value"}

    def test_selects_median_seed(self) -> None:
        # seed 0: 1.0, seed 1: 5.0, seed 2: 9.0 — median is seed 1 (value 5)
        df = _make_df(["A"], [0, 1, 2], [0], "r", lambda c, s, t: float(s * 4 + 1))
        result = median_run_curve(df, "r")
        assert result["value"][0] == pytest.approx(5.0)

    def test_multiple_conditions(self) -> None:
        df = _make_df(["A", "B"], [0, 1, 2], [0], "r", lambda c, s, t: 10.0 if c == "A" else 20.0)
        result = median_run_curve(df, "r")
        a_val = result.filter(pl.col("condition_name") == "A")["value"][0]
        b_val = result.filter(pl.col("condition_name") == "B")["value"][0]
        assert a_val == pytest.approx(10.0)
        assert b_val == pytest.approx(20.0)


class TestQuantileRunCurve:
    def test_selects_requested_quantile_seed(self) -> None:
        """Return the curve of the seed nearest the requested performance quantile."""
        df = _make_df(
            ["A"],
            [0, 1, 2],
            [0, 1],
            "r",
            lambda c, s, t: float(s * 4 + 1),
        )

        result = quantile_run_curve(df, "r", quantile=0.1)

        assert result["value"].to_list() == pytest.approx([1.0, 1.0])

    def test_median_curve_matches_quantile_half(self) -> None:
        """Keep the median curve equivalent to the general quantile contract."""
        df = _make_df(["A"], [0, 1, 2], [0, 1], "r", lambda c, s, t: float(s + t))

        assert median_run_curve(df, "r").equals(
            quantile_run_curve(df, "r", quantile=0.5)
        )


class TestToleranceBands:
    def test_returns_correct_schema(self) -> None:
        # Need >= 46 seeds for 95/90 coverage tolerance interval
        df = _make_df(["A"], list(range(46)), [0, 1], "r", lambda c, s, t: float(s + t))
        result = tolerance_bands(df, "r")
        assert set(result.columns) == {"condition_name", "step", "low", "high"}

    def test_bounds_order(self) -> None:
        df = _make_df(["A"], list(range(46)), [0, 1], "r", lambda c, s, t: float(s))
        result = tolerance_bands(df, "r")
        assert (result["low"] <= result["high"]).all()

    def test_raises_with_insufficient_seeds(self) -> None:
        df = _make_df(["A"], [0, 1, 2], [0, 1], "r", lambda c, s, t: float(s))
        with pytest.raises(ValueError, match="Insufficient seeds"):
            tolerance_bands(df, "r")

    def test_raises_lists_all_failing_conditions(self) -> None:
        df = _make_df(["A", "B"], [0, 1], [0], "r", lambda c, s, t: float(s))
        with pytest.raises(ValueError, match="'A'") as exc_info:
            tolerance_bands(df, "r")
        assert "'B'" in str(exc_info.value)


class TestEventResponse:
    def _make_event_df(self, event_step_idx: int, total_steps: int = 20) -> pl.DataFrame:
        """Two seeds, event activates at event_step_idx, outcome drops then recovers."""
        rows = []
        for seed in range(4):
            for step in range(total_steps):
                rows.append({"condition_name": "A", "seed": seed, "step": step, "metric": "mask", "value": 1.0 if step >= event_step_idx else 0.0})
                # Outcome: steady at 100, drops to 60 at event, recovers to 90 by end
                if step < event_step_idx:
                    outcome = 100.0
                elif step == event_step_idx:
                    outcome = 60.0
                else:
                    # Linear recovery from 60 toward 90
                    progress = (step - event_step_idx) / (total_steps - event_step_idx - 1)
                    outcome = 60.0 + progress * 30.0
                rows.append({"condition_name": "A", "seed": seed, "step": step, "metric": "returns", "value": outcome})
        return pl.DataFrame(rows)

    def test_detects_event_step(self) -> None:
        df = self._make_event_df(event_step_idx=5)
        result = event_response(df, "mask", "returns")
        assert result.shape[0] == 1
        assert result["event_step"][0] == 5

    def test_pre_event_value(self) -> None:
        df = self._make_event_df(event_step_idx=5)
        result = event_response(df, "mask", "returns")
        assert result["pre_event_value"][0] == pytest.approx(100.0)

    def test_drop_is_positive(self) -> None:
        df = self._make_event_df(event_step_idx=5)
        result = event_response(df, "mask", "returns")
        assert result["drop"][0] == pytest.approx(40.0)

    def test_recovery_slope_is_positive(self) -> None:
        df = self._make_event_df(event_step_idx=5)
        result = event_response(df, "mask", "returns")
        assert result["recovery_slope"][0] > 0.0

    def test_recovery_is_measured_in_logged_step_units(self) -> None:
        """Subsampled logging (every 10th step) must report recovery in steps, not sample positions."""
        dense = self._make_event_df(event_step_idx=5)
        sparse = dense.with_columns(pl.col("step") * 10)

        dense_result = event_response(dense, "mask", "returns", recovery_fraction=0.5)
        sparse_result = event_response(sparse, "mask", "returns", recovery_fraction=0.5)

        assert dense_result["steps_to_recovery"][0] == 10
        assert sparse_result["steps_to_recovery"][0] == 100
        assert sparse_result["recovery_slope"][0] == pytest.approx(dense_result["recovery_slope"][0] / 10)

    def test_no_event_returns_empty(self) -> None:
        # Event metric never exceeds threshold
        df = _make_df(["A"], [0, 1], [0, 1, 2], "mask", lambda c, s, t: 0.0)
        df_outcome = _make_df(["A"], [0, 1], [0, 1, 2], "returns", lambda c, s, t: 100.0)
        combined = pl.concat([df, df_outcome])
        result = event_response(combined, "mask", "returns")
        assert result.is_empty()

    def test_no_drop_steps_to_recovery_is_zero(self) -> None:
        # Outcome stays flat after event — no drop, steps_to_recovery = 0
        rows = []
        for seed in range(3):
            for step in range(5):
                rows.append({"condition_name": "A", "seed": seed, "step": step, "metric": "mask", "value": 1.0 if step >= 2 else 0.0})
                rows.append({"condition_name": "A", "seed": seed, "step": step, "metric": "r", "value": 100.0})
        df = pl.DataFrame(rows)
        result = event_response(df, "mask", "r")
        assert result["drop"][0] == pytest.approx(0.0)
        assert result["steps_to_recovery"][0] == 0


class TestFlatlineFlags:
    def test_flags_a_curve_that_stalls_at_the_reward_floor(self) -> None:
        """A curve that never moves off a constant floor is flatlined from the start."""
        df = _make_df(["stuck"], [0, 1, 2], list(range(10)), "r", lambda c, s, t: -1.0)
        result = flatline_flags(df, "r")
        row = result.filter(pl.col("condition_name") == "stuck")
        assert row["flatlined"][0] is True
        assert row["whole_range"][0] == pytest.approx(0.0)

    def test_does_not_flag_a_curve_still_improving_at_the_end(self) -> None:
        """A curve still moving in its last half is not flatlined."""
        df = _make_df(["learning"], [0, 1, 2], list(range(10)), "r", lambda c, s, t: float(t))
        result = flatline_flags(df, "r")
        row = result.filter(pl.col("condition_name") == "learning")
        assert row["flatlined"][0] is False

    def test_flags_a_curve_that_plateaus_before_the_run_ends(self) -> None:
        """Improves for the first half, then plateaus for the second -- the tail is flat."""
        def value_fn(c: str, s: int, t: int) -> float:
            return float(min(t, 4))

        df = _make_df(["plateaus"], [0, 1, 2], list(range(10)), "r", value_fn)
        result = flatline_flags(df, "r")
        row = result.filter(pl.col("condition_name") == "plateaus")
        assert row["flatlined"][0] is True

    def test_never_flags_a_metric_logged_at_a_single_step(self) -> None:
        """A scalar metric (one logged point per run, e.g. an AUC) has no curve that could flatline."""
        df = _make_df(["scalar"], [0, 1, 2], [0], "r", lambda c, s, t: float(s))
        result = flatline_flags(df, "r")
        row = result.filter(pl.col("condition_name") == "scalar")
        assert row["flatlined"][0] is False

    def test_returns_empty_frame_for_no_matching_conditions(self) -> None:
        df = _make_df(["A"], [0], [0, 1], "other", lambda c, s, t: 0.0)
        result = flatline_flags(df, "r")
        assert result.is_empty()
        assert result.schema.names() == ["condition_name", "flatlined", "tail_range", "whole_range"]
