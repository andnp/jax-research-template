"""Small tests for research_analysis.curves."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from research_analysis.curves import (
    event_response,
    final_values,
    mean_curves,
    median_run_curve,
    tolerance_bands,
)


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


class TestFinalValues:
    def test_returns_last_step_per_seed(self) -> None:
        df = _make_df(["A"], [0, 1], [0, 1, 2], "r", lambda c, s, t: t + s * 10)
        result = final_values(df, "r")
        assert result.shape == (2, 3)
        assert result.filter(pl.col("seed") == 0)["value"][0] == pytest.approx(2.0)
        assert result.filter(pl.col("seed") == 1)["value"][0] == pytest.approx(12.0)

    def test_sorted_by_condition_then_seed(self) -> None:
        df = _make_df(["B", "A"], [1, 0], [0, 1], "r", lambda c, s, t: 0)
        result = final_values(df, "r")
        assert result["condition_name"].to_list() == ["A", "A", "B", "B"]
        assert result["seed"].to_list() == [0, 1, 0, 1]

    def test_filters_to_requested_metric(self) -> None:
        df = _make_df(["A"], [0], [0, 1], "r", lambda c, s, t: t)
        df_other = _make_df(["A"], [0], [0, 1], "other", lambda c, s, t: t * 100)
        combined = pl.concat([df, df_other])
        result = final_values(combined, "r")
        assert result["value"][0] == pytest.approx(1.0)

    def test_multiple_conditions(self) -> None:
        df = _make_df(["A", "B"], [0], [0, 1], "r", lambda c, s, t: 10.0 if c == "A" else 20.0)
        result = final_values(df, "r")
        vals = {row["condition_name"]: row["value"] for row in result.iter_rows(named=True)}
        assert vals["A"] == pytest.approx(10.0)
        assert vals["B"] == pytest.approx(20.0)


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
