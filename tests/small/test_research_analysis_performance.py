"""Small tests for research_analysis.performance — declared per-run measures."""

import math

import polars as pl
import pytest
from research_analysis.non_finite import NonFiniteValuesError
from research_analysis.performance import FinalValue, PerformanceMeasure, ReturnRate, StoredValue, TailMean, run_scores


def _curves(rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "condition_name": ["a"] * len(rows),
            "seed": [seed for seed, _step, _value in rows],
            "step": [step for _seed, step, _value in rows],
            "metric": ["return"] * len(rows),
            "value": [value for _seed, _step, value in rows],
        }
    )


def _values(scores: pl.DataFrame) -> dict[int, float]:
    return dict(zip(scores["seed"].to_list(), scores["value"].to_list(), strict=True))


class TestFinalValue:
    def test_takes_the_highest_step_regardless_of_row_order(self) -> None:
        """Rows arrive unordered; the final value is the one at the last logged step."""
        frame = _curves([(0, 2, 9.0), (0, 0, 1.0), (0, 1, 5.0)])

        assert _values(run_scores(frame, "return", FinalValue())) == {0: 9.0}

    def test_rejects_a_diverged_final_value(self) -> None:
        """A run that ends non-finite fails loudly, naming the run, instead of feeding NaN into statistics."""
        frame = _curves([(0, 0, 1.0), (0, 1, math.nan), (1, 0, 2.0), (1, 1, 3.0)])

        with pytest.raises(NonFiniteValuesError) as raised:
            run_scores(frame, "return", FinalValue())

        assert raised.value.offenders["seed"].to_list() == [0]

    def test_rejects_two_runs_pooled_under_one_seed(self) -> None:
        """Without a run id, a repeated step for one seed means runs were pooled, not one curve."""
        frame = _curves([(0, 0, 1.0), (0, 0, 2.0)])

        with pytest.raises(ValueError, match="more than one run for seed 0"):
            run_scores(frame, "return", FinalValue())

    def test_rejects_stored_value_for_a_curve(self) -> None:
        """A curve must use a measure that reads its steps.

        StoredValue is reserved for scalar metric frames.
        """
        with pytest.raises(ValueError, match="StoredValue requires a scalar metric frame"):
            run_scores(_curves([(0, 0, 1.0)]), "return", StoredValue())


class TestTailMean:
    def test_averages_the_final_fraction_of_each_runs_step_span(self) -> None:
        """Steps 0..10 with fraction 0.2 keep steps 8, 9, and 10."""
        frame = _curves([(0, step, float(step)) for step in range(11)])

        assert _values(run_scores(frame, "return", TailMean(0.2))) == {0: pytest.approx(9.0)}

    def test_whole_fraction_matches_the_whole_curve_mean(self) -> None:
        """fraction=1 covers every logged point."""
        frame = _curves([(0, 0, 1.0), (0, 5, 2.0), (0, 10, 6.0)])

        assert _values(run_scores(frame, "return", TailMean(1.0))) == {0: pytest.approx(3.0)}

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction: float) -> None:
        """An empty or oversized window is a declaration error."""
        with pytest.raises(ValueError, match="fraction"):
            TailMean(fraction)


class TestReturnRate:
    def test_averages_every_logged_point_per_run(self) -> None:
        """Each run is summarized by its own curve mean, not pooled across runs."""
        frame = _curves([(0, 0, 1.0), (0, 1, 3.0), (1, 0, 10.0), (1, 1, 20.0)])

        assert _values(run_scores(frame, "return", ReturnRate())) == {0: 2.0, 1: 15.0}

    def test_ignores_rows_of_other_metrics(self) -> None:
        """Only the requested metric contributes to the score."""
        frame = pl.concat([_curves([(0, 0, 1.0)]), _curves([(0, 0, 100.0)]).with_columns(pl.lit("loss").alias("metric"))])

        assert _values(run_scores(frame, "return", ReturnRate())) == {0: 1.0}


class TestStoredScalarScores:
    def test_keeps_each_precomputed_run_value(self) -> None:
        """A stored run scalar is already reduced.

        The explicit StoredValue measure preserves the per-run values.
        """
        frame = pl.DataFrame(
            {
                "condition_name": ["baseline", "candidate"],
                "seed": [0, 0],
                "run_id": [10, 11],
                "execution_id": [20, 20],
                "ablation": ["default", "default"],
                "metric": ["return", "return"],
                "value": [3.0, 9.0],
            }
        )

        scores = run_scores(frame, "return", StoredValue())

        assert scores.select("run_id", "value").rows() == [(10, 3.0), (11, 9.0)]

    @pytest.mark.parametrize("measure", [FinalValue(), TailMean(0.1), ReturnRate()])
    def test_rejects_curve_measure_for_stored_scalar(self, measure: PerformanceMeasure) -> None:
        """A scalar frame requires an explicit measure that reads stored values.

        Curve reducers must not silently turn a declaration into a no-op.
        """
        frame = pl.DataFrame(
            {
                "condition_name": ["baseline"],
                "seed": [0],
                "run_id": [10],
                "metric": ["return"],
                "value": [3.0],
            }
        )

        with pytest.raises(ValueError, match="scalar metric frame.*requires StoredValue"):
            run_scores(frame, "return", measure)

    def test_rejects_duplicate_stored_values_for_a_run(self) -> None:
        """Duplicate scalar rows for one run are invalid.

        The performance layer must not silently pool stored outcomes.
        """
        frame = pl.DataFrame(
            {
                "condition_name": ["baseline", "baseline"],
                "seed": [0, 0],
                "run_id": [10, 10],
                "metric": ["return", "return"],
                "value": [3.0, 4.0],
            }
        )

        with pytest.raises(ValueError, match="duplicate stored values for run identity"):
            run_scores(frame, "return", StoredValue())

    def test_rejects_a_non_finite_stored_value(self) -> None:
        """A non-finite stored outcome follows the same divergence policy as curves.

        A stored scalar still represents the result of one completed run.
        """
        frame = pl.DataFrame(
            {
                "condition_name": ["baseline"],
                "seed": [0],
                "run_id": [10],
                "metric": ["return"],
                "value": [float("nan")],
            }
        )

        with pytest.raises(NonFiniteValuesError):
            run_scores(frame, "return", StoredValue())
