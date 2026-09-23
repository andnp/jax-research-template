"""Small tests for research_analysis.performance — declared per-run measures."""

import math

import polars as pl
import pytest
from research_analysis.performance import FinalValue, ReturnRate, TailMean, run_scores


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

    def test_keeps_a_diverged_final_value_visible(self) -> None:
        """A run that ends non-finite scores NaN rather than silently disappearing."""
        frame = _curves([(0, 0, 1.0), (0, 1, math.nan), (1, 0, 2.0), (1, 1, 3.0)])

        scores = _values(run_scores(frame, "return", FinalValue()))

        assert math.isnan(scores[0])
        assert scores[1] == 3.0

    def test_rejects_two_runs_pooled_under_one_seed(self) -> None:
        """Without a run id, a repeated step for one seed means runs were pooled, not one curve."""
        frame = _curves([(0, 0, 1.0), (0, 0, 2.0)])

        with pytest.raises(ValueError, match="more than one run for seed 0"):
            run_scores(frame, "return", FinalValue())


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
