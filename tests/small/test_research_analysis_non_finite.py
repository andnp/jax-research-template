"""Contract tests for explicit handling of diverged run values."""

from __future__ import annotations

import math

import polars as pl
import pytest
from research_analysis.dataset.types import ExperimentDataset, ExperimentDimensions
from research_analysis.non_finite import NonFiniteValuesError, require_finite, substitute_non_finite


def _dataset(values: list[float]) -> ExperimentDataset:
    n = len(values)
    runs = pl.DataFrame(
        {
            "condition_name": ["a"] * n,
            "seed": list(range(n)),
            "run_id": list(range(1, n + 1)),
            "execution_id": [1] * n,
            "ablation": [None] * n,
            "non_finite_metrics": [[] if math.isfinite(v) else ["reward"] for v in values],
        },
        schema_overrides={"ablation": pl.String},
    )
    metrics = pl.DataFrame(
        {
            "condition_name": ["a"] * n + ["a"],
            "seed": [*range(n), 0],
            "run_id": [*range(1, n + 1), 1],
            "execution_id": [1] * (n + 1),
            "step": [0] * (n + 1),
            "metric": ["reward"] * n + ["loss"],
            "value": [*values, math.nan],
        }
    )
    return ExperimentDataset("exp", ExperimentDimensions(), runs, metrics, ())


class TestRequireFinite:
    def test_finite_frames_pass(self) -> None:
        """A frame without NaN or infinity is accepted unchanged."""
        require_finite(pl.DataFrame({"run_id": [1], "value": [1.0]}), "reward")

    def test_error_names_each_diverged_run_once(self) -> None:
        """The error identifies the diverged runs, so the caller can see which seeds to explain."""
        frame = pl.DataFrame(
            {"condition_name": ["a", "a", "b"], "run_id": [1, 1, 2], "seed": [0, 0, 1], "value": [math.nan, -math.inf, 2.0]}
        )

        with pytest.raises(NonFiniteValuesError, match="final reward") as error:
            require_finite(frame, "final reward")

        assert error.value.offenders.to_dicts() == [{"condition_name": "a", "run_id": 1, "seed": 0}]


class TestSubstituteNonFinite:
    def test_replaces_only_the_named_metric(self) -> None:
        """Only the declared metric's non-finite values are replaced; other metrics stay loud."""
        dataset = _dataset([1.0, math.nan, math.inf])

        substituted = substitute_non_finite(dataset, "reward", -100.0)

        rewards = substituted.metrics.filter(pl.col("metric") == "reward")["value"].to_list()
        assert rewards == [1.0, -100.0, -100.0]
        assert math.isnan(substituted.metrics.filter(pl.col("metric") == "loss")["value"][0])

    def test_keeps_divergence_flags(self) -> None:
        """Substitution is an analysis choice; the runs still record that they diverged."""
        dataset = _dataset([1.0, math.nan])

        substituted = substitute_non_finite(dataset, "reward", 0.0)

        assert substituted.runs["non_finite_metrics"].to_list() == [[], ["reward"]]

    def test_rejects_a_non_finite_worst_case(self) -> None:
        """A worst-case value must itself be a usable number."""
        with pytest.raises(ValueError, match="worst_case must be finite"):
            substitute_non_finite(_dataset([1.0]), "reward", math.nan)
