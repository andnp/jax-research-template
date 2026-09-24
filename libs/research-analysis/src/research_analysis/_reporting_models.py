"""Private data models shared by the reporting implementation modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_analysis.comparison import BakeoffSpec
from research_analysis.diagnostics.bakeoff import BakeoffWarning
from research_analysis.performance import FinalValue, PerformanceMeasure


@dataclass(frozen=True)
class HyperparameterConfigReport:
    """Bias-corrected estimate of the best hyperparameter configuration's true performance."""

    winning_config: dict[str, Any]
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    estimate_ci: tuple[float, float]
    confidence_level: float = 0.95
    # 0 when the same config wins under every resample; approaches 1 as the winner
    # becomes as unpredictable as a uniform draw over every candidate config.
    normalized_selection_entropy: float = 0.0
    # Runs left out because their seed was not observed under every configuration.
    unpaired_run_ids: tuple[int, ...] = ()
    measure: PerformanceMeasure = FinalValue()


@dataclass(frozen=True)
class HyperparameterDifferenceReport:
    """Bias-corrected paired difference between two groups' best hyperparameter configurations."""

    minuend_config: dict[str, Any]
    subtrahend_config: dict[str, Any]
    mean_diff: float
    estimate_ci: tuple[float, float]
    confidence_level: float = 0.95
    # Runs left out because their seed was not observed under every configuration of both groups.
    unpaired_run_ids: tuple[int, ...] = ()
    measure: PerformanceMeasure = FinalValue()


@dataclass(frozen=True)
class HyperparameterSliceReport:
    """Sensitivity curve for one hyperparameter: selection-corrected best performance vs. a champion-pinned slice."""

    vary: str
    values: list[Any]
    best_performance: list[float]
    # None where the champion-pinned configuration was never run at that value.
    slice_performance: Sequence[float | None]
    plot_path: Path | None


@dataclass(frozen=True)
class CellSummary:
    """Distribution of one condition's per-run scores within one block; None where too few runs."""

    n: int
    estimate: float | None
    ci: tuple[float, float] | None
    confidence: float
    sd: float | None


@dataclass(frozen=True)
class StatisticalTestDetails:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    is_significant: bool
    assumptions: dict[str, Any]
    justification: str
    intermediate_tests: dict[str, StatisticalTestDetails]


@dataclass(frozen=True)
class BenchmarkBakeoffReport:
    algorithms: list[str]
    environments: list[str]
    normalized_scores: dict[str, dict[str, float]]
    performance_profiles: dict[str, dict[float, float]]
    probability_of_improvement: dict[str, dict[str, float | None]]
    optimality_gaps: dict[str, float]
    dispersion: dict[str, float]
    lower_tail: dict[str, float]
    supplementary_iqm: dict[str, float]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float | None]]
    difference_estimate: dict[str, dict[str, float | None]]
    """[condition_a][condition_b] = condition_a's estimand minus condition_b's, paired by
    (block, seed); None where fewer than min_samples paired observations were shared."""
    difference_ci: dict[str, dict[str, tuple[float, float] | None]]
    aggregate_plot_path: Path | None


@dataclass(frozen=True)
class BakeoffReport:
    """A repeatable bakeoff analysis: champion selection, comparison, and caveats in one call.

    ``champions`` is ``None`` when ``select_champion_over`` was not given (the metric is
    already one value per condition per seed per block, nothing to select over).
    """

    conditions: list[str]
    metric_name: str
    measure: PerformanceMeasure
    spec: BakeoffSpec
    cells: dict[str, dict[str, CellSummary]]
    """[condition][block label] -> that cell's per-run score summary, under ``spec.estimand``."""
    champions: dict[str, dict[str, HyperparameterConfigReport]] | None
    comparison: BenchmarkBakeoffReport
    warnings: tuple[BakeoffWarning, ...]


BlockKey = tuple[tuple[str, object], ...]
"""A block's coordinates as (column, value) pairs; empty when a report has no blocking axes."""
