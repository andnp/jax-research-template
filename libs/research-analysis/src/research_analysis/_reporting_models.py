"""Private data models shared by the reporting implementation modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    # Distribution-free (spec.tolerance_confidence, spec.tolerance_coverage) interval over runs;
    # None below tolerance_min_runs.
    tolerance: tuple[float, float] | None
    tolerance_min_runs: int
    # The run whose score is closest to the cell median, as a representative curve to inspect.
    median_run_id: int | None


@dataclass(frozen=True)
class PairedDifference:
    """``minuend`` minus ``subtrahend`` in one block, paired by seed; estimate None when the pair was skipped."""

    minuend: str
    subtrahend: str
    block: str
    estimate: float | None
    ci: tuple[float, float] | None
    confidence: float
    n_paired: int
    method: str
    # Holm-adjusted across the report's comparison family.
    p_value: float | None
    skipped_reason: str | None

    def reversed(self) -> PairedDifference:
        """The same comparison read as ``subtrahend`` minus ``minuend``."""
        return replace(
            self,
            minuend=self.subtrahend,
            subtrahend=self.minuend,
            estimate=None if self.estimate is None else -self.estimate,
            ci=None if self.ci is None else (-self.ci[1], -self.ci[0]),
        )


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
    omnibus_details: StatisticalTestDetails


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
    differences: tuple[PairedDifference, ...]
    """Every planned (block, pair) comparison, including pairs skipped for too few paired runs."""
    distribution_plot_paths: dict[str, Path]
    """Block label -> figure of every condition's per-run scores in that block."""
    family_size: int
    # Bonferroni per-comparison confidence keeping every difference CI jointly at spec.confidence_level.
    comparison_confidence: float
    warnings: tuple[BakeoffWarning, ...]

    def difference(self, minuend: str, subtrahend: str, block: str = "") -> PairedDifference:
        """Return the ``minuend - subtrahend`` comparison in ``block`` (a block label; "" when unblocked)."""
        for item in self.differences:
            if item.block != block:
                continue
            if (item.minuend, item.subtrahend) == (minuend, subtrahend):
                return item
            if (item.minuend, item.subtrahend) == (subtrahend, minuend):
                return item.reversed()
        raise KeyError(f"no comparison of {minuend!r} and {subtrahend!r} in block {block!r}")


BlockKey = tuple[tuple[str, object], ...]
"""A block's coordinates as (column, value) pairs; empty when a report has no blocking axes."""
