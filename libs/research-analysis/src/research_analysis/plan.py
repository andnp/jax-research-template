"""Declare an experiment's primary analysis beside its spec, before any results exist."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import ModuleType

from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import CohortSelection, ExperimentDimensions, SweepValue, VersionSelection
from research_analysis.performance import FINAL_VALUE, PerformanceMeasure


@dataclass(frozen=True)
class BakeoffAnalysis:
    """Compare ``conditions`` with ``bakeoff_report``; ``pairs`` of None plans every pair.

    ``shared_across`` reuses the runs of conditions a blocking axis does not apply to in every block
    of that axis; ``untuned`` declares single-configuration referents in a tuned-champion comparison.
    """

    name: str
    conditions: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...] | None = None
    block_by: tuple[str, ...] = ()
    select_champion_over: tuple[str, ...] | None = None
    min_seeds: int = 5
    shared_across: tuple[str, ...] = ()
    untuned: tuple[str, ...] = ()


@dataclass(frozen=True)
class BestHypersAnalysis:
    """Select the bias-corrected champion configuration, separately per ``group_by`` combination."""

    name: str
    group_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class HypersDifferenceAnalysis:
    """Paired difference between the champions of two values of ``group_axis``."""

    name: str
    group_axis: str
    minuend: SweepValue
    subtrahend: SweepValue


@dataclass(frozen=True)
class SliceHypersAnalysis:
    """Sensitivity of the champion to one swept axis."""

    name: str
    vary: str
    group_by: tuple[str, ...] = ()


type Analysis = BakeoffAnalysis | BestHypersAnalysis | HypersDifferenceAnalysis | SliceHypersAnalysis


def against_baseline(conditions: Sequence[str], baseline: str) -> tuple[tuple[str, str], ...]:
    """Plan each condition minus ``baseline``, and nothing else."""
    return tuple((condition, baseline) for condition in conditions if condition != baseline)


@dataclass(frozen=True)
class AnalysisPlan:
    """The declared primary analysis: one metric and measure, the analyses to run, and their statistics.

    ``stats`` sets alpha, resampling, minimum samples, and seed for every analysis. ``non_finite_worst_case``
    declares the value a diverged run scores; without it, a diverged run fails the report loudly.
    """

    metric: str
    analyses: tuple[Analysis, ...]
    dimensions: ExperimentDimensions
    measure: PerformanceMeasure = FINAL_VALUE
    stats: BakeoffSpec = field(default_factory=BakeoffSpec)
    higher_is_better: bool = True
    version_selection: VersionSelection = field(default_factory=VersionSelection.latest_completed)
    cohort: CohortSelection | None = None
    non_finite_worst_case: float | None = None

    def __post_init__(self) -> None:
        names = [analysis.name for analysis in self.analyses]
        if not names:
            raise ValueError("an analysis plan needs at least one analysis")
        if len(set(names)) < len(names):
            raise ValueError(f"analysis names must be unique, got {names!r}")
        if self.non_finite_worst_case is not None and not math.isfinite(self.non_finite_worst_case):
            raise ValueError(f"non_finite_worst_case must be finite, got {self.non_finite_worst_case!r}")
        hierarchical = [
            analysis.name
            for analysis in self.analyses
            if not isinstance(analysis, BakeoffAnalysis) or analysis.select_champion_over is not None
        ]
        if self.stats.interval != "percentile" and hierarchical:
            raise ValueError(
                f"analyses {hierarchical!r} use the hierarchical percentile bootstrap; "
                f"stats.interval {self.stats.interval!r} applies only to untuned bakeoffs"
            )


PLANS_ATTRIBUTE = "__research_analysis_plans__"

type PlanFactory = Callable[[], AnalysisPlan]


def analysis_plan(spec_factory: Callable[[], object]) -> Callable[[PlanFactory], PlanFactory]:
    """Register the decorated factory as the declared plan of ``spec_factory``, in their shared spec module."""

    def register(factory: PlanFactory) -> PlanFactory:
        registry: dict[str, PlanFactory] = factory.__globals__.setdefault(PLANS_ATTRIBUTE, {})
        if spec_factory.__name__ in registry:
            raise ValueError(f"spec {spec_factory.__name__!r} already has a declared analysis plan")
        registry[spec_factory.__name__] = factory
        return factory

    return register


def registered_plans(module: ModuleType) -> dict[str, PlanFactory]:
    """Spec factory name -> its declared plan factory, for a loaded spec module."""
    return dict(vars(module).get(PLANS_ATTRIBUTE, {}))
