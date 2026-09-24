"""Run an experiment's declared analysis plan into a report document."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from research_analysis._reporting_bakeoff import bakeoff_report
from research_analysis._reporting_hypers import best_hypers, slice_hypers
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExecutionProvenance, ExperimentDataset
from research_analysis.diagnostics.bakeoff import IncompleteBlocksError
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES
from research_analysis.non_finite import NonFiniteValuesError, substitute_non_finite
from research_analysis.plan import (
    Analysis,
    AnalysisPlan,
    BakeoffAnalysis,
    BestHypersAnalysis,
    HypersDifferenceAnalysis,
    SliceHypersAnalysis,
)
from research_analysis.plan_snapshots import PlanSnapshot
from research_analysis.report import (
    AnalysisFailure,
    AnalysisResult,
    BakeoffResult,
    BestHypersResult,
    HypersDifferenceResult,
    ReportDocument,
    SliceHypersResult,
    _document,
)
from research_analysis.report_json import plan_fingerprint


def _execution_problem(fingerprint: str, snapshots: Sequence[PlanSnapshot], execution: ExecutionProvenance) -> str | None:
    if execution.start_time is None:
        return "start time not recorded, so the plan cannot be shown to predate the results"
    started = datetime.fromisoformat(execution.start_time)
    in_force = [snapshot for snapshot in snapshots if snapshot.recorded_at <= started]
    if not in_force:
        return "no plan snapshot before execution: record the plan by launching runs with `research experiment run` or `submit`"
    if in_force[-1].fingerprint != fingerprint:
        return f"plan changed after results were recorded: the plan in force at these executions was {in_force[-1].fingerprint}"
    return None


def _snapshot_reasons(
    fingerprint: str, snapshots: Sequence[PlanSnapshot], executions: tuple[ExecutionProvenance, ...]
) -> tuple[str, ...]:
    """Why the plan cannot be shown to predate every included execution, naming the offending executions."""
    offenders: dict[str, list[int]] = {}
    for execution in executions:
        problem = _execution_problem(fingerprint, snapshots, execution)
        if problem is not None:
            offenders.setdefault(problem, []).append(execution.execution_id)
    return tuple(f"{problem} (executions {', '.join(map(str, ids))})" for problem, ids in offenders.items())


def _run_analysis(dataset: ExperimentDataset, plan: AnalysisPlan, analysis: Analysis) -> list[AnalysisResult]:
    stats = plan.stats
    match analysis:
        case BakeoffAnalysis():
            report = bakeoff_report(
                dataset,
                plan.metric,
                list(analysis.conditions),
                select_champion_over=None if analysis.select_champion_over is None else list(analysis.select_champion_over),
                block_by=list(analysis.block_by),
                spec=stats,
                higher_is_better=plan.higher_is_better,
                min_seeds=analysis.min_seeds,
                measure=plan.measure,
                pairs=analysis.pairs,
            )
            return [BakeoffResult(analysis.name, report)]
        case BestHypersAnalysis(group_by=()):
            champion = best_hypers(
                dataset,
                plan.metric,
                higher_is_better=plan.higher_is_better,
                confidence_level=stats.confidence_level,
                n_resamples=stats.n_resamples,
                min_samples=stats.min_samples,
                random_seed=stats.random_seed,
                measure=plan.measure,
            )
            return [BestHypersResult(analysis.name, {}, champion)]
        case BestHypersAnalysis(group_by=group_by):
            champions = best_hypers(
                dataset,
                plan.metric,
                list(group_by),
                higher_is_better=plan.higher_is_better,
                confidence_level=stats.confidence_level,
                n_resamples=stats.n_resamples,
                min_samples=stats.min_samples,
                random_seed=stats.random_seed,
                measure=plan.measure,
            )
            return [
                BestHypersResult(analysis.name, dict(zip(group_by, key, strict=True)), report)
                for key, report in champions.items()
            ]
        case HypersDifferenceAnalysis():
            difference = best_hypers(
                dataset,
                plan.metric,
                [analysis.group_axis],
                difference_between=(analysis.minuend, analysis.subtrahend),
                higher_is_better=plan.higher_is_better,
                confidence_level=stats.confidence_level,
                n_resamples=stats.n_resamples,
                min_samples=stats.min_samples,
                random_seed=stats.random_seed,
                measure=plan.measure,
            )
            return [HypersDifferenceResult(analysis.name, difference)]
        case SliceHypersAnalysis(group_by=()):
            sensitivity = slice_hypers(
                dataset,
                plan.metric,
                analysis.vary,
                higher_is_better=plan.higher_is_better,
                confidence_level=stats.confidence_level,
                n_resamples=stats.n_resamples,
                min_samples=stats.min_samples,
                random_seed=stats.random_seed,
                measure=plan.measure,
            )
            return [SliceHypersResult(analysis.name, {}, sensitivity)]
        case SliceHypersAnalysis(group_by=group_by):
            sensitivities = slice_hypers(
                dataset,
                plan.metric,
                analysis.vary,
                list(group_by),
                higher_is_better=plan.higher_is_better,
                confidence_level=stats.confidence_level,
                n_resamples=stats.n_resamples,
                min_samples=stats.min_samples,
                random_seed=stats.random_seed,
                measure=plan.measure,
            )
            return [
                SliceHypersResult(analysis.name, dict(zip(group_by, key, strict=True)), report)
                for key, report in sensitivities.items()
            ]


def run_plan(
    experiments_db: Path, slug: str, plan: AnalysisPlan, snapshots: Sequence[PlanSnapshot]
) -> ReportDocument:
    """Run the declared plan exactly as written: the only way a report is labelled declared.

    It is still labelled exploratory when ``snapshots`` cannot show this plan was in force when every
    included execution started. Loading problems and failed analyses yield a document without results.
    """
    dataset = load_experiment_dataset(
        experiments_db,
        slug,
        dimensions=plan.dimensions,
        version_selection=plan.version_selection,
        cohort=plan.cohort,
        metrics=(plan.metric,),
        partial=True,
    )
    fingerprint = plan_fingerprint(plan)
    reasons = _snapshot_reasons(fingerprint, snapshots, dataset.executions)

    def document(analyses: tuple[AnalysisResult, ...], failure: AnalysisFailure | None) -> ReportDocument:
        return _document(
            dataset,
            experiments_db,
            plan=plan,
            plan_fingerprint=fingerprint,
            exploratory_reasons=reasons,
            analyses=analyses,
            failure=failure,
        )

    if any(item.code in FATAL_DIAGNOSTIC_CODES for item in dataset.diagnostics):
        return document((), None)

    scored = dataset
    if plan.non_finite_worst_case is not None:
        scored = substitute_non_finite(dataset, plan.metric, plan.non_finite_worst_case)
    results: list[AnalysisResult] = []
    for analysis in plan.analyses:
        try:
            results.extend(_run_analysis(scored, plan, analysis))
        except NonFiniteValuesError as error:
            runs = tuple(error.offenders.to_dicts())
            return document((), AnalysisFailure(analysis.name, type(error).__name__, str(error), non_finite_runs=runs))
        except IncompleteBlocksError as error:
            failure = AnalysisFailure(analysis.name, type(error).__name__, str(error), incomplete_cells=error.cells)
            return document((), failure)
    return document(tuple(results), None)
