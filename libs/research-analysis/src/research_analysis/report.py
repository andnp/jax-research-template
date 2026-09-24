"""A versioned, self-describing report document: provenance, accounting, diagnostics, and results together."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from research_analysis._reporting_models import (
    BakeoffReport,
    HyperparameterConfigReport,
    HyperparameterDifferenceReport,
    HyperparameterSliceReport,
)
from research_analysis.dataset.types import (
    ComponentVersionProvenance,
    ExecutionProvenance,
    ExperimentDataset,
    RunAccounting,
    SweepValue,
)
from research_analysis.diagnostics.bakeoff import IncompleteCell
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES, DatasetDiagnostic, InsufficientDataCell
from research_analysis.plan import AnalysisPlan

REPORT_SCHEMA_VERSION = 1


class ReportStatus(StrEnum):
    """Whether the report holds results, or only the reason it could not."""

    OK = "ok"
    LOAD_FAILED = "load_failed"
    ANALYSIS_FAILED = "analysis_failed"


class PlanStatus(StrEnum):
    """Whether the results are the experiment's declared analysis or were chosen afterwards."""

    DECLARED = "declared"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class ReportProvenance:
    """Which experiment, database, component versions, and executions the results came from."""

    experiment: str
    experiments_db: Path
    generated_at: str
    component_versions: tuple[ComponentVersionProvenance, ...]
    executions: tuple[ExecutionProvenance, ...]


@dataclass(frozen=True)
class BakeoffResult:
    name: str
    report: BakeoffReport


@dataclass(frozen=True)
class BestHypersResult:
    """One champion selection; ``group`` is empty when the whole grid was searched jointly."""

    name: str
    group: dict[str, SweepValue]
    report: HyperparameterConfigReport


@dataclass(frozen=True)
class HypersDifferenceResult:
    name: str
    report: HyperparameterDifferenceReport


@dataclass(frozen=True)
class SliceHypersResult:
    name: str
    group: dict[str, SweepValue]
    report: HyperparameterSliceReport


type AnalysisResult = BakeoffResult | BestHypersResult | HypersDifferenceResult | SliceHypersResult


@dataclass(frozen=True)
class AnalysisFailure:
    """Why the analyses produced no results, with the runs or cells that caused it when they are known."""

    analysis: str
    error_type: str
    message: str
    non_finite_runs: tuple[dict[str, SweepValue | None], ...] = ()
    incomplete_cells: tuple[IncompleteCell, ...] = ()
    insufficient_data: tuple[InsufficientDataCell, ...] = ()


@dataclass(frozen=True)
class ReportDocument:
    """Everything a reader needs to trust or reject the results, with nothing only visible on a console.

    ``analyses`` is empty unless ``status`` is ok: a report never carries partial results.
    """

    schema_version: int
    status: ReportStatus
    plan_status: PlanStatus
    exploratory_reasons: tuple[str, ...]
    plan: AnalysisPlan | None
    plan_fingerprint: str | None
    provenance: ReportProvenance
    accounting: RunAccounting
    fatal_diagnostics: tuple[DatasetDiagnostic, ...]
    non_fatal_diagnostics: tuple[DatasetDiagnostic, ...]
    analyses: tuple[AnalysisResult, ...]
    failure: AnalysisFailure | None
    plots: dict[str, Path]
    """Figure name -> path relative to the report's directory; empty until plots are written."""


def _document(
    dataset: ExperimentDataset,
    experiments_db: Path,
    *,
    plan: AnalysisPlan | None,
    plan_fingerprint: str | None,
    exploratory_reasons: tuple[str, ...],
    analyses: tuple[AnalysisResult, ...],
    failure: AnalysisFailure | None,
) -> ReportDocument:
    fatal = tuple(item for item in dataset.diagnostics if item.code in FATAL_DIAGNOSTIC_CODES)
    status = ReportStatus.LOAD_FAILED if fatal else ReportStatus.ANALYSIS_FAILED if failure else ReportStatus.OK
    return ReportDocument(
        schema_version=REPORT_SCHEMA_VERSION,
        status=status,
        plan_status=PlanStatus.EXPLORATORY if exploratory_reasons else PlanStatus.DECLARED,
        exploratory_reasons=exploratory_reasons,
        plan=plan,
        plan_fingerprint=plan_fingerprint,
        provenance=ReportProvenance(
            experiment=dataset.experiment_name,
            experiments_db=experiments_db.resolve(),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            component_versions=dataset.provenance,
            executions=dataset.executions,
        ),
        accounting=dataset.accounting,
        fatal_diagnostics=fatal,
        non_fatal_diagnostics=tuple(item for item in dataset.diagnostics if item not in fatal),
        analyses=analyses if status is ReportStatus.OK else (),
        failure=failure,
        plots={},
    )


def exploratory_report(
    dataset: ExperimentDataset, experiments_db: Path, analyses: Sequence[AnalysisResult]
) -> ReportDocument:
    """Wrap analyses chosen by hand in a report document, always labelled exploratory."""
    return _document(
        dataset,
        experiments_db,
        plan=None,
        plan_fingerprint=None,
        exploratory_reasons=("no declared plan: the analyses were chosen by hand",),
        analyses=tuple(analyses),
        failure=None,
    )
