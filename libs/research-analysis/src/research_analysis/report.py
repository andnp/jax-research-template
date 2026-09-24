"""A versioned, self-describing report document: provenance, accounting, diagnostics, and results together."""

from __future__ import annotations

from dataclasses import dataclass
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
    RunAccounting,
    SweepValue,
)
from research_analysis.diagnostics.types import DatasetDiagnostic
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
    """Why the analyses produced no results; ``offenders`` lists the runs that caused it, when known."""

    analysis: str
    error_type: str
    message: str
    offenders: tuple[dict[str, SweepValue | None], ...]


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
