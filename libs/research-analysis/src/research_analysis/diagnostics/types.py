"""Structured diagnostics for experiment dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiagnosticCode(StrEnum):
    """Structured dataset-loading diagnostic categories."""

    MISSING_METRIC_ARTIFACT = "missing_metric_artifact"
    EXCLUDED_VERSION = "excluded_version"
    AMBIGUOUS_COHORT = "ambiguous_cohort"
    INVALID_VERSION_SELECTION = "invalid_version_selection"
    RUN_FAILED = "run_failed"
    RUN_PENDING = "run_pending"
    MALFORMED_ARTIFACT = "malformed_artifact"
    MISSING_HYPER_CONFIG = "missing_hyper_config"


@dataclass(frozen=True)
class DatasetDiagnostic:
    """A structured issue encountered while selecting or materializing data."""

    code: DiagnosticCode
    message: str
    run_id: int | None = None
    execution_id: int | None = None


class DatasetLoadError(ValueError):
    """Raised when strict dataset loading encounters a fatal diagnostic."""

    def __init__(self, diagnostics: tuple[DatasetDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__("experiment dataset loading failed: " + "; ".join(d.message for d in diagnostics))


FATAL_DIAGNOSTIC_CODES = frozenset(
    {
        DiagnosticCode.AMBIGUOUS_COHORT,
        DiagnosticCode.MISSING_METRIC_ARTIFACT,
        DiagnosticCode.INVALID_VERSION_SELECTION,
        DiagnosticCode.MALFORMED_ARTIFACT,
        DiagnosticCode.MISSING_HYPER_CONFIG,
    }
)
