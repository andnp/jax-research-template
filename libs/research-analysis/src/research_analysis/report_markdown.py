"""Render a report document as markdown: accounting and diagnostics first, then results with stated confidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from research_analysis.dataset.types import RunAccounting
from research_analysis.diagnostics.types import DatasetDiagnostic
from research_analysis.report import AnalysisFailure, PlanStatus, ReportDocument

_LISTED_MESSAGES = 10


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    escaped = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in escaped),
    ]


def _status_section(document: ReportDocument) -> list[str]:
    lines = [f"# Report: {document.provenance.experiment}", "", f"- Status: **{document.status}**"]
    if document.plan_status is PlanStatus.DECLARED:
        lines.append(f"- Plan: **declared** ({document.plan_fingerprint})")
    else:
        lines.append("- Plan: **exploratory**")
        lines.extend(f"  - {reason}" for reason in document.exploratory_reasons)
    return lines


def _provenance_section(document: ReportDocument) -> list[str]:
    provenance = document.provenance
    commits = sorted({item.git_commit or "unrecorded" for item in provenance.executions})
    dirty = sum(item.dirty for item in provenance.executions)
    lines = [
        "## Provenance",
        "",
        f"- Experiments database: `{provenance.experiments_db}`",
        f"- Generated: {provenance.generated_at} (report schema v{document.schema_version})",
        f"- Executions: {len(provenance.executions)}; git commits: {', '.join(commits) or 'none'}",
    ]
    if dirty:
        lines.append(f"- **{dirty} execution(s) ran with uncommitted changes**; their commit alone does not reproduce them.")
    if provenance.component_versions:
        rows = [
            [item.component_type, item.component_name, str(item.version_number), item.code_snapshot_hash[:12]]
            for item in provenance.component_versions
        ]
        lines += ["", *_table(["Type", "Component", "Version", "Code hash"], rows)]
    return lines


def _accounting_section(accounting: RunAccounting) -> list[str]:
    rows = [
        [
            ", ".join(f"{axis}={value}" for axis, value in cell.cell),
            str(cell.planned),
            str(cell.completed),
            str(cell.failed),
            str(cell.pending),
            ", ".join(f"{metric}: {count}" for metric, count in cell.diverged.items()) or "0",
            ", ".join(f"{reason}: {count}" for reason, count in cell.excluded.items()) or "0",
        ]
        for cell in accounting.cells
    ]
    header = ["Cell", "Planned", "Completed", "Failed", "Pending", "Diverged (counted as completed)", "Excluded"]
    lines = ["## Run accounting", "", *_table(header, rows)]
    if accounting.unassigned:
        placed = ", ".join(f"{reason}: {count}" for reason, count in accounting.unassigned.items())
        lines += ["", f"Planned runs placed in no cell: {placed}"]
    return lines


def _diagnostic_lines(diagnostics: tuple[DatasetDiagnostic, ...]) -> list[str]:
    lines: list[str] = []
    for code, count in Counter(item.code for item in diagnostics).items():
        messages = [item.message for item in diagnostics if item.code is code]
        lines.append(f"- `{code}` ({count})")
        lines.extend(f"  - {message}" for message in messages[:_LISTED_MESSAGES])
        if count > _LISTED_MESSAGES:
            lines.append(f"  - ... {count - _LISTED_MESSAGES} more in the JSON report")
    return lines


def _diagnostics_section(document: ReportDocument) -> list[str]:
    lines = ["## Diagnostics", ""]
    if document.fatal_diagnostics:
        lines += ["**Fatal: no results are reported.**", "", *_diagnostic_lines(document.fatal_diagnostics), ""]
    if document.non_fatal_diagnostics:
        lines += ["Non-fatal:", "", *_diagnostic_lines(document.non_fatal_diagnostics)]
    if not document.fatal_diagnostics and not document.non_fatal_diagnostics:
        lines.append("None.")
    return lines


def _failure_section(failure: AnalysisFailure) -> list[str]:
    return ["## Analysis failed", "", f"`{failure.analysis}` raised {failure.error_type}: {failure.message}"]


def render_markdown(document: ReportDocument) -> str:
    """The whole document as markdown; every number shown is also in the JSON report."""
    sections = [
        _status_section(document),
        _provenance_section(document),
        _accounting_section(document.accounting),
        _diagnostics_section(document),
    ]
    if document.failure is not None:
        sections.append(_failure_section(document.failure))
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
