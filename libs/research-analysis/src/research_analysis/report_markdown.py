"""Render a report document as markdown: accounting and diagnostics first, then results with stated confidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from research_analysis._reporting_models import BakeoffReport, CellSummary, HyperparameterConfigReport, PairedDifference
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import RunAccounting
from research_analysis.diagnostics.bakeoff import BakeoffWarning
from research_analysis.diagnostics.types import DatasetDiagnostic
from research_analysis.performance import FinalValue, PerformanceMeasure, ReturnRate, TailMean
from research_analysis.plan import (
    Analysis,
    AnalysisPlan,
    BakeoffAnalysis,
    BestHypersAnalysis,
    HypersDifferenceAnalysis,
    SliceHypersAnalysis,
)
from research_analysis.report import (
    AnalysisFailure,
    AnalysisResult,
    BakeoffResult,
    BestHypersResult,
    HypersDifferenceResult,
    PlanStatus,
    ReportDocument,
    SliceHypersResult,
)
from research_analysis.toolchain import AnalysisToolchain

_LISTED_MESSAGES = 10


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    escaped = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in escaped),
    ]


def _number(value: float) -> str:
    return f"{value:.4g}"


def _interval(interval: tuple[float, float] | None) -> str:
    return "—" if interval is None else f"[{interval[0]:.4g}, {interval[1]:.4g}]"


def _percent(confidence: float) -> str:
    return f"{confidence * 100:.4g}%"


def _describe_measure(measure: PerformanceMeasure) -> str:
    match measure:
        case FinalValue():
            return "final logged value"
        case TailMean(fraction=fraction):
            return f"mean over the final {fraction:.0%} of logged steps"
        case ReturnRate():
            return "return rate (mean over every logged point)"


def _status_section(document: ReportDocument) -> list[str]:
    lines = [f"# Report: {document.provenance.experiment}", "", f"- Status: **{document.status}**"]
    if document.plan_status is PlanStatus.DECLARED:
        lines.append(f"- Plan: **declared** ({document.plan_fingerprint})")
    else:
        lines.append("- Plan: **exploratory**")
        lines.extend(f"  - {reason}" for reason in document.exploratory_reasons)
    return lines


def _describe_analysis(analysis: Analysis) -> str:
    if analysis.measure is None:
        return _describe_analysis_kind(analysis)
    return f"{_describe_analysis_kind(analysis)}; per run: {_describe_measure(analysis.measure)}"


def _describe_analysis_kind(analysis: Analysis) -> str:
    match analysis:
        case BakeoffAnalysis(name=name, conditions=conditions, pairs=pairs, block_by=block_by):
            planned = "every pair" if pairs is None else ", ".join(f"{a} − {b}" for a, b in pairs)
            champions = "none" if analysis.select_champion_over is None else ", ".join(analysis.select_champion_over)
            return (
                f"`{name}`: bakeoff of {', '.join(conditions)}; pairs: {planned}; blocked by: "
                f"{', '.join(block_by) or 'nothing'}; shared across: {', '.join(analysis.shared_across) or 'nothing'}; "
                f"champion over: {champions}; untuned: {', '.join(analysis.untuned) or 'none'}; "
                f"min seeds {analysis.min_seeds}"
            )
        case BestHypersAnalysis(name=name, group_by=group_by):
            return f"`{name}`: champion per {', '.join(group_by) or 'whole grid'}"
        case HypersDifferenceAnalysis(name=name, group_axis=axis, minuend=minuend, subtrahend=subtrahend):
            return f"`{name}`: champion difference {axis}={minuend} minus {axis}={subtrahend}"
        case SliceHypersAnalysis(name=name, vary=vary, group_by=group_by):
            return f"`{name}`: sensitivity to {vary} per {', '.join(group_by) or 'whole grid'}"


def _describe_interval(spec: BakeoffSpec) -> str:
    return f"{spec.interval.replace('_', ' ')} bootstrap"


def _plan_section(plan: AnalysisPlan) -> list[str]:
    stats = plan.stats
    dimensions = plan.dimensions
    swept = ", ".join(f"{axis.name} ({axis.value_type})" for axis in dimensions.sweep_axes) or "none"
    diverged = (
        "fail the report" if plan.non_finite_worst_case is None else f"score {_number(plan.non_finite_worst_case)}"
    )
    return [
        "## Declared plan",
        "",
        f"- Metric: `{plan.metric}` ({'higher' if plan.higher_is_better else 'lower'} is better), per run: "
        f"{_describe_measure(plan.measure)}; estimand: {stats.estimand}",
        f"- Dimensions: condition `{dimensions.condition}`, environment `{dimensions.environment or 'none'}`, "
        f"swept: {swept}",
        f"- Statistics: {_percent(stats.confidence_level)} {_describe_interval(stats)}, {stats.n_resamples} resamples, "
        f"seed {stats.random_seed}, min samples {stats.min_samples}; missingness: {stats.missingness}; tolerance "
        f"({_percent(stats.tolerance_confidence)} confidence, {_percent(stats.tolerance_coverage)} coverage)",
        f"- Diverged runs: {diverged}",
        "- Analyses:",
        *(f"  - {_describe_analysis(analysis)}" for analysis in plan.analyses),
    ]


def _describe_toolchain(toolchain: AnalysisToolchain) -> str:
    changes = {True: " with uncommitted changes", False: "", None: ", uncommitted changes unknown"}[toolchain.dirty]
    return f"research-analysis {toolchain.package_version} at commit {toolchain.git_commit}{changes}"


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
        f"- Analysis code: {_describe_toolchain(provenance.toolchain)}",
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
    lines = ["## Analysis failed: no results are reported", "", f"`{failure.analysis}` raised {failure.error_type}: {failure.message}"]
    if failure.non_finite_runs:
        rows = [[", ".join(f"{key}={value}" for key, value in run.items())] for run in failure.non_finite_runs]
        lines += ["", *_table(["Diverged run"], rows)]
    if failure.incomplete_cells:
        rows = [
            [cell.condition, cell.block or "—", ", ".join(map(str, cell.missing_seeds)), str(cell.failed), str(cell.pending), str(cell.excluded)]
            for cell in failure.incomplete_cells
        ]
        lines += ["", *_table(["Condition", "Block", "Missing seeds", "Failed", "Pending", "Excluded"], rows)]
    if failure.insufficient_data:
        rows = [
            [", ".join(f"{key}={value}" for key, value in cell.coordinates.items()) or "—", cell.required, cell.observed]
            for cell in failure.insufficient_data
        ]
        lines += ["", *_table(["Cell", "Required", "Observed"], rows)]
    return lines


def _cell_row(condition: str, block: str, cell: CellSummary) -> list[str]:
    return [
        condition,
        block or "—",
        str(cell.n),
        "—" if cell.estimate is None else _number(cell.estimate),
        _interval(cell.ci) if cell.ci_skipped_reason is None else f"none: {cell.ci_skipped_reason}",
        "—" if cell.sd is None else _number(cell.sd),
        _interval(cell.tolerance) if cell.tolerance is not None else f"needs n ≥ {cell.tolerance_min_runs}",
        "—" if cell.median_run_id is None else str(cell.median_run_id),
    ]


def _difference_row(item: PairedDifference) -> list[str]:
    if item.estimate is None:
        return [item.block or "—", f"{item.minuend} − {item.subtrahend}", "—", "—", str(item.n_paired), "—", f"skipped: {item.skipped_reason}"]
    p_value = "—" if item.p_value is None else f"{item.p_value:.4g}"
    return [
        item.block or "—",
        f"{item.minuend} − {item.subtrahend}",
        _number(item.estimate),
        f"{_interval(item.ci)} at {_percent(item.confidence)}",
        str(item.n_paired),
        p_value,
        item.method,
    ]


def _warning_lines(warnings: tuple[BakeoffWarning, ...]) -> list[str]:
    lines = [f"- `{item.code}` {item.condition or '(all)'} {dict(item.cell) or ''}: {item.message}" for item in warnings]
    return ["", f"Warnings ({len(warnings)}):", "", *(lines or ["- none"])]


def _reuse_lines(report: BakeoffReport) -> list[str]:
    lines: list[str] = []
    for axis, conditions in report.shared_across.items():
        reused = [run for run in report.runs if run.condition in conditions]
        lines.append(
            f"- Shared across `{axis}`: {', '.join(conditions) or 'no condition'} lack `{axis}`, so their runs are reused "
            f"in every `{axis}` block; {len({run.run_id for run in reused})} distinct runs appear in the reported cells "
            f"({len(reused)} cell entries); run accounting counts each once"
        )
    if report.untuned:
        lines.append(f"- Untuned referents (one configuration, no selection stage): {', '.join(report.untuned)}")
    return lines


def _bakeoff_section(name: str, report: BakeoffReport) -> list[str]:
    spec = report.spec
    lines = [
        f"## {name}: bakeoff of {', '.join(report.conditions)}",
        "",
        f"- Metric: `{report.metric_name}`, per run: {_describe_measure(report.measure)}; estimand: {spec.estimand}",
        f"- {_describe_interval(spec).capitalize()}, {spec.n_resamples} resamples, seed {spec.random_seed}; "
        f"missingness: {spec.missingness}",
        *_reuse_lines(report),
        "",
        f"Cells: {spec.estimand} with its {_percent(spec.confidence_level)} CI; SD and the "
        f"({_percent(spec.tolerance_confidence)} confidence, {_percent(spec.tolerance_coverage)} coverage) "
        "tolerance interval describe run-to-run variability.",
        "",
    ]
    cell_rows = [
        _cell_row(condition, block, report.cells[condition][block])
        for block in report.blocks
        for condition in report.conditions
    ]
    header = ["Condition", "Block", "n", spec.estimand.capitalize(), "CI", "SD", "Tolerance interval", "Median run"]
    lines += _table(header, cell_rows)
    lines += [
        "",
        f"Paired differences: {report.family_size} comparisons, each CI at {_percent(report.comparison_confidence)} "
        f"so all hold jointly at {_percent(spec.confidence_level)} (Bonferroni); p-values are Holm-adjusted.",
        "",
        *_table(["Block", "Difference", "Estimate", "CI", "n paired", "Holm p", "Method"], [_difference_row(item) for item in report.differences]),
    ]
    if report.omnibus is not None:
        lines += ["", f"Omnibus: {report.omnibus.test_name}, p = {report.omnibus.p_value:.4g}."]
    if report.omnibus_skipped_reason is not None:
        lines += ["", f"No omnibus test: {report.omnibus_skipped_reason}."]
    if report.tests_skipped_reason is not None:
        lines += ["", f"No hypothesis tests: {report.tests_skipped_reason}."]
    if report.champions is not None:
        champion_rows = [
            [
                block or "—",
                condition,
                ", ".join(f"{key}={value}" for key, value in champion.winning_config.items()),
                _number(champion.corrected_mean),
                f"{_interval(champion.estimate_ci)} at {_percent(champion.confidence_level)}",
                _number(champion.maximization_bias),
                f"{champion.normalized_selection_entropy:.2f}",
            ]
            for block, per_condition in report.champions.items()
            for condition, champion in per_condition.items()
        ]
        header = ["Block", "Condition", "Champion", "Corrected mean", "CI", "Maximization bias", "Selection entropy"]
        lines += ["", "Champions (their cells above carry selection bias; differences do not):", "", *_table(header, champion_rows)]
    return lines + _warning_lines(report.warnings)


def _group_label(group: Mapping[str, object]) -> str:
    return ", ".join(f"{axis}={value}" for axis, value in group.items()) or "whole grid"


def _config(config: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in config.items())


def _champion_lines(report: HyperparameterConfigReport) -> list[str]:
    return [
        f"- Champion: {_config(report.winning_config)}, scored per run by {_describe_measure(report.measure)}",
        f"- Corrected mean {_number(report.corrected_mean)}, {_percent(report.confidence_level)} CI "
        f"{_interval(report.estimate_ci)} (hierarchical bootstrap over the selection distribution)",
        f"- Raw champion mean {_number(report.raw_mean)}; maximization bias {_number(report.maximization_bias)}; "
        f"normalized selection entropy {report.normalized_selection_entropy:.2f}",
        f"- Runs outside the shared seeds, left out: {len(report.unpaired_run_ids)}",
        *_warning_lines(report.warnings),
    ]


def _analysis_section(result: AnalysisResult) -> list[str]:
    match result:
        case BakeoffResult(name=name, report=report):
            return _bakeoff_section(name, report)
        case BestHypersResult(name=name, group=group, report=report):
            return [f"## {name}: champion for {_group_label(group)}", "", *_champion_lines(report)]
        case HypersDifferenceResult(name=name, report=report):
            return [
                f"## {name}: champion difference",
                "",
                f"- {_config(report.minuend_config)} minus {_config(report.subtrahend_config)}",
                f"- Estimate {_number(report.mean_diff)}, {_percent(report.confidence_level)} CI "
                f"{_interval(report.estimate_ci)} (hierarchical bootstrap, selection included)",
                f"- Runs outside the shared seeds, left out: {len(report.unpaired_run_ids)}",
                *_warning_lines(report.warnings),
            ]
        case SliceHypersResult(name=name, group=group, report=report):
            rows = [
                [str(value), _number(best), "not run" if pinned is None else _number(pinned)]
                for value, best, pinned in zip(report.values, report.best_performance, report.slice_performance, strict=True)
            ]
            header = [report.vary, "Best achievable (selection-corrected)", "Champion-pinned slice (raw mean)"]
            return [f"## {name}: sensitivity to {report.vary} for {_group_label(group)}", "", *_table(header, rows)]


def _figures_section(document: ReportDocument) -> list[str]:
    return ["## Figures", "", *(f"- {name}: ![{name}]({path})" for name, path in document.plots.items())]


def render_markdown(document: ReportDocument) -> str:
    """The whole document as markdown; every number shown is also in the JSON report."""
    sections = [_status_section(document)]
    if document.plan is not None:
        sections.append(_plan_section(document.plan))
    sections += [
        _provenance_section(document),
        _accounting_section(document.accounting),
        _diagnostics_section(document),
    ]
    if document.failure is not None:
        sections.append(_failure_section(document.failure))
    sections.extend(_analysis_section(result) for result in document.analyses)
    if document.plots:
        sections.append(_figures_section(document))
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def _headline_lines(result: AnalysisResult) -> list[str]:
    match result:
        case BakeoffResult(name=name, report=report):
            lines = [f"- `{name}`: {len(report.warnings)} warning(s)"]
            for item in report.differences:
                block = f" [{item.block}]" if item.block else ""
                if item.estimate is None:
                    lines.append(f"  - {item.minuend} − {item.subtrahend}{block}: skipped, {item.skipped_reason}")
                    continue
                lines.append(
                    f"  - {item.minuend} − {item.subtrahend}{block}: {_number(item.estimate)} "
                    f"{_interval(item.ci)} at {_percent(item.confidence)} (n={item.n_paired})"
                )
            return lines
        case BestHypersResult(name=name, group=group, report=report):
            return [
                f"- `{name}` ({_group_label(group)}): {_config(report.winning_config)}, corrected mean "
                f"{_number(report.corrected_mean)} {_interval(report.estimate_ci)} at {_percent(report.confidence_level)}"
            ]
        case HypersDifferenceResult(name=name, report=report):
            return [
                f"- `{name}`: {_number(report.mean_diff)} {_interval(report.estimate_ci)} at {_percent(report.confidence_level)}"
            ]
        case SliceHypersResult(name=name, group=group, report=report):
            return [f"- `{name}` ({_group_label(group)}): sensitivity to {report.vary} over {len(report.values)} values"]


def render_summary(document: ReportDocument) -> str:
    """A few lines for a console: status, what the runs amount to, what went wrong, and the headline numbers."""
    cells = document.accounting.cells
    totals = {
        "planned": sum(cell.planned for cell in cells),
        "completed": sum(cell.completed for cell in cells),
        "failed": sum(cell.failed for cell in cells),
        "pending": sum(cell.pending for cell in cells),
        "diverged": sum(sum(cell.diverged.values()) for cell in cells),
        "excluded": sum(sum(cell.excluded.values()) for cell in cells) + sum(document.accounting.unassigned.values()),
    }
    lines = _status_section(document)[2:]
    lines.append("- Runs: " + ", ".join(f"{count} {state}" for state, count in totals.items()))
    for label, diagnostics in (("Fatal", document.fatal_diagnostics), ("Non-fatal", document.non_fatal_diagnostics)):
        if diagnostics:
            counts = Counter(str(item.code) for item in diagnostics)
            lines.append(f"- {label} diagnostics: " + ", ".join(f"{code} ({count})" for code, count in counts.items()))
    if document.failure is not None:
        lines.append(f"- Analysis `{document.failure.analysis}` failed: {document.failure.error_type}")
    for result in document.analyses:
        lines.extend(_headline_lines(result))
    return "\n".join(lines) + "\n"
