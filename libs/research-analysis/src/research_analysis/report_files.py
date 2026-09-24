"""Write one report document's files into a directory under the durable results root."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from research_analysis.report import ReportDocument
from research_analysis.report_json import report_to_json
from research_analysis.report_markdown import render_markdown
from research_analysis.report_plots import write_report_plots


def report_directory(results_root: Path, experiment: str) -> Path:
    """A new ``<results_root>/reports/<experiment>/<UTC time>/`` path, the default home of one report."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return results_root / "reports" / re.sub(r"[^A-Za-z0-9._-]+", "_", experiment) / stamp


def write_report_document(document: ReportDocument, output_dir: Path) -> ReportDocument:
    """Write the figures, ``report.json``, and ``report.md``, returning the document that names its figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = write_report_plots(document, output_dir)
    (output_dir / "report.json").write_text(report_to_json(document))
    (output_dir / "report.md").write_text(render_markdown(document))
    return document
