"""Identify the analysis code that produced a report, so a changed default shows up in its provenance."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

UNKNOWN = "unknown"
_SOURCE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AnalysisToolchain:
    """The research-analysis version and the commit of the repository holding its source.

    ``dirty`` says whether that source has uncommitted changes; it is None when git cannot tell.
    """

    package_version: str
    git_commit: str
    dirty: bool | None


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_SOURCE), *arguments], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_version() -> str:
    try:
        return version("research-analysis")
    except PackageNotFoundError:
        return UNKNOWN


@cache
def analysis_toolchain() -> AnalysisToolchain:
    """This process's analysis code; an installed copy outside a git checkout has an unknown commit."""
    commit = _git("rev-parse", "HEAD")
    changes = None if commit is None else _git("status", "--porcelain", "--", ".")
    return AnalysisToolchain(_package_version(), commit or UNKNOWN, None if changes is None else bool(changes))
