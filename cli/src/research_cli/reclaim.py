"""Release the executions of dead owners before a launch plans new ones (ADR 014)."""

from __future__ import annotations

from experiment_definition.db import DatabaseManager, ReclaimReport
from research_cluster.ownership import owner_state


def describe_reclaim(report: ReclaimReport) -> str:
    line = f"Reclaimed {len(report.failed)} dead execution(s)"
    stuck = [
        f"{len(ids)} {reason}"
        for ids, reason in ((report.unknown, "with an owner that cannot be checked"), (report.ownerless, "without an owner"))
        if ids
    ]
    if not stuck:
        return line
    return f"{line}; {' and '.join(stuck)} stay live: release them with `research experiment invalidate --execution <id>`."


def reclaim_dead(database: DatabaseManager, experiment_id: int) -> str:
    """Fail the experiment's executions whose owner is dead and describe the outcome in one line."""
    return describe_reclaim(database.reclaim_dead_executions(experiment_id, owner_state))
