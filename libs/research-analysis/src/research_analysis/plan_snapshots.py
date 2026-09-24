"""Append-only history of an experiment's declared analysis plans, recorded when its runs are launched."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from research_analysis.plan import AnalysisPlan
from research_analysis.report_json import plan_fingerprint, to_jsonable

PLAN_HISTORY_FILE = "analysis-plans.jsonl"


@dataclass(frozen=True)
class PlanSnapshot:
    """One plan as declared at a moment; ``plan_json`` is its canonical JSON."""

    fingerprint: str
    recorded_at: datetime
    plan_json: str


def read_plan_snapshots(results_root: Path) -> tuple[PlanSnapshot, ...]:
    """Every snapshot recorded under ``results_root``, oldest first; empty when none was ever recorded."""
    path = results_root / PLAN_HISTORY_FILE
    if not path.exists():
        return ()
    snapshots = []
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        snapshots.append(
            PlanSnapshot(entry["fingerprint"], datetime.fromisoformat(entry["recorded_at"]), json.dumps(entry["plan"], sort_keys=True))
        )
    return tuple(snapshots)


def record_plan_snapshot(plan: AnalysisPlan, results_root: Path) -> PlanSnapshot:
    """Append ``plan`` to the history unless it is already the latest entry; earlier entries are never rewritten."""
    fingerprint = plan_fingerprint(plan)
    history = read_plan_snapshots(results_root)
    if history and history[-1].fingerprint == fingerprint:
        return history[-1]

    snapshot = PlanSnapshot(fingerprint, datetime.now(UTC), json.dumps(to_jsonable(plan), sort_keys=True))
    entry = {"fingerprint": fingerprint, "recorded_at": snapshot.recorded_at.isoformat(), "plan": to_jsonable(plan)}
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / PLAN_HISTORY_FILE).open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
    return snapshot
