"""Append-only history of an experiment's declared analysis plans, recorded when its runs are launched."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from research_analysis.declared_plan import PlanInterpretationError, plan_fingerprint, recorded_plan_fingerprint
from research_analysis.plan import AnalysisPlan
from research_analysis.report_json import JsonValue, to_jsonable

PLAN_HISTORY_FILE = "analysis-plans.jsonl"


@dataclass(frozen=True)
class PlanSnapshot:
    """One plan as declared at a moment; ``fingerprint`` is recomputed from ``plan_json``, its canonical JSON.

    ``recorded_at`` is None when the persisted timestamp could not be trusted to order
    this snapshot against execution start times, and ``fingerprint`` is None when the
    recorded plan does not fit the current schema; ``invalid_reason`` then says why.
    """

    fingerprint: str | None
    recorded_at: datetime | None
    plan_json: str
    invalid_reason: str | None = None


def _parse_recorded_at(text: str) -> tuple[datetime | None, str | None]:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError:
        return None, f"recorded_at {text!r} is not an ISO time"
    if instant.tzinfo is None:
        return None, f"recorded_at {text!r} is not an ISO time with a UTC offset"
    return instant.astimezone(UTC), None


def _recorded_fingerprint(plan_json: JsonValue) -> tuple[str | None, str | None]:
    try:
        return recorded_plan_fingerprint(plan_json), None
    except PlanInterpretationError as error:
        return None, f"recorded plan does not fit the current plan schema: {error}"


def read_plan_snapshots(results_root: Path) -> tuple[PlanSnapshot, ...]:
    """Every snapshot recorded under ``results_root``, oldest first; empty when none was ever recorded.

    A snapshot whose ``recorded_at`` is naive or malformed (a hand-edited history file), or whose
    plan the current schema cannot read, is never guessed at: it is returned with its ``invalid_reason``.
    """
    path = results_root / PLAN_HISTORY_FILE
    if not path.exists():
        return ()
    snapshots = []
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        recorded_at, time_problem = _parse_recorded_at(entry["recorded_at"])
        fingerprint, plan_problem = _recorded_fingerprint(entry["plan"])
        invalid_reason = "; ".join(problem for problem in (time_problem, plan_problem) if problem is not None) or None
        plan_json = json.dumps(entry["plan"], sort_keys=True)
        snapshots.append(PlanSnapshot(fingerprint, recorded_at, plan_json, invalid_reason=invalid_reason))
    return tuple(snapshots)


def record_plan_snapshot(plan: AnalysisPlan, results_root: Path) -> PlanSnapshot:
    """Append ``plan`` to the history unless it is already the latest entry; earlier entries are never rewritten."""
    fingerprint = plan_fingerprint(plan)
    history = read_plan_snapshots(results_root)
    if history and history[-1].fingerprint == fingerprint:
        return history[-1]

    recorded_at = datetime.now(UTC)
    snapshot = PlanSnapshot(fingerprint, recorded_at, json.dumps(to_jsonable(plan), sort_keys=True))
    entry = {"fingerprint": fingerprint, "recorded_at": recorded_at.isoformat(), "plan": to_jsonable(plan)}
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / PLAN_HISTORY_FILE).open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
    return snapshot
