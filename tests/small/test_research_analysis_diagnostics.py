"""Behavior tests for the diagnostics accumulator."""

from __future__ import annotations

from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode


def test_add_appends_a_diagnostic_with_the_given_fields() -> None:
    """add() records one diagnostic carrying the code, message, and run identity."""
    accumulator = DiagnosticAccumulator()

    accumulator.add(DiagnosticCode.MISSING_DIMENSION, "missing condition key", run_id=1, execution_id=2)

    assert accumulator.as_tuple() == (
        DatasetDiagnostic(DiagnosticCode.MISSING_DIMENSION, "missing condition key", 1, 2),
    )


def test_extend_appends_diagnostics_in_order() -> None:
    """extend() appends an iterable of diagnostics after any already collected."""
    accumulator = DiagnosticAccumulator()
    accumulator.add(DiagnosticCode.EXCLUDED_VERSION, "first")

    accumulator.extend(
        [
            DatasetDiagnostic(DiagnosticCode.EXCLUDED_VERSION, "second"),
            DatasetDiagnostic(DiagnosticCode.AMBIGUOUS_COHORT, "third"),
        ]
    )

    assert [item.message for item in accumulator.as_tuple()] == ["first", "second", "third"]


def test_fatal_returns_only_diagnostics_matching_the_given_codes() -> None:
    """fatal() filters collected diagnostics down to a caller-supplied fatal-code set."""
    accumulator = DiagnosticAccumulator()
    accumulator.add(DiagnosticCode.MISSING_DIMENSION, "fatal")
    accumulator.add(DiagnosticCode.EXCLUDED_VERSION, "not fatal")

    fatal = accumulator.fatal(frozenset({DiagnosticCode.MISSING_DIMENSION}))

    assert [item.message for item in fatal] == ["fatal"]


def test_fatal_returns_empty_tuple_when_no_codes_match() -> None:
    """fatal() with a code set that matches nothing returns an empty tuple, not None."""
    accumulator = DiagnosticAccumulator()
    accumulator.add(DiagnosticCode.EXCLUDED_VERSION, "not fatal")

    assert accumulator.fatal(frozenset({DiagnosticCode.MISSING_DIMENSION})) == ()


def test_separate_instances_do_not_share_state() -> None:
    """Two accumulators start independent instead of sharing one mutable default list."""
    first = DiagnosticAccumulator()
    second = DiagnosticAccumulator()

    first.add(DiagnosticCode.EXCLUDED_VERSION, "only in first")

    assert second.as_tuple() == ()
