"""Behavior tests for the bakeoff-warning accumulator."""

from __future__ import annotations

from research_analysis.diagnostics.bakeoff import (
    BakeoffWarning,
    BakeoffWarningAccumulator,
    BakeoffWarningCode,
)


def test_add_appends_a_warning_with_the_given_fields() -> None:
    """add() records one warning carrying the code, message, condition, and cell."""
    accumulator = BakeoffWarningAccumulator()

    accumulator.add(
        BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY,
        "learning_rate at the upper edge of the swept grid",
        condition="sac",
        cell=(("env_name", "pendulum"),),
    )

    assert accumulator.as_tuple() == (
        BakeoffWarning(
            BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY,
            "learning_rate at the upper edge of the swept grid",
            "sac",
            (("env_name", "pendulum"),),
        ),
    )


def test_extend_appends_warnings_in_order() -> None:
    """extend() appends an iterable of warnings after any already collected."""
    accumulator = BakeoffWarningAccumulator()
    accumulator.add(BakeoffWarningCode.ZERO_VARIANCE, "first")

    accumulator.extend(
        [
            BakeoffWarning(BakeoffWarningCode.INSUFFICIENT_SEEDS, "second"),
            BakeoffWarning(BakeoffWarningCode.NON_NORMAL_DISTRIBUTION, "third"),
        ]
    )

    assert [item.message for item in accumulator.as_tuple()] == ["first", "second", "third"]


def test_separate_instances_do_not_share_state() -> None:
    """Two accumulators start independent instead of sharing one mutable default list."""
    first = BakeoffWarningAccumulator()
    second = BakeoffWarningAccumulator()

    first.add(BakeoffWarningCode.FLATLINED_LEARNING_CURVE, "only in first")

    assert second.as_tuple() == ()
