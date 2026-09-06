"""Unit tests for standalone conditional sweeps."""

from __future__ import annotations

import pytest
from experiment_definition.sweep import Sweep, values, when


def test_resolve_preserves_axis_order_and_full_configurations() -> None:
    """Resolve valid combinations without dropping declared axis keys."""
    sweep = Sweep(
        axes={"agent": values("dqn", "ppo"), "learning_rate": values(1e-3, 3e-4)},
        constraints=[when("agent", "dqn").require("learning_rate", 1e-3)],
    )

    assert sweep.resolve() == [
        {"agent": "dqn", "learning_rate": 1e-3},
        {"agent": "ppo", "learning_rate": 1e-3},
        {"agent": "ppo", "learning_rate": 3e-4},
    ]


def test_resolve_rejects_unknown_constraint_axes() -> None:
    """Reject malformed constraints instead of treating them as non-matches."""
    with pytest.raises(ValueError, match="unknown axis: optimizer"):
        Sweep(axes={"agent": values("dqn")}, constraints=[when("optimizer", "adam").require("agent", "dqn")])
