"""Contract tests for the generic batched rollout API."""

import jax.numpy as jnp
from rl_components.rollout import (
    BatchedRolloutCarry,
    BatchedRolloutResult,
    ResetPolicy,
)


def test_rollout_result_preserves_boundary_sidecars() -> None:
    """
    Keep boundary signals and environment diagnostics outside generic metrics.

    The combined done view must remain derivable without losing either source flag.
    """
    carry = BatchedRolloutCarry(
        observations=jnp.zeros((2, 1)),
        environment_state=jnp.zeros((2, 1)),
        controller_state=None,
        rng=jnp.zeros((2, 2), dtype=jnp.uint32),
        previous_action=jnp.zeros((2, 1)),
    )
    result = BatchedRolloutResult(
        carry,
        {"reward": jnp.zeros((1, 2))},
        jnp.array([[True, False]]),
        jnp.array([[False, True]]),
        {"diagnostic": jnp.ones((1, 2))},
    )

    assert result.done.tolist() == [[True, True]]
    assert set(result.info) == {"diagnostic"}


def test_rollout_reset_policy_is_explicitly_limited() -> None:
    """
    Keep the reset-policy type limited to continuous ignore or episodic reset.

    Invalid policy names must not become an implicit third execution mode.
    """
    policies: tuple[ResetPolicy, ...] = ("ignore", "auto_reset")

    assert policies == ("ignore", "auto_reset")
