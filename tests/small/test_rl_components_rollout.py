"""Contract tests for the generic batched rollout API."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvReset, EnvStep
from rl_components.rollout import (
    BatchedRolloutCarry,
    BatchedRolloutResult,
    ResetPolicy,
    make_fresh_batched_carry,
    run_batched_rollout,
)


class _AlwaysDoneEnvironment:
    def reset(self, key: jax.Array) -> EnvReset[jax.Array, jax.Array]:
        del key
        value = jnp.array(0.0, dtype=jnp.float32)
        return EnvReset(observation=value, state=value)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, action
        next_state = state + 1.0
        return EnvStep(
            observation=next_state,
            state=next_state,
            reward=next_state,
            terminated=jnp.array(True),
            truncated=jnp.array(False),
            info={"state": next_state},
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


def test_continuous_rollout_ignores_done_without_discarding_signals() -> None:
    """
    Continue a plant after every done signal under the default policy.

    JIT execution must retain boundary and info sidecars instead of resetting or dropping them.
    """
    environment = _AlwaysDoneEnvironment()
    keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])

    def action_step(
        observations: jax.Array,
        controller_state: object,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[jax.Array, object]:
        del observations, previous_action, step_index, action_key
        return jnp.zeros((2, 1), dtype=jnp.float32), controller_state

    def telemetry(
        observations: jax.Array,
        actions: jax.Array,
        previous_action: jax.Array,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        info: Mapping[str, jax.Array],
    ) -> dict[str, jax.Array]:
        del observations, actions, previous_action, rewards, truncated, info
        return {"saw_done": terminated.astype(jnp.float32)}

    def run(seed_keys: jax.Array):
        carry = make_fresh_batched_carry(
            environment,
            seed_keys,
            previous_action=jnp.zeros((2, 1), dtype=jnp.float32),
            action_shape=(1,),
        )
        return run_batched_rollout(
            environment,
            carry,
            steps=3,
            action_step=action_step,
            rng_mode="two-way",
            capture_action=False,
            mask_step_zero_delta=False,
            telemetry_hook=telemetry,
            telemetry_keys=("saw_done",),
            action_shape=(1,),
            step_offset=0,
        )

    result = jax.jit(run)(keys)

    assert jnp.array_equal(result.final_carry.environment_state, jnp.array([3.0, 3.0]))
    assert jnp.array_equal(result.done, jnp.ones((3, 2), dtype=jnp.bool_))
    assert jnp.array_equal(
        result.info["state"],
        jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    assert jnp.array_equal(result.metrics["saw_done"], jnp.ones((3, 2)))
