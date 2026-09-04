"""Contract tests for the generic batched rollout API."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
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


class _ProbeEnvironment:
    def __init__(self, action_shape: tuple[int, ...] = (1,)) -> None:
        self.action_shape = action_shape

    def reset(self, key: jax.Array) -> EnvReset[jax.Array, jax.Array]:
        del key
        value = jnp.zeros(self.action_shape, dtype=jnp.float32)
        return EnvReset(observation=value, state=value)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> EnvStep[jax.Array, jax.Array]:
        next_state = state + action
        return EnvStep(
            observation=next_state,
            state=next_state,
            reward=jnp.sum(action),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={"step_key": key, "seen_action": action},
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


def test_rollout_preserves_three_way_rng_streams_under_jit() -> None:
    """
    Keep action, environment, and continuation RNG streams independent.

    The expected keys are calculated from the public split contract rather than implementation details.
    """
    environment = _ProbeEnvironment()
    seed_keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])

    def action_step(
        observations: jax.Array,
        controller_state: object,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[jax.Array, object]:
        del observations, controller_state, previous_action, step_index
        assert action_key is not None
        return jax.vmap(lambda key: jax.random.uniform(key, shape=(1,)))(action_key), None

    carry = make_fresh_batched_carry(
        environment,
        seed_keys,
        previous_action=jnp.zeros((2, 1), dtype=jnp.float32),
        action_shape=(1,),
    )
    result = jax.jit(
        lambda initial: run_batched_rollout(
            environment,
            initial,
            steps=3,
            action_step=action_step,
            rng_mode="three-way",
            capture_action=True,
            mask_step_zero_delta=True,
            action_shape=(1,),
            step_offset=0,
        )
    )(carry)

    expected_actions = []
    expected_keys = []
    expected_rng = []
    for seed_key in seed_keys:
        _, rng = jax.random.split(seed_key)
        actions = []
        keys = []
        for _ in range(3):
            action_key, environment_key, rng = jax.random.split(rng, 3)
            actions.append(jax.random.uniform(action_key, shape=(1,)))
            keys.append(environment_key)
        expected_actions.append(jnp.stack(actions))
        expected_keys.append(jnp.stack(keys))
        expected_rng.append(rng)

    assert jnp.array_equal(result.metrics["action"], jnp.stack(expected_actions, axis=1)[..., 0])
    assert jnp.array_equal(result.info["step_key"], jnp.stack(expected_keys, axis=1))
    assert jnp.array_equal(result.final_carry.rng, jnp.stack(expected_rng))
    assert jnp.array_equal(result.metrics["action_delta_norm"][0], jnp.zeros((2,)))


def test_rollout_continuation_matches_uninterrupted_scan() -> None:
    """
    Continue from a prior carry without changing state, RNG, or callback indexing.

    A split scan must produce the same suffix as one uninterrupted scan.
    """
    environment = _ProbeEnvironment()
    seed_keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])

    def action_step(
        observations: jax.Array,
        controller_state: object,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[jax.Array, object]:
        del observations, controller_state, action_key
        return previous_action + step_index + 1.0, None

    initial = make_fresh_batched_carry(
        environment,
        seed_keys,
        previous_action=jnp.zeros((2,), dtype=jnp.float32),
        action_shape=(1,),
    )

    def run(
        carry: BatchedRolloutCarry[jax.Array, object],
        steps: int,
        offset: int,
    ):
        return run_batched_rollout(
            environment,
            carry,
            steps=steps,
            action_step=action_step,
            rng_mode="two-way",
            capture_action=True,
            mask_step_zero_delta=False,
            action_shape=(1,),
            step_offset=offset,
        )

    first = jax.jit(lambda carry: run(carry, 1, 0))(initial)
    continuation = jax.jit(lambda carry: run(carry, 1, 1))(first.final_carry)
    uninterrupted = jax.jit(lambda carry: run(carry, 2, 0))(initial)

    assert jnp.array_equal(
        continuation.metrics["action"], uninterrupted.metrics["action"][1:]
    )
    assert jnp.array_equal(
        continuation.metrics["reward"], uninterrupted.metrics["reward"][1:]
    )
    assert jnp.array_equal(
        continuation.final_carry.environment_state,
        uninterrupted.final_carry.environment_state,
    )
    assert jnp.array_equal(continuation.final_carry.rng, uninterrupted.final_carry.rng)


def test_rollout_preserves_scalar_and_multidimensional_action_shapes() -> None:
    """
    Preserve action dimensions while computing one movement norm per seed.

    Scalar actions must not create an invalid norm axis or broadcast across seeds.
    """
    for action_shape in ((), (2, 2)):
        environment = _ProbeEnvironment(action_shape)
        seed_keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
        previous_action = jnp.zeros((2, *action_shape), dtype=jnp.float32)
        if action_shape == ():
            previous_action = jnp.zeros((2,), dtype=jnp.float32)
        carry = make_fresh_batched_carry(
            environment,
            seed_keys,
            previous_action=previous_action,
            action_shape=action_shape,
        )

        def action_step(
            observations: jax.Array,
            controller_state: object,
            previous_action: jax.Array,
            step_index: jax.Array,
            action_key: jax.Array | None,
        ) -> tuple[jax.Array, object]:
            del observations, controller_state, step_index, action_key
            return previous_action + 1.0, None

        def run_one(
            initial: BatchedRolloutCarry[jax.Array, object],
            environment: _ProbeEnvironment = environment,
            action_shape: tuple[int, ...] = action_shape,
        ):
            return run_batched_rollout(
                environment,
                initial,
                steps=1,
                action_step=action_step,
                rng_mode="two-way",
                capture_action=True,
                mask_step_zero_delta=False,
                action_shape=action_shape,
                step_offset=0,
            )

        result = jax.jit(run_one)(carry)

        assert result.metrics["action_delta_norm"].shape == (1, 2)
        assert result.metrics["action"].shape == (1, 2, *action_shape)
        expected_norm = 1.0 if action_shape == () else 2.0
        assert jnp.array_equal(
            result.metrics["action_delta_norm"],
            jnp.full((1, 2), expected_norm),
        )


def test_auto_reset_is_explicit_and_resets_environment_and_controller() -> None:
    """
    Reset completed batch rows only when episodic policy is selected.

    The terminating transition remains in the result while the next action sees reset state.
    """
    environment = _AlwaysDoneEnvironment()
    seed_keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    initial = make_fresh_batched_carry(
        environment,
        seed_keys,
        previous_action=jnp.zeros((2, 1), dtype=jnp.float32),
        controller_state=jnp.zeros((2,), dtype=jnp.float32),
        action_shape=(1,),
    )

    def action_step(
        observations: jax.Array,
        controller_state: jax.Array,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        del observations, previous_action, step_index, action_key
        return controller_state[:, None], controller_state + 1.0

    result = jax.jit(
        lambda carry: run_batched_rollout(
            environment,
            carry,
            steps=2,
            action_step=action_step,
            rng_mode="two-way",
            capture_action=True,
            mask_step_zero_delta=False,
            action_shape=(1,),
            step_offset=0,
            reset_policy="auto_reset",
            controller_reset=lambda key: jnp.array(10.0),
        )
    )(initial)

    assert jnp.array_equal(result.metrics["action"], jnp.array([[0.0, 0.0], [10.0, 10.0]]))
    assert jnp.array_equal(result.info["state"], jnp.ones((2, 2)))
    assert jnp.array_equal(result.final_carry.environment_state, jnp.array([0.0, 0.0]))
    assert jnp.array_equal(result.final_carry.controller_state, jnp.array([10.0, 10.0]))

    with pytest.raises(ValueError, match="auto_reset requires controller_reset"):
        run_batched_rollout(
            environment,
            initial,
            steps=1,
            action_step=action_step,
            rng_mode="two-way",
            capture_action=False,
            mask_step_zero_delta=False,
            action_shape=(1,),
            step_offset=0,
            reset_policy="auto_reset",
        )
