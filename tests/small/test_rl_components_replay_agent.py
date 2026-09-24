"""Public behavior of the shared replay-agent step helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.timestep import Timestep


def test_replay_agent_step_preserves_transition_and_training_order_under_jit() -> None:
    """The helper skips the first insertion, gates learning, and closes boundary data."""
    spec = EnvSpec(
        id="replay-agent",
        observation_shape=(1,),
        action_shape=(1,),
        observation_dtype=jnp.float32,
        action_dtype=jnp.float32,
        action_low=jnp.array([-1.0], dtype=jnp.float32),
        action_high=jnp.array([1.0], dtype=jnp.float32),
    )
    buffer = ReplayBuffer(
        capacity=8,
        obs_shape=spec.observation_shape,
        action_shape=spec.action_shape,
        action_dtype=spec.action_dtype,
        obs_dtype=spec.observation_dtype,
    )
    initial_state = init_replay_agent_state(jnp.array(0, jnp.int32), buffer.init(), jax.random.key(0), spec)
    timesteps = Timestep(
        reward=jnp.array([0.0, 11.0, 22.0], dtype=jnp.float32),
        discount=jnp.array([1.0, 0.0, 0.75], dtype=jnp.float32),
        bootstrap_observation=jnp.array([[0.0], [99.0], [101.0]], dtype=jnp.float32),
        episode_end=jnp.array([False, True, False]),
        observation=jnp.array([[1.0], [100.0], [101.0]], dtype=jnp.float32),
    )

    def close_transition(
        replay_state: ReplayBufferState,
        last_obs: jax.Array,
        last_action: jax.Array,
        timestep: Timestep[jax.Array],
    ) -> ReplayBufferState:
        return buffer.add(
            replay_state,
            last_obs[None, ...],
            last_action[None, ...],
            timestep.reward[None, ...],
            timestep.bootstrap_observation[None, ...],
            timestep.discount[None, ...],
        )

    def learn(
        algorithm_state: jax.Array,
        replay_state: ReplayBufferState,
        keys: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ReplayBufferState, dict[str, jax.Array]]:
        del keys
        return algorithm_state + 1, replay_state, {
            "loss": jnp.array(3.0, jnp.float32),
            "histogram": jnp.array([4, 5], jnp.int32),
        }

    def select_action(
        algorithm_state: jax.Array,
        observation: jax.Array,
        step_index: jax.Array,
        keys: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del observation, step_index, keys
        return jnp.full((1,), algorithm_state, dtype=jnp.float32), {
            "policy_value": algorithm_state.astype(jnp.float32) + 10.0
        }

    def after_learn(algorithm_state: jax.Array, step_index: jax.Array, did_learn: jax.Array) -> jax.Array:
        del step_index
        return algorithm_state + did_learn.astype(jnp.int32)

    def scan_step(
        state: ReplayAgentState[jax.Array, ReplayBufferState],
        inputs: tuple[jax.Array, Timestep[jax.Array]],
    ) -> tuple[ReplayAgentState[jax.Array, ReplayBufferState], tuple[jax.Array, dict[str, jax.Array]]]:
        step_index, timestep = inputs
        carry_key, learn_key, action_key = jax.random.split(state.key, 3)
        result = replay_agent_step(
            state,
            timestep,
            step_index,
            carry_key=carry_key,
            learn_keys=(learn_key,),
            action_keys=(action_key,),
            learning_starts=None,
            train_frequency=1,
            warmup_steps=1,
            close_transition=close_transition,
            learn=learn,
            select_action=select_action,
            minimum_replay_size=2,
            after_learn=after_learn,
        )
        return result.state, (result.action, result.metrics)

    def scan(
        state: ReplayAgentState[jax.Array, ReplayBufferState], steps: Timestep[jax.Array]
    ) -> tuple[ReplayAgentState[jax.Array, ReplayBufferState], tuple[jax.Array, dict[str, jax.Array]]]:
        return jax.lax.scan(scan_step, state, (jnp.arange(3), steps))

    final_state, (actions, metrics) = jax.jit(scan)(initial_state, timesteps)
    replay = final_state.replay_state

    assert initial_state.last_obs.shape == spec.observation_shape
    assert initial_state.last_action.shape == spec.action_shape
    assert jnp.array_equal(replay.count, 2)
    assert jnp.array_equal(replay.obs[:2], jnp.array([[1.0], [100.0]], dtype=jnp.float32))
    assert jnp.array_equal(replay.actions[:2], actions[:2])
    assert jnp.array_equal(replay.rewards[:2], jnp.array([11.0, 22.0], dtype=jnp.float32))
    assert jnp.array_equal(replay.next_obs[:2], jnp.array([[99.0], [101.0]], dtype=jnp.float32))
    assert jnp.array_equal(replay.discount[:2], jnp.array([0.0, 0.75], dtype=jnp.float32))
    assert jnp.array_equal(metrics["loss"], jnp.array([0.0, 0.0, 3.0], dtype=jnp.float32))
    assert jnp.array_equal(metrics["histogram"], jnp.array([[0, 0], [0, 0], [4, 5]], jnp.int32))
    assert metrics["histogram"].dtype == jnp.int32
    assert jnp.array_equal(metrics["policy_value"], jnp.array([0.0, 10.0, 12.0], jnp.float32))
    assert jnp.array_equal(final_state.algorithm_state, 2)


def test_replay_agent_step_starts_learning_after_the_configured_index() -> None:
    """The step-index training threshold stays exclusive, matching current agents."""
    spec = EnvSpec(
        id="replay-agent-start",
        observation_shape=(1,),
        action_shape=(1,),
        observation_dtype=jnp.float32,
        action_dtype=jnp.float32,
    )
    buffer = ReplayBuffer(
        capacity=4,
        obs_shape=spec.observation_shape,
        action_shape=spec.action_shape,
        action_dtype=spec.action_dtype,
        obs_dtype=spec.observation_dtype,
    )
    initial_state = init_replay_agent_state(jnp.array(0, jnp.int32), buffer.init(), jax.random.key(0), spec)
    timestep = Timestep(
        reward=jnp.zeros((), jnp.float32),
        discount=jnp.zeros((), jnp.float32),
        bootstrap_observation=jnp.zeros((1,), jnp.float32),
        episode_end=jnp.zeros((), jnp.bool_),
        observation=jnp.ones((1,), jnp.float32),
    )

    def close_transition(
        replay_state: ReplayBufferState,
        last_obs: jax.Array,
        last_action: jax.Array,
        timestep: Timestep[jax.Array],
    ) -> ReplayBufferState:
        del last_obs, last_action, timestep
        return replay_state

    def learn(
        algorithm_state: jax.Array,
        replay_state: ReplayBufferState,
        keys: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ReplayBufferState, dict[str, jax.Array]]:
        del keys
        return algorithm_state + 1, replay_state, {"loss": jnp.ones((), jnp.float32)}

    def select_action(
        algorithm_state: jax.Array,
        observation: jax.Array,
        step_index: jax.Array,
        keys: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del algorithm_state, observation, step_index, keys
        return jnp.zeros((1,), jnp.float32), {}

    def step(state: ReplayAgentState[jax.Array, ReplayBufferState]):
        carry_key, learn_key, action_key = jax.random.split(state.key, 3)
        return replay_agent_step(
            state,
            timestep,
            jnp.array(1, jnp.int32),
            carry_key=carry_key,
            learn_keys=(learn_key,),
            action_keys=(action_key,),
            learning_starts=1,
            train_frequency=1,
            warmup_steps=None,
            close_transition=close_transition,
            learn=learn,
            select_action=select_action,
        )

    result = jax.jit(step)(initial_state)

    assert jnp.array_equal(result.state.algorithm_state, 0)
    assert jnp.array_equal(result.metrics["loss"], 0.0)
