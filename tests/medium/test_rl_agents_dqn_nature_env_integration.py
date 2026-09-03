"""Medium integration test for ``DQNAgent`` against a Nature-CNN-shaped environment.

The point is the construction path, not the arithmetic: an injected environment whose
``EnvSpec`` reports stacked ``uint8`` image observations must drive the whole port --
spec-derived network selection, a ``uint8`` replay buffer, and a ``lax.scan`` that traces
under ``jit`` -- with nothing but the spec to go on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
STEPS = 4
GAMMA = 0.99


class FakeAtariLikeEnv:
    """A counter environment wearing Atari's observation shape and dtype."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="fake-atari",
            observation_shape=OBSERVATION_SHAPE,
            action_shape=(),
            observation_dtype=jnp.dtype(jnp.uint8),
            action_dtype=jnp.dtype(jnp.int32),
            num_actions=NUM_ACTIONS,
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.uint8),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        return EnvStep(
            observation=jnp.full(OBSERVATION_SHAPE, next_state, dtype=jnp.uint8),
            state=next_state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={},
        )


class TestDQNNatureEnvIntegration:
    def test_run_drives_the_agent_against_an_injected_atari_like_env(self) -> None:
        config = DQNConfig(
            NETWORK_PRESET="nature_cnn",
            TOTAL_TIMESTEPS=STEPS,
            LEARNING_STARTS=100,
            BUFFER_SIZE=16,
            BATCH_SIZE=4,
        )
        agent = DQNAgent(config)
        env = FakeAtariLikeEnv()

        final_state, metrics = jax.jit(
            lambda key: run(agent, env, key, steps=STEPS, gamma=GAMMA)
        )(jax.random.key(0))

        for key_name in ("loss", "epsilon", "q_max", "loop/reward", "loop/episode_end"):
            assert metrics[key_name].shape == (STEPS,), key_name

        buffer_state = final_state.agent_state.buffer_state
        assert buffer_state.obs.dtype == jnp.uint8, "the spec's observation dtype must reach the buffer"
        assert int(buffer_state.count) == STEPS - 1, "the final action opens a transition that never closes"
        assert jnp.all(metrics["loss"] == 0.0), "LEARNING_STARTS above the budget must keep the learn path shut"

    def test_the_spec_selects_the_nature_network_without_an_observation_argument(self) -> None:
        """``init`` sees only the spec, so the CNN choice must follow from it alone."""
        agent = DQNAgent(DQNConfig(NETWORK_PRESET="nature_cnn", BUFFER_SIZE=8))

        state = agent.init(jax.random.key(0), FakeAtariLikeEnv().spec())

        q_values = state.train_state.apply_fn(
            state.train_state.params,
            jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.uint8),
        )
        assert q_values.shape == (NUM_ACTIONS,)
        assert state.last_obs.shape == OBSERVATION_SHAPE
        assert state.last_obs.dtype == jnp.uint8
        assert state.last_action.shape == ()
        assert int(state.num_actions) == NUM_ACTIONS
