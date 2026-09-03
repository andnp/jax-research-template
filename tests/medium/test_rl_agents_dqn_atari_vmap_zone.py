"""One compiled kernel must train a batch of DQN Atari hyperparameter arms independently."""

import dataclasses

import jax
import jax.numpy as jnp
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariAgentState,
    DQNAtariConfig,
    DQNAtariHypers,
    dqn_atari_hypers,
    dqn_atari_runtime_from_dqn_zoo,
)
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
SPEC = EnvSpec(
    id="toy-atari",
    observation_shape=OBSERVATION_SHAPE,
    action_shape=(),
    observation_dtype=jnp.dtype(jnp.uint8),
    action_dtype=jnp.dtype(jnp.int32),
    num_actions=NUM_ACTIONS,
)
NUM_STEPS = 8


def _run_arm(
    agent: DQNAtariAgent, base_hypers: DQNAtariHypers, learning_rate: jax.Array, key: jax.Array
) -> jax.Array:
    hypers = dataclasses.replace(base_hypers, LEARNING_RATE=learning_rate)
    state = dataclasses.replace(agent.init(key, SPEC), hypers=hypers)

    def one_step(state: DQNAtariAgentState, step_index: jax.Array):
        observation = jnp.full(OBSERVATION_SHAPE, (step_index + 1) % 255, dtype=jnp.uint8)
        timestep = Timestep(
            observation=observation,
            reward=jnp.float32(1.0),
            discount=jnp.float32(1.0),
            bootstrap_observation=observation,
            episode_end=jnp.bool_(False),
        )
        out = agent.step(state, timestep, step_index)
        return out.state, out.metrics["loss"]

    state, _ = jax.lax.scan(one_step, state, jnp.arange(NUM_STEPS))
    return state.train_state.params


def _agent() -> DQNAtariAgent:
    config = DQNAtariConfig(
        REPLAY_CAPACITY=8,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=2,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=40_000,
    )
    runtime_config = dqn_atari_runtime_from_dqn_zoo(config, num_iterations=1, num_train_frames_per_iteration=32)
    return DQNAtariAgent(config, runtime_config)


class TestDQNAtariVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept rate
        never reached the kernel -- the failure mode this fixes if the learning
        rate is left closed over ``optax.rmsprop`` instead of riding the state.
        """
        agent = _agent()
        base_hypers = dqn_atari_hypers(agent.config)
        learning_rates = jnp.array([1e-1, 1e-4], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]

        fast, slow = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(fast, slow)
        assert jnp.linalg.norm(fast) > jnp.linalg.norm(slow)

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same rate and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept
        value, not from the batching itself.
        """
        agent = _agent()
        base_hypers = dqn_atari_hypers(agent.config)
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])
