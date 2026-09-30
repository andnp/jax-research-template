"""Exploration behavior of the shared replay-Q agent's epsilon-greedy action selection."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

NUM_ACTIONS = 4
EPSILON = 0.25
NUM_KEYS = 8_000


def _actions_and_greedy(num_keys: int) -> tuple[jax.Array, jax.Array]:
    config = DQNConfig(
        BUFFER_SIZE=8,
        LEARNING_STARTS=1_000,
        EPSILON_START=EPSILON,
        EPSILON_END=EPSILON,
        TOTAL_TIMESTEPS=100,
    )
    agent = DQNAgent(config)
    spec = EnvSpec(id="toy", observation_shape=(2,), action_shape=(), num_actions=NUM_ACTIONS)
    state = agent.init(jax.random.key(0), spec)
    observation = jnp.array([0.3, -0.7])
    timestep = Timestep(
        reward=jnp.asarray(0.0),
        discount=jnp.asarray(0.99),
        bootstrap_observation=observation,
        episode_end=jnp.asarray(False),
        observation=observation,
    )
    q_values = state.algorithm_state.train_state.apply_fn(state.algorithm_state.train_state.params, observation)

    def act(key: jax.Array) -> jax.Array:
        return agent.step(replace(state, key=key), timestep, jnp.asarray(1)).action

    actions = jax.vmap(act)(jax.random.split(jax.random.key(1), num_keys))
    return actions, jnp.argmax(q_values)


class TestEpsilonGreedyExploration:
    def test_exploratory_actions_are_uniform_over_all_actions(self) -> None:
        """Each non-greedy action occurs at epsilon / num_actions, so the explore coin does not bias the draw."""
        actions, greedy = _actions_and_greedy(NUM_KEYS)

        expected = EPSILON / NUM_ACTIONS
        for action in range(NUM_ACTIONS):
            if action == greedy:
                continue
            frequency = float(jnp.mean(actions == action))
            assert abs(frequency - expected) < 0.02, (action, frequency, expected)
