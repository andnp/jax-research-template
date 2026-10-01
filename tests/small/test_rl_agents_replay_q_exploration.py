"""Exploration behavior of the replay-Q and QRC agents' epsilon-greedy action selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.qrc import QRCAgent, QRCConfig
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

NUM_ACTIONS = 4
EPSILON = 0.25
NUM_KEYS = 8_000

EXPLORATION = dict(
    BUFFER_SIZE=8, LEARNING_STARTS=1_000, EPSILON_START=EPSILON, EPSILON_END=EPSILON, total_timesteps=100
)

SPEC = EnvSpec(id="toy", observation_shape=(2,), action_shape=(), num_actions=NUM_ACTIONS)
OBSERVATION = jnp.array([0.3, -0.7])
TIMESTEP = Timestep(
    reward=jnp.asarray(0.0),
    discount=jnp.asarray(0.99),
    bootstrap_observation=OBSERVATION,
    episode_end=jnp.asarray(False),
    observation=OBSERVATION,
)


def _assert_exploration_uniform(act: Callable[[jax.Array], jax.Array], greedy: jax.Array) -> None:
    keys = jax.random.split(jax.random.key(1), NUM_KEYS)
    actions = jax.vmap(act)(keys)

    expected = EPSILON / NUM_ACTIONS
    for action in range(NUM_ACTIONS):
        if action == greedy:
            continue
        frequency = float(jnp.mean(actions == action))
        assert abs(frequency - expected) < 0.02, (action, frequency, expected)


class TestEpsilonGreedyExploration:
    def test_replay_q_exploratory_actions_are_uniform(self) -> None:
        """Each non-greedy action occurs at epsilon / num_actions, so the explore coin does not bias the draw."""
        agent = DQNAgent(DQNConfig(**EXPLORATION))
        state = agent.init(jax.random.key(0), SPEC)
        train_state = state.algorithm_state.train_state
        greedy = jnp.argmax(train_state.apply_fn(train_state.params, OBSERVATION))

        _assert_exploration_uniform(lambda key: agent.step(replace(state, key=key), TIMESTEP, jnp.asarray(1)).action, greedy)

    def test_qrc_exploratory_actions_are_uniform(self) -> None:
        """QRC's exploratory actions are uniform over the non-greedy actions at epsilon / num_actions."""
        agent = QRCAgent(QRCConfig(**EXPLORATION))
        state = agent.init(jax.random.key(0), SPEC)
        train_state = state.algorithm_state.train_state
        greedy = jnp.argmax(train_state.apply_fn(train_state.params, OBSERVATION)[0])

        _assert_exploration_uniform(lambda key: agent.step(replace(state, key=key), TIMESTEP, jnp.asarray(1)).action, greedy)
