"""Behavioural gate for the ported Atari DQN, which the shared learn-path gate cannot cover.

``tests/medium/test_rl_agents_learn_path.py`` drives every other replay agent against a
two-element observation and keys its learning/warmup pair off ``LEARNING_STARTS``. Neither
fits here: the Nature CNN needs image observations, and this agent's can-train predicate
gates on replay occupancy instead. So the pair below is keyed off
``MIN_REPLAY_CAPACITY_FRACTION`` -- one run whose warmup threshold the horizon can reach and
one whose it cannot -- and everything else it asserts is the same property the shared gate
asserts for the rest of the family.

No loss, target or update rule is re-implemented here. The agent is driven through
``AgentProtocol`` plus :func:`rl_components.loop.run` and inspected only through the replay
it wrote and the metrics it published, so a broken agent cannot satisfy these assertions by
agreeing with a copy of itself.

``GAMMA`` is deliberately 0.9 rather than 0.99. The deleted ``ADDITIONAL_DISCOUNT`` was
0.99, so a run that still multiplied by it would store 0.891 where the loop's discount is
0.9, and the exact-value assertion below is what makes that visible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariConfig,
    dqn_atari_runtime_from_dqn_zoo,
    dqn_zoo_atari_min_replay_capacity,
    dqn_zoo_atari_total_train_env_steps,
)
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
EPISODE_LENGTH = 5
TOTAL_FRAMES = 48
"""48 frames over 4 action repeats is a 12-step horizon: two boundaries and 11 transitions."""

REPLAY_CAPACITY = 16
BATCH_SIZE = 4
GAMMA = 0.9
SEED = 0

REACHABLE_WARMUP_FRACTION = 0.25
"""A threshold of 4 transitions, which the 11 the horizon closes clears."""

UNREACHABLE_WARMUP_FRACTION = 0.75
"""A threshold of 12 transitions, which the 11 the horizon closes never clears."""


def _observation(counter: jax.Array) -> jax.Array:
    """Encode the episode counter in every pixel, so each step is distinguishable."""
    return jnp.full(OBSERVATION_SHAPE, counter, dtype=jnp.uint8)


TERMINAL_OBSERVATION = _observation(jnp.asarray(EPISODE_LENGTH, dtype=jnp.uint8))
"""The true final observation of an episode, distinct from the post-reset ``_observation(0)``."""

EPISODE_START_OBSERVATION = _observation(jnp.asarray(0, dtype=jnp.uint8))


class ToyAtariEpisodeEnv:
    """An Atari-shaped counter environment that terminates without resetting itself.

    Reaching the boundary leaves the counter at ``EPISODE_LENGTH`` rather than wrapping to
    zero, so ``EnvStep.observation`` is the state the transition actually reached and the
    loop's reset is the only thing that starts the next episode.
    """

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-atari-episode",
            observation_shape=OBSERVATION_SHAPE,
            action_shape=(),
            observation_dtype=jnp.dtype(jnp.uint8),
            action_dtype=jnp.dtype(jnp.int32),
            num_actions=NUM_ACTIONS,
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        counter = jnp.asarray(0, dtype=jnp.int32)
        return EnvReset(observation=_observation(counter), state=counter)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_counter = state + jnp.asarray(1, dtype=jnp.int32)
        return EnvStep(
            observation=_observation(next_counter),
            state=next_counter,
            reward=action.astype(jnp.float32) - 0.25 * state.astype(jnp.float32),
            terminated=next_counter >= EPISODE_LENGTH,
            truncated=jnp.bool_(False),
            info={},
        )


class Run(NamedTuple):
    """One finished training run, reduced to what the gate inspects."""

    params: list[jax.Array]
    discounts: jax.Array
    bootstrap_observations: jax.Array
    observations: jax.Array
    metrics: dict[str, jax.Array]
    transitions: int


def _run(warmup_fraction: float) -> Run:
    config = DQNAtariConfig(
        REPLAY_CAPACITY=REPLAY_CAPACITY,
        MIN_REPLAY_CAPACITY_FRACTION=warmup_fraction,
        BATCH_SIZE=BATCH_SIZE,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
    )
    runtime_config = dqn_atari_runtime_from_dqn_zoo(
        config,
        num_iterations=1,
        num_train_frames_per_iteration=TOTAL_FRAMES,
    )
    steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
    agent = DQNAtariAgent(config, runtime_config)

    final_state, metrics = jax.jit(
        lambda key: run(agent, ToyAtariEpisodeEnv(), key, steps=steps, gamma=GAMMA)
    )(jax.random.key(SEED))

    agent_state = final_state.agent_state
    count = int(agent_state.buffer_state.count)
    return Run(
        params=jax.tree_util.tree_leaves(agent_state.train_state.params),
        discounts=agent_state.buffer_state.discount[:count],
        bootstrap_observations=agent_state.buffer_state.next_obs[:count],
        observations=agent_state.buffer_state.obs[:count],
        metrics=metrics,
        transitions=steps - 1,
    )


@pytest.fixture(scope="module")
def learning_run() -> Run:
    return _run(REACHABLE_WARMUP_FRACTION)


@pytest.fixture(scope="module")
def warmup_only_run() -> Run:
    return _run(UNREACHABLE_WARMUP_FRACTION)


def test_the_two_runs_straddle_the_replay_warmup_threshold(learning_run: Run) -> None:
    """The pair is only a control if one threshold is reachable and the other is not."""
    reachable = dqn_zoo_atari_min_replay_capacity(
        DQNAtariConfig(REPLAY_CAPACITY=REPLAY_CAPACITY, MIN_REPLAY_CAPACITY_FRACTION=REACHABLE_WARMUP_FRACTION)
    )
    unreachable = dqn_zoo_atari_min_replay_capacity(
        DQNAtariConfig(REPLAY_CAPACITY=REPLAY_CAPACITY, MIN_REPLAY_CAPACITY_FRACTION=UNREACHABLE_WARMUP_FRACTION)
    )

    assert reachable <= learning_run.transitions < unreachable


def test_learn_path_moves_parameters(learning_run: Run, warmup_only_run: Run) -> None:
    unchanged = [
        bool(jnp.array_equal(before, after))
        for before, after in zip(warmup_only_run.params, learning_run.params, strict=True)
    ]

    assert not all(unchanged), "dqn_atari left every parameter untouched: its update never fired"


def test_learn_path_reports_nonzero_loss(learning_run: Run, warmup_only_run: Run) -> None:
    loss = learning_run.metrics["loss"]

    assert jnp.any(loss != 0.0), "dqn_atari reported an all-zero loss"
    assert jnp.all(jnp.isfinite(loss))
    assert jnp.all(warmup_only_run.metrics["loss"] == 0.0), (
        "dqn_atari reported a nonzero loss without ever taking a gradient step"
    )


def test_metrics_use_a_fixed_schema_outside_the_loop_namespace(learning_run: Run) -> None:
    agent_keys = {key for key in learning_run.metrics if not key.startswith("loop/")}

    assert agent_keys == {"loss", "epsilon"}
    for key in agent_keys:
        assert learning_run.metrics[key].shape == (learning_run.transitions + 1,), key


def test_stored_discounts_carry_the_loops_gamma_and_nothing_else(learning_run: Run) -> None:
    """The deleted ``ADDITIONAL_DISCOUNT`` would show up here as ``GAMMA * 0.99``."""
    discounts = learning_run.discounts

    assert discounts.size == learning_run.transitions
    assert jnp.any(discounts == 0.0), "termination never reached the stored transitions"
    assert jnp.allclose(discounts[discounts > 0.0], GAMMA), (
        "a surviving bootstrap must carry exactly the loop's gamma, not a second discount factor"
    )


def test_terminal_transitions_store_the_true_final_observation(learning_run: Run) -> None:
    """The bootstrap must be taken at the boundary, never across it."""
    terminal = learning_run.discounts == 0.0

    assert bool(jnp.any(terminal)), "no terminal transition was stored to check"
    assert jnp.array_equal(
        learning_run.bootstrap_observations[terminal],
        jnp.broadcast_to(TERMINAL_OBSERVATION, (int(jnp.sum(terminal)), *OBSERVATION_SHAPE)),
    ), "dqn_atari stored the post-reset observation at a boundary instead of the true final state"
    assert not jnp.array_equal(TERMINAL_OBSERVATION, EPISODE_START_OBSERVATION), (
        "the toy environment must make the two observations distinguishable"
    )


def test_the_transition_after_a_boundary_starts_the_new_episode(learning_run: Run) -> None:
    """The mirror of the terminal-observation property, and it fails the other way.

    Insertion must read ``timestep.bootstrap_observation`` while the agent's carried
    ``last_obs`` must become ``timestep.observation``. Getting the first wrong bootstraps a
    terminal transition from the next episode; getting the second wrong pairs the previous
    episode's final state with an action chosen from the new episode's first state, which is
    a transition that never happened in any episode.
    """
    terminal = jnp.flatnonzero(learning_run.discounts == 0.0)
    count = int(learning_run.observations.shape[0])
    following = [int(index) + 1 for index in terminal if int(index) + 1 < count]

    assert following, "no transition closed after a boundary to check"
    for index in following:
        assert jnp.array_equal(learning_run.observations[index], EPISODE_START_OBSERVATION), (
            "dqn_atari acted from the post-reset state but stored the previous episode's "
            f"final state as the observation of transition {index}"
        )
