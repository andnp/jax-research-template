"""Integration gate: Rainbow's n-step replay, written by the shared loop's boundaries.

The agent is driven through ``AgentProtocol`` plus :func:`rl_components.loop.run` against
an Atari-shaped counter environment, and inspected only through the replay it wrote. No
n-step arithmetic is re-implemented here; every expected value below is hand-computed from
``gamma`` and the reward sequence, so an accumulator that agreed with a copy of itself
would still fail.

``gamma`` is ``0.5`` so that every expected reward and coefficient is exact in binary, and
the reward is a constant ``1.0`` so that none of them depends on the actions a noisy
network happens to pick.

The two environments differ only in which boundary flag they raise, and that is the whole
point of the file. A termination drives the coefficient to ``0`` and the shortened rows are
indistinguishable from each other. A truncation leaves the coefficient at ``gamma``, and
the same flush then has to produce three *different* coefficients from one step -- which is
the case the deleted compile-time ``ADDITIONAL_DISCOUNT ** N_STEP`` could not express, and
which the boolean ``done`` the entry used to store masked away entirely.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax_replay.types import PERBufferState
from rl_agents.rainbow import RainbowAgent, RainbowConfig
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
EPISODE_LENGTH = 4
STEPS = 13
"""Thirteen iterations close twelve transitions: three whole episodes of four."""

GAMMA = 0.5
N_STEP = 3
REPLAY_CAPACITY = 16
BATCH_SIZE = 4
SEED = 0


def _observation(counter: jax.Array) -> jax.Array:
    """Encode the episode counter in every pixel, so each step is distinguishable."""
    return jnp.full(OBSERVATION_SHAPE, counter, dtype=jnp.uint8)


TERMINAL_OBSERVATION = _observation(jnp.asarray(EPISODE_LENGTH, dtype=jnp.uint8))
EPISODE_START_OBSERVATION = _observation(jnp.asarray(0, dtype=jnp.uint8))


class CounterEnv:
    """An Atari-shaped counter that raises one boundary flag and never resets itself.

    Reaching the boundary leaves the counter at ``EPISODE_LENGTH`` rather than wrapping to
    zero, so ``EnvStep.observation`` is the state the transition actually reached and the
    loop's reset is the only thing that starts the next episode.
    """

    def __init__(self, *, truncates: bool) -> None:
        self.truncates = truncates

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-atari-counter",
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
        del key, action, params
        next_counter = state + jnp.asarray(1, dtype=jnp.int32)
        at_boundary = next_counter >= EPISODE_LENGTH
        return EnvStep(
            observation=_observation(next_counter),
            state=next_counter,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=at_boundary & jnp.bool_(not self.truncates),
            truncated=at_boundary & jnp.bool_(self.truncates),
            info={},
        )


class Replay(NamedTuple):
    """One finished run's replay contents, keyed by field rather than by leaf index."""

    obs: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    discounts: jax.Array
    metrics: dict[str, jax.Array]
    buffer_state: PERBufferState


def _run(*, truncates: bool) -> Replay:
    config = RainbowConfig(
        REPLAY_CAPACITY=REPLAY_CAPACITY,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=BATCH_SIZE,
        LEARN_PERIOD_FRAMES=8,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=16,
        N_STEP=N_STEP,
    )
    agent = RainbowAgent(config)

    def train_fn(key: jax.Array):
        return run(agent, CounterEnv(truncates=truncates), key, steps=STEPS, gamma=GAMMA)

    final_state, metrics = jax.jit(train_fn)(jax.random.key(SEED))

    buffer_state = final_state.agent_state.buffer_state
    count = int(buffer_state.count)
    return Replay(
        obs=buffer_state.data["0"][:count],
        rewards=buffer_state.data["2"][:count],
        next_obs=buffer_state.data["3"][:count],
        discounts=buffer_state.data["4"][:count],
        metrics=metrics,
        buffer_state=buffer_state,
    )


@pytest.fixture(scope="module")
def terminating() -> Replay:
    return _run(truncates=False)


@pytest.fixture(scope="module")
def truncating() -> Replay:
    return _run(truncates=True)


def test_every_closed_transition_reaches_replay(terminating: Replay) -> None:
    """A boundary flushes the window whole, so the entry count is the transition count.

    An accumulator that dropped its pending rows at a boundary instead of flushing them
    would lose ``N_STEP - 1`` entries per episode and still train.
    """
    assert int(terminating.buffer_state.count) == STEPS - 1


def test_replay_leaves_keep_the_environment_dtypes(terminating: Replay) -> None:
    assert terminating.buffer_state.data["0"].dtype == jnp.uint8
    assert terminating.buffer_state.data["3"].dtype == jnp.uint8
    assert terminating.buffer_state.data["4"].dtype == jnp.float32
    assert int(terminating.buffer_state.logical_capacity) == REPLAY_CAPACITY


def test_metrics_use_a_fixed_schema_outside_the_loop_namespace(terminating: Replay) -> None:
    agent_keys = {key for key in terminating.metrics if not key.startswith("loop/")}

    assert agent_keys == {"loss", "max_q"}
    for key in agent_keys:
        assert terminating.metrics[key].shape == (STEPS,), key
        assert jnp.all(jnp.isfinite(terminating.metrics[key])), key


def test_termination_stores_the_n_step_return_and_a_zero_coefficient(terminating: Replay) -> None:
    """Each episode contributes one full window plus the three its boundary flushes.

    ``1.75 = 1 + 0.5 + 0.25`` is the full three-step return; ``1.75, 1.5, 1.0`` are the
    same window seen from one, two and three steps later. Every one of them ends in the
    terminal step, so every coefficient is an exact ``0.0`` -- the tail of the return is
    erased by the product itself, with no ``done`` mask anywhere.
    """
    npt.assert_allclose(terminating.rewards, jnp.tile(jnp.array([1.75, 1.75, 1.5, 1.0], jnp.float32), 3))
    npt.assert_allclose(terminating.discounts, jnp.tile(jnp.array([0.125, 0.0, 0.0, 0.0], jnp.float32), 3))


def test_truncation_flushes_the_same_window_at_three_horizons(truncating: Replay) -> None:
    """The bootstrap survives, so the three flushed rows must differ in their coefficient.

    Rows leaving at offsets 0, 1 and 2 of the flush bootstrap over three, two and one step
    respectively, so they carry ``0.5 ** 3``, ``0.5 ** 2`` and ``0.5``. The old entry
    stored ``done = True`` for all three and the learner multiplied by a single
    ``gamma ** N_STEP``, which is right for none of the last two and destroyed the
    bootstrap of all three.
    """
    npt.assert_allclose(truncating.rewards, jnp.tile(jnp.array([1.75, 1.75, 1.5, 1.0], jnp.float32), 3))
    npt.assert_allclose(truncating.discounts, jnp.tile(jnp.array([0.125, 0.125, 0.25, 0.5], jnp.float32), 3))
    assert jnp.all(truncating.discounts > 0.0), "a truncation must not erase any bootstrap"


def test_a_flushed_window_bootstraps_from_the_true_boundary_state(truncating: Replay) -> None:
    """Every row of one flush shares one bootstrap state, and it is the pre-reset one.

    The rows leave at different horizons but they all stop at the same step, so the state
    they bootstrap from is the same -- the true final observation. Storing the post-reset
    observation instead would bootstrap a whole window from the next episode's first state,
    and with the coefficient nonzero here nothing else would hide it.
    """
    # A shortened window bootstraps over fewer steps, so its coefficient is LARGER than a
    # full window's `gamma ** N_STEP`, not smaller.
    boundary = truncating.discounts > 0.125
    assert int(jnp.sum(boundary)) == 6, "expected two shortened rows per episode"
    assert jnp.array_equal(
        truncating.next_obs[boundary],
        jnp.broadcast_to(TERMINAL_OBSERVATION, (int(jnp.sum(boundary)), *OBSERVATION_SHAPE)),
    )
    assert not jnp.array_equal(TERMINAL_OBSERVATION, EPISODE_START_OBSERVATION), (
        "the toy environment must make the two observations distinguishable"
    )


def test_the_window_after_a_boundary_starts_the_new_episode(truncating: Replay) -> None:
    """The mirror property: the agent's carried ``last_obs`` must be the post-reset state.

    The first entry of each episode's flush group opens at the episode's first state.
    Getting this wrong pairs the previous episode's final state with an action chosen from
    the new episode's first state -- a transition that happened in no episode at all.
    """
    starts = jnp.flatnonzero(truncating.discounts == 0.5) + 1
    following = [int(index) for index in starts if int(index) < truncating.obs.shape[0]]

    assert following, "no window opened after a boundary to check"
    for index in following:
        assert jnp.array_equal(truncating.obs[index], EPISODE_START_OBSERVATION), (
            f"entry {index} opened from the previous episode's final state"
        )
