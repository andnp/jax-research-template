"""Behavioural gate for the ported Rainbow, which the shared learn-path gate cannot cover.

``tests/medium/test_rl_agents_learn_path.py`` drives every other replay agent against a
two-element observation, reads its replay through ``ReplayBufferState`` fields, and keys
its learning/warmup pair off ``LEARNING_STARTS``. None of the three fits here: the Nature
CNN needs image observations, Rainbow's prioritised buffer stores its leaves keyed by
flatten index rather than by name, and this agent's can-train predicate gates on replay
occupancy. So the pair below is keyed off ``MIN_REPLAY_CAPACITY_FRACTION`` -- one run whose
warmup threshold the horizon can reach and one whose it cannot -- and everything else it
asserts is the same property the shared gate asserts for the rest of the family.

No loss, target or update rule is re-implemented here. The agent is driven through
``AgentProtocol`` plus :func:`rl_components.loop.run` and inspected only through the replay
it wrote and the metrics it published, so a broken agent cannot satisfy these assertions by
agreeing with a copy of itself.

``GAMMA`` is deliberately 0.9 rather than 0.99. The deleted ``ADDITIONAL_DISCOUNT`` was
0.99, so a window that still multiplied by it would store ``0.729 * 0.970`` where the
loop's three-step coefficient is ``0.729``, and the exact-value assertion below is what
makes that visible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from rl_agents.rainbow import (
    RainbowAgent,
    RainbowConfig,
    rainbow_zoo_atari_min_replay_capacity,
)
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
EPISODE_LENGTH = 4
STEPS = 13
"""Thirteen iterations close twelve transitions, and every one of them reaches replay."""

N_STEP = 3
REPLAY_CAPACITY = 16
BATCH_SIZE = 4
GAMMA = 0.9
SEED = 0

REACHABLE_WARMUP_FRACTION = 0.25
"""A threshold of 4 entries, which the 12 the horizon writes clears."""

UNREACHABLE_WARMUP_FRACTION = 0.875
"""A threshold of 14 entries, which the 12 the horizon writes never clears."""


def _observation(counter: jax.Array) -> jax.Array:
    return jnp.full(OBSERVATION_SHAPE, counter, dtype=jnp.uint8)


class ToyAtariEpisodeEnv:
    """An Atari-shaped counter environment that terminates without resetting itself."""

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
    metrics: dict[str, jax.Array]
    entries: int


def _run(warmup_fraction: float) -> Run:
    config = RainbowConfig(
        REPLAY_CAPACITY=REPLAY_CAPACITY,
        MIN_REPLAY_CAPACITY_FRACTION=warmup_fraction,
        BATCH_SIZE=BATCH_SIZE,
        LEARN_PERIOD_FRAMES=8,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=16,
        N_STEP=N_STEP,
    )
    final_state, metrics = jax.jit(
        lambda key: run(RainbowAgent(config), ToyAtariEpisodeEnv(), key, steps=STEPS, gamma=GAMMA)
    )(jax.random.key(SEED))

    agent_state = final_state.agent_state
    count = int(agent_state.buffer_state.count)
    return Run(
        params=jax.tree_util.tree_leaves(agent_state.train_state.params),
        discounts=agent_state.buffer_state.data["4"][:count],
        metrics=metrics,
        entries=count,
    )


@pytest.fixture(scope="module")
def learning_run() -> Run:
    return _run(REACHABLE_WARMUP_FRACTION)


@pytest.fixture(scope="module")
def warmup_only_run() -> Run:
    return _run(UNREACHABLE_WARMUP_FRACTION)


def test_the_two_runs_straddle_the_replay_warmup_threshold(learning_run: Run) -> None:
    """The pair is only a control if one threshold is reachable and the other is not."""
    reachable = rainbow_zoo_atari_min_replay_capacity(
        RainbowConfig(REPLAY_CAPACITY=REPLAY_CAPACITY, MIN_REPLAY_CAPACITY_FRACTION=REACHABLE_WARMUP_FRACTION)
    )
    unreachable = rainbow_zoo_atari_min_replay_capacity(
        RainbowConfig(REPLAY_CAPACITY=REPLAY_CAPACITY, MIN_REPLAY_CAPACITY_FRACTION=UNREACHABLE_WARMUP_FRACTION)
    )

    assert reachable <= learning_run.entries < unreachable


def test_learn_path_moves_parameters(learning_run: Run, warmup_only_run: Run) -> None:
    unchanged = [
        bool(jnp.array_equal(before, after))
        for before, after in zip(warmup_only_run.params, learning_run.params, strict=True)
    ]

    assert not all(unchanged), "rainbow left every parameter untouched: its update never fired"


def test_learn_path_reports_nonzero_loss(learning_run: Run, warmup_only_run: Run) -> None:
    loss = learning_run.metrics["loss"]

    assert jnp.any(loss != 0.0), "rainbow reported an all-zero loss"
    assert jnp.all(jnp.isfinite(loss))
    assert jnp.all(warmup_only_run.metrics["loss"] == 0.0), (
        "rainbow reported a nonzero loss without ever taking a gradient step"
    )


def test_metrics_use_a_fixed_schema_outside_the_loop_namespace(learning_run: Run) -> None:
    agent_keys = {key for key in learning_run.metrics if not key.startswith("loop/")}

    assert agent_keys == {"loss", "max_q"}
    for key in agent_keys:
        assert learning_run.metrics[key].shape == (STEPS,), key


def test_stored_discounts_carry_the_loops_gamma_and_nothing_else(learning_run: Run) -> None:
    """A resurrected ``ADDITIONAL_DISCOUNT`` would show up here as ``0.9 ** 3 * 0.99 ** 3``.

    Only a full-length window survives with a nonzero coefficient in a terminating
    environment, so every nonzero entry must be exactly ``gamma ** N_STEP``: one factor of
    the loop's gamma per step the window spanned, and no factor from anywhere else.
    """
    discounts = learning_run.discounts

    assert discounts.size == learning_run.entries
    assert jnp.any(discounts == 0.0), "termination never reached the stored transitions"
    assert jnp.allclose(discounts[discounts > 0.0], GAMMA**N_STEP), (
        "a surviving n-step bootstrap must carry exactly the loop's gamma per step, not a second factor"
    )
