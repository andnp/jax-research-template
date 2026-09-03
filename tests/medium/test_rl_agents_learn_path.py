"""Behavioural gate: every replay-buffer agent must actually reach its learn path.

Each case drives an agent through its public ``make_train`` entry point only. No loss,
target or update rule is re-implemented here, so a broken agent cannot satisfy these
assertions by agreeing with a copy of itself.

Every agent is run twice against the same seed and the same toy environment:

- a *learning* run, whose ``LEARNING_STARTS`` sits strictly below the step budget, so the
  ``can_train`` branch genuinely fires;
- a *warmup-only* run, whose ``LEARNING_STARTS`` equals the step budget, so the branch can
  never fire.

The pair is the negative control: parameters must move in the first run and must stay
bit-identical in the second, which is only possible if the gradient update ran. Agents that
publish a loss metric must additionally report a nonzero loss in the learning run and an
all-zero loss in the warmup-only run, so a zero-filled placeholder metric fails.

The toy environment terminates every ``EPISODE_LENGTH`` steps and auto-resets, so each run
crosses several episode boundaries with a nonzero ``done``.

Excluded agents: ``dqn_atari`` needs the ALE construction path and its own runtime config,
and ``rainbow`` samples from its own prioritised buffer rather than
``rl_components.buffers.ReplayBuffer``. Both need their own gate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import pytest
from flax.training.train_state import TrainState
from rl_agents import double_dqn, dqn, dueling_dqn, greedy_ac, qrc, sac, td3
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gym_env import ContinuousActionSpace, DiscreteActionSpace, GymEnv
from rl_components.gymnax_bridge import make_gymnax_compat_env

TOTAL_TIMESTEPS = 12
"""Step budget for every case: short enough to stay inside the medium tier."""

LEARNING_STARTS = 4
"""Strictly below the budget, so ``t > LEARNING_STARTS`` holds for seven of the steps."""

EPISODE_LENGTH = 5
"""Terminates at step 4 (before learning starts) and step 9 (after it does)."""

BUFFER_SIZE = 16
BATCH_SIZE = 4
SEED = 0


def _observation(step: jax.Array) -> jax.Array:
    scaled = step.astype(jnp.float32) / EPISODE_LENGTH
    return jnp.stack([scaled, 1.0 - scaled])


def _advance(step: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Step the counter, terminating and auto-resetting at the episode boundary."""
    nxt = step + jnp.int32(1)
    terminated = nxt >= EPISODE_LENGTH
    return jnp.where(terminated, jnp.int32(0), nxt), terminated


class ToyDiscreteEnv:
    """Two-action counter environment with a reward that depends on the action."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(id="toy-discrete", observation_shape=(2,), action_shape=(), num_actions=2)

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        step = jnp.int32(0)
        return EnvReset(observation=_observation(step), state=step)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_step, terminated = _advance(state)
        reward = action.astype(jnp.float32) - 0.25 * state.astype(jnp.float32)
        return EnvStep(
            observation=_observation(next_step),
            state=next_step,
            reward=reward,
            terminated=terminated,
            truncated=jnp.bool_(False),
            info={},
        )


class ToyContinuousEnv:
    """One-dimensional counter environment with a reward that depends on the action."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-continuous",
            observation_shape=(2,),
            action_shape=(1,),
            action_dtype=jnp.float32,
            action_low=jnp.full((1,), -1.0, dtype=jnp.float32),
            action_high=jnp.full((1,), 1.0, dtype=jnp.float32),
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        step = jnp.int32(0)
        return EnvReset(observation=_observation(step), state=step)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_step, terminated = _advance(state)
        reward = -jnp.square(action[0] - 0.5) - 0.25 * state.astype(jnp.float32)
        return EnvStep(
            observation=_observation(next_step),
            state=next_step,
            reward=reward,
            terminated=terminated,
            truncated=jnp.bool_(False),
            info={},
        )


def _discrete_env() -> GymEnv[DiscreteActionSpace]:
    # The bridge reports a union action space because EnvSpec decides at runtime;
    # the toy environment is always discrete.
    return cast(
        GymEnv[DiscreteActionSpace],
        make_gymnax_compat_env(cast(EnvProtocol[jax.Array, jax.Array, jax.Array, None], ToyDiscreteEnv())),
    )


def _continuous_env() -> GymEnv[ContinuousActionSpace]:
    return cast(
        GymEnv[ContinuousActionSpace],
        make_gymnax_compat_env(cast(EnvProtocol[jax.Array, jax.Array, jax.Array, None], ToyContinuousEnv())),
    )


class Run(NamedTuple):
    """One agent's finished training run, reduced to what the gate inspects."""

    params: list[jax.Array]
    metrics: dict[str, jax.Array]


def _finish(runner_state: tuple[object, ...], metrics: dict[str, jax.Array]) -> Run:
    params = [
        leaf
        for field in runner_state
        if isinstance(field, TrainState)
        for leaf in jax.tree_util.tree_leaves(field.params)
    ]
    return Run(params=params, metrics=metrics)


def _run_dqn(learning_starts: int) -> Run:
    config = dqn.DQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    out = jax.jit(dqn.make_train(config, env=_discrete_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_double_dqn(learning_starts: int) -> Run:
    config = double_dqn.DoubleDQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    out = jax.jit(double_dqn.make_train(config, env=_discrete_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_dueling_dqn(learning_starts: int) -> Run:
    config = dueling_dqn.DuelingDQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    out = jax.jit(dueling_dqn.make_train(config, env=_discrete_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_qrc(learning_starts: int) -> Run:
    config = qrc.QRCConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    out = jax.jit(qrc.make_train(config, env=_discrete_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_sac(learning_starts: int) -> Run:
    config = sac.SACConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    out = jax.jit(sac.make_train(config, env=_continuous_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_td3(learning_starts: int) -> Run:
    config = td3.TD3Config(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        POLICY_DELAY=1,
    )
    out = jax.jit(td3.make_train(config, env=_continuous_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


def _run_greedy_ac(learning_starts: int) -> Run:
    config = greedy_ac.GACConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        NUM_SAMPLES=4,
        NUM_RAND_ACTIONS=2,
    )
    out = jax.jit(greedy_ac.make_train(config, env=_continuous_env()))(jax.random.key(SEED))
    return _finish(tuple(out["runner_state"]), out["metrics"])


AGENT_RUNS: dict[str, Callable[[int], Run]] = {
    "dqn": _run_dqn,
    "double_dqn": _run_double_dqn,
    "dueling_dqn": _run_dueling_dqn,
    "qrc": _run_qrc,
    "sac": _run_sac,
    "td3": _run_td3,
    "greedy_ac": _run_greedy_ac,
}

LOSS_METRIC_AGENTS = ("sac", "td3", "greedy_ac")
"""Agents that publish their losses. The DQN family and ``qrc`` discard theirs entirely:
their ``_update_step`` returns the raw environment ``info``, so the learn path can only be
observed through the parameters."""


@pytest.fixture(scope="module")
def learning_runs() -> dict[str, Run]:
    return {name: run(LEARNING_STARTS) for name, run in AGENT_RUNS.items()}


@pytest.fixture(scope="module")
def warmup_only_runs() -> dict[str, Run]:
    return {name: run(TOTAL_TIMESTEPS) for name, run in AGENT_RUNS.items()}


@pytest.mark.parametrize("agent", list(AGENT_RUNS))
def test_learn_path_moves_parameters(
    agent: str,
    learning_runs: dict[str, Run],
    warmup_only_runs: dict[str, Run],
) -> None:
    learned = learning_runs[agent]
    untrained = warmup_only_runs[agent]

    unchanged = [
        bool(jnp.array_equal(before, after))
        for before, after in zip(untrained.params, learned.params, strict=True)
    ]
    assert not all(unchanged), f"{agent} left every parameter untouched: the learn path never fired"


@pytest.mark.parametrize("agent", LOSS_METRIC_AGENTS)
def test_learn_path_reports_nonzero_loss(
    agent: str,
    learning_runs: dict[str, Run],
    warmup_only_runs: dict[str, Run],
) -> None:
    learned = learning_runs[agent].metrics
    losses = {key: value for key, value in learned.items() if key.endswith("_loss")}
    assert losses, f"{agent} published no loss metric"

    for key, value in losses.items():
        assert jnp.any(value != 0.0), f"{agent} reported an all-zero {key}"
        assert jnp.all(jnp.isfinite(value)), f"{agent} reported a non-finite {key}: {value}"
        assert jnp.all(warmup_only_runs[agent].metrics[key] == 0.0), (
            f"{agent} reported a nonzero {key} without ever taking a gradient step"
        )


@pytest.mark.parametrize("agent", list(AGENT_RUNS))
def test_run_crosses_episode_boundaries(agent: str, learning_runs: dict[str, Run]) -> None:
    metrics = learning_runs[agent].metrics

    terminated = metrics["terminated"]
    assert terminated.shape == (TOTAL_TIMESTEPS,)
    assert int(jnp.sum(terminated)) >= 2, f"{agent} never crossed an episode boundary"

    for key, value in metrics.items():
        if jnp.issubdtype(value.dtype, jnp.floating):
            assert jnp.all(jnp.isfinite(value)), f"{agent} reported a non-finite {key} across a boundary"
