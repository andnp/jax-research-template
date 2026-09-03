"""Behavioural gate: every replay-buffer agent must actually reach its learn path.

Each case drives an agent through its public entry point only. No loss, target or update
rule is re-implemented here, so a broken agent cannot satisfy these assertions by agreeing
with a copy of itself.

There are two such entry points during the migration, and each agent names its own driver.
A ported agent goes through ``AgentProtocol`` plus :func:`rl_components.loop.run`; the rest
still go through their private ``make_train``. Every assertion below holds for both, which
is what makes the pair comparable: a port that loses the learn path fails the same test its
``make_train`` predecessor passed. Two facts differ by construction rather than by defect,
so the driver reports them instead of the assertions assuming them -- the loop closes
``steps - 1`` transitions where ``make_train`` closes ``steps``, and boundary flags arrive
under the loop's reserved ``loop/`` prefix rather than in the environment's ``info``.

Every agent is run twice against the same seed and the same toy environment:

- a *learning* run, whose ``LEARNING_STARTS`` sits strictly below the step budget, so the
  ``can_train`` branch genuinely fires;
- a *warmup-only* run, whose ``LEARNING_STARTS`` equals the step budget, so the branch can
  never fire.

The pair is the negative control: parameters must move in the first run and must stay
bit-identical in the second, which is only possible if the gradient update ran. Agents that
publish a loss metric must additionally report a nonzero loss in the learning run and an
all-zero loss in the warmup-only run, so a zero-filled placeholder metric fails.

The toy environment terminates every ``EPISODE_LENGTH`` steps, so each run crosses several
episode boundaries with a nonzero ``done``. The stored discounts must then carry an exact
``0.0``, which fails any agent that ignores ``done`` when writing to its buffer. Who
performs the reset differs by driver: the ``make_train`` agents have no reset of their own,
so their environment auto-resets inside ``step``; the loop owns the boundary, so its
environment does not, and the true terminal observation therefore survives to be asserted
on. That the zero *target* also collapses to the bare reward is gated separately, through
each agent's real loss, in ``test_rl_agents_terminal_bootstrap.py`` and
``test_rl_agents_qrc_gradient.py``.

Excluded agents: ``dqn_atari`` needs image observations for its Nature CNN and gates
learning on replay occupancy rather than on ``LEARNING_STARTS``, and ``rainbow`` samples
from its own prioritised buffer rather than ``rl_components.buffers.ReplayBuffer``. Both
need their own gate; ``dqn_atari`` has one in
``tests/medium/test_rl_agents_dqn_atari_learn_path.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple, cast, override

import jax
import jax.numpy as jnp
import pytest
from flax.training.train_state import TrainState
from rl_agents import double_dqn, dqn, dueling_dqn, greedy_ac, qrc, sac, sac_rc, td3
from rl_components.buffers import ReplayBufferState
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.gymnax_bridge import make_gymnax_compat_env
from rl_components.loop import run

TOTAL_TIMESTEPS = 12
"""Step budget for every case: short enough to stay inside the medium tier."""

LEARNING_STARTS = 4
"""Strictly below the budget, so ``t > LEARNING_STARTS`` holds for seven of the steps."""

EPISODE_LENGTH = 5
"""Terminates at step 4 (before learning starts) and step 9 (after it does)."""

BUFFER_SIZE = 16
BATCH_SIZE = 4
SEED = 0
GAMMA = 0.99
"""Discount for the ported drivers, where it belongs to ``loop.run`` rather than a config."""


def _observation(step: jax.Array) -> jax.Array:
    scaled = step.astype(jnp.float32) / EPISODE_LENGTH
    return jnp.stack([scaled, 1.0 - scaled])


def _advance(step: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Step the counter, terminating and auto-resetting at the episode boundary."""
    nxt = step + jnp.int32(1)
    terminated = nxt >= EPISODE_LENGTH
    return jnp.where(terminated, jnp.int32(0), nxt), terminated


TERMINAL_OBSERVATION = _observation(jnp.int32(EPISODE_LENGTH))
"""The true final observation of an episode, distinct from the post-reset ``_observation(0)``.

An agent that inserts the post-reset observation instead of the terminal one stores
``_observation(0)`` here, which is what makes the distinction assertable.
"""


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


class ToyDiscreteEpisodeEnv(ToyDiscreteEnv):
    """``ToyDiscreteEnv`` without the auto-reset, for the driver whose loop owns boundaries.

    Reaching the boundary leaves the counter at ``EPISODE_LENGTH`` rather than wrapping to
    zero, so ``EnvStep.observation`` is the state the transition actually reached.
    """

    @override
    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_step = state + jnp.int32(1)
        reward = action.astype(jnp.float32) - 0.25 * state.astype(jnp.float32)
        return EnvStep(
            observation=_observation(next_step),
            state=next_step,
            reward=reward,
            terminated=next_step >= EPISODE_LENGTH,
            truncated=jnp.bool_(False),
            info={},
        )


class ToyContinuousEpisodeEnv(ToyContinuousEnv):
    """``ToyContinuousEnv`` without the auto-reset, for the driver whose loop owns boundaries."""

    @override
    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_step = state + jnp.int32(1)
        reward = -jnp.square(action[0] - 0.5) - 0.25 * state.astype(jnp.float32)
        return EnvStep(
            observation=_observation(next_step),
            state=next_step,
            reward=reward,
            terminated=next_step >= EPISODE_LENGTH,
            truncated=jnp.bool_(False),
            info={},
        )


def _continuous_env() -> GymEnv[ContinuousActionSpace]:
    return cast(
        GymEnv[ContinuousActionSpace],
        make_gymnax_compat_env(cast(EnvProtocol[jax.Array, jax.Array, jax.Array, None], ToyContinuousEnv())),
    )


class Run(NamedTuple):
    """One agent's finished training run, reduced to what the gate inspects."""

    params: dict[str, list[jax.Array]]
    """Parameter leaves per ``TrainState``, keyed by its ``RunnerState`` field name.

    Grouping matters: a flat list would let an agent that updates only its critic and
    never its actor satisfy an "at least one leaf moved" assertion.
    """

    discounts: jax.Array
    """The discount stored for every transition the run actually wrote to the buffer."""

    next_observations: jax.Array
    """The bootstrap observation stored for every transition the run wrote to the buffer."""

    observations: jax.Array
    """The acting observation stored for every transition the run wrote to the buffer."""

    metrics: dict[str, jax.Array]

    terminated: jax.Array
    """Per-step environment termination flag, wherever this driver reports it."""

    transitions: int
    """Transitions this driver closes over the budget, which the loop and ``make_train``
    disagree about by one: the action taken on the loop's final iteration opens a
    transition that never closes."""


def _finish(
    agent_fields: Mapping[str, object],
    metrics: dict[str, jax.Array],
    *,
    terminated: jax.Array,
    transitions: int,
) -> Run:
    params = {
        name: jax.tree_util.tree_leaves(field.params)
        for name, field in agent_fields.items()
        if isinstance(field, TrainState)
    }
    buffer_state = agent_fields["buffer_state"]
    assert isinstance(buffer_state, ReplayBufferState)
    count = int(buffer_state.count)
    return Run(
        params=params,
        discounts=buffer_state.discount[:count],
        next_observations=buffer_state.next_obs[:count],
        observations=buffer_state.obs[:count],
        metrics=metrics,
        terminated=terminated,
        transitions=transitions,
    )


def _run_dqn(learning_starts: int) -> Run:
    config = dqn.DQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = dqn.DQNAgent(config)
    env = ToyDiscreteEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_double_dqn(learning_starts: int) -> Run:
    config = double_dqn.DoubleDQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = double_dqn.DoubleDQNAgent(config)
    env = ToyDiscreteEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_dueling_dqn(learning_starts: int) -> Run:
    config = dueling_dqn.DuelingDQNConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = dueling_dqn.DuelingDQNAgent(config)
    env = ToyDiscreteEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_qrc(learning_starts: int) -> Run:
    config = qrc.QRCConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = qrc.QRCAgent(config)
    env = ToyDiscreteEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_sac(learning_starts: int) -> Run:
    config = sac.SACConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = sac.SACAgent(config)
    env = ToyContinuousEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_td3(learning_starts: int) -> Run:
    config = td3.TD3Config(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        POLICY_DELAY=1,
    )
    agent = td3.TD3Agent(config)
    env = ToyContinuousEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


def _run_sac_rc(learning_starts: int) -> Run:
    config = sac_rc.SACRCConfig(
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        LEARNING_STARTS=learning_starts,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )
    agent = sac_rc.SACRCAgent(config)
    env = ToyContinuousEpisodeEnv()
    final_state, metrics = jax.jit(
        lambda key: run(agent, env, key, steps=TOTAL_TIMESTEPS, gamma=GAMMA)
    )(jax.random.key(SEED))
    return _finish(
        dict(final_state.agent_state),
        metrics,
        terminated=metrics["loop/terminated"],
        transitions=TOTAL_TIMESTEPS - 1,
    )


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
    return _finish(
        out["runner_state"]._asdict(),
        out["metrics"],
        terminated=out["metrics"]["terminated"],
        transitions=TOTAL_TIMESTEPS,
    )


AGENT_RUNS: dict[str, Callable[[int], Run]] = {
    "dqn": _run_dqn,
    "double_dqn": _run_double_dqn,
    "dueling_dqn": _run_dueling_dqn,
    "qrc": _run_qrc,
    "sac": _run_sac,
    "td3": _run_td3,
    "sac_rc": _run_sac_rc,
    "greedy_ac": _run_greedy_ac,
}

LOOP_DRIVER_AGENTS = ("dqn", "double_dqn", "dueling_dqn", "qrc", "sac", "td3", "sac_rc")
"""Agents driven through ``AgentProtocol`` plus ``loop.run`` rather than ``make_train``.

Each port adds itself here. When the tuple holds every agent, the ``make_train`` drivers,
``ToyDiscreteEnv``'s auto-reset and the Gymnax compatibility bridge all go together."""

LOSS_METRIC_AGENTS = tuple(AGENT_RUNS)
"""Agents that publish their losses, which is now all of them.

``qrc`` used to be the exception: it computed a real loss and then discarded it, returning
the raw environment ``info``, so it was observable only through its parameters. Its port
publishes the loss it already computed, so the allowlist is no longer a subset and this
name exists only to say so. It goes when the last ``make_train`` driver does."""


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

    assert learned.params.keys() == untrained.params.keys()
    for name, after in learned.params.items():
        unchanged = [
            bool(jnp.array_equal(before, leaf))
            for before, leaf in zip(untrained.params[name], after, strict=True)
        ]
        assert not all(unchanged), (
            f"{agent}.{name} left every parameter untouched: its update never fired"
        )


@pytest.mark.parametrize("agent", LOSS_METRIC_AGENTS)
def test_learn_path_reports_nonzero_loss(
    agent: str,
    learning_runs: dict[str, Run],
    warmup_only_runs: dict[str, Run],
) -> None:
    learned = learning_runs[agent].metrics
    losses = {key: value for key, value in learned.items() if key == "loss" or key.endswith("_loss")}
    assert losses, f"{agent} published no loss metric"

    for key, value in losses.items():
        assert jnp.any(value != 0.0), f"{agent} reported an all-zero {key}"
        assert jnp.all(jnp.isfinite(value)), f"{agent} reported a non-finite {key}: {value}"
        assert jnp.all(warmup_only_runs[agent].metrics[key] == 0.0), (
            f"{agent} reported a nonzero {key} without ever taking a gradient step"
        )


@pytest.mark.parametrize("agent", list(AGENT_RUNS))
def test_terminal_transitions_store_a_zero_discount(agent: str, learning_runs: dict[str, Run]) -> None:
    """The agent must fuse ``done`` into the discount it stores, not just observe it.

    Whether a terminal transition reaches any particular minibatch is seed luck, so this
    asserts on what the agent wrote instead: an agent that ignores termination stores the
    bare ``GAMMA`` everywhere and never produces the exact ``0.0`` demanded here.
    """
    learned = learning_runs[agent]
    discounts = learned.discounts

    assert discounts.size == learned.transitions
    assert jnp.any(discounts == 0.0), (
        f"{agent} stored no zero discount: termination never reached the stored transitions"
    )
    assert jnp.any(discounts > 0.0), f"{agent} stored a zero discount for every transition"


@pytest.mark.parametrize("agent", list(AGENT_RUNS))
def test_run_crosses_episode_boundaries(agent: str, learning_runs: dict[str, Run]) -> None:
    learned = learning_runs[agent]
    metrics = learned.metrics

    terminated = learned.terminated
    assert terminated.shape == (TOTAL_TIMESTEPS,)
    assert int(jnp.sum(terminated)) >= 2, f"{agent} never crossed an episode boundary"

    for key, value in metrics.items():
        if jnp.issubdtype(value.dtype, jnp.floating):
            assert jnp.all(jnp.isfinite(value)), f"{agent} reported a non-finite {key} across a boundary"


@pytest.mark.parametrize("agent", LOOP_DRIVER_AGENTS)
def test_terminal_transitions_store_the_true_final_observation(agent: str, learning_runs: dict[str, Run]) -> None:
    """The bootstrap must be taken at the boundary, never across it.

    The fused auto-reset this port removes replaced the terminal observation with the
    post-reset one, so a terminal transition bootstrapped from the start of the next
    episode. The discount is zero there, which is why the defect was invisible for
    one-step targets and corrupting for every n-step and multi-network agent that follows.
    """
    learned = learning_runs[agent]
    terminal = learned.discounts == 0.0

    assert bool(jnp.any(terminal)), f"{agent} stored no terminal transition to check"
    assert jnp.allclose(learned.next_observations[terminal], TERMINAL_OBSERVATION), (
        f"{agent} stored the post-reset observation at a boundary instead of the true final state"
    )
    assert not jnp.allclose(TERMINAL_OBSERVATION, _observation(jnp.int32(0))), (
        "the toy environment must make the two observations distinguishable"
    )


@pytest.mark.parametrize("agent", LOOP_DRIVER_AGENTS)
def test_the_transition_after_a_boundary_starts_the_new_episode(agent: str, learning_runs: dict[str, Run]) -> None:
    """The mirror of the terminal-observation property, and it fails the other way.

    Insertion must read ``timestep.bootstrap_observation`` while the agent's carried
    ``last_obs`` must become ``timestep.observation``. Getting the first wrong bootstraps a
    terminal transition from the next episode; getting the second wrong pairs the previous
    episode's final state with an action chosen from the new episode's first state, which
    is a transition that never happened in any episode. Both stay plausible under training
    and only this assertion separates them.
    """
    learned = learning_runs[agent]
    terminal = jnp.flatnonzero(learned.discounts == 0.0)
    count = int(learned.observations.shape[0])
    following = [int(index) + 1 for index in terminal if int(index) + 1 < count]

    assert following, f"{agent} closed no transition after a boundary to check"
    episode_start = _observation(jnp.int32(0))
    for index in following:
        assert jnp.allclose(learned.observations[index], episode_start), (
            f"{agent} acted from the post-reset state but stored the previous episode's "
            f"final state as the observation of transition {index}"
        )
