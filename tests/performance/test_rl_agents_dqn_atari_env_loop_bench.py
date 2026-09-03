"""Benchmark the Atari DQN layers: environment, policy, and the learner, on fake and real Pong.

Every benchmark drives the real :class:`~rl_agents.dqn_atari.DQNAtariAgent` through the real
:func:`rl_components.loop.run`, or a real public component. The earlier version carried a
hand-copy of the agent's loss and two hand-rolled scan rollouts that re-implemented the loop
body; those measured a copy of the agent rather than the agent, and they are gone. Isolating
the learner's sub-phases now happens by construction instead: the same ``agent.step`` is
timed at a ``step_index`` where its can-train predicate holds and at one where it does not,
so the difference between the two is the learner and neither number comes from copied code.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp
import pytest
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariConfig,
    DQNAtariRuntimeConfig,
    dqn_atari_runtime_from_dqn_zoo,
    dqn_zoo_atari_total_train_env_steps,
)
from rl_components.agent_protocol import AgentProtocol, AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.loop import run
from rl_components.timestep import Timestep

BENCHMARK_ROUNDS = 5
TRAIN_BENCHMARK_ROUNDS = 3
UPDATE_BENCHMARK_ROUNDS = 3
ROLLOUT_STEPS = 64
RUN_KEY = jax.random.key(1)
AGENT_INIT_KEY = jax.random.key(13)
REPLAY_SAMPLE_KEY = jax.random.key(4)
BENCHMARK_ENV_VAR = "ALE_BENCHMARKS"
PONG_GAME = "Pong"
ATARI_OBSERVATION_SHAPE = (4, 84, 84, 1)
ATARI_ACTIONS = 6
GAMMA = 0.99
LEARNER_REPLAY_CAPACITY = 128
LEARNER_BATCH_SIZE = 32


class _BenchmarkFixture(Protocol):
    def pedantic(self, target: Callable[[], object], *, rounds: int) -> object: ...


class FakeAtariEnv:
    """A counter environment wearing Atari's observation shape, dtype and action count."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="fake-atari-benchmark",
            observation_shape=ATARI_OBSERVATION_SHAPE,
            action_shape=(),
            observation_dtype=jnp.dtype(jnp.uint8),
            action_dtype=jnp.dtype(jnp.int32),
            num_actions=ATARI_ACTIONS,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros(ATARI_OBSERVATION_SHAPE, dtype=jnp.uint8),
            state=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(
        self,
        key: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_state = state + jnp.asarray(action, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
        return EnvStep(
            observation=jnp.full(ATARI_OBSERVATION_SHAPE, next_state % 256, dtype=jnp.uint8),
            state=next_state,
            reward=jnp.asarray(action, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={},
        )


class ConstantActionAgent:
    """The cheapest possible ``AgentProtocol``, so a run measures the environment alone."""

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> jax.Array:
        del key
        return jnp.zeros(tuple(spec.action_shape), dtype=jnp.dtype(spec.action_dtype))

    def step(
        self,
        state: jax.Array,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[jax.Array, jax.Array]:
        del step_index
        return AgentStep(
            state=state,
            action=state,
            metrics={"observation_probe": jnp.ravel(timestep.observation)[0].astype(jnp.float32)},
        )


def _require_real_benchmarks() -> None:
    if os.environ.get(BENCHMARK_ENV_VAR) != "1":
        pytest.skip(f"Set {BENCHMARK_ENV_VAR}=1 to run real ALE benchmarks.")


def _make_real_env_or_skip() -> EnvProtocol[jax.Array, object, jax.Array, None]:
    _require_real_benchmarks()
    from rl_components.atari_ale import AleAtariConfig, make_atari_adapter

    try:
        return cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            make_atari_adapter(AleAtariConfig(game=PONG_GAME)),
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"ALE Pong benchmark environment unavailable: {exc}")


def _benchmark_compiled(benchmark: object, target: Callable[[], object], *, rounds: int = BENCHMARK_ROUNDS) -> None:
    benchmark_fixture = cast(_BenchmarkFixture, benchmark)
    warm_result = target()
    jax.block_until_ready(warm_result)

    def run_once() -> object:
        result = target()
        jax.block_until_ready(result)
        return result

    benchmark_fixture.pedantic(run_once, rounds=rounds)


def _fake_env() -> EnvProtocol[jax.Array, object, jax.Array, None]:
    return cast(EnvProtocol[jax.Array, object, jax.Array, None], FakeAtariEnv())


def _learner_config() -> DQNAtariConfig:
    """The learner configuration the sub-phase benchmarks share.

    Its can-train predicate gates on replay occupancy, and the warmup threshold sits above
    what a ``ROLLOUT_STEPS`` run can fill, so the acting-path benchmarks never learn while a
    benchmark handed a pre-filled buffer does.
    """
    return DQNAtariConfig(
        REPLAY_CAPACITY=LEARNER_REPLAY_CAPACITY,
        MIN_REPLAY_CAPACITY_FRACTION=0.75,
        BATCH_SIZE=LEARNER_BATCH_SIZE,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
        LEARN_PERIOD_FRAMES=4,
        EXPLORATION_EPSILON_DECAY_FRAME_FRACTION=0.25,
    )


def _runtime_config(config: DQNAtariConfig) -> DQNAtariRuntimeConfig:
    return dqn_atari_runtime_from_dqn_zoo(
        config,
        num_iterations=1,
        num_train_frames_per_iteration=ROLLOUT_STEPS * config.NUM_ACTION_REPEATS,
    )


def _compiled_run[AgentStateT](
    agent: AgentProtocol[AgentStateT, jax.Array, jax.Array],
    env: EnvProtocol[jax.Array, object, jax.Array, None],
    steps: int,
) -> Callable[[jax.Array], object]:
    return jax.jit(lambda key: run(agent, env, key, steps=steps, gamma=GAMMA))


def _fake_replay_buffer_state(capacity: int) -> ReplayBufferState:
    flat_size = capacity * int(jnp.prod(jnp.asarray(ATARI_OBSERVATION_SHAPE)))
    obs = jnp.arange(flat_size, dtype=jnp.int32).reshape((capacity, *ATARI_OBSERVATION_SHAPE))
    return ReplayBufferState(
        obs=jnp.asarray(obs % 256, dtype=jnp.uint8),
        actions=jnp.arange(capacity, dtype=jnp.int32) % ATARI_ACTIONS,
        rewards=jnp.linspace(0.0, 1.0, capacity, dtype=jnp.float32),
        next_obs=jnp.asarray((obs + 1) % 256, dtype=jnp.uint8),
        discount=0.99 * (1.0 - (jnp.arange(capacity, dtype=jnp.int32) % 7 == 0).astype(jnp.float32)),
        pointer=jnp.asarray(0, dtype=jnp.int32),
        count=jnp.asarray(capacity, dtype=jnp.int32),
    )


def _agent_step_target(*, learning: bool) -> Callable[[], object]:
    """Compile one ``agent.step``, with the replay warmup either satisfied or not."""
    config = _learner_config()
    agent = DQNAtariAgent(config, _runtime_config(config))
    env = FakeAtariEnv()
    state = agent.init(AGENT_INIT_KEY, env.spec())
    if learning:
        state = dataclasses.replace(state, buffer_state=_fake_replay_buffer_state(config.REPLAY_CAPACITY))
    observation = env.reset(RUN_KEY).observation
    timestep = Timestep(
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(GAMMA, dtype=jnp.float32),
        bootstrap_observation=observation,
        episode_end=jnp.bool_(False),
        observation=observation,
    )
    compiled = jax.jit(lambda agent_state, step_index: agent.step(agent_state, timestep, step_index))
    step_index = jnp.asarray(4, dtype=jnp.int32)
    return lambda: compiled(state, step_index)


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_env_only_rollout_speed(benchmark: object) -> None:
    compiled = _compiled_run(ConstantActionAgent(), _fake_env(), ROLLOUT_STEPS)
    _benchmark_compiled(benchmark, lambda: compiled(RUN_KEY))


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_real_pong_env_only_rollout_speed(benchmark: object) -> None:
    compiled = _compiled_run(ConstantActionAgent(), _make_real_env_or_skip(), ROLLOUT_STEPS)
    _benchmark_compiled(benchmark, lambda: compiled(RUN_KEY))


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_policy_and_env_rollout_speed(benchmark: object) -> None:
    config = _learner_config()
    agent = DQNAtariAgent(config, _runtime_config(config))
    compiled = _compiled_run(agent, _fake_env(), ROLLOUT_STEPS)
    _benchmark_compiled(benchmark, lambda: compiled(RUN_KEY))


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_real_pong_policy_and_env_rollout_speed(benchmark: object) -> None:
    config = _learner_config()
    agent = DQNAtariAgent(config, _runtime_config(config))
    compiled = _compiled_run(agent, _make_real_env_or_skip(), ROLLOUT_STEPS)
    _benchmark_compiled(benchmark, lambda: compiled(RUN_KEY))


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_micro_train_replay_and_update_speed(benchmark: object) -> None:
    config = DQNAtariConfig(
        REPLAY_CAPACITY=16,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
        LEARN_PERIOD_FRAMES=4,
        EXPLORATION_EPSILON_DECAY_FRAME_FRACTION=0.25,
    )
    runtime_config = _runtime_config(config)
    agent = DQNAtariAgent(config, runtime_config)
    compiled = _compiled_run(agent, _fake_env(), dqn_zoo_atari_total_train_env_steps(runtime_config))

    warm_result = cast(tuple[object, dict[str, jax.Array]], compiled(RUN_KEY))
    jax.block_until_ready(warm_result)
    learn_steps = int(jax.device_get(jnp.count_nonzero(warm_result[1]["loss"])))
    assert learn_steps > 0, "the benchmark must measure a run that actually learns"

    _benchmark_compiled(benchmark, lambda: compiled(RUN_KEY), rounds=TRAIN_BENCHMARK_ROUNDS)


@pytest.mark.benchmark(group="dqn-atari-update-subphases")
def test_fake_replay_sampling_only_speed(benchmark: object) -> None:
    buffer = ReplayBuffer(
        LEARNER_REPLAY_CAPACITY,
        ATARI_OBSERVATION_SHAPE,
        (),
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.uint8),
    )
    buffer_state = _fake_replay_buffer_state(LEARNER_REPLAY_CAPACITY)
    compiled = jax.jit(lambda state, key: buffer.sample(state, key, LEARNER_BATCH_SIZE))
    _benchmark_compiled(
        benchmark,
        lambda: compiled(buffer_state, REPLAY_SAMPLE_KEY),
        rounds=UPDATE_BENCHMARK_ROUNDS,
    )


@pytest.mark.benchmark(group="dqn-atari-update-subphases")
def test_fake_agent_step_without_learning_speed(benchmark: object) -> None:
    _benchmark_compiled(benchmark, _agent_step_target(learning=False), rounds=UPDATE_BENCHMARK_ROUNDS)


@pytest.mark.benchmark(group="dqn-atari-update-subphases")
def test_fake_agent_step_with_learning_speed(benchmark: object) -> None:
    _benchmark_compiled(benchmark, _agent_step_target(learning=True), rounds=UPDATE_BENCHMARK_ROUNDS)
