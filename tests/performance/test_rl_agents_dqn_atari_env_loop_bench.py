"""Benchmark DQN Atari environment-loop layers with fake and real Pong rollouts."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal, Protocol, cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax_nn.heads import epsilon_greedy_action
from rl_agents.dqn import NatureQNetwork
from rl_agents.dqn_atari import DQNAtariConfig, make_train
from rl_components.atari import JAXAtariConfig, make_atari_adapter
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import make_gymnax_compat_env

ROLLOUT_STEPS = 64
BENCHMARK_ROUNDS = 5
TRAIN_BENCHMARK_ROUNDS = 3
RESET_KEY = jax.random.key(0)
STEP_KEYS = jax.random.split(jax.random.key(1), ROLLOUT_STEPS)
ENV_ONLY_ACTIONS = jnp.arange(ROLLOUT_STEPS, dtype=jnp.int32) % 6
POLICY_KEYS = jax.random.split(jax.random.key(2), ROLLOUT_STEPS)
TRAIN_KEY = jax.random.key(3)
BENCHMARK_ENV_VAR = "JAXATARI_BENCHMARKS"
PONG_GAME = "pong"
ATARI_OBSERVATION_SHAPE = (4, 84, 84, 1)
ATARI_ACTIONS = 6
EPSILON = jnp.asarray(0.05, dtype=jnp.float32)


class _BenchmarkFixture(Protocol):
    def pedantic(self, target: Callable[[], object], *, rounds: int) -> object: ...


class _ObservationSpace(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class _ActionSpace(Protocol):
    n: int


class _EnvLike(Protocol):
    def observation_space(self, params: object | None = None) -> _ObservationSpace: ...

    def action_space(self, params: object | None = None) -> _ActionSpace: ...

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, object]: ...

    def step(
        self,
        key: jax.Array,
        state: object,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, dict[str, jax.Array]]: ...


class FakeAtariEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="fake-atari-benchmark",
            observation_shape=ATARI_OBSERVATION_SHAPE,
            action_shape=(),
            observation_dtype=jnp.uint8,
            action_dtype=jnp.int32,
            num_actions=ATARI_ACTIONS,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros(ATARI_OBSERVATION_SHAPE, dtype=jnp.uint8),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        key: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_state = state + jnp.asarray(action, dtype=jnp.int32) + jnp.array(1, dtype=jnp.int32)
        pixel_value = jnp.asarray(next_state % 256, dtype=jnp.uint8)
        return EnvStep(
            observation=jnp.full(ATARI_OBSERVATION_SHAPE, pixel_value, dtype=jnp.uint8),
            state=next_state,
            reward=jnp.asarray(action, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={
                "returned_episode": jnp.array(False),
                "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
            },
        )


def _observation_probe(observation: jax.Array) -> jax.Array:
    return jnp.ravel(jnp.asarray(observation, dtype=jnp.float32))[0]


def _infer_observation_layout(observation_shape: tuple[int, ...]) -> Literal["hwc", "fhwc"]:
    if len(observation_shape) == 3:
        return "hwc"
    if len(observation_shape) == 4:
        return "fhwc"
    raise ValueError(f"Unexpected Atari observation shape for NatureQNetwork: {observation_shape!r}")


def _make_fake_env() -> _EnvLike:
    return cast(
        _EnvLike,
        make_gymnax_compat_env(cast(EnvProtocol[jax.Array, jax.Array, jax.Array, None], FakeAtariEnv())),
    )


def _require_real_benchmarks() -> None:
    if os.environ.get(BENCHMARK_ENV_VAR) != "1":
        pytest.skip(f"Set {BENCHMARK_ENV_VAR}=1 to run real JAXAtari benchmarks.")


def _make_real_env_or_skip() -> _EnvLike:
    _require_real_benchmarks()
    try:
        return cast(
            _EnvLike,
            make_gymnax_compat_env(
                cast(
                    EnvProtocol[jax.Array, object, jax.Array, None],
                    make_atari_adapter(JAXAtariConfig(game=PONG_GAME)),
                )
            ),
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"JAXAtari Pong benchmark environment unavailable: {exc}")


def _benchmark_compiled(benchmark: object, target: Callable[[], object], *, rounds: int = BENCHMARK_ROUNDS) -> None:
    benchmark_fixture = cast(_BenchmarkFixture, benchmark)
    warm_result = target()
    jax.block_until_ready(warm_result)

    def run_once() -> object:
        result = target()
        jax.block_until_ready(result)
        return result

    benchmark_fixture.pedantic(run_once, rounds=rounds)


def _benchmark_rollout(
    benchmark: object,
    rollout_fn: Callable[[jax.Array, jax.Array, jax.Array], object],
    xs: jax.Array,
) -> None:
    compiled = jax.jit(rollout_fn)
    _benchmark_compiled(benchmark, lambda: compiled(RESET_KEY, STEP_KEYS, xs))


def _make_env_only_rollout(env: _EnvLike) -> Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    def rollout(reset_key: jax.Array, step_keys: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
        observation, env_state = env.reset(reset_key, None)

        def _step(carry: tuple[object, jax.Array], xs: tuple[jax.Array, jax.Array]) -> tuple[tuple[object, jax.Array], jax.Array]:
            state, _last_observation = carry
            step_key, action = xs
            next_observation, next_state, reward, done, _info = env.step(step_key, state, action, None)
            metric = reward + _observation_probe(next_observation) + jnp.asarray(done, dtype=jnp.float32)
            return (next_state, next_observation), metric

        (_, final_observation), metrics = jax.lax.scan(_step, (env_state, observation), (step_keys, actions))
        return final_observation, metrics.sum()

    return rollout


def _make_policy_rollout(env: _EnvLike) -> Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    observation_space = env.observation_space(None)
    network = NatureQNetwork(
        action_dim=env.action_space(None).n,
        observation_layout=_infer_observation_layout(tuple(observation_space.shape)),
    )
    initial_observation = jnp.zeros(tuple(observation_space.shape), dtype=observation_space.dtype)
    params = network.init(jax.random.key(7), initial_observation)

    def rollout(reset_key: jax.Array, step_keys: jax.Array, action_keys: jax.Array) -> tuple[jax.Array, jax.Array]:
        observation, env_state = env.reset(reset_key, None)

        def _step(carry: tuple[object, jax.Array], xs: tuple[jax.Array, jax.Array]) -> tuple[tuple[object, jax.Array], jax.Array]:
            state, last_observation = carry
            step_key, action_key = xs
            q_values = network.apply(params, last_observation)
            action = epsilon_greedy_action(q_values, EPSILON, key=action_key)
            next_observation, next_state, reward, done, _info = env.step(step_key, state, action, None)
            metric = reward + _observation_probe(next_observation) + jnp.max(q_values) + jnp.asarray(done, dtype=jnp.float32)
            return (next_state, next_observation), metric

        (_, final_observation), metrics = jax.lax.scan(_step, (env_state, observation), (step_keys, action_keys))
        return final_observation, metrics.sum()

    return rollout


def _make_fake_micro_train() -> Callable[[jax.Array], object]:
    return make_train(
        DQNAtariConfig(
            GAME=PONG_GAME,
            REPLAY_CAPACITY=16,
            MIN_REPLAY_CAPACITY_FRACTION=0.25,
            BATCH_SIZE=4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
            LEARN_PERIOD_FRAMES=4,
            NUM_ITERATIONS=1,
            NUM_TRAIN_FRAMES_PER_ITERATION=64,
            EXPLORATION_EPSILON_DECAY_FRAME_FRACTION=0.25,
        ),
        env=_make_fake_env(),
    )


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_env_only_rollout_speed(benchmark: object) -> None:
    _benchmark_rollout(benchmark, _make_env_only_rollout(_make_fake_env()), ENV_ONLY_ACTIONS)


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_real_pong_env_only_rollout_speed(benchmark: object) -> None:
    _benchmark_rollout(benchmark, _make_env_only_rollout(_make_real_env_or_skip()), ENV_ONLY_ACTIONS)


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_policy_and_env_rollout_speed(benchmark: object) -> None:
    _benchmark_rollout(benchmark, _make_policy_rollout(_make_fake_env()), POLICY_KEYS)


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_fake_micro_train_replay_and_update_speed(benchmark: object) -> None:
    compiled = jax.jit(_make_fake_micro_train())
    warm_result = cast(dict[str, object], compiled(TRAIN_KEY))
    jax.block_until_ready(warm_result)

    metrics = cast(dict[str, jax.Array], warm_result["metrics"])
    learn_steps = int(jax.device_get(jnp.count_nonzero(metrics["loss"])))
    assert learn_steps > 0

    _benchmark_compiled(benchmark, lambda: compiled(TRAIN_KEY), rounds=TRAIN_BENCHMARK_ROUNDS)


@pytest.mark.benchmark(group="dqn-atari-env-loop")
def test_real_pong_policy_and_env_rollout_speed(benchmark: object) -> None:
    _benchmark_rollout(benchmark, _make_policy_rollout(_make_real_env_or_skip()), POLICY_KEYS)