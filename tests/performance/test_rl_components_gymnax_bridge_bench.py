"""Performance benchmarks for canonical env rollout vs Gymnax bridge overhead."""

import chex
import gymnax.wrappers
import jax
import jax.numpy as jnp
import pytest
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import GymnaxCompatibilityBridge

ROLLOUT_STEPS = 512
BENCHMARK_ROUNDS = 5
RESET_KEY = jax.random.key(0)
STEP_KEYS = jax.random.split(jax.random.key(1), ROLLOUT_STEPS)
ACTIONS = jnp.ones((ROLLOUT_STEPS,), dtype=jnp.int32)


class DummyCanonicalEnv:
    def spec(self, params: None = None):
        del params
        return EnvSpec(
            id="dummy-benchmark",
            observation_shape=(4,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
        )

    def reset(self, key: chex.PRNGKey, params: None = None):
        del key, params
        return EnvReset(
            observation=jnp.zeros((4,), dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None):
        del key, params
        next_state = state + jnp.asarray(action, dtype=jnp.int32)
        next_state_f32 = jnp.asarray(next_state, dtype=jnp.float32)
        return EnvStep(
            observation=jnp.full((4,), next_state_f32, dtype=jnp.float32),
            state=next_state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={"custom_metric": next_state_f32},
        )


def _benchmark_rollout(benchmark, rollout_fn):
    compiled = jax.jit(rollout_fn)
    warm_result = compiled(RESET_KEY, STEP_KEYS, ACTIONS)
    jax.block_until_ready(warm_result)

    def run_once():
        result = compiled(RESET_KEY, STEP_KEYS, ACTIONS)
        jax.block_until_ready(result)
        return result

    benchmark.pedantic(run_once, rounds=BENCHMARK_ROUNDS)


def _canonical_rollout(reset_key: chex.PRNGKey, step_keys: jax.Array, actions: jax.Array):
    env = DummyCanonicalEnv()
    reset = env.reset(reset_key)

    def _step(state: jax.Array, xs: tuple[jax.Array, jax.Array]):
        step_key, action = xs
        transition = env.step(step_key, state, action)
        metric = (
            transition.reward
            + transition.observation[0]
            + jnp.asarray(transition.terminated, dtype=jnp.float32)
            + jnp.asarray(transition.truncated, dtype=jnp.float32)
            + jnp.asarray(transition.info["custom_metric"], dtype=jnp.float32)
        )
        return transition.state, metric

    final_state, metrics = jax.lax.scan(_step, reset.state, (step_keys, actions))
    return final_state, metrics.sum()


def _bridge_rollout(reset_key: chex.PRNGKey, step_keys: jax.Array, actions: jax.Array):
    env = GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyCanonicalEnv())
    _, state = env.reset(reset_key, None)

    def _step(carry_state: jax.Array, xs: tuple[jax.Array, jax.Array]):
        step_key, action = xs
        observation, next_state, reward, done, info = env.step(step_key, carry_state, action, None)
        metric = (
            reward
            + observation[0]
            + jnp.asarray(done, dtype=jnp.float32)
            + jnp.asarray(info["terminated"], dtype=jnp.float32)
            + jnp.asarray(info["truncated"], dtype=jnp.float32)
            + jnp.asarray(info["custom_metric"], dtype=jnp.float32)
        )
        return next_state, metric

    final_state, metrics = jax.lax.scan(_step, state, (step_keys, actions))
    return final_state, metrics.sum()


def _bridge_log_wrapper_rollout(reset_key: chex.PRNGKey, step_keys: jax.Array, actions: jax.Array):
    env = gymnax.wrappers.LogWrapper(GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyCanonicalEnv()))
    _, state = env.reset(reset_key, None)  # type: ignore[not-iterable, invalid-argument-type, too-many-positional-arguments]  # gymnax JitWrapped

    def _step(carry_state: object, xs: tuple[jax.Array, jax.Array]):
        step_key, action = xs
        observation, next_state, reward, done, info = env.step(step_key, carry_state, action, None)  # type: ignore[not-iterable, invalid-argument-type, too-many-positional-arguments]  # gymnax JitWrapped
        metric = (
            reward
            + observation[0]
            + jnp.asarray(done, dtype=jnp.float32)
            + jnp.asarray(info["returned_episode"], dtype=jnp.float32)
            + jnp.asarray(info["returned_episode_returns"], dtype=jnp.float32)
            + jnp.asarray(info["returned_episode_lengths"], dtype=jnp.float32)
            + jnp.asarray(info["custom_metric"], dtype=jnp.float32)
        )
        return next_state, metric

    final_state, metrics = jax.lax.scan(_step, state, (step_keys, actions))
    return final_state, metrics.sum()


@pytest.mark.benchmark(group="env-seam-rollout")
def test_canonical_env_rollout_speed(benchmark):
    _benchmark_rollout(benchmark, _canonical_rollout)


@pytest.mark.benchmark(group="env-seam-rollout")
def test_gymnax_bridge_rollout_speed(benchmark):
    _benchmark_rollout(benchmark, _bridge_rollout)


@pytest.mark.benchmark(group="env-seam-rollout")
def test_gymnax_bridge_log_wrapper_rollout_speed(benchmark):
    _benchmark_rollout(benchmark, _bridge_log_wrapper_rollout)
