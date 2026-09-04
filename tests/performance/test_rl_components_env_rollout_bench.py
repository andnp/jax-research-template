"""Performance benchmark for canonical env rollout speed.

Used to also benchmark GymnaxCompatibilityBridge overhead, now deleted along with the
bridge; the canonical rollout is the only seam left to measure.
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import pytest
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep

ROLLOUT_STEPS = 512
BENCHMARK_ROUNDS = 5
RESET_KEY = jax.random.key(0)
STEP_KEYS = jax.random.split(jax.random.key(1), ROLLOUT_STEPS)
ACTIONS = jnp.ones((ROLLOUT_STEPS,), dtype=jnp.int32)


class DummyCanonicalEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="dummy-benchmark",
            observation_shape=(4,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros((4,), dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
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


def _benchmark_rollout(benchmark: Any, rollout_fn: Callable[..., Any]) -> None:
    compiled = jax.jit(rollout_fn)
    warm_result = compiled(RESET_KEY, STEP_KEYS, ACTIONS)
    jax.block_until_ready(warm_result)

    def run_once() -> Any:
        result = compiled(RESET_KEY, STEP_KEYS, ACTIONS)
        jax.block_until_ready(result)
        return result

    benchmark.pedantic(run_once, rounds=BENCHMARK_ROUNDS)


def _canonical_rollout(reset_key: chex.PRNGKey, step_keys: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
    env = DummyCanonicalEnv()
    reset = env.reset(reset_key)

    def _step(state: jax.Array, xs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
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


@pytest.mark.benchmark(group="env-seam-rollout")
def test_canonical_env_rollout_speed(benchmark: Any) -> None:
    _benchmark_rollout(benchmark, _canonical_rollout)
