"""Process control benchmark adapter for the Gymnax-style GymEnv protocol.

Wraps a ``make_*_benchmark(config) -> (reset, step)`` function pair into
the tuple-returning GymEnv[ContinuousActionSpace] interface expected by
RL agents (TD3, SAC, etc.).

Usage::

    from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, make_bsm1_benchmark
    from rl_components.process_control_bridge import ProcessControlAdapter

    config = BSM1BenchmarkConfig()
    reset_fn, step_fn = make_bsm1_benchmark(config)
    env = ProcessControlAdapter(reset_fn, step_fn, obs_dim=9, action_dim=2, env_id="bsm1")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from process_control.benchmarks.registry import get_benchmark_entry


@dataclass(frozen=True)
class _ObsSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32


@dataclass(frozen=True)
class _ActionSpace:
    shape: tuple[int, ...]
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


class ProcessControlAdapter:
    """Adapts a process control benchmark to the GymEnv[ContinuousActionSpace] protocol.

    Process-control benchmarks use:
        reset(rng) -> (state, obs)
        step(state, action, rng) -> (state, obs, reward, done, info)

    GymEnv protocol expects:
        reset(key, params) -> (obs, state)
        step(key, state, action, params) -> (obs, state, reward, done, info)
    """

    def __init__(
        self,
        reset_fn: Callable,
        step_fn: Callable,
        obs_dim: int,
        action_dim: int,
        env_id: str = "process_control",
        max_steps: int = 10000,
        scalar_action: bool = False,
        action_low: jax.Array | float | None = None,
        action_high: jax.Array | float | None = None,
    ):
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be provided together")
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._env_id = env_id
        self._max_steps = max_steps
        self._scalar_action = scalar_action
        self._obs_space = _ObsSpace(shape=(obs_dim,))
        if action_low is None:
            normalized_low = normalized_high = None
        else:
            normalized_low = jnp.broadcast_to(jnp.asarray(action_low, dtype=jnp.float32), (action_dim,))
            normalized_high = jnp.broadcast_to(jnp.asarray(action_high, dtype=jnp.float32), (action_dim,))
        self._action_space = _ActionSpace(
            shape=(action_dim,),
            action_low=normalized_low,
            action_high=normalized_high,
        )

    def observation_space(self, params: object | None = None):
        return self._obs_space

    def action_space(self, params: object | None = None):
        return self._action_space

    def reset(self, key: jax.Array, params: object | None = None):
        state, obs = self._reset_fn(key)
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ):
        if self._scalar_action:
            action = jnp.squeeze(action)
        new_state, obs, reward, done, info = self._step_fn(state, action, key)
        return obs, new_state, reward, done, info


def make_adapter(name: str, **config_overrides: Any) -> ProcessControlAdapter:
    """Create a ProcessControlAdapter by benchmark name.

    Args:
        name: benchmark name (e.g. "bsm1", "chlorine", "membrane_fouling")
        **config_overrides: keyword overrides for the benchmark config

    Returns:
        ProcessControlAdapter wrapping the named benchmark.
    """
    import importlib

    entry = get_benchmark_entry(name)
    mod = importlib.import_module(entry.module)
    config_cls = getattr(mod, entry.config_cls)
    make_fn = getattr(mod, entry.make_fn)

    config = config_cls(**config_overrides) if config_overrides else config_cls()
    reset_fn, step_fn = make_fn(config)

    return ProcessControlAdapter(
        reset_fn=reset_fn,
        step_fn=step_fn,
        obs_dim=entry.obs_dim,
        action_dim=entry.action_dim,
        env_id=f"process_control:{entry.name}",
        scalar_action=entry.scalar_action,
        action_low=entry.action_low,
        action_high=entry.action_high,
    )
