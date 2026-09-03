"""Process control benchmark adapters.

Wraps a ``make_*_benchmark(config) -> (reset, step)`` function pair into either of
two interfaces: ``ProcessControlEnv`` implements the canonical
``rl_components.env_protocol.EnvProtocol``, and ``ProcessControlAdapter`` implements
the tuple-returning Gymnax-style ``GymEnv[ContinuousActionSpace]`` still consumed by
``projects/process-control-baselines``.

Usage::

    from rl_components.process_control_bridge import make_process_control_env

    env = make_process_control_env("bsm1")
    spec = env.spec()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp
from process_control.benchmarks.registry import BenchmarkEntry, get_action_bounds, get_benchmark_entry

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep


def _broadcast_action_bounds(
    action_low: jax.Array | float | None,
    action_high: jax.Array | float | None,
    action_dim: int,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Broadcast optional scalar or vector action bounds to ``(action_dim,)``."""
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is None or action_high is None:
        return None, None
    low = jnp.broadcast_to(jnp.asarray(action_low, dtype=jnp.float32), (action_dim,))
    high = jnp.broadcast_to(jnp.asarray(action_high, dtype=jnp.float32), (action_dim,))
    return low, high


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
        normalized_low, normalized_high = _broadcast_action_bounds(action_low, action_high, action_dim)
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._env_id = env_id
        self._max_steps = max_steps
        self._scalar_action = scalar_action
        self._obs_space = _ObsSpace(shape=(obs_dim,))
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


type ProcessControlResetFn[StateT] = Callable[[chex.PRNGKey], tuple[StateT, jax.Array]]
type ProcessControlStepFn[StateT] = Callable[
    [StateT, jax.Array, chex.PRNGKey],
    tuple[StateT, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]],
]


class ProcessControlEnv[StateT]:
    """Adapts a process control benchmark to ``rl_components.env_protocol.EnvProtocol``.

    Process-control benchmarks use::

        reset(rng) -> (state, obs)
        step(state, action, rng) -> (state, obs, reward, done, info)

    The benchmarks FUSE termination and truncation into that single ``done`` flag and
    expose nothing that distinguishes the two, so ``step`` reports ``terminated=done``
    and ``truncated=False`` always. No distinction is invented here: an optimistic
    mapping would bootstrap across real terminations. A benchmark that grows a genuine
    time limit must surface it explicitly before ``truncated`` can become nonzero.

    Observation and action dtypes are ``float32``, matching the arrays the benchmarks
    return; shapes and the optional action bounds come from the benchmark registry.
    """

    def __init__(
        self,
        reset_fn: ProcessControlResetFn[StateT],
        step_fn: ProcessControlStepFn[StateT],
        obs_dim: int,
        action_dim: int,
        env_id: str = "process_control",
        scalar_action: bool = False,
        action_low: jax.Array | float | None = None,
        action_high: jax.Array | float | None = None,
    ) -> None:
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._env_id = env_id
        self._scalar_action = scalar_action
        self._action_low, self._action_high = _broadcast_action_bounds(action_low, action_high, action_dim)

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id=self._env_id,
            observation_shape=(self._obs_dim,),
            action_shape=(self._action_dim,),
            observation_dtype=jnp.float32,
            action_dtype=jnp.float32,
            action_low=self._action_low,
            action_high=self._action_high,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, StateT]:
        del params
        state, observation = self._reset_fn(key)
        return EnvReset(observation=observation, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, StateT]:
        del params
        if self._scalar_action:
            action = jnp.squeeze(action)
        next_state, observation, reward, done, info = self._step_fn(state, action, key)
        terminated = jnp.asarray(done, dtype=bool)
        return EnvStep(
            observation=observation,
            state=next_state,
            reward=jnp.asarray(reward),
            terminated=terminated,
            truncated=jnp.zeros_like(terminated),
            info=info,
        )


def _build_benchmark(
    name: str,
    config_overrides: dict[str, Any],
) -> tuple[BenchmarkEntry, Callable, Callable, float | None, float | None]:
    """Resolve a registry entry into its ``(reset, step)`` pair and action bounds."""
    import importlib

    entry = get_benchmark_entry(name)
    mod = importlib.import_module(entry.module)
    config_cls = getattr(mod, entry.config_cls)
    make_fn = getattr(mod, entry.make_fn)

    config = config_cls(**config_overrides) if config_overrides else config_cls()
    reset_fn, step_fn = make_fn(config)
    action_low, action_high = get_action_bounds(entry, config)
    return entry, reset_fn, step_fn, action_low, action_high


def make_process_control_env(name: str, **config_overrides: Any) -> ProcessControlEnv[object]:
    """Create an ``EnvProtocol`` process-control environment by benchmark name.

    Args:
        name: benchmark name (e.g. "bsm1", "chlorine", "membrane_fouling")
        **config_overrides: keyword overrides for the benchmark config

    Returns:
        ProcessControlEnv wrapping the named benchmark.
    """
    entry, reset_fn, step_fn, action_low, action_high = _build_benchmark(name, config_overrides)

    return ProcessControlEnv(
        reset_fn=reset_fn,
        step_fn=step_fn,
        obs_dim=entry.obs_dim,
        action_dim=entry.action_dim,
        env_id=f"process_control:{entry.name}",
        scalar_action=entry.scalar_action,
        action_low=action_low,
        action_high=action_high,
    )


def make_adapter(name: str, **config_overrides: Any) -> ProcessControlAdapter:
    """Create a ProcessControlAdapter by benchmark name.

    Args:
        name: benchmark name (e.g. "bsm1", "chlorine", "membrane_fouling")
        **config_overrides: keyword overrides for the benchmark config

    Returns:
        ProcessControlAdapter wrapping the named benchmark.
    """
    entry, reset_fn, step_fn, action_low, action_high = _build_benchmark(name, config_overrides)

    return ProcessControlAdapter(
        reset_fn=reset_fn,
        step_fn=step_fn,
        obs_dim=entry.obs_dim,
        action_dim=entry.action_dim,
        env_id=f"process_control:{entry.name}",
        scalar_action=entry.scalar_action,
        action_low=action_low,
        action_high=action_high,
    )
