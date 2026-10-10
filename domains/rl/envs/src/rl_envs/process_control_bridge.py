"""Process control benchmark environment bridge.

Wraps a ``make_*_benchmark(config) -> (reset, step)`` function pair into the
canonical ``rl_components.env_protocol.EnvProtocol`` interface.

Usage::

    from rl_envs.process_control_bridge import make_process_control_env

    env = make_process_control_env("bsm1")
    spec = env.spec()
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from process_control.benchmarks.registry import BenchmarkEntry, get_action_bounds, get_action_dim, get_benchmark_entry, get_observation_dim
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep


def _broadcast_action_bounds(
    action_low: jax.Array | float | tuple[float, ...] | None,
    action_high: jax.Array | float | tuple[float, ...] | None,
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
        action_low: jax.Array | float | tuple[float, ...] | None = None,
        action_high: jax.Array | float | tuple[float, ...] | None = None,
    ) -> None:
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._env_id = env_id
        self._scalar_action = scalar_action
        self._action_low, self._action_high = _broadcast_action_bounds(action_low, action_high, action_dim)

    def spec(self, params: object | None = None) -> EnvSpec:
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

    def reset(self, key: chex.PRNGKey, params: object | None = None) -> EnvReset[jax.Array, StateT]:
        del params
        state, observation = self._reset_fn(key)
        return EnvReset(observation=observation, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: object | None = None,
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
) -> tuple[BenchmarkEntry, Callable, Callable, int, jax.Array | float | tuple[float, ...] | None, jax.Array | float | tuple[float, ...] | None]:
    """Resolve a registry entry into its ``(reset, step)`` pair, action dimension and action bounds."""
    import importlib

    entry = get_benchmark_entry(name)
    mod = importlib.import_module(entry.module)
    config_cls = getattr(mod, entry.config_cls)
    make_fn = getattr(mod, entry.make_fn)

    config = config_cls(**config_overrides) if config_overrides else config_cls()
    reset_fn, step_fn = make_fn(config)
    action_low, action_high = get_action_bounds(entry, config)
    return entry, reset_fn, step_fn, get_action_dim(entry, config), action_low, action_high


def make_process_control_env(name: str, **config_overrides: Any) -> ProcessControlEnv[object]:
    """Create an ``EnvProtocol`` process-control environment by benchmark name.

    Args:
        name: benchmark name (e.g. "bsm1", "chlorine", "membrane_fouling")
        **config_overrides: keyword overrides for the benchmark config

    Returns:
        ProcessControlEnv wrapping the named benchmark.
    """
    entry, reset_fn, step_fn, action_dim, action_low, action_high = _build_benchmark(name, config_overrides)

    scenario = config_overrides.get("scenario")
    env_id = f"process_control:{entry.name}"
    if scenario is not None:
        env_id = f"{env_id}:{scenario}"

    return ProcessControlEnv(
        reset_fn=reset_fn,
        step_fn=step_fn,
        obs_dim=get_observation_dim(reset_fn),
        action_dim=action_dim,
        env_id=env_id,
        scalar_action=entry.scalar_action,
        action_low=action_low,
        action_high=action_high,
    )
