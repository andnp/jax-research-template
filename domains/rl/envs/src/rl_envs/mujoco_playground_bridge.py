"""MuJoCo Playground bridge for the canonical single-environment protocol.

Wraps ``mujoco_playground.MjxEnv`` environments into the ``EnvProtocol``
interface, making dm_control_suite and locomotion environments available to
:func:`rl_components.loop.run`.

Usage::

    from mujoco_playground import dm_control_suite
    from rl_components.loop import run
    from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

    env = MujocoPlaygroundAdapter(dm_control_suite.load("CheetahRun"))
    final_state, metrics = run(agent, env, key, steps=100_000, gamma=0.99)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep


class _PlaygroundModel(Protocol):
    @property
    def actuator_ctrlrange(self) -> ArrayLike | None: ...


class _PlaygroundState(Protocol):
    @property
    def obs(self) -> jax.Array | Mapping[str, jax.Array]: ...

    @property
    def reward(self) -> jax.Array: ...

    @property
    def done(self) -> jax.Array: ...

    @property
    def info(self) -> Mapping[str, object]: ...


class _PlaygroundEnv[StateT: _PlaygroundState](Protocol):
    @property
    def action_size(self) -> int: ...

    @property
    def observation_size(self) -> int | Mapping[str, int | tuple[int, ...]]: ...

    @property
    def mj_model(self) -> _PlaygroundModel: ...

    def reset(self, rng: jax.Array) -> StateT: ...

    def step(self, state: StateT, action: jax.Array) -> StateT: ...


def _flat_observation(observation: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
    if isinstance(observation, Mapping):
        raise TypeError("MujocoPlaygroundAdapter requires flat array observations")
    return observation


def _action_bounds(mj_model: _PlaygroundModel) -> tuple[jax.Array | None, jax.Array | None]:
    """Read the actuator control range as trace-time constants.

    ``spec`` is called from inside :func:`rl_components.loop.run`, which the caller
    jits, so every ``jnp`` call here would otherwise be staged: the unbounded test
    below would raise ``TracerBoolConversionError`` and the bounds would reach
    :class:`EnvSpec` as tracers no agent can read.

    Args:
        mj_model: The MuJoCo model carrying ``actuator_ctrlrange``.

    Returns:
        The lower and upper bounds, or ``(None, None)`` when the model declares no
        usable range.
    """
    ctrl_range = mj_model.actuator_ctrlrange
    if ctrl_range is None:
        return None, None
    with jax.ensure_compile_time_eval():
        ctrl_range_array = jnp.asarray(ctrl_range, dtype=jnp.float32)
        if ctrl_range_array.ndim != 2 or ctrl_range_array.shape[1] != 2:
            return None, None
        low, high = ctrl_range_array[:, 0], ctrl_range_array[:, 1]
        if jnp.all(low == -jnp.inf) and jnp.all(high == jnp.inf):
            return None, None
        return low, high


class MujocoPlaygroundAdapter[StateT: _PlaygroundState]:
    """Adapts a ``MjxEnv`` to the canonical ``EnvProtocol`` interface."""

    def __init__(self, env: _PlaygroundEnv[StateT]) -> None:
        self._env = env

    def spec(self, params: None = None) -> EnvSpec:
        del params
        obs_size = self._env.observation_size
        if not isinstance(obs_size, int):
            raise TypeError(
                f"MujocoPlaygroundAdapter requires flat (int) observation_size, "
                f"got {type(obs_size).__name__}. Dict-observation environments are not supported."
            )
        action_low, action_high = _action_bounds(self._env.mj_model)
        return EnvSpec(
            id=f"playground:{type(self._env).__name__}",
            observation_shape=(obs_size,),
            action_shape=(self._env.action_size,),
            observation_dtype=jnp.float32,
            action_dtype=jnp.float32,
            action_low=action_low,
            action_high=action_high,
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, StateT]:
        del params
        state = self._env.reset(key)
        return EnvReset(observation=_flat_observation(state.obs), state=state)

    def step(
        self,
        key: jax.Array,
        state: StateT,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, StateT]:
        del key, params
        next_state = self._env.step(state, action)
        info: dict[str, jax.Array] = {}
        for k, v in next_state.info.items():
            if isinstance(v, jax.Array):
                info[k] = v
        truncated = jnp.asarray(info.get("truncation", False), dtype=bool)
        terminated = jnp.asarray(next_state.done, dtype=bool) & ~truncated
        return EnvStep(
            observation=_flat_observation(next_state.obs),
            state=next_state,
            reward=next_state.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
