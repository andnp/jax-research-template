"""MuJoCo Playground bridge for the canonical single-environment protocol.

Wraps ``mujoco_playground.MjxEnv`` environments into the ``EnvProtocol``
interface, making dm_control_suite and locomotion environments available to
any ``make_train`` function that accepts ``GymEnv``.

Usage::

    from mujoco_playground import dm_control_suite
    from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter
    from rl_components.gymnax_bridge import make_gymnax_compat_env

    env = make_gymnax_compat_env(MujocoPlaygroundAdapter(dm_control_suite.load("CheetahRun")))
    train = make_train(config, env)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep


class _PlaygroundState(Protocol):
    obs: jax.Array | Mapping[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    info: Mapping[str, Any]


class _PlaygroundEnv(Protocol):
    @property
    def action_size(self) -> int: ...

    @property
    def observation_size(self) -> int | Mapping[str, Any]: ...

    @property
    def mj_model(self) -> Any: ...

    def reset(self, rng: jax.Array) -> Any: ...

    def step(self, state: Any, action: jax.Array) -> Any: ...


def _action_bounds(mj_model: Any) -> tuple[jax.Array | None, jax.Array | None]:
    ctrl_range = getattr(mj_model, "actuator_ctrlrange", None)
    if ctrl_range is None:
        return None, None
    ctrl_range_array = jnp.asarray(ctrl_range, dtype=jnp.float32)
    if ctrl_range_array.ndim != 2 or ctrl_range_array.shape[1] != 2:
        return None, None
    low, high = ctrl_range_array[:, 0], ctrl_range_array[:, 1]
    if jnp.all(low == -jnp.inf) and jnp.all(high == jnp.inf):
        return None, None
    return low, high


class MujocoPlaygroundAdapter:
    """Adapts a ``MjxEnv`` to the canonical ``EnvProtocol`` interface."""

    def __init__(self, env: _PlaygroundEnv) -> None:
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

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, _PlaygroundState]:
        del params
        state = self._env.reset(key)
        return EnvReset(observation=state.obs, state=state)

    def step(
        self,
        key: jax.Array,
        state: _PlaygroundState,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, _PlaygroundState]:
        del key, params
        next_state = self._env.step(state, action)
        info: dict[str, jax.Array] = {}
        for k, v in next_state.info.items():
            if isinstance(v, jax.Array):
                info[k] = v
        return EnvStep(
            observation=next_state.obs,
            state=next_state,
            reward=next_state.reward,
            terminated=next_state.done,
            truncated=jnp.asarray(False),
            info=info,
        )

