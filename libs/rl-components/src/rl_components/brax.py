"""Brax adapter for the canonical single-environment protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.structs import chex_struct


class _BraxState(Protocol):
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Mapping[str, object]


class _BraxEnv[StateT](Protocol):
    action_size: int
    observation_size: int
    sys: object

    def reset(self, key: chex.PRNGKey) -> StateT: ...

    def step(self, state: StateT, action: jax.Array) -> StateT: ...


@chex_struct(frozen=True)
class BraxConfig:
    env_name: str
    backend: str | None = None
    episode_length: int = 1000
    auto_reset: bool = True


def _as_brax_env(env: object) -> _BraxEnv[_BraxState]:
    return cast(_BraxEnv[_BraxState], env)


def _make_brax_env(config: BraxConfig) -> _BraxEnv[_BraxState]:
    from brax import envs as brax_envs

    if config.backend is None:
        env = brax_envs.create(
            config.env_name,
            episode_length=config.episode_length,
            auto_reset=config.auto_reset,
        )
    else:
        env = brax_envs.create(
            config.env_name,
            backend=config.backend,
            episode_length=config.episode_length,
            auto_reset=config.auto_reset,
        )
    return _as_brax_env(env)


def _info_value_as_array(value: object) -> jax.Array | None:
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, (bool, int, float)):
        return jnp.asarray(value)
    return None


def _coerce_info(info: Mapping[str, object]) -> dict[str, jax.Array]:
    converted: dict[str, jax.Array] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            raise TypeError(f"expected step info keys to be str, got {type(key)!r}")
        array_value = _info_value_as_array(value)
        if array_value is not None:
            converted[key] = array_value
    return converted


def _action_bounds(env: _BraxEnv[_BraxState]) -> tuple[jax.Array | None, jax.Array | None]:
    actuator = getattr(env.sys, "actuator", None)
    ctrl_range = getattr(actuator, "ctrl_range", None)
    if ctrl_range is None:
        return None, None

    ctrl_range_array = jnp.asarray(ctrl_range, dtype=jnp.float32)
    expected_shape = (int(env.action_size), 2)
    if ctrl_range_array.shape != expected_shape:
        raise ValueError(
            f"expected Brax actuator ctrl_range shape {expected_shape}, got {ctrl_range_array.shape}"
        )
    return ctrl_range_array[:, 0], ctrl_range_array[:, 1]


class BraxAdapter:
    _env: _BraxEnv[_BraxState]

    def __init__(self, config: BraxConfig) -> None:
        self.config = config
        self._env = _make_brax_env(config)

    def spec(self, params: None = None) -> EnvSpec:
        del params
        action_low, action_high = _action_bounds(self._env)
        return EnvSpec(
            id=f"brax:{self.config.env_name}",
            observation_shape=(int(self._env.observation_size),),
            action_shape=(int(self._env.action_size),),
            observation_dtype=jnp.float32,
            action_dtype=jnp.float32,
            action_low=action_low,
            action_high=action_high,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, _BraxState]:
        del params
        state = self._env.reset(key)
        return EnvReset(observation=state.obs, state=state)

    def step(self, key: chex.PRNGKey, state: _BraxState, action: jax.Array, params: None = None) -> EnvStep[jax.Array, _BraxState]:
        del key, params
        next_state = self._env.step(state, action)
        info = _coerce_info(next_state.info)
        truncated = info.get("truncated", info.get("truncation", jnp.asarray(False)))
        return EnvStep(
            observation=next_state.obs,
            state=next_state,
            reward=jnp.asarray(next_state.reward),
            terminated=jnp.asarray(next_state.done),
            truncated=jnp.asarray(truncated),
            info=info,
        )


def make_brax_adapter(config: BraxConfig) -> BraxAdapter:
    return BraxAdapter(config)