"""Continuous canonical action normalization wrapper."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError

from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


def _validated_continuous_spec(spec: EnvSpec) -> EnvSpec:
    if spec.num_actions is not None:
        raise ValueError("action normalization requires a continuous environment spec")
    if spec.action_low is None or spec.action_high is None:
        raise ValueError("action normalization requires continuous action_low and action_high bounds")

    action_dtype = jnp.dtype(spec.action_dtype)
    if not jnp.issubdtype(action_dtype, jnp.floating):
        raise TypeError(f"action normalization requires a floating action_dtype, got {action_dtype}")
    if spec.action_low.shape != spec.action_shape:
        raise ValueError(f"action_low shape must match action_shape {spec.action_shape}, got {spec.action_low.shape}")
    if spec.action_high.shape != spec.action_shape:
        raise ValueError(f"action_high shape must match action_shape {spec.action_shape}, got {spec.action_high.shape}")

    low_dtype = jnp.dtype(spec.action_low.dtype)
    high_dtype = jnp.dtype(spec.action_high.dtype)
    if low_dtype != action_dtype:
        raise TypeError(f"action_low dtype must match action_dtype {action_dtype}, got {low_dtype}")
    if high_dtype != action_dtype:
        raise TypeError(f"action_high dtype must match action_dtype {action_dtype}, got {high_dtype}")
    try:
        bounds_are_ordered = bool(jnp.all(spec.action_low <= spec.action_high))
    except TracerBoolConversionError:
        return spec
    if not bounds_are_ordered:
        raise ValueError("continuous action bounds must satisfy action_low <= action_high elementwise")
    return spec


def _normalized_spec(spec: EnvSpec) -> EnvSpec:
    validated_spec = _validated_continuous_spec(spec)
    action_dtype = jnp.dtype(validated_spec.action_dtype)
    return EnvSpec(
        id=validated_spec.id,
        observation_shape=tuple(validated_spec.observation_shape),
        action_shape=tuple(validated_spec.action_shape),
        observation_dtype=jnp.dtype(validated_spec.observation_dtype),
        action_dtype=action_dtype,
        action_low=jnp.full(validated_spec.action_shape, -1.0, dtype=action_dtype),
        action_high=jnp.full(validated_spec.action_shape, 1.0, dtype=action_dtype),
    )


def _denormalize_action(action: jax.Array, spec: EnvSpec) -> jax.Array:
    validated_spec = _validated_continuous_spec(spec)
    if validated_spec.action_low is None or validated_spec.action_high is None:
        raise ValueError("action normalization requires continuous action_low and action_high bounds")

    action_dtype = jnp.dtype(validated_spec.action_dtype)
    normalized_action = jnp.asarray(action, dtype=action_dtype)
    if normalized_action.shape != validated_spec.action_shape:
        raise ValueError(
            f"normalized action shape must match action_shape {validated_spec.action_shape}, got {normalized_action.shape}"
        )

    action_low = validated_spec.action_low
    action_high = validated_spec.action_high
    action_center = (action_low + action_high) / jnp.asarray(2.0, dtype=action_dtype)
    action_half_range = (action_high - action_low) / jnp.asarray(2.0, dtype=action_dtype)
    return action_center + normalized_action * action_half_range


class ActionNormalizationWrapper[ObservationT, StateT, ParamsT]:
    def __init__(self, env: EnvProtocol[ObservationT, StateT, jax.Array, ParamsT]) -> None:
        self._env = env

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)

    def spec(self, params: ParamsT | None = None) -> EnvSpec:
        return _normalized_spec(self._env.spec(params))

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> EnvReset[ObservationT, StateT]:
        return self._env.reset(key, params)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: ParamsT | None = None,
    ) -> EnvStep[ObservationT, StateT]:
        native_action = _denormalize_action(action, self._env.spec(params))
        return self._env.step(key, state, native_action, params)


def make_action_normalization_wrapper[ObservationT, StateT, ParamsT](
    env: EnvProtocol[ObservationT, StateT, jax.Array, ParamsT],
) -> ActionNormalizationWrapper[ObservationT, StateT, ParamsT]:
    return ActionNormalizationWrapper(env)