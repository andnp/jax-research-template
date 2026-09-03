"""Gymnax adapters, in both directions.

:class:`GymnaxEnv` is the driven adapter: it presents a raw Gymnax environment as an
:class:`~rl_components.env_protocol.EnvProtocol` for the shared training loop.

:class:`GymnaxCompatibilityBridge` is the opposite, and is legacy: it presents an
``EnvProtocol`` environment through the Gymnax tuple surface, for the agents that
still own private training loops. It dies with the last of them.
"""

from __future__ import annotations

from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class GymnaxSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


@chex_struct(frozen=True)
class GymnaxDiscreteSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    n: int
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


def _observation_space_from_spec(spec: EnvSpec) -> GymnaxSpace:
    return GymnaxSpace(
        shape=tuple(spec.observation_shape),
        dtype=jnp.dtype(spec.observation_dtype),
    )


def _action_space_from_spec(spec: EnvSpec) -> GymnaxSpace | GymnaxDiscreteSpace:
    if spec.num_actions is not None:
        return GymnaxDiscreteSpace(
            shape=tuple(spec.action_shape),
            dtype=jnp.dtype(spec.action_dtype),
            n=int(spec.num_actions),
        )
    return GymnaxSpace(
        shape=tuple(spec.action_shape),
        dtype=jnp.dtype(spec.action_dtype),
        action_low=spec.action_low,
        action_high=spec.action_high,
    )


class GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    def __init__(self, env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT]) -> None:
        self._env = env

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)

    def observation_space(self, params: object | None = None) -> GymnaxSpace:
        return _observation_space_from_spec(self._env.spec(cast(ParamsT | None, params)))

    def action_space(self, params: object | None = None) -> GymnaxSpace | GymnaxDiscreteSpace:
        return _action_space_from_spec(self._env.spec(cast(ParamsT | None, params)))

    def reset(self, key: chex.PRNGKey, params: object | None = None) -> tuple[ObservationT, StateT]:
        reset = self._env.reset(key, cast(ParamsT | None, params))
        return reset.observation, reset.state

    def step(
        self,
        key: chex.PRNGKey,
        state: object,
        action: ActionT,
        params: object | None = None,
    ) -> tuple[ObservationT, object, jax.Array, jax.Array, dict[str, jax.Array]]:
        transition = self._env.step(key, cast(StateT, state), action, cast(ParamsT | None, params))
        done = jnp.logical_or(transition.terminated, transition.truncated)
        info = dict(transition.info)
        info.setdefault("terminated", jnp.asarray(transition.terminated))
        info.setdefault("truncated", jnp.asarray(transition.truncated))
        return transition.observation, transition.state, transition.reward, done, info


def make_gymnax_compat_env[ObservationT, StateT, ActionT, ParamsT](
    env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT],
) -> GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    return GymnaxCompatibilityBridge(env)

class _GymnaxState(Protocol):
    time: jax.Array


class _GymnaxParams(Protocol):
    max_steps_in_episode: int


class _GymnaxObservationSpace(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class _GymnaxActionSpace(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class _GymnaxBoundedActionSpace(Protocol):
    low: jax.Array | float
    high: jax.Array | float


def _continuous_bounds(
    action_space: _GymnaxActionSpace,
    action_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array] | tuple[None, None]:
    """Read a continuous Gymnax action space's bounds, broadcast to the action shape.

    Args:
        action_space: The Gymnax action space.
        action_shape: The shape a bound must take to satisfy :class:`EnvSpec`.

    Returns:
        The lower and upper bounds, or ``(None, None)`` when the space declares neither;
        :class:`EnvSpec` accepts both bounds or neither, never one.
    """
    if not (hasattr(action_space, "low") and hasattr(action_space, "high")):
        return None, None
    bounded = cast(_GymnaxBoundedActionSpace, action_space)
    return (
        jnp.broadcast_to(jnp.asarray(bounded.low, dtype=jnp.float32), action_shape),
        jnp.broadcast_to(jnp.asarray(bounded.high, dtype=jnp.float32), action_shape),
    )


class _GymnaxEnv[StateT, ParamsT](Protocol):
    name: str

    @property
    def default_params(self) -> ParamsT: ...

    def observation_space(self, params: ParamsT) -> _GymnaxObservationSpace: ...

    def action_space(self, params: ParamsT) -> _GymnaxActionSpace: ...

    def reset_env(self, key: chex.PRNGKey, params: ParamsT) -> tuple[jax.Array, StateT]: ...

    def step_env(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: ParamsT,
    ) -> tuple[jax.Array, StateT, jax.Array, jax.Array, dict[str, jax.Array]]: ...


class GymnaxEnv[StateT: _GymnaxState, ParamsT: _GymnaxParams]:
    """Adapts a raw Gymnax environment to :class:`EnvProtocol`.

    Two things separate this from handing the Gymnax environment to a training loop
    directly.

    **It does not auto-reset.** Gymnax's ``Environment.step`` selects between the stepped
    state and a freshly reset one, so the observation it returns on a ``done`` step is the
    post-reset observation and the true final state is unreachable. This adapter calls
    ``step_env`` and ``reset_env`` -- the un-fused halves -- so the boundary belongs to the
    loop and ``EnvStep.observation`` is always the state the transition actually reached.

    **It splits ``done``.** Gymnax fuses the time limit into ``is_terminal``, so the
    single flag cannot say whether the bootstrap should survive. The split uses the two
    fields every Gymnax environment carries: ``EnvState.time`` and
    ``EnvParams.max_steps_in_episode``. The limitation is that an environment reaching a
    genuinely terminal state on exactly the step the limit fires is reported as a
    truncation; Gymnax exposes no way to tell the two apart, and no environment in the
    registry distinguishes them either.

    Both action-space kinds are specified. A space exposing an integer ``n`` is discrete;
    any other space is continuous and is described by its own ``shape`` plus, when it
    carries them, its ``low``/``high`` bounds broadcast to that shape.
    """

    _env: _GymnaxEnv[StateT, ParamsT]

    def __init__(self, env: object) -> None:
        """Wrap a Gymnax environment.

        Args:
            env: A Gymnax ``Environment``, unwrapped. ``LogWrapper`` is neither needed nor
                wanted: it wraps the fused ``step``, and the loop already reports episode
                returns and lengths under its own metric prefix.
        """
        self._env = cast(_GymnaxEnv[StateT, ParamsT], env)

    def _resolve(self, params: ParamsT | None) -> ParamsT:
        return self._env.default_params if params is None else params

    def spec(self, params: ParamsT | None = None) -> EnvSpec:
        resolved = self._resolve(params)
        observation_space = self._env.observation_space(resolved)
        action_space = self._env.action_space(resolved)
        num_actions = getattr(action_space, "n", None)
        if isinstance(num_actions, int):
            return EnvSpec(
                id=f"gymnax:{self._env.name}",
                observation_shape=tuple(observation_space.shape),
                action_shape=(),
                observation_dtype=jnp.dtype(observation_space.dtype),
                action_dtype=jnp.dtype(jnp.int32),
                num_actions=num_actions,
            )
        action_shape = tuple(action_space.shape)
        action_low, action_high = _continuous_bounds(action_space, action_shape)
        return EnvSpec(
            id=f"gymnax:{self._env.name}",
            observation_shape=tuple(observation_space.shape),
            action_shape=action_shape,
            observation_dtype=jnp.dtype(observation_space.dtype),
            action_dtype=jnp.dtype(jnp.float32),
            action_low=action_low,
            action_high=action_high,
        )

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> EnvReset[jax.Array, StateT]:
        observation, state = self._env.reset_env(key, self._resolve(params))
        return EnvReset(observation=observation, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: ParamsT | None = None,
    ) -> EnvStep[jax.Array, StateT]:
        resolved = self._resolve(params)
        observation, next_state, reward, done, info = self._env.step_env(key, state, action, resolved)
        done = jnp.asarray(done, dtype=jnp.bool_)
        truncated = jnp.asarray(next_state.time >= resolved.max_steps_in_episode, dtype=jnp.bool_)
        return EnvStep(
            observation=observation,
            state=next_state,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            terminated=done & ~truncated,
            truncated=truncated,
            info={key_name: jnp.asarray(value) for key_name, value in info.items()},
        )


def make_gymnax_env[StateT: _GymnaxState, ParamsT: _GymnaxParams](env: object) -> GymnaxEnv[StateT, ParamsT]:
    """Wrap a raw Gymnax environment as an :class:`EnvProtocol` environment."""
    return GymnaxEnv(env)
