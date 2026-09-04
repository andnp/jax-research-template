"""Adapts a raw Gymnax environment to the canonical single-environment protocol.

:class:`GymnaxEnv` presents a raw Gymnax environment as an
:class:`~rl_components.env_protocol.EnvProtocol` for the shared training loop.
"""

from __future__ import annotations

from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import BoundaryResetProtocol, EnvReset, EnvSpec, EnvStep


class _GymnaxState(Protocol):
    time: jax.Array


class _GymnaxParams(Protocol):
    max_steps_in_episode: int

    def replace(self, **changes: object) -> "_GymnaxParams": ...


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

        The bounds are forced to evaluate rather than stage. ``spec`` is called from
        inside :func:`rl_components.loop.run`, which the caller jits, so these ``jnp``
        calls would otherwise be traced and the agent reading the spec could not compare
        the bounds it was handed. A bound that genuinely derives from traced ``params``
        stays traced, which the agent-side check treats as unverifiable.
    """
    if not (hasattr(action_space, "low") and hasattr(action_space, "high")):
        return None, None
    bounded = cast(_GymnaxBoundedActionSpace, action_space)
    with jax.ensure_compile_time_eval():
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
    ``EnvParams.max_steps_in_episode``.

    The collision case -- an environment reaching a genuinely terminal state on exactly
    the step the limit fires -- is resolved rather than conceded. The adapter hands Gymnax
    a limit one step higher than the real one, so Gymnax's own cutoff cannot contribute to
    ``done`` on the final step, and ``done`` there is purely the environment's terminal
    signal. The truncation comes from this adapter's own ``time >= limit`` instead. Without
    the offset a real termination landing on the limit step reads as a truncation and keeps
    its bootstrap under ``truncation_policy="bootstrap"``, biasing the target upward on
    exactly the transitions that should have no bootstrap at all.

    Raising the limit shifts Gymnax's ``info["discount"]`` at the boundary, which is inert
    here: nothing in ``rl_agents`` or :mod:`rl_components.loop` reads it, and the loop
    derives its own coefficient in :func:`rl_components.timestep.bootstrap_terms`.

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

    def _deferred_limit(self, resolved: ParamsT) -> ParamsT:
        """Return ``resolved`` with Gymnax's own cutoff moved one step out of reach."""
        return cast(ParamsT, resolved.replace(max_steps_in_episode=resolved.max_steps_in_episode + 1))

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> EnvReset[jax.Array, StateT]:
        resolved = self._resolve(params)
        observation, state = self._env.reset_env(key, self._deferred_limit(resolved))
        return EnvReset(observation=observation, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: ParamsT | None = None,
    ) -> EnvStep[jax.Array, StateT]:
        resolved = self._resolve(params)
        limit = resolved.max_steps_in_episode
        observation, next_state, reward, done, info = self._env.step_env(
            key, state, action, self._deferred_limit(resolved)
        )
        done = jnp.asarray(done, dtype=jnp.bool_)
        # ``done`` is a pure terminal signal because ``_deferred_limit`` put Gymnax's own
        # cutoff out of reach, so termination dominates the limit rather than yielding to
        # it. Masking ``done`` by ``~truncated`` here would reintroduce the collision the
        # deferred limit exists to resolve.
        reached_limit = jnp.asarray(next_state.time >= limit, dtype=jnp.bool_)
        return EnvStep(
            observation=observation,
            state=next_state,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            terminated=done,
            truncated=reached_limit & ~done,
            info={key_name: jnp.asarray(value) for key_name, value in info.items()},
        )


class _BoundaryResetGymnaxEnv[StateT: _GymnaxState, ParamsT: _GymnaxParams](GymnaxEnv[StateT, ParamsT]):
    def reset_after_episode(
        self,
        key: chex.PRNGKey,
        previous_state: StateT,
        params: ParamsT | None = None,
    ) -> EnvReset[jax.Array, StateT]:
        resolved = self._resolve(params)
        if not isinstance(self._env, BoundaryResetProtocol):
            raise TypeError("boundary-capable Gymnax adapter requires a boundary-reset environment")
        return self._env.reset_after_episode(key, previous_state, self._deferred_limit(resolved))


def make_gymnax_env[StateT: _GymnaxState, ParamsT: _GymnaxParams](env: object) -> GymnaxEnv[StateT, ParamsT]:
    """Wrap a raw Gymnax environment as an :class:`EnvProtocol` environment."""
    if isinstance(env, BoundaryResetProtocol):
        return _BoundaryResetGymnaxEnv(env)
    return GymnaxEnv(env)
