"""Gymnax adapter: wraps native gymnax environments into EnvProtocol with correct trunc/term signals.

Gymnax treats time-limit truncations as terminations (old OpenAI gym convention), which
biases return estimates. This adapter intercepts truncation at ``max_steps`` by telling
gymnax its limit is ``T + 1``, then checking ``state.time >= T`` itself and emitting
proper separated ``terminated`` / ``truncated`` signals.
"""

from __future__ import annotations

from typing import Any, Protocol

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep


class _GymnaxEnv(Protocol):
    name: str

    @property
    def default_params(self) -> Any: ...

    def observation_space(self, params: Any) -> Any: ...

    def action_space(self, params: Any) -> Any: ...

    def reset_env(self, key: jax.Array, params: Any) -> tuple[jax.Array, Any]: ...

    def step_env(
        self,
        key: jax.Array,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]: ...


class GymnaxEnvAdapter:
    """Wraps a native gymnax ``Environment`` into ``EnvProtocol``.

    Sets gymnax's ``max_steps_in_episode = max_steps + 1`` so its own time-limit
    check cannot fire at step ``T``.  Calls ``step_env`` / ``reset_env`` directly
    (bypassing gymnax's JIT-wrapped ``step``), checks truncation itself, and
    handles auto-reset on any episode end.
    """

    def __init__(
        self,
        env: _GymnaxEnv,
        max_steps: int | None = None,
        default_params: Any = None,
    ) -> None:
        self._env = env
        self._default_params = default_params if default_params is not None else env.default_params
        self._max_steps: int = (
            max_steps if max_steps is not None else int(self._default_params.max_steps_in_episode)
        )

    def _gymnax_params(self, params: Any) -> Any:
        p = params if params is not None else self._default_params
        return p.replace(max_steps_in_episode=self._max_steps + 1)

    def spec(self, params: Any = None) -> EnvSpec:
        gp = self._gymnax_params(params)
        obs_space = self._env.observation_space(gp)
        act_space = self._env.action_space(gp)
        num_actions: int | None = getattr(act_space, "n", None)
        return EnvSpec(
            id=self._env.name,
            observation_shape=tuple(obs_space.shape),
            action_shape=() if num_actions is not None else tuple(act_space.shape),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32 if num_actions is not None else jnp.float32,
            num_actions=num_actions,
        )

    def reset(self, key: chex.PRNGKey, params: Any = None) -> EnvReset[jax.Array, Any]:
        gp = self._gymnax_params(params)
        obs, state = self._env.reset_env(key, gp)
        return EnvReset(observation=obs, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        action: Any,
        params: Any = None,
    ) -> EnvStep[jax.Array, Any]:
        gp = self._gymnax_params(params)
        key_step, key_reset = jax.random.split(key)

        obs_st, state_st, reward, done, info = self._env.step_env(key_step, state, action, gp)
        obs_re, state_re = self._env.reset_env(key_reset, gp)

        # state_st.time is already incremented by step_env.
        # Since gymnax max_steps = T+1, done only fires for true env termination here.
        truncated = jnp.asarray(state_st.time >= self._max_steps)
        terminated = jnp.asarray(done)
        episode_done = jnp.logical_or(terminated, truncated)

        final_state = jax.tree.map(
            lambda x, y: jax.lax.select(episode_done, x, y), state_re, state_st
        )
        final_obs = jax.lax.select(episode_done, obs_re, obs_st)

        return EnvStep(
            observation=final_obs,
            state=final_state,
            reward=jnp.asarray(reward),
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )


def make_gymnax_env(env_name: str, max_steps: int | None = None) -> GymnaxEnvAdapter:
    """Create a ``GymnaxEnvAdapter`` from a registered gymnax environment name."""
    import gymnax  # noqa: PLC0415

    env, default_params = gymnax.make(env_name)
    return GymnaxEnvAdapter(env, max_steps=max_steps, default_params=default_params)


