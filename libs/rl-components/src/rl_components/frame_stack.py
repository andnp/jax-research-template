"""Composable frame-stacking wrapper for any EnvProtocol.

``step`` always rolls. It does not refill the stack at a boundary, because the boundary
belongs to :func:`rl_components.loop.run`: the loop calls ``reset``, which stacks the fresh
observation ``n_frames`` times. A wrapper that refilled on its own would disagree with the
inner environment about where the episode ended -- it would hand the agent a stack of
post-terminal frames while the emulator was still on the terminal one.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class FrameStackState[StateT]:
    inner_state: StateT
    frames: jax.Array


class FrameStackWrapper[ObsT: jax.Array, StateT, ActionT, ParamsT]:
    def __init__(
        self,
        env: EnvProtocol[ObsT, StateT, ActionT, ParamsT],
        n_frames: int,
    ) -> None:
        self._env = env
        self._n_frames = n_frames

    def spec(self, params: ParamsT | None = None) -> EnvSpec:
        inner_spec = self._env.spec(params)
        stacked_obs_shape = (self._n_frames, *inner_spec.observation_shape)
        return EnvSpec(
            id=inner_spec.id,
            observation_shape=stacked_obs_shape,
            action_shape=inner_spec.action_shape,
            observation_dtype=inner_spec.observation_dtype,
            action_dtype=inner_spec.action_dtype,
            num_actions=inner_spec.num_actions,
            action_low=inner_spec.action_low,
            action_high=inner_spec.action_high,
        )

    def reset(
        self, key: chex.PRNGKey, params: ParamsT | None = None
    ) -> EnvReset[jax.Array, FrameStackState[StateT]]:
        inner_reset = self._env.reset(key, params)
        obs: jax.Array = inner_reset.observation
        frames = jnp.stack([obs] * self._n_frames, axis=0)
        state = FrameStackState(inner_state=inner_reset.state, frames=frames)
        return EnvReset(observation=frames, state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: FrameStackState[StateT],
        action: ActionT,
        params: ParamsT | None = None,
    ) -> EnvStep[jax.Array, FrameStackState[StateT]]:
        inner_step = self._env.step(key, state.inner_state, action, params)
        new_obs: jax.Array = inner_step.observation
        new_frames = jnp.roll(state.frames, shift=-1, axis=0).at[-1].set(new_obs)
        new_state = FrameStackState(inner_state=inner_step.state, frames=new_frames)
        return EnvStep(
            observation=new_frames,
            state=new_state,
            reward=inner_step.reward,
            terminated=inner_step.terminated,
            truncated=inner_step.truncated,
            info=inner_step.info,
        )
