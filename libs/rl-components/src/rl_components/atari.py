"""JAXAtari adapter for the canonical single-environment protocol."""

from __future__ import annotations

from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp
from jaxatari.core import make as make_jaxatari_env
from jaxatari.wrappers import AtariWrapper, LogWrapper, PixelObsWrapper

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.structs import chex_struct


class _Space(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class _DiscreteSpace(_Space, Protocol):
    n: int


class _ImageSpace(Protocol):
    shape: tuple[int, int, int]


class _WrappedAtariEnv[StateT](Protocol):
    def observation_space(self) -> _Space: ...

    def action_space(self) -> _DiscreteSpace: ...

    def image_space(self) -> _ImageSpace: ...

    def reset(self, key: chex.PRNGKey) -> tuple[jax.Array, StateT]: ...

    def step(self, state: StateT, action: jax.Array) -> tuple[jax.Array, StateT, jax.Array, jax.Array, dict[str, object]]: ...


@chex_struct(frozen=True)
class JAXAtariConfig:
    game: str
    frame_stack: int = 4
    frame_skip: int = 4
    grayscale: bool = True
    max_pooling: bool = True
    life_loss_terminal: bool = True
    resize_shape: tuple[int, int] = (84, 84)
    log_returns: bool = True


def _resize_is_required[StateT](env: _WrappedAtariEnv[StateT], resize_shape: tuple[int, int]) -> bool:
    image_height, image_width, _ = env.image_space().shape
    return (image_height, image_width) != resize_shape


def _as_wrapped_atari_env(env: object) -> _WrappedAtariEnv[object]:
    return cast(_WrappedAtariEnv[object], env)


def _coerce_info(info: object) -> dict[str, jax.Array]:
    if not isinstance(info, dict):
        raise TypeError(f"expected step info to be dict[str, jax.Array], got {type(info)!r}")
    converted: dict[str, jax.Array] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            raise TypeError(f"expected step info keys to be str, got {type(key)!r}")
        converted[key] = jnp.asarray(value)
    return converted


class JAXAtariAdapter:
    _env: _WrappedAtariEnv[object]

    def __init__(self, config: JAXAtariConfig) -> None:
        self.config = config
        base_env = make_jaxatari_env(config.game)
        atari_env = _as_wrapped_atari_env(
            AtariWrapper(
                base_env,
                frame_stack_size=config.frame_stack,
                frame_skip=config.frame_skip,
                episodic_life=config.life_loss_terminal,
                max_pooling=config.max_pooling,
            )
        )
        pixel_env = PixelObsWrapper(
            atari_env,
            do_pixel_resize=_resize_is_required(atari_env, config.resize_shape),
            pixel_resize_shape=config.resize_shape,
            grayscale=config.grayscale,
        )
        wrapped_env = _as_wrapped_atari_env(LogWrapper(pixel_env) if config.log_returns else pixel_env)
        self._env = wrapped_env

    def spec(self, params: None = None) -> EnvSpec:
        del params
        observation_space = self._env.observation_space()
        action_space = self._env.action_space()
        return EnvSpec(
            id=f"jaxatari:{self.config.game}",
            observation_shape=tuple(observation_space.shape),
            action_shape=tuple(action_space.shape),
            observation_dtype=jnp.dtype(observation_space.dtype),
            action_dtype=jnp.dtype(action_space.dtype),
            num_actions=int(action_space.n),
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, object]:
        del params
        observation, state = self._env.reset(key)
        return EnvReset(observation=observation, state=state)

    def step(self, key: chex.PRNGKey, state: object, action: jax.Array, params: None = None) -> EnvStep[jax.Array, object]:
        del key, params
        observation, next_state, reward, done, info = self._env.step(state, action)
        info_arrays = _coerce_info(info)
        terminated = info_arrays.get("terminated", jnp.asarray(done))
        truncated = info_arrays.get("truncated", jnp.asarray(False))
        return EnvStep(
            observation=observation,
            state=next_state,
            reward=jnp.asarray(reward),
            terminated=terminated,
            truncated=truncated,
            info=info_arrays,
        )


def make_atari_adapter(
    config: JAXAtariConfig,
) -> JAXAtariAdapter:
    return JAXAtariAdapter(config)