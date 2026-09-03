from __future__ import annotations

from typing import Callable, Protocol, cast

import chex
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep

FINAL_OBSERVATION_INFO_KEY = "final_observation"
FINAL_OBSERVATION_VALID_INFO_KEY = "final_observation_valid"


class _ALEHandle(Protocol):
    """The slice of ale-py's ALE interface this bridge depends on."""

    def cloneState(self) -> object: ...

    def restoreState(self, state: object, /) -> None: ...


class _ALEUnwrapped(Protocol):
    """An unwrapped gymnasium ALE env, which exposes the raw ALE interface."""

    ale: _ALEHandle


class PythonEnvBridge:
    """Wraps a gymnasium ALE environment as a JAX-compatible EnvProtocol.

    StateT = jnp.uint8[1] — a dummy state token (actual ALE state is stored internally).
    Calls to reset() and step() cross the JIT boundary via jax.experimental.io_callback.
    The env's step is side-effectful; use ordered=True to guarantee sequential ordering
    when this is called inside jax.lax.scan.

    One bridge instance owns one mutable host environment. Calls must be serialized;
    JAX transformations do not make the bridge reentrant, safely batchable, or
    independently usable under vmap.

    The factory callable is invoked once at construction time to produce the gym env.
    """

    _env: gymnasium.Env
    _obs_shape: tuple[int, ...]
    _obs_dtype: np.dtype
    _num_actions: int
    _state_size: int
    _env_id: str
    _ale: _ALEHandle
    _ale_state: object  # opaque ALEState from cloneState()

    def __init__(
        self,
        make_env: Callable[[], gymnasium.Env],
        env_id: str = "python_env",
    ) -> None:
        self._env = make_env()
        self._env_id = env_id
        # Determine obs shape/dtype from a probe reset
        raw_obs, _ = self._env.reset(seed=0)
        obs_arr = np.asarray(raw_obs)
        self._obs_shape = obs_arr.shape
        self._obs_dtype = obs_arr.dtype
        # ale-py 0.11+ uses opaque ALEState; store internally, pass dummy token through JAX
        self._state_size = 1
        self._ale = cast(_ALEUnwrapped, self._env.unwrapped).ale
        self._ale_state = self._ale.cloneState()
        # Determine num actions
        action_space = self._env.action_space
        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise TypeError(f"{type(self).__name__} requires a Discrete action space, got {type(action_space).__name__}")
        self._num_actions = int(action_space.n)
        # Re-sync to a clean state for actual use
        self._env.reset(seed=0)

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id=self._env_id,
            observation_shape=self._obs_shape,
            action_shape=(),
            observation_dtype=jnp.dtype(self._obs_dtype),
            action_dtype=jnp.int32,
            num_actions=self._num_actions,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del params
        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)

        def _do_reset(seed_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raw_obs, _ = self._env.reset(seed=int(seed_arr))
            self._ale_state = self._ale.cloneState()
            obs = np.asarray(raw_obs, dtype=self._obs_dtype)
            state = np.zeros(1, dtype=np.uint8)  # dummy token
            return obs, state

        obs, state = jax.experimental.io_callback(
            _do_reset,
            (
                jax.ShapeDtypeStruct(self._obs_shape, self._obs_dtype),
                jax.ShapeDtypeStruct((self._state_size,), jnp.uint8),
            ),
            seed,
            ordered=True,
        )
        return EnvReset(observation=obs, state=state)

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, params

        def _do_step(
            state_arr: np.ndarray, action_arr: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            self._ale.restoreState(self._ale_state)
            raw_obs, reward, terminated, truncated, _ = self._env.step(int(action_arr))
            final_obs = np.asarray(raw_obs, dtype=self._obs_dtype)
            if terminated or truncated:
                raw_obs, _ = self._env.reset()
            obs = np.asarray(raw_obs, dtype=self._obs_dtype)
            self._ale_state = self._ale.cloneState()
            next_state = np.zeros(1, dtype=np.uint8)  # dummy token
            return (
                obs,
                np.array(reward, dtype=np.float32),
                np.array(terminated, dtype=np.bool_),
                np.array(truncated, dtype=np.bool_),
                next_state,
                final_obs,
            )

        obs, reward, terminated, truncated, next_state, final_obs = jax.experimental.io_callback(
            _do_step,
            (
                jax.ShapeDtypeStruct(self._obs_shape, self._obs_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((self._state_size,), jnp.uint8),
                jax.ShapeDtypeStruct(self._obs_shape, self._obs_dtype),
            ),
            state,
            action,
            ordered=True,
        )
        return EnvStep(
            observation=obs,
            state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                FINAL_OBSERVATION_INFO_KEY: final_obs,
                FINAL_OBSERVATION_VALID_INFO_KEY: jnp.logical_or(terminated, truncated),
            },
        )
