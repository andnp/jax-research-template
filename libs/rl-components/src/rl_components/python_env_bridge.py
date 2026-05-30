from __future__ import annotations

import functools
from typing import Callable

import chex
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


class PythonEnvBridge:
    """Wraps a gymnasium ALE environment as a JAX-compatible EnvProtocol.

    StateT = jnp.uint8[state_size] — the serialized ALE emulator state bytes.
    Calls to reset() and step() cross the JIT boundary via jax.experimental.io_callback.
    The env's step is side-effectful; use ordered=True to guarantee sequential ordering
    when this is called inside jax.lax.scan.

    The factory callable is invoked once at construction time to produce the gym env.
    """

    _env: gymnasium.Env
    _obs_shape: tuple[int, ...]
    _obs_dtype: np.dtype
    _num_actions: int
    _state_size: int
    _env_id: str

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
        # Determine serialized state size
        state_bytes = self._env.unwrapped.ale.getState()  # type: ignore[attr-defined]
        self._state_size = len(state_bytes)
        # Determine num actions
        self._num_actions = int(self._env.action_space.n)  # type: ignore[union-attr]
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
            state_bytes = self._env.unwrapped.ale.getState()  # type: ignore[attr-defined]
            obs = np.asarray(raw_obs, dtype=self._obs_dtype)
            state = np.frombuffer(state_bytes, dtype=np.uint8).copy()
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

        def _do_step(state_arr: np.ndarray, action_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            self._env.unwrapped.ale.setState(state_arr.tobytes())  # type: ignore[attr-defined]
            raw_obs, reward, terminated, truncated, _ = self._env.step(int(action_arr))
            if terminated or truncated:
                raw_obs, _ = self._env.reset()
            obs = np.asarray(raw_obs, dtype=self._obs_dtype)
            next_state = np.frombuffer(self._env.unwrapped.ale.getState(), dtype=np.uint8).copy()  # type: ignore[attr-defined]
            return (
                obs,
                np.array(reward, dtype=np.float32),
                np.array(terminated, dtype=np.bool_),
                np.array(truncated, dtype=np.bool_),
                next_state,
            )

        obs, reward, terminated, truncated, next_state = jax.experimental.io_callback(
            _do_step,
            (
                jax.ShapeDtypeStruct(self._obs_shape, self._obs_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((self._state_size,), jnp.uint8),
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
            info={},
        )
