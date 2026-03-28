"""Medium integration tests for the JAXAtari adapter under JIT."""

from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import rl_components.atari as atari_module
from rl_components.atari import JAXAtariConfig, make_atari_adapter


@dataclass(frozen=True)
class FakeSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    n: int | None = None


class FakeWrappedEnv:
    def __init__(self, *, image_shape: tuple[int, int, int]) -> None:
        self._image_shape = image_shape

    def observation_space(self) -> FakeSpace:
        return FakeSpace(shape=(4, 84, 84, 1), dtype=jnp.uint8)

    def action_space(self) -> FakeSpace:
        return FakeSpace(shape=(), dtype=jnp.int32, n=6)

    def image_space(self) -> FakeSpace:
        return FakeSpace(shape=self._image_shape, dtype=jnp.uint8)

    def reset(self, key: chex.PRNGKey) -> tuple[jax.Array, jax.Array]:
        del key
        return jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8), jnp.array(0, dtype=jnp.int32)

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del action
        next_state = state + jnp.array(1, dtype=jnp.int32)
        observation = jnp.full((4, 84, 84, 1), next_state, dtype=jnp.uint8)
        return (
            observation,
            next_state,
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(True),
            {
                "terminated": jnp.array(True),
                "truncated": jnp.array(False),
                "returned_episode_returns": jnp.array(1.0, dtype=jnp.float32),
            },
        )


class FakeAtariBackend:
    def make(self, game: str) -> object:
        del game
        return FakeWrappedEnv(image_shape=(96, 96, 3))

    def atari_wrapper(self, env: FakeWrappedEnv, *, frame_stack_size: int, frame_skip: int, episodic_life: bool, max_pooling: bool) -> FakeWrappedEnv:
        del frame_stack_size, frame_skip, episodic_life, max_pooling
        return env

    def pixel_wrapper(
        self,
        env: FakeWrappedEnv,
        *,
        do_pixel_resize: bool,
        pixel_resize_shape: tuple[int, int],
        grayscale: bool,
    ) -> FakeWrappedEnv:
        del do_pixel_resize, pixel_resize_shape, grayscale
        return env

    def log_wrapper(self, env: FakeWrappedEnv) -> FakeWrappedEnv:
        return env


def _install_fake_backend(monkeypatch, backend: FakeAtariBackend) -> None:
    monkeypatch.setattr(atari_module, "make_jaxatari_env", backend.make)
    monkeypatch.setattr(atari_module, "AtariWrapper", backend.atari_wrapper)
    monkeypatch.setattr(atari_module, "PixelObsWrapper", backend.pixel_wrapper)
    monkeypatch.setattr(atari_module, "LogWrapper", backend.log_wrapper)


class TestJAXAtariAdapterJIT:
    def test_reset_and_step_paths_are_jittable(self, monkeypatch):
        _install_fake_backend(monkeypatch, FakeAtariBackend())
        adapter = make_atari_adapter(JAXAtariConfig(game="pong"))

        reset = jax.jit(adapter.reset)(jax.random.key(0))
        transition = jax.jit(adapter.step)(jax.random.key(1), reset.state, jnp.array(2, dtype=jnp.int32))

        assert reset.observation.shape == (4, 84, 84, 1)
        assert int(transition.state) == 1
        assert float(transition.reward) == 1.0
        assert bool(transition.terminated) is True
        assert bool(transition.truncated) is False
        assert float(transition.info["returned_episode_returns"]) == 1.0