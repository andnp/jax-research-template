"""Small tests for the JAXAtari canonical adapter."""

from dataclasses import dataclass
from typing import cast

import chex
import jax
import jax.numpy as jnp
import rl_components.atari as atari_module
from rl_components.atari import JAXAtariAdapter, JAXAtariConfig
from rl_components.env_protocol import EnvProtocol


@dataclass(frozen=True)
class FakeSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    n: int | None = None


class FakeWrappedEnv:
    def __init__(self, *, image_shape: tuple[int, int, int], wrappers: tuple[str, ...] = ()) -> None:
        self._image_shape = image_shape
        self.wrappers = wrappers

    def with_wrapper(self, name: str) -> "FakeWrappedEnv":
        return FakeWrappedEnv(image_shape=self._image_shape, wrappers=self.wrappers + (name,))

    def observation_space(self) -> FakeSpace:
        channels = 1
        return FakeSpace(shape=(4, 84, 84, channels), dtype=jnp.uint8)

    def action_space(self) -> FakeSpace:
        return FakeSpace(shape=(), dtype=jnp.int32, n=6)

    def image_space(self) -> FakeSpace:
        return FakeSpace(shape=self._image_shape, dtype=jnp.uint8)

    def reset(self, key: chex.PRNGKey) -> tuple[jax.Array, jax.Array]:
        del key
        observation = jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8)
        state = jnp.array(0, dtype=jnp.int32)
        return observation, state

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del action
        next_state = state + jnp.array(1, dtype=jnp.int32)
        observation = jnp.full((4, 84, 84, 1), next_state, dtype=jnp.uint8)
        info = {
            "returned_episode": jnp.array(True),
            "returned_episode_returns": jnp.array(3.0, dtype=jnp.float32),
        }
        return observation, next_state, jnp.array(1.5, dtype=jnp.float32), jnp.array(True), info


class FakeAtariBackend:
    def __init__(self, *, image_shape: tuple[int, int, int] = (96, 96, 3)) -> None:
        self.calls: list[tuple[str, object]] = []
        self.image_shape = image_shape

    def make(self, game: str) -> object:
        self.calls.append(("make", game))
        return FakeWrappedEnv(image_shape=self.image_shape)

    def atari_wrapper(self, env: FakeWrappedEnv, *, frame_stack_size: int, frame_skip: int, episodic_life: bool, max_pooling: bool) -> FakeWrappedEnv:
        self.calls.append(
            (
                "atari",
                {
                    "frame_stack": frame_stack_size,
                    "frame_skip": frame_skip,
                    "max_pooling": max_pooling,
                    "life_loss_terminal": episodic_life,
                },
            )
        )
        return env.with_wrapper("atari")

    def pixel_wrapper(
        self,
        env: FakeWrappedEnv,
        *,
        do_pixel_resize: bool,
        pixel_resize_shape: tuple[int, int],
        grayscale: bool,
    ) -> FakeWrappedEnv:
        self.calls.append(
            (
                "pixels",
                {
                    "grayscale": grayscale,
                    "resize_shape": pixel_resize_shape,
                    "do_resize": do_pixel_resize,
                },
            )
        )
        return env.with_wrapper("pixels")

    def log_wrapper(self, env: FakeWrappedEnv) -> FakeWrappedEnv:
        self.calls.append(("log", None))
        return env.with_wrapper("log")


def _install_fake_backend(monkeypatch, backend: FakeAtariBackend) -> None:
    monkeypatch.setattr(atari_module, "make_jaxatari_env", backend.make)
    monkeypatch.setattr(atari_module, "AtariWrapper", backend.atari_wrapper)
    monkeypatch.setattr(atari_module, "PixelObsWrapper", backend.pixel_wrapper)
    monkeypatch.setattr(atari_module, "LogWrapper", backend.log_wrapper)


class TestJAXAtariConfig:
    def test_defaults_match_issue_10_preprocessing_contract(self):
        config = JAXAtariConfig(game="pong")

        assert config.frame_stack == 4
        assert config.frame_skip == 4
        assert config.grayscale is True
        assert config.max_pooling is True
        assert config.life_loss_terminal is True
        assert config.resize_shape == (84, 84)
        assert config.log_returns is True


class TestJAXAtariAdapter:
    def test_builds_deepmind_style_wrapper_chain(self, monkeypatch):
        backend = FakeAtariBackend()
        _install_fake_backend(monkeypatch, backend)
        adapter = JAXAtariAdapter(JAXAtariConfig(game="pong"))

        spec = adapter.spec()

        assert backend.calls == [
            ("make", "pong"),
            (
                "atari",
                {
                    "frame_stack": 4,
                    "frame_skip": 4,
                    "max_pooling": True,
                    "life_loss_terminal": True,
                },
            ),
            (
                "pixels",
                {
                    "grayscale": True,
                    "resize_shape": (84, 84),
                    "do_resize": True,
                },
            ),
            ("log", None),
        ]
        assert spec.id == "jaxatari:pong"
        assert spec.observation_shape == (4, 84, 84, 1)
        assert spec.num_actions == 6

    def test_skips_resize_when_image_pipeline_is_already_84_square(self, monkeypatch):
        backend = FakeAtariBackend(image_shape=(84, 84, 3))
        _install_fake_backend(monkeypatch, backend)

        JAXAtariAdapter(JAXAtariConfig(game="breakout"))

        assert backend.calls[2] == (
            "pixels",
            {
                "grayscale": True,
                "resize_shape": (84, 84),
                "do_resize": False,
            },
        )

    def test_adapts_reset_and_step_to_canonical_protocol(self, monkeypatch):
        backend = FakeAtariBackend()
        _install_fake_backend(monkeypatch, backend)
        adapter: EnvProtocol[jax.Array, object, jax.Array, None] = JAXAtariAdapter(JAXAtariConfig(game="pong"))

        reset = adapter.reset(jax.random.key(0))
        transition = adapter.step(jax.random.key(1), reset.state, jnp.array(2, dtype=jnp.int32))

        assert isinstance(adapter, EnvProtocol)
        assert reset.observation.shape == (4, 84, 84, 1)
        assert int(cast(jax.Array, transition.state)) == 1
        assert float(transition.reward) == 1.5
        assert bool(transition.terminated) is True
        assert bool(transition.truncated) is False
        assert bool(transition.info["returned_episode"]) is True
        assert float(transition.info["returned_episode_returns"]) == 3.0