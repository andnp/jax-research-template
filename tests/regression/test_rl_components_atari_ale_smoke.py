"""Smoke test for the real ALE adapter: shapes, and the rolling the loop now depends on."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_components.atari_ale import AleAtariConfig, make_atari_adapter

FRAME_STACK = 4
SCREEN_SIZE = 84
OBSERVATION_SHAPE = (FRAME_STACK, SCREEN_SIZE, SCREEN_SIZE, 1)


def test_ale_adapter_reset_and_step_shapes() -> None:
    config = AleAtariConfig(game="Pong", frame_stack=FRAME_STACK, frame_skip=4)
    env = make_atari_adapter(config)

    spec = env.spec()
    assert spec.num_actions is not None and spec.num_actions > 0
    assert tuple(spec.observation_shape) == OBSERVATION_SHAPE

    reset_out = env.reset(jax.random.key(0))
    assert reset_out.observation.shape == OBSERVATION_SHAPE
    assert jnp.all(reset_out.observation[0] == reset_out.observation[-1]), (
        "reset stacks the fresh observation n_frames times, which is what makes the loop's "
        "boundary reset sufficient without a wrapper-side refill"
    )

    action = jnp.asarray(0, dtype=jnp.int32)
    step_out = env.step(jax.random.key(1), reset_out.state, action)
    assert step_out.observation.shape == OBSERVATION_SHAPE
    assert step_out.reward.shape == ()
    assert step_out.terminated.shape == ()
    assert step_out.truncated.shape == ()


def test_ale_adapter_step_rolls_the_frame_stack() -> None:
    """``step`` shifts the stack and never refills it, so the loop owns the boundary alone."""
    env = make_atari_adapter(AleAtariConfig(game="Pong", frame_stack=FRAME_STACK, frame_skip=4))

    reset_out = env.reset(jax.random.key(0))
    step_out = env.step(jax.random.key(1), reset_out.state, jnp.asarray(0, dtype=jnp.int32))

    assert jnp.array_equal(step_out.observation[:-1], reset_out.observation[1:])
    assert step_out.info == {}, "the bridge's observation is the true one, so it smuggles nothing in info"
