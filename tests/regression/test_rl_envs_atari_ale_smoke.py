"""Smoke test for the real ALE adapter: shapes, and the rolling the loop now depends on."""

from __future__ import annotations

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from ale_py.env import AtariEnv
from rl_envs.atari_ale import AleAtariConfig, _make_ale_env, make_atari_adapter

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


def test_episode_ending_step_returns_the_current_ale_screen() -> None:
    """Every terminal skip offset returns the current screen with only a valid pooled frame."""
    early_offsets: set[int] = set()
    late_offsets: set[int] = set()

    for seed in range(32):
        config = AleAtariConfig(
            game="Breakout",
            frame_skip=4,
            screen_size=SCREEN_SIZE,
            terminal_on_life_loss=True,
        )
        env = _make_ale_env(config)
        env.noop_max = 0
        oracle = AtariEnv(
            "breakout",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        env.reset(seed=seed)
        oracle.reset(seed=seed)
        actions = np.random.default_rng(seed)

        for _ in range(10_000):
            lives_before = oracle.ale.lives()
            first_frame = env.ale.getEpisodeFrameNumber()
            action = int(actions.integers(4))
            observation, _, terminated, truncated, _ = env.step(action)

            raw_frames = []
            for _ in range(config.frame_skip):
                _, _, raw_terminated, raw_truncated, _ = oracle.step(action)
                screen = np.empty((210, 160), dtype=np.uint8)
                oracle.ale.getScreenGrayscale(screen)
                raw_frames.append(screen.copy())
                if raw_terminated or raw_truncated or oracle.ale.lives() < lives_before:
                    break

            frames_advanced = env.ale.getEpisodeFrameNumber() - first_frame
            if terminated or truncated:
                assert frames_advanced == len(raw_frames)
                current_screen = raw_frames[-1]
                if frames_advanced == config.frame_skip:
                    current_screen = np.maximum(current_screen, raw_frames[-2])
                current = cv2.resize(current_screen, (SCREEN_SIZE, SCREEN_SIZE), interpolation=cv2.INTER_AREA)
                expected = current[..., None]
                assert np.array_equal(observation, expected)
                if frames_advanced < config.frame_skip - 1:
                    early_offsets.add(frames_advanced)
                else:
                    late_offsets.add(frames_advanced)
                break

            assert len(raw_frames) == config.frame_skip
        env.close()
        oracle.close()

    assert early_offsets, "Breakout did not lose a life on an early frame-skip sub-step"
    assert late_offsets, "Breakout did not lose a life on the final or second-to-last sub-step"


def test_losing_a_life_does_not_end_the_atari_episode() -> None:
    """Life loss continues the ALE episode until Breakout reaches game over."""
    env = _make_ale_env(AleAtariConfig(game="Breakout"))
    env.reset(seed=0)
    previous_lives = env.ale.lives()
    observed_life_loss = False
    reached_game_over = False
    action_key = np.random.default_rng(1)

    for _ in range(10_000):
        _, _, terminated, truncated, _ = env.step(int(action_key.integers(4)))
        lives = env.ale.lives()
        if lives < previous_lives:
            observed_life_loss |= not terminated and not truncated
        if terminated or truncated:
            assert terminated
            assert lives == 0
            reached_game_over = True
            break
        previous_lives = lives

    assert observed_life_loss, "Breakout did not lose a life during the run"
    assert reached_game_over, "Breakout did not reach game over during the run"
