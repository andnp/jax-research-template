"""Small contract tests for frame-deduplicating replay storage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.buffers import FrameReplayBuffer, FrameReplayBufferState, ReplayBuffer, ReplayBufferState

Transition = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, bool]


def _stream(n_frames: int, frame_shape: tuple[int, ...], lengths: tuple[int, ...]) -> list[Transition]:
    frame = jnp.zeros(frame_shape, dtype=jnp.uint8)
    stream = []
    value = 1
    for length in lengths:
        frame = jnp.full(frame_shape, value, dtype=jnp.uint8)
        value += 1
        obs = jnp.repeat(frame[None], n_frames, axis=0)
        for step in range(length):
            next_frame = jnp.full(frame_shape, value, dtype=jnp.uint8)
            value += 1
            next_obs = jnp.concatenate((obs[1:], next_frame[None]), axis=0)
            stream.append((obs, jnp.asarray(len(stream), jnp.int32), jnp.asarray(float(step)), next_obs, jnp.asarray(0.9), step == length - 1))
            obs = next_obs
        frame = obs[-1]
    return stream


def _add_full(buffer: ReplayBuffer, state: ReplayBufferState, transition: Transition) -> ReplayBufferState:
    obs, action, reward, next_obs, discount, episode_end = transition
    return buffer.add(state, obs[None], action[None], jnp.asarray([reward]), next_obs[None], jnp.asarray([discount]))


def _add_frame(buffer: FrameReplayBuffer, state: FrameReplayBufferState, transition: Transition) -> FrameReplayBufferState:
    obs, action, reward, next_obs, discount, episode_end = transition
    return buffer.add(state, obs[None], action[None], jnp.asarray([reward]), next_obs[None], jnp.asarray([discount]), jnp.asarray([episode_end]))


class TestFrameReplayBuffer:
    def test_reconstructs_replay_buffer_transitions_byte_for_byte(self) -> None:
        """Reconstructed valid slots equal full-stack ReplayBuffer transitions."""
        n_frames = 4
        capacity = 7
        stream = _stream(n_frames, (3, 2), (2, 1, 1, 5, 1, 3, 4, 2))
        full = ReplayBuffer(capacity, (n_frames, 3, 2), (), action_dtype=jnp.int32, obs_dtype=jnp.uint8)
        frames = FrameReplayBuffer(capacity, (n_frames, 3, 2), (), action_dtype=jnp.int32, obs_dtype=jnp.uint8)
        full_state = full.init()
        frame_state = frames.init()

        def assert_equivalent(full_state: ReplayBufferState, frame_state: FrameReplayBufferState) -> None:
            oldest = (int(frame_state.pointer) - int(frame_state.count)) % capacity
            first = n_frames - 1 if int(frame_state.count) == capacity else 0
            for age in range(first, int(frame_state.count)):
                index = (oldest + age) % capacity
                expected = tuple(
                    array[index]
                    for array in (full_state.obs, full_state.actions, full_state.rewards, full_state.next_obs, full_state.discount)
                )
                actual = (
                    frames._observation(frame_state, jnp.asarray(index)),
                    frame_state.actions[index],
                    frame_state.rewards[index],
                    jnp.concatenate(
                        (frames._observation(frame_state, jnp.asarray(index))[1:], frame_state.next_frame[index][None]),
                        axis=0,
                    ),
                    frame_state.discount[index],
                )
                for got, want in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(got, want)

        for transition in stream:
            full_state = _add_full(full, full_state, transition)
            frame_state = _add_frame(frames, frame_state, transition)
            assert_equivalent(full_state, frame_state)

    def test_sample_excludes_overwritten_history(self) -> None:
        """Sampling after wrap never returns slots lacking predecessor history."""
        buffer = FrameReplayBuffer(6, (4, 2), (), action_dtype=jnp.int32, obs_dtype=jnp.uint8)
        state = buffer.init()
        for transition in _stream(4, (2,), (20,)):
            state = _add_frame(buffer, state, transition)
        samples = buffer.sample(state, jax.random.key(0), 256)[1]
        oldest = (int(state.pointer) - int(state.count)) % buffer.capacity
        valid_ages = range(buffer.n_frames - 1, int(state.count))
        valid_actions = {int(state.actions[(oldest + age) % buffer.capacity]) for age in valid_ages}
        assert all(int(action) in valid_actions for action in samples)

    def test_jit_and_leading_size_contract(self) -> None:
        """JIT preserves the add/sample API and rejects batched additions."""
        buffer = FrameReplayBuffer(8, (4, 2), (), action_dtype=jnp.int32, obs_dtype=jnp.uint8)
        state = buffer.init()
        def add(current: FrameReplayBufferState) -> FrameReplayBufferState:
            return _add_frame(buffer, current, _stream(4, (2,), (1,))[0])

        add = jax.jit(add)
        state = add(state)

        def sample(current: FrameReplayBufferState) -> tuple[jax.Array, ...]:
            return buffer.sample(current, jax.random.key(1), 1)

        sample = jax.jit(sample)
        assert sample(state)[0].shape == (1, 4, 2)
        transition = _stream(4, (2,), (1,))[0]
        with pytest.raises(ValueError, match="exactly one"):
            buffer.add(
                state,
                transition[0][None].repeat(2, axis=0),
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((2,)),
                transition[3][None].repeat(2, axis=0),
                jnp.ones((2,)),
                jnp.zeros((2,), bool),
            )
