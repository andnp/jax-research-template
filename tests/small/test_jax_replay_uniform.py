"""Small (unit) tests for jax_replay uniform buffer.

Verifies init shapes, add/sample correctness, pointer wrapping, and count capping.
Target duration: << 1 ms per test (no JIT).
"""

from __future__ import annotations

from typing import NamedTuple

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.types import BufferState
from jax_replay.uniform import add, init_buffer, sample


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array


def _make_prototype() -> Transition:
    return Transition(
        obs=jnp.zeros((4,), dtype=jnp.float32),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
    )


class TestInitBuffer:
    def test_returns_buffer_state(self) -> None:
        state = init_buffer(_make_prototype(), capacity=8)
        assert isinstance(state, BufferState)

    def test_data_shapes_match_prototype(self) -> None:
        state = init_buffer(_make_prototype(), capacity=16)
        # 3 leaves in Transition: obs(4,), action(), reward()
        assert state.data["0"].shape == (16, 4)  # obs
        assert state.data["1"].shape == (16,)  # action
        assert state.data["2"].shape == (16,)  # reward

    def test_data_dtypes_match_prototype(self) -> None:
        state = init_buffer(_make_prototype(), capacity=8)
        assert state.data["0"].dtype == jnp.float32
        assert state.data["1"].dtype == jnp.int32
        assert state.data["2"].dtype == jnp.float32

    def test_pointer_and_count_start_at_zero(self) -> None:
        state = init_buffer(_make_prototype(), capacity=8)
        assert int(state.pointer) == 0
        assert int(state.count) == 0


class TestAdd:
    def test_single_add_increments_pointer_and_count(self) -> None:
        state = init_buffer(_make_prototype(), capacity=8)
        t = Transition(obs=jnp.ones(4), action=jnp.int32(2), reward=jnp.float32(1.0))
        state = add(state, t)
        assert int(state.pointer) == 1
        assert int(state.count) == 1

    def test_data_written_at_correct_index(self) -> None:
        state = init_buffer(_make_prototype(), capacity=8)
        t = Transition(obs=jnp.array([1.0, 2.0, 3.0, 4.0]), action=jnp.int32(5), reward=jnp.float32(0.5))
        state = add(state, t)
        assert jnp.allclose(state.data["0"][0], jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert int(state.data["1"][0]) == 5
        assert float(state.data["2"][0]) == 0.5

    def test_pointer_wraps_at_capacity(self) -> None:
        capacity = 4
        state = init_buffer(_make_prototype(), capacity=capacity)
        for i in range(capacity + 2):
            t = Transition(obs=jnp.full(4, float(i)), action=jnp.int32(i), reward=jnp.float32(i))
            state = add(state, t)
        # pointer should be capacity + 2 = 6 (raw), wraps to index 2 on next write
        assert int(state.pointer) == capacity + 2
        # count capped at capacity
        assert int(state.count) == capacity

    def test_count_caps_at_capacity(self) -> None:
        capacity = 4
        state = init_buffer(_make_prototype(), capacity=capacity)
        for _i in range(10):
            t = Transition(obs=jnp.zeros(4), action=jnp.int32(0), reward=jnp.float32(0.0))
            state = add(state, t)
        assert int(state.count) == capacity

    def test_overwrites_oldest_data(self) -> None:
        capacity = 4
        state = init_buffer(_make_prototype(), capacity=capacity)
        for i in range(capacity):
            t = Transition(obs=jnp.full(4, float(i)), action=jnp.int32(i), reward=jnp.float32(i))
            state = add(state, t)
        # Overwrite index 0
        t = Transition(obs=jnp.full(4, 99.0), action=jnp.int32(99), reward=jnp.float32(99.0))
        state = add(state, t)
        assert jnp.allclose(state.data["0"][0], jnp.full(4, 99.0))


class TestSample:
    def test_sample_returns_correct_structure(self) -> None:
        proto = _make_prototype()
        state = init_buffer(proto, capacity=8)
        for i in range(4):
            t = Transition(obs=jnp.full(4, float(i)), action=jnp.int32(i), reward=jnp.float32(i))
            state = add(state, t)
        batch = sample(state, jax.random.key(0), batch_size=2, prototype=proto)
        assert isinstance(batch, Transition)
        assert batch.obs.shape == (2, 4)
        assert batch.action.shape == (2,)
        assert batch.reward.shape == (2,)

    def test_sample_returns_valid_data(self) -> None:
        proto = _make_prototype()
        state = init_buffer(proto, capacity=8)
        t = Transition(obs=jnp.array([1.0, 2.0, 3.0, 4.0]), action=jnp.int32(7), reward=jnp.float32(3.14))
        state = add(state, t)
        # With only 1 element, all samples must be that element
        batch = sample(state, jax.random.key(42), batch_size=3, prototype=proto)
        for row in range(3):
            assert jnp.allclose(batch.obs[row], jnp.array([1.0, 2.0, 3.0, 4.0]))
            assert int(batch.action[row]) == 7
