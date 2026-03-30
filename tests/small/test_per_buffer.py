"""Small (unit) tests for jax_replay PER buffer.

Verifies priority insertion, IS weight correctness, and priority updates.
Target duration: << 1 ms per test (no JIT).
"""

from __future__ import annotations

from typing import NamedTuple

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.per import init_per_buffer, per_add, per_sample, per_update_priorities


class Transition(NamedTuple):
    obs: jax.Array
    reward: jax.Array


def _proto() -> Transition:
    return Transition(obs=jnp.zeros((2,)), reward=jnp.zeros(()))


class TestInitPerBuffer:
    def test_tree_initialized_to_zeros(self) -> None:
        state = init_per_buffer(_proto(), capacity=4)
        assert jnp.all(state.tree == 0.0)

    def test_max_priority_starts_at_one(self) -> None:
        state = init_per_buffer(_proto(), capacity=4)
        assert float(state.max_priority) == 1.0

    def test_non_power_of_two_capacity_pads_internal_storage(self) -> None:
        state = init_per_buffer(_proto(), capacity=5)
        assert int(state.logical_capacity) == 5
        assert int(state.storage_capacity) == 8
        assert state.data["0"].shape == (8, 2)
        assert state.data["1"].shape == (8,)


class TestPerAdd:
    def test_new_transition_gets_max_priority(self) -> None:
        state = init_per_buffer(_proto(), capacity=4)
        t = Transition(obs=jnp.array([1.0, 2.0]), reward=jnp.float32(0.5))
        state = per_add(state, t, alpha=1.0)
        # max_priority=1.0, alpha=1.0 → leaf priority = 1.0^1.0 = 1.0
        assert float(state.tree[4]) == 1.0  # leaf at capacity + 0

    def test_root_sum_equals_sum_of_leaf_priorities(self) -> None:
        state = init_per_buffer(_proto(), capacity=4)
        for i in range(3):
            t = Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i))
            state = per_add(state, t, alpha=1.0)
        # 3 leaves with priority 1.0 each
        assert float(state.tree[1]) == 3.0

    def test_new_transition_uses_current_max_raw_priority(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(2):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=0.5)

        state = per_update_priorities(
            state,
            jnp.array([0], dtype=jnp.uint32),
            jnp.array([9.0]),
            alpha=0.5,
            epsilon=0.0,
        )

        state = per_add(state, Transition(obs=jnp.array([9.0, 9.0]), reward=jnp.float32(9.0)), alpha=0.5)
        assert jnp.allclose(state.tree[6], 3.0)

    def test_non_power_of_two_capacity_wraps_at_logical_capacity(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=5)

        for i in range(7):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=1.0)

        assert int(state.count) == 5
        assert jnp.allclose(state.data["0"][0], jnp.array([5.0, 5.0], dtype=jnp.float32))
        assert jnp.allclose(state.data["0"][1], jnp.array([6.0, 6.0], dtype=jnp.float32))


class TestPerSample:
    def test_returns_correct_shapes(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)))
        batch, weights, indices = per_sample(state, jax.random.key(0), batch_size=2, beta=1.0, prototype=proto)
        assert batch.obs.shape == (2, 2)
        assert batch.reward.shape == (2,)
        assert weights.shape == (2,)
        assert indices.shape == (2,)

    def test_is_weights_max_is_one(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)))
        _, weights, _ = per_sample(state, jax.random.key(42), batch_size=4, beta=1.0, prototype=proto)
        assert jnp.allclose(jnp.max(weights), 1.0)

    def test_uniform_priorities_give_equal_weights(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=1.0)
        # All priorities equal → all IS weights should be 1.0
        _, weights, _ = per_sample(state, jax.random.key(0), batch_size=4, beta=1.0, prototype=proto)
        assert jnp.allclose(weights, 1.0)

    def test_is_weights_normalize_against_global_min_probability(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=1.0)

        state = per_update_priorities(
            state,
            jnp.array([0, 1, 2, 3], dtype=jnp.uint32),
            jnp.array([100.0, 1.0, 1.0, 1.0]),
            alpha=1.0,
            epsilon=0.0,
        )

        _, weights, indices = per_sample(state, jax.random.key(0), batch_size=4, beta=1.0, prototype=proto)
        assert jnp.array_equal(indices, jnp.zeros((4,), dtype=indices.dtype))
        assert jnp.allclose(weights, jnp.full((4,), 0.01, dtype=weights.dtype))

    def test_sampling_with_logical_capacity_never_returns_padded_indices(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=5)
        for i in range(5):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=1.0)

        _, _, indices = per_sample(state, jax.random.key(0), batch_size=16, beta=1.0, prototype=proto)
        assert jnp.all(indices < 5)


class TestPerUpdatePriorities:
    def test_priorities_change_after_update(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)))

        old_root = float(state.tree[1])
        indices = jnp.array([0, 1], dtype=jnp.uint32)
        td_errors = jnp.array([10.0, 20.0])
        state = per_update_priorities(state, indices, td_errors, alpha=1.0, epsilon=0.0)
        new_root = float(state.tree[1])
        assert new_root != old_root

    def test_max_priority_updates(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        for i in range(2):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)))

        indices = jnp.array([0], dtype=jnp.uint32)
        td_errors = jnp.array([100.0])
        state = per_update_priorities(state, indices, td_errors, alpha=1.0, epsilon=0.01)
        assert jnp.allclose(state.max_priority, 100.01)

    def test_tree_stores_alpha_scaled_priority_while_max_priority_stays_raw(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=4)
        state = per_add(state, Transition(obs=jnp.ones(2), reward=jnp.float32(1.0)), alpha=0.5)

        state = per_update_priorities(
            state,
            jnp.array([0], dtype=jnp.uint32),
            jnp.array([16.0]),
            alpha=0.5,
            epsilon=0.0,
        )

        assert jnp.allclose(state.tree[4], 4.0)
        assert jnp.allclose(state.max_priority, 16.0)
