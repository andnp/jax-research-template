"""Small (unit) tests for jax_replay SumTree.

Verifies update propagation, root-sum invariant, sample validity, and batch sizing.
Target duration: << 1 ms per test (no JIT).
"""

from __future__ import annotations

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.sum_tree import _tree_find, tree_init, tree_sample, tree_sample_batch, tree_update


class TestTreeInit:
    def test_shape_is_double_capacity(self) -> None:
        tree = tree_init(8)
        assert tree.shape == (16,)

    def test_all_zeros(self) -> None:
        tree = tree_init(8)
        assert jnp.all(tree == 0.0)

    def test_rejects_non_power_of_2(self) -> None:
        try:
            tree_init(5)
            raise AssertionError("Should have raised")
        except AssertionError as e:
            assert "power of 2" in str(e)


class TestTreeUpdate:
    def test_sets_leaf_value(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(3.0))
        # Leaf 0 is at index 4 (capacity + 0)
        assert float(tree[4]) == 3.0

    def test_root_equals_sum_of_all_leaves(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(1.0))
        tree = tree_update(tree, jnp.uint32(1), jnp.float32(2.0))
        tree = tree_update(tree, jnp.uint32(2), jnp.float32(3.0))
        tree = tree_update(tree, jnp.uint32(3), jnp.float32(4.0))
        assert float(tree[1]) == 10.0  # root = total sum

    def test_parent_sums_correct(self) -> None:
        tree = tree_init(4)
        # Leaves at indices 4,5,6,7
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(1.0))  # idx 4
        tree = tree_update(tree, jnp.uint32(1), jnp.float32(2.0))  # idx 5
        tree = tree_update(tree, jnp.uint32(2), jnp.float32(3.0))  # idx 6
        tree = tree_update(tree, jnp.uint32(3), jnp.float32(4.0))  # idx 7
        # Parent of (4,5) is 2: should be 1+2=3
        assert float(tree[2]) == 3.0
        # Parent of (6,7) is 3: should be 3+4=7
        assert float(tree[3]) == 7.0

    def test_update_overwrites_and_propagates(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(5.0))
        tree = tree_update(tree, jnp.uint32(1), jnp.float32(2.0))
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(1.0))  # overwrite
        assert float(tree[4]) == 1.0
        assert float(tree[2]) == 3.0
        assert float(tree[1]) == 3.0


class TestTreeSample:
    def test_exact_prefix_boundaries_select_expected_leaf(self) -> None:
        tree = tree_init(4)
        priorities = [1.0, 2.0, 3.0, 4.0]
        for i, priority in enumerate(priorities):
            tree = tree_update(tree, jnp.uint32(i), jnp.float32(priority))

        assert int(_tree_find(tree, jnp.float32(0.0), 4)) == 0
        assert int(_tree_find(tree, jnp.float32(0.999), 4)) == 0
        assert int(_tree_find(tree, jnp.float32(1.0), 4)) == 1
        assert int(_tree_find(tree, jnp.float32(2.999), 4)) == 1
        assert int(_tree_find(tree, jnp.float32(3.0), 4)) == 2
        assert int(_tree_find(tree, jnp.float32(5.999), 4)) == 2
        assert int(_tree_find(tree, jnp.float32(6.0), 4)) == 3
        assert int(_tree_find(tree, jnp.float32(9.999), 4)) == 3

    def test_returns_valid_index(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(1.0))
        tree = tree_update(tree, jnp.uint32(1), jnp.float32(1.0))
        idx = tree_sample(tree, jax.random.key(0), 4)
        assert 0 <= int(idx) < 4

    def test_single_nonzero_always_sampled(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(2), jnp.float32(5.0))
        for seed in range(20):
            idx = tree_sample(tree, jax.random.key(seed), 4)
            assert int(idx) == 2

    def test_zero_priority_never_sampled(self) -> None:
        tree = tree_init(4)
        tree = tree_update(tree, jnp.uint32(0), jnp.float32(0.0))
        tree = tree_update(tree, jnp.uint32(1), jnp.float32(10.0))
        tree = tree_update(tree, jnp.uint32(2), jnp.float32(0.0))
        tree = tree_update(tree, jnp.uint32(3), jnp.float32(0.0))
        for seed in range(20):
            idx = tree_sample(tree, jax.random.key(seed), 4)
            assert int(idx) == 1


class TestTreeSampleBatch:
    def test_uniform_tree_stratification_covers_each_segment_once(self) -> None:
        tree = tree_init(4)
        for i in range(4):
            tree = tree_update(tree, jnp.uint32(i), jnp.float32(1.0))

        indices = tree_sample_batch(tree, jax.random.key(0), 4, batch_size=4)
        assert jnp.array_equal(indices, jnp.arange(4, dtype=indices.dtype))

    def test_batch_size_matches(self) -> None:
        tree = tree_init(8)
        for i in range(8):
            tree = tree_update(tree, jnp.uint32(i), jnp.float32(1.0))
        indices = tree_sample_batch(tree, jax.random.key(0), 8, batch_size=4)
        assert indices.shape == (4,)

    def test_all_indices_valid(self) -> None:
        tree = tree_init(8)
        for i in range(8):
            tree = tree_update(tree, jnp.uint32(i), jnp.float32(float(i + 1)))
        indices = tree_sample_batch(tree, jax.random.key(0), 8, batch_size=6)
        for i in range(6):
            assert 0 <= int(indices[i]) < 8
