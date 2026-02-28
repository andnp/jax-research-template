"""Small tests for jax_utils.pytree — math operations on pytrees."""

import jax.numpy as jnp
from jax_utils.pytree import (
    tree_add,
    tree_inner_product,
    tree_lerp,
    tree_mean,
    tree_norm,
    tree_ones_like,
    tree_scalar_mul,
    tree_std,
    tree_sub,
    tree_zeros_like,
)


class TestTreeZerosOnesLike:
    def test_zeros_like_dict(self):
        tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
        result = tree_zeros_like(tree)
        assert jnp.allclose(result["a"], jnp.zeros(2))
        assert float(result["b"]) == 0.0

    def test_ones_like_dict(self):
        tree = {"a": jnp.array([1.0, 2.0])}
        result = tree_ones_like(tree)
        assert jnp.allclose(result["a"], jnp.ones(2))

    def test_preserves_structure(self):
        tree = {"x": {"y": jnp.array([1.0])}}
        result = tree_zeros_like(tree)
        assert "x" in result
        assert "y" in result["x"]


class TestTreeArithmetic:
    def test_add(self):
        a = {"w": jnp.array([1.0, 2.0])}
        b = {"w": jnp.array([3.0, 4.0])}
        result = tree_add(a, b)
        assert jnp.allclose(result["w"], jnp.array([4.0, 6.0]))

    def test_sub(self):
        a = {"w": jnp.array([5.0, 3.0])}
        b = {"w": jnp.array([1.0, 1.0])}
        result = tree_sub(a, b)
        assert jnp.allclose(result["w"], jnp.array([4.0, 2.0]))

    def test_scalar_mul(self):
        tree = {"w": jnp.array([2.0, 3.0])}
        result = tree_scalar_mul(2.0, tree)
        assert jnp.allclose(result["w"], jnp.array([4.0, 6.0]))


class TestTreeStatistics:
    def test_mean_single_leaf(self):
        tree = {"w": jnp.array([1.0, 2.0, 3.0])}
        assert jnp.allclose(tree_mean(tree), 2.0)

    def test_mean_multiple_leaves(self):
        tree = {"a": jnp.array([1.0, 3.0]), "b": jnp.array([2.0, 4.0])}
        # mean of [1, 3, 2, 4] = 2.5
        assert jnp.allclose(tree_mean(tree), 2.5)

    def test_std_uniform(self):
        tree = {"w": jnp.array([2.0, 2.0, 2.0])}
        assert jnp.allclose(tree_std(tree), 0.0, atol=1e-6)

    def test_std_known(self):
        tree = {"w": jnp.array([1.0, 3.0])}
        # mean=2, var=((1-2)²+(3-2)²)/2 = 1, std=1
        assert jnp.allclose(tree_std(tree), 1.0)


class TestTreeNorm:
    def test_single_leaf(self):
        tree = {"w": jnp.array([3.0, 4.0])}
        assert jnp.allclose(tree_norm(tree), 5.0)

    def test_multiple_leaves(self):
        tree = {"a": jnp.array([1.0]), "b": jnp.array([2.0]), "c": jnp.array([2.0])}
        # sqrt(1 + 4 + 4) = 3
        assert jnp.allclose(tree_norm(tree), 3.0)


class TestTreeInnerProduct:
    def test_basic(self):
        a = {"w": jnp.array([1.0, 2.0])}
        b = {"w": jnp.array([3.0, 4.0])}
        assert jnp.allclose(tree_inner_product(a, b), 11.0)


class TestTreeLerp:
    def test_t_zero(self):
        a = {"w": jnp.array([1.0, 2.0])}
        b = {"w": jnp.array([3.0, 4.0])}
        result = tree_lerp(a, b, 0.0)
        assert jnp.allclose(result["w"], a["w"])

    def test_t_one(self):
        a = {"w": jnp.array([1.0, 2.0])}
        b = {"w": jnp.array([3.0, 4.0])}
        result = tree_lerp(a, b, 1.0)
        assert jnp.allclose(result["w"], b["w"])

    def test_midpoint(self):
        a = {"w": jnp.array([0.0, 0.0])}
        b = {"w": jnp.array([4.0, 6.0])}
        result = tree_lerp(a, b, 0.5)
        assert jnp.allclose(result["w"], jnp.array([2.0, 3.0]))
