"""Small (unit) tests for jax_nn initializers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax_nn.initializers import output_orthogonal, stable_orthogonal

SEED = 42


class TestStableOrthogonal:
    def test_output_shape(self) -> None:
        init_fn = stable_orthogonal()
        w = init_fn(jax.random.key(SEED), (4, 8))
        assert w.shape == (4, 8)

    def test_default_dtype_float32(self) -> None:
        init_fn = stable_orthogonal()
        w = init_fn(jax.random.key(SEED), (4, 4))
        assert w.dtype == jnp.float32

    def test_default_scale_sqrt2(self) -> None:
        init_fn = stable_orthogonal()
        w = init_fn(jax.random.key(SEED), (64, 64))
        col_norms = jnp.linalg.norm(w, axis=0)
        npt.assert_allclose(col_norms, jnp.sqrt(2.0), atol=1e-5)

    def test_custom_scale(self) -> None:
        init_fn = stable_orthogonal(scale=3.0)
        w = init_fn(jax.random.key(SEED), (64, 64))
        col_norms = jnp.linalg.norm(w, axis=0)
        npt.assert_allclose(col_norms, 3.0, atol=1e-5)

    def test_orthogonality_of_columns(self) -> None:
        init_fn = stable_orthogonal()
        w = init_fn(jax.random.key(SEED), (64, 64))
        gram = w.T @ w
        expected = 2.0 * jnp.eye(64)
        npt.assert_allclose(gram, expected, atol=1e-5)

    def test_deterministic_with_same_key(self) -> None:
        init_fn = stable_orthogonal()
        w1 = init_fn(jax.random.key(SEED), (8, 8))
        w2 = init_fn(jax.random.key(SEED), (8, 8))
        npt.assert_array_equal(w1, w2)


class TestOutputOrthogonal:
    def test_output_shape(self) -> None:
        init_fn = output_orthogonal()
        w = init_fn(jax.random.key(SEED), (64, 4))
        assert w.shape == (64, 4)

    def test_small_scale(self) -> None:
        init_fn = output_orthogonal()
        w = init_fn(jax.random.key(SEED), (64, 64))
        col_norms = jnp.linalg.norm(w, axis=0)
        npt.assert_allclose(col_norms, 0.01, atol=1e-5)

    def test_near_zero_values(self) -> None:
        init_fn = output_orthogonal()
        w = init_fn(jax.random.key(SEED), (64, 4))
        assert jnp.max(jnp.abs(w)) < 0.1
