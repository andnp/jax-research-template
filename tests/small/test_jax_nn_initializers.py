"""Small (unit) tests for jax_nn initializers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax_nn.initializers import legacy_dqn_bound, legacy_dqn_uniform, output_orthogonal, stable_orthogonal

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


class TestLegacyDQNInitializers:
    def test_bound_matches_inverse_sqrt_fan_in(self) -> None:
        assert legacy_dqn_bound(16) == 0.25

    def test_uniform_infers_dense_kernel_fan_in(self) -> None:
        init_fn = legacy_dqn_uniform()
        w = init_fn(jax.random.key(SEED), (4, 8))
        bound = math.sqrt(1.0 / 4.0)
        assert w.shape == (4, 8)
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)

    def test_uniform_infers_conv_kernel_fan_in(self) -> None:
        init_fn = legacy_dqn_uniform()
        w = init_fn(jax.random.key(SEED), (8, 8, 4, 32))
        bound = math.sqrt(1.0 / (8 * 8 * 4))
        assert w.shape == (8, 8, 4, 32)
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)

    def test_uniform_supports_bias_with_explicit_fan_in(self) -> None:
        init_fn = legacy_dqn_uniform(num_input_units=256)
        b = init_fn(jax.random.key(SEED), (32,))
        bound = math.sqrt(1.0 / 256.0)
        assert b.shape == (32,)
        assert jnp.all(b >= -bound)
        assert jnp.all(b <= bound)

    def test_bias_shape_requires_explicit_fan_in(self) -> None:
        init_fn = legacy_dqn_uniform()
        with pytest.raises(ValueError, match="explicit num_input_units"):
            init_fn(jax.random.key(SEED), (32,))

    def test_uniform_is_deterministic_for_same_key(self) -> None:
        init_fn = legacy_dqn_uniform()
        w1 = init_fn(jax.random.key(SEED), (4, 8))
        w2 = init_fn(jax.random.key(SEED), (4, 8))
        npt.assert_array_equal(w1, w2)
