"""Small (unit) tests for rl_networks legacy DQN-matching initializers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from rl_networks.initializers import legacy_dqn_bound, legacy_dqn_uniform

SEED = 42


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
