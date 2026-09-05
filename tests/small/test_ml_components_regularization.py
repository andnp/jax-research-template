"""Small tests for ml_components.regularization — L2 penalty on a named params head."""

import jax.numpy as jnp
from ml_components.regularization import l2_regularizer


class TestL2Regularizer:
    def test_sums_squares_under_head_key(self) -> None:
        params = {"params": {"h_head": {"kernel": jnp.array([3.0, 4.0])}, "trunk": {"kernel": jnp.array([10.0])}}}
        assert jnp.allclose(l2_regularizer(params, "h_head"), 25.0)

    def test_missing_head_key_is_zero(self) -> None:
        params = {"params": {"trunk": {"kernel": jnp.array([10.0])}}}
        assert jnp.allclose(l2_regularizer(params, "h_head"), 0.0)

    def test_unwraps_bare_params_dict(self) -> None:
        params = {"h_heads": {"kernel": jnp.array([1.0, 2.0, 2.0])}}
        assert jnp.allclose(l2_regularizer(params, "h_heads"), 9.0)
