"""Small (unit) tests for DuelingHead."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax_nn.heads import DuelingHead

SEED = 42


class TestDuelingHeadShape:
    def test_single_observation(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,))
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.shape == (4,)

    def test_batched_observations(self) -> None:
        model = DuelingHead(action_dim=6, hidden_features=32)
        x = jnp.ones((8, 16))
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.shape == (8, 6)


class TestDuelingHeadMeanSubtraction:
    def test_advantage_mean_is_zero(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((8, 16))
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        centered = q - jnp.mean(q, axis=-1, keepdims=True)
        npt.assert_allclose(jnp.mean(centered, axis=-1), 0.0, atol=1e-6)


class TestDuelingHeadDtype:
    def test_float32_output(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,), dtype=jnp.float32)
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.dtype == jnp.float32

    def test_bfloat16_output(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32, dtype=jnp.bfloat16)
        x = jnp.ones((16,), dtype=jnp.bfloat16)
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.dtype == jnp.bfloat16


class TestDuelingHeadParams:
    def test_has_four_dense_layers(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,))
        variables = model.init(jax.random.key(SEED), x)
        params = variables["params"]
        dense_keys = [k for k in params if k.startswith("Dense_")]
        assert len(dense_keys) == 4
