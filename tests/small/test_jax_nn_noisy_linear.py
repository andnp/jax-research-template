"""Small (unit) tests for NoisyLinear."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax_nn.layers import NoisyLinear

SEED = 42


class TestNoisyLinearShapes:
    def test_output_shape_1d(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y = model.apply(variables, x, rngs={"noise": jax.random.key(SEED + 2)})
        assert y.shape == (8,)

    def test_output_shape_batched(self) -> None:
        model = NoisyLinear(features=16)
        x = jnp.ones((32, 4))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y = model.apply(variables, x, rngs={"noise": jax.random.key(SEED + 2)})
        assert y.shape == (32, 16)

    def test_param_keys_present(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        params = cast(dict[str, jax.Array], variables["params"])
        assert set(params.keys()) == {"mu_w", "mu_b", "sigma_w", "sigma_b"}

    def test_param_shapes(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        params = cast(dict[str, jax.Array], variables["params"])
        assert params["mu_w"].shape == (4, 8)
        assert params["mu_b"].shape == (8,)
        assert params["sigma_w"].shape == (4, 8)
        assert params["sigma_b"].shape == (8,)


class TestNoisyLinearNoise:
    def test_different_noise_keys_produce_different_outputs(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y1 = model.apply(variables, x, rngs={"noise": jax.random.key(100)})
        y2 = model.apply(variables, x, rngs={"noise": jax.random.key(200)})
        assert not jnp.allclose(y1, y2)

    def test_same_noise_key_produces_same_output(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y1 = model.apply(variables, x, rngs={"noise": jax.random.key(100)})
        y2 = model.apply(variables, x, rngs={"noise": jax.random.key(100)})
        npt.assert_array_equal(y1, y2)

    def test_zero_sigma_recovers_standard_linear(self) -> None:
        model = NoisyLinear(features=8, sigma_init=0.0)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y1 = model.apply(variables, x, rngs={"noise": jax.random.key(100)})
        y2 = model.apply(variables, x, rngs={"noise": jax.random.key(200)})
        npt.assert_allclose(y1, y2, atol=1e-6)


class TestNoisyLinearDtype:
    def test_default_dtype_float32(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,), dtype=jnp.float32)
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y = model.apply(variables, x, rngs={"noise": jax.random.key(SEED + 2)})
        assert y.dtype == jnp.float32

    def test_bfloat16_output(self) -> None:
        model = NoisyLinear(features=8, dtype=jnp.bfloat16)
        x = jnp.ones((4,), dtype=jnp.bfloat16)
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)
        y = model.apply(variables, x, rngs={"noise": jax.random.key(SEED + 2)})
        assert y.dtype == jnp.bfloat16
