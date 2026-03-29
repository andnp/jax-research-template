"""Small (unit) tests for NatureCNN."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from jax_nn.initializers import legacy_dqn_bound
from jax_nn.layers import NatureCNN

SEED = 42


class TestNatureCNNShapes:
    def test_output_shape_unbatched(self) -> None:
        model = NatureCNN()
        x = jnp.zeros((84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x)
        y = cast(jax.Array, model.apply(variables, x))
        assert y.shape == (3136,)

    def test_output_shape_batched(self) -> None:
        model = NatureCNN()
        x = jnp.zeros((2, 84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x)
        y = cast(jax.Array, model.apply(variables, x))
        assert y.shape == (2, 3136)


class TestNatureCNNContracts:
    def test_legacy_dqn_init_bounds_match_each_conv_layer(self) -> None:
        model = NatureCNN()
        x = jnp.zeros((84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x)
        params = cast(dict[str, dict[str, jax.Array]], unfreeze(variables["params"]))

        expected_fan_in = {
            "Conv_0": 8 * 8 * 4,
            "Conv_1": 4 * 4 * 32,
            "Conv_2": 3 * 3 * 64,
        }

        for layer_name, fan_in in expected_fan_in.items():
            bound = legacy_dqn_bound(fan_in)
            kernel = params[layer_name]["kernel"]
            bias = params[layer_name]["bias"]
            assert jnp.all(kernel >= -bound)
            assert jnp.all(kernel <= bound)
            assert jnp.all(bias >= -bound)
            assert jnp.all(bias <= bound)

    def test_uint8_inputs_are_scaled_before_convolution(self) -> None:
        model = NatureCNN()
        x_zero = jnp.zeros((84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x_zero)
        params = cast(dict[str, dict[str, jax.Array]], unfreeze(variables["params"]))

        for layer_params in params.values():
            layer_params["kernel"] = jnp.ones_like(layer_params["kernel"])
            layer_params["bias"] = jnp.zeros_like(layer_params["bias"])

        scaled_variables = freeze({"params": params})
        y_zero = cast(jax.Array, model.apply(scaled_variables, x_zero))
        y_full = cast(jax.Array, model.apply(scaled_variables, jnp.full((84, 84, 4), 255, dtype=jnp.uint8)))

        assert jnp.allclose(y_zero, 0.0)
        assert jnp.all(y_full > 0.0)