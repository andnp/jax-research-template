"""Small tests for MultiHeadReadout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_utils.pytree import shift_head_readout
from rl_components.heads import MultiHeadReadout

SEED = 42
WIDTH = 8


class TestMultiHeadReadoutShape:
    def test_scalar_head_shape(self) -> None:
        model = MultiHeadReadout(num_heads=5, head_shape=())
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)
        out = model.apply(variables, phi)
        assert out.shape == (5,)

    def test_vector_head_shape(self) -> None:
        model = MultiHeadReadout(num_heads=5, head_shape=(3,))
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)
        out = model.apply(variables, phi)
        assert out.shape == (5, 3)

    def test_batched_input_keeps_leading_dims(self) -> None:
        """Every current caller vmaps over unbatched transitions, but the module itself must
        accept a leading batch dimension like any other Flax layer."""
        model = MultiHeadReadout(num_heads=5, head_shape=(3,))
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)

        batched_phi = jnp.ones((7, WIDTH))
        out = model.apply(variables, batched_phi)
        assert out.shape == (7, 5, 3)
        assert jnp.allclose(out[0], model.apply(variables, phi))


class TestMultiHeadReadoutBias:
    def test_use_bias_false_omits_bias_param(self) -> None:
        model = MultiHeadReadout(num_heads=5, head_shape=(3,), use_bias=False)
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)
        assert set(variables["params"]) == {"kernel"}

    def test_use_bias_true_adds_bias_param(self) -> None:
        model = MultiHeadReadout(num_heads=5, head_shape=(3,), use_bias=True)
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)
        assert set(variables["params"]) == {"kernel", "bias"}


class TestMultiHeadReadoutShift:
    def test_shift_head_readout_advances_every_head_by_one(self) -> None:
        model = MultiHeadReadout(num_heads=4, head_shape=(2,))
        phi = jnp.ones((WIDTH,))
        variables = model.init(jax.random.key(SEED), phi)
        out = model.apply(variables, phi)

        shifted_params = shift_head_readout(variables["params"], kernel_name="kernel", kernel_axis=1, bias_name="bias")
        shifted_out = model.apply({**variables, "params": shifted_params}, phi)

        assert jnp.allclose(shifted_out[:-1], out[1:])
        assert jnp.allclose(shifted_out[-1], out[-1])
