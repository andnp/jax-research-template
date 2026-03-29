"""Medium integration tests for NoisyLinear under JIT."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax_nn.layers import NoisyLinear

SEED = 42


class TestNoisyLinearJIT:
    def test_jit_forward(self) -> None:
        model = NoisyLinear(features=8)
        x = jnp.ones((4,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)

        @jax.jit
        def forward(variables, x, noise_key):
            return cast(jax.Array, model.apply(variables, x, rngs={"noise": noise_key}))

        y = forward(variables, x, jax.random.key(99))
        assert y.shape == (8,)

    def test_jit_gradient_flows(self) -> None:
        model = NoisyLinear(features=4)
        x = jnp.ones((8,))
        variables = model.init({"params": jax.random.key(SEED), "noise": jax.random.key(SEED + 1)}, x)

        @jax.jit
        def loss_fn(params, x, noise_key):
            y = cast(jax.Array, model.apply({"params": params}, x, rngs={"noise": noise_key}))
            return jnp.sum(y**2)

        grads = jax.grad(loss_fn)(variables["params"], x, jax.random.key(99))
        for name in ("mu_w", "mu_b", "sigma_w", "sigma_b"):
            assert jnp.any(grads[name] != 0), f"Zero gradient for {name}"
