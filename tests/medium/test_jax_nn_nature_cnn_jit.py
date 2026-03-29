"""Medium integration tests for NatureCNN under JIT."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax_nn.layers import NatureCNN

SEED = 42


class TestNatureCNNJIT:
    def test_jit_forward_unbatched(self) -> None:
        model = NatureCNN()
        x = jnp.zeros((84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x)

        @jax.jit
        def forward(variables, x):
            return model.apply(variables, x)

        y = cast(jax.Array, forward(variables, x))
        assert y.shape == (3136,)

    def test_jit_forward_batched(self) -> None:
        model = NatureCNN()
        x = jnp.zeros((2, 84, 84, 4), dtype=jnp.uint8)
        variables = model.init(jax.random.key(SEED), x)

        @jax.jit
        def forward(variables, x):
            return model.apply(variables, x)

        y = cast(jax.Array, forward(variables, x))
        assert y.shape == (2, 3136)