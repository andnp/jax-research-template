"""Medium integration tests for epsilon_greedy_action under JIT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_nn.heads import epsilon_greedy_action

SEED = 42


class TestEpsilonGreedyJIT:
    def test_jit_compiles(self) -> None:
        q = jnp.array([1.0, 2.0, 3.0])

        @jax.jit
        def select(q, eps, key):
            return epsilon_greedy_action(q, eps, key=key)

        action = select(q, 0.1, jax.random.key(SEED))
        assert action.shape == ()

    def test_jit_with_traced_epsilon(self) -> None:
        q = jnp.array([1.0, 5.0, 2.0])

        @jax.jit
        def select(q, eps, key):
            return epsilon_greedy_action(q, eps, key=key)

        eps = jnp.float32(0.0)
        action = select(q, eps, jax.random.key(SEED))
        assert action == 1
