"""Small (unit) tests for jax_nn public API surface."""

from __future__ import annotations


class TestPublicAPI:
    def test_submodule_imports(self) -> None:
        from jax_nn.heads import DuelingHead, epsilon_greedy_action
        from jax_nn.initializers import output_orthogonal, stable_orthogonal
        from jax_nn.layers import NoisyLinear

        for obj in (DuelingHead, epsilon_greedy_action, output_orthogonal, stable_orthogonal, NoisyLinear):
            assert obj is not None
