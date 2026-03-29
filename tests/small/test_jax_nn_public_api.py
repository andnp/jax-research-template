"""Small (unit) tests for jax_nn public API surface."""

from __future__ import annotations


class TestPublicAPI:
    def test_submodule_imports(self) -> None:
        from jax_nn.heads import DuelingHead, epsilon_greedy_action
        from jax_nn.initializers import legacy_dqn_bound, legacy_dqn_uniform, output_orthogonal, stable_orthogonal
        from jax_nn.layers import NatureCNN, NoisyLinear

        for obj in (
            DuelingHead,
            epsilon_greedy_action,
            legacy_dqn_bound,
            legacy_dqn_uniform,
            output_orthogonal,
            stable_orthogonal,
            NatureCNN,
            NoisyLinear,
        ):
            assert obj is not None
