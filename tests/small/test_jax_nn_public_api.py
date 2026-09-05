"""Small (unit) tests for jax_nn public API surface."""

from __future__ import annotations


class TestPublicAPI:
    def test_submodule_imports(self) -> None:
        from jax_nn.distributional import CategoricalValueHead, categorical_cross_entropy, categorical_expected_value, categorical_l2_project
        from jax_nn.heads import DuelingHead, epsilon_greedy_action
        from jax_nn.initializers import output_orthogonal, stable_orthogonal
        from jax_nn.layers import NoisyLinear

        for obj in (
            CategoricalValueHead,
            categorical_cross_entropy,
            categorical_expected_value,
            categorical_l2_project,
            DuelingHead,
            epsilon_greedy_action,
            output_orthogonal,
            stable_orthogonal,
            NoisyLinear,
        ):
            assert obj is not None
