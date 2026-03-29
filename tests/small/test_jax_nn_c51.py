"""Small (unit) tests for categorical distributional helpers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax_nn.distributional import (
    CategoricalValueHead,
    categorical_cross_entropy,
    categorical_expected_value,
    categorical_l2_project,
)

SEED = 42


class TestCategoricalProjection:
    def test_identity_projection_on_matching_support(self) -> None:
        support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        probabilities = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)

        projected = categorical_l2_project(support, probabilities, support)

        npt.assert_allclose(projected, probabilities, atol=1e-6)

    def test_projection_splits_mass_between_neighboring_atoms(self) -> None:
        support = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        target_support = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
        probabilities = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)

        projected = categorical_l2_project(target_support, probabilities, support)

        npt.assert_allclose(projected, jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32), atol=1e-6)

    def test_projection_clips_out_of_range_atoms_and_preserves_mass(self) -> None:
        support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        target_support = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
        probabilities = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)

        projected = categorical_l2_project(target_support, probabilities, support)

        npt.assert_allclose(projected, jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32), atol=1e-6)
        npt.assert_allclose(jnp.sum(projected), 1.0, atol=1e-6)


class TestCategoricalLossAndExpectation:
    def test_cross_entropy_matches_manual_value(self) -> None:
        target_probabilities = jnp.array([0.25, 0.75], dtype=jnp.float32)
        logits = jnp.log(target_probabilities)

        loss = categorical_cross_entropy(logits, target_probabilities)

        expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        npt.assert_allclose(loss, expected, atol=1e-6)

    def test_expected_value_reduces_over_atom_axis(self) -> None:
        support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        probabilities = jnp.array(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )

        values = categorical_expected_value(probabilities, support)

        npt.assert_allclose(values, jnp.array([0.6, -0.5], dtype=jnp.float32), atol=1e-6)


class TestCategoricalValueHead:
    def test_head_emits_action_atom_logits(self) -> None:
        model = CategoricalValueHead(action_dim=4, num_atoms=51)
        x = jnp.ones((2, 16), dtype=jnp.float32)

        variables = model.init(jax.random.key(SEED), x)
        logits = jnp.asarray(model.apply(variables, x))

        assert logits.shape == (2, 4, 51)
        assert logits.dtype == jnp.float32