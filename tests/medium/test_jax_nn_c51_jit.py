"""Medium integration tests for categorical distributional helpers under JIT."""

from __future__ import annotations

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


class TestCategoricalDistributionalJIT:
    def test_jit_projection_and_expected_value(self) -> None:
        support = jnp.linspace(-10.0, 10.0, 51, dtype=jnp.float32)
        target_support = 0.9 * support + 1.5
        target_probabilities = jax.nn.softmax(jnp.linspace(-1.0, 1.0, 51, dtype=jnp.float32))

        @jax.jit
        def project_and_value(target_support, target_probabilities, support):
            projected = categorical_l2_project(target_support, target_probabilities, support)
            value = categorical_expected_value(projected, support)
            return projected, value

        projected, value = project_and_value(target_support, target_probabilities, support)

        assert projected.shape == (51,)
        npt.assert_allclose(jnp.sum(projected), 1.0, atol=1e-6)
        assert value.shape == ()

    def test_jit_head_and_cross_entropy(self) -> None:
        model = CategoricalValueHead(action_dim=4, num_atoms=51)
        x = jnp.ones((2, 16), dtype=jnp.float32)
        target_probabilities = jnp.full((2, 4, 51), 1.0 / 51.0, dtype=jnp.float32)
        variables = model.init(jax.random.key(SEED), x)

        @jax.jit
        def forward_and_loss(variables, x, target_probabilities):
            logits = jnp.asarray(model.apply(variables, x))
            loss = categorical_cross_entropy(logits, target_probabilities)
            return logits, loss

        logits, loss = forward_and_loss(variables, x, target_probabilities)

        assert logits.shape == (2, 4, 51)
        assert loss.shape == (2, 4)
        assert jnp.all(loss > 0.0)