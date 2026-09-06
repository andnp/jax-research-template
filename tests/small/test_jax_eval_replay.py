"""Small behavioral tests for jax-eval replay diagnostics."""

import jax.numpy as jnp
from jax_eval.replay import priority_utilization


def test_priority_utilization_normalizes_entropy() -> None:
    """Return one for uniform priorities and normalized entropy otherwise."""
    uniform = priority_utilization(jnp.ones(4))
    uneven = priority_utilization(jnp.array([1.0, 3.0]))

    expected = -(0.25 * jnp.log(0.25) + 0.75 * jnp.log(0.75)) / jnp.log(2.0)
    assert jnp.allclose(uniform, 1.0)
    assert jnp.allclose(uneven, expected)


def test_priority_utilization_ignores_zero_probability_entries() -> None:
    """Exclude zero entries from entropy while retaining their capacity slots."""
    result = priority_utilization(jnp.array([1.0, 0.0, 1.0, 0.0]))

    assert jnp.allclose(result, 0.5)
