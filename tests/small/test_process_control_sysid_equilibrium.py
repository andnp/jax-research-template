"""Small contracts for fixed-point steady-state analysis."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.equilibrium import relax_to_equilibrium


def test_relax_to_equilibrium_converges_to_the_known_fixed_point() -> None:
    """A contracting linear map should settle near its analytic fixed point."""

    def step(state: jax.Array) -> jax.Array:
        return 0.5 * state + 3.0

    result = relax_to_equilibrium(step, jnp.array(0.0), max_steps=200, tolerance=1e-9)

    assert result.converged
    assert result.steps_taken < 200
    assert jnp.allclose(result.state, 6.0, atol=1e-4)


def test_relax_to_equilibrium_reports_non_convergence_for_a_diverging_map() -> None:
    """A growing map must exhaust the step budget rather than report a false fixed point."""

    def step(state: jax.Array) -> jax.Array:
        return 1.01 * state + 1.0

    result = relax_to_equilibrium(step, jnp.array(1.0), max_steps=50, tolerance=1e-9)

    assert not result.converged
    assert result.steps_taken == 50


def test_relax_to_equilibrium_rejects_invalid_budgets() -> None:
    """Reject inputs that would silently skip the relaxation or divide by zero tolerance."""

    def step(state: jax.Array) -> jax.Array:
        return state

    with pytest.raises(ValueError, match="max_steps must be positive"):
        relax_to_equilibrium(step, jnp.array(0.0), max_steps=0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        relax_to_equilibrium(step, jnp.array(0.0), max_steps=10, tolerance=0.0)
