"""Focused checks for process-control certification helpers."""

import jax
import jax.numpy as jnp
import pytest
from process_control.certification import (
    certify_reset_step,
    check_mass_balance,
    check_timestep_refinement,
)


def _reset(_key: jax.Array) -> tuple[jax.Array]:
    return (jnp.asarray(0.0),)


def _step(
    state: object,
    action: object,
    _key: jax.Array,
) -> tuple[jax.Array]:
    return (jnp.asarray(state) + jnp.asarray(action),)


def test_certify_reset_step_checks_finite_deterministic_outputs() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
        steps=2,
    )

    assert report.passed
    assert report.failed == ()


def test_certify_reset_step_checks_declared_action_bounds() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
        action_low=0.0,
        action_high=1.0,
    )

    assert report.passed


def test_certify_reset_step_reports_action_outside_declared_bounds() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(2.0),
        action_low=0.0,
        action_high=1.0,
    )

    assert not report.passed
    assert report.failed[0].name == "bounded_action_no_nan"
    assert "above action_high" in report.failed[0].message


def test_certify_reset_step_reports_nonfinite_outputs() -> None:
    def nonfinite_step(
        state: object,
        _action: object,
        _key: jax.Array,
    ) -> tuple[jax.Array]:
        return (jnp.asarray(state) + jnp.nan,)

    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=nonfinite_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
    )

    assert not report.passed
    assert report.failed[0].name == "finite_reset_and_step"
    with pytest.raises(AssertionError, match="certification failed"):
        report.raise_for_failure()


def test_timestep_refinement_and_mass_balance_checks() -> None:
    refinement = check_timestep_refinement(
        name="linear",
        initial_state=jnp.asarray(0.0),
        coarse_step=lambda state, dt: jnp.asarray(state) + dt,
        fine_step=lambda state, dt: jnp.asarray(state) + dt,
        state_value=lambda state: jnp.asarray(state),
        coarse_dt=1.0,
        fine_dt=0.5,
        tolerance=1e-6,
    )
    balance = check_mass_balance(
        name="inventory",
        initial_inventory=10.0,
        final_inventory=11.0,
        inlet_flow=jnp.asarray([2.0, 2.0]),
        realized_outlet_flow=jnp.asarray([1.0, 1.0]),
        overflow_flow=jnp.asarray([0.0, 0.0]),
        dt=0.5,
    )

    assert refinement.passed
    assert balance.passed
