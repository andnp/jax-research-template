"""Focused tests for membrane backwash timing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from process_control.units.membrane import MembraneParams, MembraneState
from process_control.units.membrane import reset as membrane_reset
from process_control.units.membrane import step as membrane_step


def _cleaning_params() -> MembraneParams:
    return MembraneParams(
        bw_duration=0.03,
        bw_recovery=0.75,
        alpha_cake=0.0,
        cip_interval=1e6,
    )


def _step(state: MembraneState, params: MembraneParams, dt: float, trigger: float):
    return membrane_step(
        state,
        jnp.array(50.0),
        jnp.array(0.03),
        jnp.array(0.0),
        jnp.array(trigger),
        params,
        jnp.array(dt),
    )


def test_backwash_uses_duration_and_ignores_held_trigger() -> None:
    params = _cleaning_params()
    state = replace(
        membrane_reset(params, jax.random.PRNGKey(0)),
        r_reversible=jnp.array(100.0),
    )

    for _ in range(3):
        state, _, _, flow = _step(state, params, 0.01, 1.0)
        assert float(flow) == pytest.approx(0.0)

    assert not bool(state.is_backwashing)
    assert float(state.backwash_remaining) == pytest.approx(0.0)
    assert float(state.r_reversible) == pytest.approx(25.0, rel=1e-6)

    state, _, _, flow = _step(state, params, 0.01, 1.0)
    assert float(flow) == pytest.approx(3.0)
    assert float(state.r_reversible) == pytest.approx(25.0, rel=1e-6)


def test_backwash_is_timestep_invariant() -> None:
    params = _cleaning_params()
    initial = replace(
        membrane_reset(params, jax.random.PRNGKey(1)),
        r_reversible=jnp.array(100.0),
    )

    coarse, _, _, _ = _step(initial, params, 0.05, 1.0)

    fine = initial
    for index in range(5):
        fine, _, _, _ = _step(fine, params, 0.01, float(index == 0))

    assert float(coarse.r_reversible) == pytest.approx(float(fine.r_reversible), rel=1e-6)
    assert float(coarse.permeate_volume) == pytest.approx(float(fine.permeate_volume), rel=1e-6)
    assert float(coarse.hours_since_bw) == pytest.approx(float(fine.hours_since_bw), rel=1e-6)
    assert float(coarse.backwash_remaining) == pytest.approx(0.0)
    assert not bool(coarse.is_backwashing)


def test_backwash_retrigger_requires_command_release() -> None:
    params = replace(_cleaning_params(), bw_duration=0.01)
    state = replace(
        membrane_reset(params, jax.random.PRNGKey(2)),
        r_reversible=jnp.array(100.0),
    )

    state, _, _, _ = _step(state, params, 0.01, 1.0)
    state, _, _, _ = _step(state, params, 0.01, 1.0)
    assert float(state.r_reversible) == pytest.approx(25.0, rel=1e-6)

    state, _, _, _ = _step(state, params, 0.01, 0.0)
    state, _, _, _ = _step(state, params, 0.01, 1.0)
    assert float(state.r_reversible) == pytest.approx(6.25, rel=1e-6)


def _foul(flux: float, params: MembraneParams, steps: int = 300) -> MembraneState:
    state = membrane_reset(params, jax.random.PRNGKey(3))
    for _ in range(steps):
        state, _, _, _ = membrane_step(
            state, jnp.array(50.0), jnp.array(flux), jnp.array(0.0), jnp.array(0.0), params, jnp.array(0.01)
        )
    return state


def test_higher_flux_fouls_faster_for_same_feed() -> None:
    """Fouling must depend on flux, not just feed load and time: the same
    feed TSS over the same duration should deposit more cake at higher flux."""
    params = MembraneParams(cip_interval=1e6)

    low_flux_state = _foul(0.01, params)
    high_flux_state = _foul(0.05, params)

    assert float(high_flux_state.r_reversible) > float(low_flux_state.r_reversible)
    assert float(high_flux_state.r_irreversible) > float(low_flux_state.r_irreversible)


def test_irreversible_fouling_scales_with_deposited_mass() -> None:
    """Irreversible fouling should track the reversible deposit it derives
    from, not accumulate independently of it."""
    params = MembraneParams(cip_interval=1e6)

    state = _foul(0.05, params)

    assert float(state.r_irreversible) == pytest.approx(
        params.irr_fraction * float(state.r_reversible), rel=1e-4
    )
