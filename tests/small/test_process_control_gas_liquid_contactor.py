"""Tests for the gas-liquid contactor module."""

import jax.numpy as jnp
import pytest
from process_control.units.gas_liquid_contactor import (
    ContactorParams,
    GasInlet,
    compute_removal,
)


@pytest.fixture
def params() -> ContactorParams:
    return ContactorParams()


@pytest.fixture
def nominal_gas() -> GasInlet:
    return GasInlet(
        gas_flow=jnp.array(500.0),
        h2s_ppm=jnp.array(50.0),
        temperature=jnp.array(25.0),
    )


def test_result_shapes(params: ContactorParams, nominal_gas: GasInlet):
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert result.removal_efficiency.shape == ()
    assert result.outlet_h2s_ppm.shape == ()
    assert result.sulfide_load.shape == ()


def test_efficiency_bounded_zero_one(params: ContactorParams, nominal_gas: GasInlet):
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert 0.0 <= float(result.removal_efficiency) <= 1.0


def test_outlet_h2s_consistent_with_efficiency(params: ContactorParams, nominal_gas: GasInlet):
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    expected_outlet = float(nominal_gas.h2s_ppm) * (1.0 - float(result.removal_efficiency))
    assert jnp.isclose(result.outlet_h2s_ppm, expected_outlet, atol=1e-4)


def test_higher_oxidant_improves_removal(params: ContactorParams, nominal_gas: GasInlet):
    low = compute_removal(
        nominal_gas,
        oxidant=jnp.array(1.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(10.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert float(high.removal_efficiency) > float(low.removal_efficiency)


def test_higher_alkalinity_improves_removal(params: ContactorParams, nominal_gas: GasInlet):
    low = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(0.5),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(8.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert float(high.removal_efficiency) > float(low.removal_efficiency)


def test_higher_gas_flow_reduces_removal(params: ContactorParams):
    low_flow = GasInlet(gas_flow=jnp.array(200.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(25.0))
    high_flow = GasInlet(gas_flow=jnp.array(800.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(25.0))

    r_low = compute_removal(
        low_flow,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    r_high = compute_removal(
        high_flow,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert float(r_low.removal_efficiency) > float(r_high.removal_efficiency)


def test_higher_recirc_improves_removal(params: ContactorParams, nominal_gas: GasInlet):
    low = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(10.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(100.0),
        params=params,
    )
    assert float(high.removal_efficiency) > float(low.removal_efficiency)


def test_zero_oxidant_still_has_some_removal(params: ContactorParams, nominal_gas: GasInlet):
    """Even with depleted oxidant, alkalinity can provide some removal."""
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(0.0),
        alkalinity=jnp.array(5.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    # Should not be zero — alkalinity still contributes
    assert float(result.removal_efficiency) > 0.0


def test_sulfide_load_positive(params: ContactorParams, nominal_gas: GasInlet):
    """Sulfide load transferred to liquid should be positive when removing H2S."""
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert float(result.sulfide_load) > 0.0


def test_no_h2s_no_removal_needed(params: ContactorParams):
    """With zero inlet H₂S, outlet should be zero and no sulfide load."""
    clean_gas = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(0.0), temperature=jnp.array(25.0))
    result = compute_removal(
        clean_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        params=params,
    )
    assert jnp.isclose(result.outlet_h2s_ppm, 0.0, atol=1e-6)
    assert jnp.isclose(result.sulfide_load, 0.0, atol=1e-6)


def test_efficiency_capped_at_max(params: ContactorParams, nominal_gas: GasInlet):
    """Efficiency should never exceed max_efficiency."""
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(100.0),
        alkalinity=jnp.array(100.0),
        recirc_flow=jnp.array(1000.0),
        params=params,
    )
    assert float(result.removal_efficiency) <= params.max_efficiency
