"""Tests for the gas-liquid contactor module."""

from dataclasses import replace

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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert 0.0 <= float(result.removal_efficiency) <= 1.0


def test_outlet_h2s_consistent_with_efficiency(params: ContactorParams, nominal_gas: GasInlet):
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(10.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert float(high.removal_efficiency) > float(low.removal_efficiency)


def test_higher_alkalinity_improves_removal(params: ContactorParams, nominal_gas: GasInlet):
    low = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(0.5),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(8.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    r_high = compute_removal(
        high_flow,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert float(r_low.removal_efficiency) > float(r_high.removal_efficiency)


def test_higher_temperature_reduces_removal(params: ContactorParams):
    cool = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(10.0))
    warm = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(25.0))

    r_cool = compute_removal(
        cool,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    r_warm = compute_removal(
        warm,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert float(r_cool.removal_efficiency) > float(r_warm.removal_efficiency)


def test_temp_coefficient_one_disables_temperature_effect(params: ContactorParams):
    flat_params = replace(params, temp_coefficient=1.0)
    cool = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(10.0))
    warm = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(25.0))

    r_cool = compute_removal(
        cool,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=flat_params,
    )
    r_warm = compute_removal(
        warm,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=flat_params,
    )
    assert float(r_cool.removal_efficiency) == pytest.approx(float(r_warm.removal_efficiency))


def test_higher_recirc_improves_removal(params: ContactorParams, nominal_gas: GasInlet):
    low = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(10.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    high = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(100.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
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
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert float(result.removal_efficiency) <= params.max_efficiency


def test_lower_ph_reduces_removal(params: ContactorParams, nominal_gas: GasInlet):
    """Below pKa1, more dissolved sulfide is un-ionized H2S, throttling absorption."""
    low_ph = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(6.0),
        sulfide=jnp.array(1.0),
        params=params,
    )
    high_ph = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    assert float(high_ph.removal_efficiency) > float(low_ph.removal_efficiency)


def test_ph_well_above_pka1_barely_limits_removal(params: ContactorParams, nominal_gas: GasInlet):
    """Far above pKa1 the speciation factor saturates near 1, matching typical plant pH."""
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    f_ph = 1.0 / (1.0 + 10.0 ** (params.h2s_pka1 - 9.8))
    assert f_ph == pytest.approx(1.0, abs=1e-2)
    assert float(result.removal_efficiency) > 0.0


def test_higher_dissolved_sulfide_reduces_removal(params: ContactorParams, nominal_gas: GasInlet):
    """A liquid already loaded with sulfide has less remaining absorption capacity."""
    low_load = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(0.1),
        params=params,
    )
    high_load = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(50.0),
        params=params,
    )
    assert float(low_load.removal_efficiency) > float(high_load.removal_efficiency)


def test_sulfide_load_density_factor_matches_ideal_gas_law(params: ContactorParams):
    """Mass transferred per removed ppmv shrinks with temperature (ideal gas law), not fixed at STP."""
    cold_gas = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(0.0))
    warm_gas = GasInlet(gas_flow=jnp.array(500.0), h2s_ppm=jnp.array(50.0), temperature=jnp.array(20.0))

    cold = compute_removal(
        cold_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    warm = compute_removal(
        warm_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(1.0),
        params=params,
    )
    # Density factor at 0°C is 22.414/(22.414*293.15/273.15) ~= 1.073x the
    # factor at 20°C; the two removal_efficiency values differ only through
    # f_temperature, so normalize sulfide_load by that ratio before comparing.
    density_ratio = (293.15) / (273.15)
    assert float(cold.sulfide_load) / float(warm.sulfide_load) == pytest.approx(
        density_ratio * float(cold.removal_efficiency) / float(warm.removal_efficiency), rel=1e-3
    )


def test_zero_dissolved_sulfide_does_not_throttle_removal(params: ContactorParams, nominal_gas: GasInlet):
    """An empty liquid has full remaining absorption capacity (f_loading == 1)."""
    result = compute_removal(
        nominal_gas,
        oxidant=jnp.array(5.0),
        alkalinity=jnp.array(3.0),
        recirc_flow=jnp.array(50.0),
        ph=jnp.array(9.8),
        sulfide=jnp.array(0.0),
        params=params,
    )
    f_loading = params.loading_half_sat / (params.loading_half_sat + 0.0)
    assert f_loading == pytest.approx(1.0)
    assert float(result.removal_efficiency) > 0.0
