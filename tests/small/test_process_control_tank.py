import jax
import jax.numpy as jnp
from process_control.units.tank import TankParams, TankState, step, step_with_result

PARAMS = TankParams(max_level=10.0, min_level=2.0, cross_section_area=5.0)


def test_step_preserves_state_only_api() -> None:
    state = TankState.create(5.0)

    new_state = step(state, jnp.array(4.0), jnp.array(3.0), PARAMS, jnp.array(2.0))

    assert jnp.isclose(new_state.level, 5.4)


def test_outlet_is_limited_by_available_inventory() -> None:
    result = step_with_result(
        TankState.create(2.0),
        jnp.array(3.0),
        jnp.array(10.0),
        PARAMS,
        jnp.array(2.0),
    )

    assert jnp.isclose(result.state.level, 2.0)
    assert jnp.isclose(result.realized_outlet_flow, 3.0)
    assert jnp.isclose(result.unmet_outlet_flow, 7.0)
    assert jnp.isclose(result.overflow_flow, 0.0)
    assert result.constraint_status == -1


def test_excess_inlet_is_reported_as_overflow() -> None:
    result = step_with_result(
        TankState.create(9.0),
        jnp.array(8.0),
        jnp.array(1.0),
        PARAMS,
        jnp.array(1.0),
    )

    assert jnp.isclose(result.state.level, 10.0)
    assert jnp.isclose(result.realized_outlet_flow, 1.0)
    assert jnp.isclose(result.overflow_flow, 2.0)
    assert jnp.isclose(result.unmet_outlet_flow, 0.0)
    assert result.constraint_status == 1


def test_cumulative_mass_balance_over_sequence() -> None:
    state = TankState.create(6.0)
    initial_volume = (state.level - PARAMS.min_level) * PARAMS.cross_section_area
    inlet_flows = jnp.array([3.0, 20.0, 0.0, 2.0, 1.0])
    requested_outlets = jnp.array([1.0, 0.0, 30.0, 1.0, 5.0])
    dt = jnp.array(1.0)
    total_inlet = jnp.array(0.0)
    total_outlet = jnp.array(0.0)
    total_overflow = jnp.array(0.0)

    for inlet, requested_outlet in zip(inlet_flows, requested_outlets, strict=True):
        result = step_with_result(state, inlet, requested_outlet, PARAMS, dt)
        state = result.state
        total_inlet += inlet * dt
        total_outlet += result.realized_outlet_flow * dt
        total_overflow += result.overflow_flow * dt

    final_volume = (state.level - PARAMS.min_level) * PARAMS.cross_section_area
    assert jnp.isclose(initial_volume + total_inlet, final_volume + total_outlet + total_overflow)


def test_step_result_is_jit_compatible() -> None:
    jit_step = jax.jit(step_with_result, static_argnames=("params",))

    result = jit_step(
        TankState.create(9.0),
        jnp.array(8.0),
        jnp.array(1.0),
        PARAMS,
        jnp.array(1.0),
    )

    assert jnp.isclose(result.state.level, 10.0)
    assert jnp.isclose(result.overflow_flow, 2.0)
