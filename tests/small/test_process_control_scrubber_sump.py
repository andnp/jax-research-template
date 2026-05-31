"""Tests for the scrubber sump chemistry model."""

import jax
import jax.numpy as jnp
import pytest
from process_control.units.scrubber_sump import (
    ScrubberSumpParams,
    ScrubberSumpState,
    SumpInputs,
    compute_orp,
    compute_ph,
    reset,
    step,
)


@pytest.fixture
def params() -> ScrubberSumpParams:
    return ScrubberSumpParams()


@pytest.fixture
def rng() -> jax.Array:
    return jax.random.key(42)


@pytest.fixture
def zero_inputs() -> SumpInputs:
    return SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )


def test_reset_produces_nominal_volume(params: ScrubberSumpParams, rng: jax.Array):
    state = reset(rng, params)
    assert jnp.isclose(state.volume, params.nominal_volume)


def test_reset_state_shapes(params: ScrubberSumpParams, rng: jax.Array):
    state = reset(rng, params)
    for field in ["oxidant", "alkalinity", "volume", "sulfide", "temperature"]:
        arr = getattr(state, field)
        assert arr.shape == (), f"{field} should be scalar"


def test_zero_inputs_oxidant_decays(params: ScrubberSumpParams, rng: jax.Array):
    """With no dosing or load, oxidant should slowly decay."""
    state = reset(rng, params)
    dt = jnp.array(1.0 / 12.0)  # 5-minute step in hours

    for _ in range(100):
        state = step(
            state,
            SumpInputs(
                bleach_flow=jnp.array(0.0),
                caustic_flow=jnp.array(0.0),
                makeup_flow=jnp.array(0.0),
                sulfide_load=jnp.array(0.0),
            ),
            params,
            dt,
        )

    assert state.oxidant < 5.0, "Oxidant should decay from initial 5.0"
    assert state.oxidant >= 0.0, "Oxidant should not go negative"


def test_bleach_raises_oxidant(params: ScrubberSumpParams, rng: jax.Array):
    """Bleach dosing should increase oxidant concentration."""
    state = ScrubberSumpState.create(oxidant=1.0, sulfide=0.0)
    dt = jnp.array(1.0 / 12.0)

    dosing_inputs = SumpInputs(
        bleach_flow=jnp.array(50.0),  # 50 mL/min bleach
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(10):
        state = step(state, dosing_inputs, params, dt)

    assert state.oxidant > 1.0, "Bleach should raise oxidant"


def test_caustic_raises_alkalinity(params: ScrubberSumpParams, rng: jax.Array):
    """Caustic dosing should increase alkalinity."""
    state = ScrubberSumpState.create(alkalinity=1.0, sulfide=0.0)
    dt = jnp.array(1.0 / 12.0)

    dosing_inputs = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(30.0),  # 30 mL/min caustic
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(10):
        state = step(state, dosing_inputs, params, dt)

    assert state.alkalinity > 1.0, "Caustic should raise alkalinity"


def test_sulfide_load_increases_sulfide(params: ScrubberSumpParams, rng: jax.Array):
    """Sulfide entering from contactor should raise dissolved sulfide."""
    state = ScrubberSumpState.create(sulfide=0.1, oxidant=0.0)
    dt = jnp.array(1.0 / 12.0)

    loading_inputs = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(10.0),  # 10 mg/min sulfide
    )

    for _ in range(10):
        state = step(state, loading_inputs, params, dt)

    assert state.sulfide > 0.1, "Sulfide load should increase dissolved sulfide"


def test_oxidant_consumes_sulfide(params: ScrubberSumpParams, rng: jax.Array):
    """When both oxidant and sulfide are present, oxidation should consume both."""
    state = ScrubberSumpState.create(oxidant=10.0, sulfide=5.0)
    dt = jnp.array(1.0 / 12.0)

    zero = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(50):
        state = step(state, zero, params, dt)

    assert state.sulfide < 5.0, "Oxidation should consume sulfide"
    assert state.oxidant < 10.0, "Oxidation should consume oxidant"


def test_makeup_dilutes_concentrations(params: ScrubberSumpParams, rng: jax.Array):
    """Makeup water should dilute dissolved species."""
    state = ScrubberSumpState.create(oxidant=10.0, alkalinity=5.0, sulfide=3.0)
    dt = jnp.array(1.0 / 12.0)

    makeup_inputs = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(20.0),  # 20 L/min makeup
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(20):
        state = step(state, makeup_inputs, params, dt)

    # All concentrations should decrease due to dilution
    assert state.oxidant < 10.0, "Makeup should dilute oxidant"
    assert state.alkalinity < 5.0, "Makeup should dilute alkalinity"


def test_volume_overflow(params: ScrubberSumpParams, rng: jax.Array):
    """Volume should not grow unbounded — overflow drains excess."""
    state = ScrubberSumpState.create(volume=params.volume_max)
    dt = jnp.array(1.0 / 12.0)

    high_makeup = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(100.0),  # very high makeup
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(50):
        state = step(state, high_makeup, params, dt)

    # Volume should stay near max, not grow to infinity
    assert state.volume <= params.volume_max * 1.5 + 0.1


def test_non_negative_concentrations(params: ScrubberSumpParams, rng: jax.Array):
    """All concentrations should remain non-negative under extreme conditions."""
    state = ScrubberSumpState.create(oxidant=0.01, alkalinity=0.01, sulfide=100.0)
    dt = jnp.array(1.0 / 12.0)

    zero = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(200):
        state = step(state, zero, params, dt)

    assert state.oxidant >= 0.0
    assert state.alkalinity >= 0.0
    assert state.sulfide >= 0.0


def test_ph_monotonic_in_alkalinity():
    """Higher alkalinity should give higher pH."""
    low = ScrubberSumpState.create(alkalinity=0.5)
    mid = ScrubberSumpState.create(alkalinity=3.0)
    high = ScrubberSumpState.create(alkalinity=8.0)

    assert compute_ph(low) < compute_ph(mid) < compute_ph(high)


def test_orp_increases_with_oxidant():
    """Higher oxidant (at constant sulfide) should give higher ORP."""
    low = ScrubberSumpState.create(oxidant=1.0, sulfide=1.0)
    high = ScrubberSumpState.create(oxidant=10.0, sulfide=1.0)

    assert compute_orp(low) < compute_orp(high)


def test_orp_decreases_with_sulfide():
    """Higher sulfide (at constant oxidant) should give lower ORP."""
    low_s = ScrubberSumpState.create(oxidant=5.0, sulfide=0.5)
    high_s = ScrubberSumpState.create(oxidant=5.0, sulfide=5.0)

    assert compute_orp(low_s) > compute_orp(high_s)


def test_temperature_relaxes_to_ambient(params: ScrubberSumpParams, rng: jax.Array):
    """Temperature should drift toward ambient when no reactions occur."""
    state = ScrubberSumpState.create(temperature=40.0, oxidant=0.0, sulfide=0.0)
    dt = jnp.array(1.0 / 12.0)

    zero = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    for _ in range(500):
        state = step(state, zero, params, dt)

    assert abs(float(state.temperature) - params.ambient_temp) < 1.0


def test_jit_compatible(params: ScrubberSumpParams, rng: jax.Array):
    """The step function should be JIT-compilable."""
    state = reset(rng, params)
    dt = jnp.array(1.0 / 12.0)
    inputs = SumpInputs(
        bleach_flow=jnp.array(10.0),
        caustic_flow=jnp.array(5.0),
        makeup_flow=jnp.array(2.0),
        sulfide_load=jnp.array(1.0),
    )

    jit_step = jax.jit(lambda s: step(s, inputs, params, dt))
    result = jit_step(state)

    # Should produce valid output
    assert jnp.isfinite(result.oxidant)
    assert jnp.isfinite(result.sulfide)
