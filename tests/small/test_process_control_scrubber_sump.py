"""Tests for the scrubber sump chemistry model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from process_control.integration import PROJECTION
from process_control.integration_records import AdvanceResult, AdvanceStatus
from process_control.units.scrubber_sump import (
    CLAMP_OVERFLOW,
    ScrubberSumpParams,
    ScrubberSumpState,
    SumpInputs,
    compute_orp,
    compute_ph,
    reset,
    step,
)


def _accepted(result: AdvanceResult[ScrubberSumpState]) -> ScrubberSumpState:
    assert int(result.status) == AdvanceStatus.SUCCESS
    return result.state


@jax.jit
def _scan(
    state: ScrubberSumpState,
    flows: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    params: ScrubberSumpParams,
    dt: jax.Array,
    n_steps: jax.Array | int,
) -> tuple[ScrubberSumpState, jax.Array]:
    inputs = SumpInputs(*flows)

    def body(_step: jax.Array, carry: tuple[ScrubberSumpState, jax.Array]) -> tuple[ScrubberSumpState, jax.Array]:
        result = step(carry[0], inputs, params, dt)
        return result.state, carry[1] | (result.status != AdvanceStatus.SUCCESS)

    return jax.lax.fori_loop(0, n_steps, body, (state, jnp.array(False)))


def _run(
    state: ScrubberSumpState, inputs: SumpInputs, params: ScrubberSumpParams, dt: jax.Array, n_steps: int
) -> ScrubberSumpState:
    flows = (inputs.bleach_flow, inputs.caustic_flow, inputs.makeup_flow, inputs.sulfide_load)
    final, failed = _scan(state, flows, params, dt, n_steps)
    assert not bool(failed)
    return final


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

    state = _run(
        state,
        SumpInputs(
            bleach_flow=jnp.array(0.0),
            caustic_flow=jnp.array(0.0),
            makeup_flow=jnp.array(0.0),
            sulfide_load=jnp.array(0.0),
        ),
        params,
        dt,
        100,
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
        state = _accepted(step(state, dosing_inputs, params, dt))

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
        state = _accepted(step(state, dosing_inputs, params, dt))

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
        state = _accepted(step(state, loading_inputs, params, dt))

    assert state.sulfide > 0.1, "Sulfide load should increase dissolved sulfide"


def test_stripping_throttled_at_high_ph(params: ScrubberSumpParams, rng: jax.Array):
    """Only un-ionized H2S(aq) strips; at high pH almost all sulfide is HS-, so stripping stalls."""
    no_oxidation = replace(params, oxidation_rate=0.0)
    dt = jnp.array(1.0 / 12.0)
    zero = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    low_ph_state = ScrubberSumpState.create(sulfide=5.0, alkalinity=0.0, oxidant=0.0)
    high_ph_state = ScrubberSumpState.create(sulfide=5.0, alkalinity=50.0, oxidant=0.0)
    for _ in range(20):
        low_ph_state = _accepted(step(low_ph_state, zero, no_oxidation, dt))
        high_ph_state = _accepted(step(high_ph_state, zero, no_oxidation, dt))

    assert low_ph_state.sulfide < high_ph_state.sulfide, (
        "Low-pH (more un-ionized H2S) sulfide should strip out faster than high-pH sulfide"
    )


def test_oxidation_consumes_alkalinity_independent_of_absorption(params: ScrubberSumpParams, rng: jax.Array):
    """Oxidizing sulfide to sulfate releases acid that must draw down alkalinity too."""
    dt = jnp.array(1.0 / 12.0)
    zero_load = SumpInputs(
        bleach_flow=jnp.array(0.0),
        caustic_flow=jnp.array(0.0),
        makeup_flow=jnp.array(0.0),
        sulfide_load=jnp.array(0.0),
    )

    oxidizing = ScrubberSumpState.create(oxidant=10.0, sulfide=5.0, alkalinity=10.0)
    not_oxidizing = ScrubberSumpState.create(oxidant=0.0, sulfide=5.0, alkalinity=10.0)
    for _ in range(20):
        oxidizing = _accepted(step(oxidizing, zero_load, params, dt))
        not_oxidizing = _accepted(step(not_oxidizing, zero_load, params, dt))

    assert oxidizing.alkalinity < not_oxidizing.alkalinity, (
        "Active oxidation must consume more alkalinity than absorption alone"
    )


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
        state = _accepted(step(state, zero, params, dt))

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
        state = _accepted(step(state, makeup_inputs, params, dt))

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
        makeup_flow=jnp.array(1000.0),  # very high makeup
        sulfide_load=jnp.array(0.0),
    )

    state = _run(state, high_makeup, params, dt, 50)

    # Volume should stay near max, not grow to infinity
    assert state.volume <= params.volume_max * 1.5 + 0.1

    overflow = step(
        ScrubberSumpState.create(volume=params.volume_max * 1.5), high_makeup, params, dt,
    ).quadratures[PROJECTION][CLAMP_OVERFLOW]
    assert overflow > 0.0

    no_overflow = step(
        ScrubberSumpState.create(volume=params.volume_max), SumpInputs(
            bleach_flow=jnp.array(0.0), caustic_flow=jnp.array(0.0),
            makeup_flow=jnp.array(0.0), sulfide_load=jnp.array(0.0),
        ), params, dt,
    ).quadratures[PROJECTION][CLAMP_OVERFLOW]
    assert no_overflow == 0.0


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

    state = _run(state, zero, params, dt, 200)

    assert state.oxidant >= 0.0
    assert state.alkalinity >= 0.0
    assert state.sulfide >= 0.0


def test_more_substeps_refines_without_changing_a_well_behaved_answer(
    params: ScrubberSumpParams, rng: jax.Array
) -> None:
    """Sub-stepping must refine, not alter, the integrated result for ordinary dynamics."""
    inputs = SumpInputs(
        bleach_flow=jnp.array(10.0),
        caustic_flow=jnp.array(5.0),
        makeup_flow=jnp.array(20.0),
        sulfide_load=jnp.array(50.0),
    )
    dt = jnp.array(1.0 / 12.0)
    initial = reset(rng, params)

    coarse = _accepted(step(initial, inputs, params, dt, substeps=1))
    fine = _accepted(step(initial, inputs, params, dt, substeps=50))

    assert jnp.allclose(coarse.alkalinity, fine.alkalinity, atol=1e-2)
    # oxidant_sulfide_rate's corrected (sulfate-route) value makes oxidant
    # consumption per unit of oxidation steeper, widening the coarse-vs-fine
    # gap at substeps=1; still tight relative to the ~2.0 oxidant scale here.
    assert jnp.allclose(coarse.oxidant, fine.oxidant, atol=0.25)
    assert jnp.allclose(coarse.sulfide, fine.sulfide, atol=0.05)


def test_ph_monotonic_in_alkalinity():
    """Higher alkalinity should give higher pH."""
    low = ScrubberSumpState.create(alkalinity=0.5)
    mid = ScrubberSumpState.create(alkalinity=3.0)
    high = ScrubberSumpState.create(alkalinity=8.0)

    assert compute_ph(low) < compute_ph(mid) < compute_ph(high)


def test_ph_amplitude_raises_the_saturation_ceiling():
    """A wider amplitude must raise the achievable pH ceiling, not just rescale it."""
    high = ScrubberSumpState.create(alkalinity=50.0)

    assert compute_ph(high, amplitude=3.0) < compute_ph(high, amplitude=7.0)


def test_orp_increases_with_oxidant(params: ScrubberSumpParams):
    """Higher oxidant (at constant sulfide) should give higher ORP."""
    low = ScrubberSumpState.create(oxidant=1.0, sulfide=1.0)
    high = ScrubberSumpState.create(oxidant=10.0, sulfide=1.0)
    ph = jnp.array(params.orp_ph_reference)

    assert compute_orp(low, ph, params) < compute_orp(high, ph, params)


def test_orp_decreases_with_sulfide(params: ScrubberSumpParams):
    """Higher sulfide (at constant oxidant) should give lower ORP."""
    low_s = ScrubberSumpState.create(oxidant=5.0, sulfide=0.5)
    high_s = ScrubberSumpState.create(oxidant=5.0, sulfide=5.0)
    ph = jnp.array(params.orp_ph_reference)

    assert compute_orp(low_s, ph, params) > compute_orp(high_s, ph, params)


def test_orp_decreases_with_ph(params: ScrubberSumpParams):
    """Real hypochlorite ORP falls as pH rises for fixed oxidant (Nernst pH term)."""
    state = ScrubberSumpState.create(oxidant=5.0, sulfide=1.0)

    assert compute_orp(state, jnp.array(11.0), params) < compute_orp(state, jnp.array(8.0), params)


def test_orp_matches_baseline_at_ph_reference(params: ScrubberSumpParams):
    """At orp_ph_reference the pH term must vanish, leaving the original oxidant/sulfide formula."""
    state = ScrubberSumpState.create(oxidant=5.0, sulfide=1.0)
    ratio = (state.oxidant + 0.01) / (state.sulfide + 0.01)
    expected = 450.0 + 100.0 * jnp.log10(ratio)

    result = compute_orp(state, jnp.array(params.orp_ph_reference), params)

    assert jnp.allclose(result, expected)


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

    state = _run(state, zero, params, dt, 500)

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

    def step_fn(s: ScrubberSumpState):
        return step(s, inputs, params, dt)

    result = _accepted(jax.jit(step_fn)(state))

    # Should produce valid output
    assert jnp.isfinite(result.oxidant)
    assert jnp.isfinite(result.sulfide)
