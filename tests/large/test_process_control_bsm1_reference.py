"""BSM1 reference results of Alex et al. (2008), Appendices 4 and 5, reproduced by the configurable benchmark.

https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

import dataclasses
from collections.abc import Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.benchmarks.bsm1 import (
    BSM1Handles,
    BSM1State,
    Fixed,
    OperatingPoint,
    StepFn,
    bsm1_reference_closed_loop,
    make_bsm1_benchmark,
)
from process_control.plants.activated_sludge import BSM1_LAYOUT, BSM1_OPEN_LOOP_REACTORS, activated_sludge_flowsheet, settled_state
from process_control.scenarios.tabular_influent import bsm1_influent_profiles
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA

STEPS_PER_DAY = 96
# The reference evaluates days 7-14 of a dry-weather run that itself starts from 14 dry days.
DRY_DAYS = 28
EVALUATION = slice(21 * STEPS_PER_DAY, 28 * STEPS_PER_DAY)

# Appendix 4, open loop, dry weather: flow-weighted effluent averages (g/m³) and indices (per day).
OPEN_LOOP_DRY = {
    "effluent_s_nh": 4.7589,
    "effluent_s_no": 8.8238,
    "effluent_tss": 12.9917,
    "effluent_tkn": 6.7448,
    "effluent_n_tot": 15.5686,
    "effluent_cod": 48.2958,
    "effluent_bod5": 2.7746,
    "effluent_quality": 6690.1066,
    "aeration_energy": 3341.3867,
    "pumping_energy": 388.17,
    "sludge_production": 2435.6901,
    "mixing_energy": 240.0,
    "operational_cost": 16148.0073,
}
# Appendix 5, closed loop, dry weather.
CLOSED_LOOP_DRY = {
    "effluent_s_nh": 2.5392,
    "effluent_s_no": 12.4199,
    "effluent_tss": 13.0038,
    "effluent_tkn": 4.5046,
    "effluent_n_tot": 16.9245,
    "effluent_cod": 48.2201,
    "effluent_bod5": 2.7568,
    "effluent_quality": 6123.0182,
    "aeration_energy": 3698.3438,
    "pumping_energy": 241.0305,
    "sludge_production": 2440.6057,
    "mixing_energy": 240.0,
    "operational_cost": 16382.4027,
}
# Appendix 5, closed-loop steady state under constant influent and ideal sensors: reactors 1-5.
CLOSED_LOOP_STEADY_STATE = (
    (30.0, 3.2439, 1149.1683, 98.6029, 2552.1095, 151.6721, 446.9249, 0.0076964, 3.5133, 11.8312, 1.3621, 6.1775, 5.3399),
    (30.0, 1.6707, 1149.1683, 91.7032, 2552.3711, 151.5303, 448.0839, 6.0271e-05, 1.0, 12.5482, 0.78899, 5.9537, 5.5706),
    (30.0, 1.2195, 1149.1683, 69.6594, 2560.2025, 152.6873, 449.6336, 1.635, 6.2289, 7.3197, 0.8307, 4.7131, 4.8236),
    (30.0, 0.97326, 1149.1683, 54.4484, 2563.3104, 153.7108, 451.1853, 2.4745, 11.0693, 2.7825, 0.75276, 3.8403, 4.1538),
    (30.0, 0.80801, 1149.1683, 44.4828, 2562.8514, 154.163, 452.7367, 2.0, 13.5243, 0.67193, 0.6645, 3.2605, 3.8277),
)
# Open loop is deterministic, so only the 15-minute sampling of the reference evaluation separates the two.
OPEN_LOOP_RTOL = 0.005
# Closed loop adds sensor noise drawn differently from the reference's noise file; the controllers average
# it out over the week, leaving differences an order of magnitude below this.
CLOSED_LOOP_RTOL = 0.01


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _run(step: StepFn, state: BSM1State, days: int) -> tuple[BSM1State, dict[str, jax.Array]]:
    def body(current: BSM1State, key: jax.Array) -> tuple[BSM1State, dict[str, jax.Array]]:
        current, _, _, _, info = step(current, jnp.zeros(0), key)
        return current, info

    def run(start: BSM1State) -> tuple[BSM1State, dict[str, jax.Array]]:
        return jax.lax.scan(body, start, jax.random.split(jax.random.key(7), days * STEPS_PER_DAY))

    return jax.jit(run)(state)


def _evaluate(info: Mapping[str, jax.Array]) -> dict[str, float]:
    """Flow-weighted effluent concentrations and mean indices over the evaluation week."""
    window = {name: np.asarray(values[EVALUATION]) for name, values in info.items()}
    flow = window["effluent_flow"]
    concentrations = [name for name in OPEN_LOOP_DRY if name.startswith("effluent_") and name != "effluent_quality"]
    weighted = {name: float(np.sum(window[name] * flow) / np.sum(flow)) for name in concentrations}
    return weighted | {name: float(np.mean(window[name])) for name in OPEN_LOOP_DRY if name not in weighted}


def _array(value: object) -> jax.Array:
    assert isinstance(value, jax.Array)
    return value


def _relative_errors(measured: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {name: abs(measured[name] - value) / value for name, value in reference.items()}


class TestOpenLoop:
    def test_dry_weather_matches_appendix_4(self) -> None:
        """Every handle at the open-loop point, the dry-weather week reproduces the open-loop effluent and cost indices."""
        handles = BSM1Handles(**{name: Fixed(value) for name, value in OperatingPoint().values().items()})
        config = dataclasses.replace(bsm1_reference_closed_loop("dry"), handles=handles)
        reset, step = make_bsm1_benchmark(config)

        _, info = _run(step, reset(jax.random.key(0))[0], DRY_DAYS)

        errors = _relative_errors(_evaluate(info), OPEN_LOOP_DRY)
        assert max(errors.values()) < OPEN_LOOP_RTOL, errors

    def test_constant_influent_settles_to_tables_6_to_8_from_anywhere(self) -> None:
        """From a uniform plant, 150 days of the stabilisation influent reach the published open-loop steady state."""
        flowsheet = activated_sludge_flowsheet(BSM1_LAYOUT)
        constant = bsm1_influent_profiles()["constant"]
        inlets = {"influent": Stream(flow=jnp.array(constant.flow[0]), components=jnp.array(constant.components[0]), schema=ASM1_SCHEMA)}
        signals = {name: jnp.asarray(value) for name, value in OperatingPoint().values().items() if name != "q_w"}
        signals |= {"q_underflow": signals["q_r"] + 385.0 / 24.0, "temperature": jnp.array(15.0)}
        uniform = jnp.tile(jnp.array([30.0, 5.0, 1000.0, 100.0, 2000.0, 100.0, 400.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0]), (5, 1))
        start = settled_state(BSM1_LAYOUT, uniform, jnp.full(10, 1000.0))

        def hour(_: int, state: Mapping[str, object]) -> Mapping[str, object]:
            return flowsheet.step(state, inlets, signals, jnp.array(1.0), 240)

        def settle(state: Mapping[str, object]) -> Mapping[str, object]:
            return jax.lax.fori_loop(0, 150 * 24, hour, state)

        end = jax.jit(settle)(start)

        reactors = jnp.stack([_array(end[name]) for name in BSM1_LAYOUT.reactors])
        assert jnp.allclose(reactors, jnp.array(BSM1_OPEN_LOOP_REACTORS), rtol=0.002, atol=1e-4)


@pytest.fixture(scope="module")
def stabilised(float64: None) -> BSM1State:
    """The default closed loop after 100 days of constant influent with ideal sensors."""
    reset, step = make_bsm1_benchmark(bsm1_reference_closed_loop("constant", sensor_noise=False))
    end, _ = _run(step, reset(jax.random.key(0))[0], 100)
    return end


class TestClosedLoop:
    def test_stabilisation_reaches_the_closed_loop_steady_state(self, stabilised: BSM1State) -> None:
        """The default controllers settle the plant at the closed-loop steady state of Appendix 5."""
        reactors = jnp.stack([stabilised.plant[name] for name in BSM1_LAYOUT.reactors])

        assert jnp.allclose(reactors, jnp.array(CLOSED_LOOP_STEADY_STATE), rtol=0.002, atol=1e-4)

    def test_dry_weather_matches_appendix_5(self, stabilised: BSM1State) -> None:
        """From the stabilised plant, the noisy dry-weather week reproduces the closed-loop effluent and cost indices."""
        reset, step = make_bsm1_benchmark(bsm1_reference_closed_loop("dry"))
        fresh, _ = reset(jax.random.key(3))
        start = dataclasses.replace(
            stabilised,
            influent=fresh.influent,
            time=fresh.time,
            controller_sensors={
                name: dataclasses.replace(sensor, noise_key=fresh.controller_sensors[name].noise_key)
                for name, sensor in stabilised.controller_sensors.items()
            },
        )

        _, info = _run(step, start, DRY_DAYS)

        errors = _relative_errors(_evaluate(info), CLOSED_LOOP_DRY)
        assert max(errors.values()) < CLOSED_LOOP_RTOL, errors
        assert np.allclose(np.mean(np.asarray(info["q_a"][EVALUATION])) * 24.0, 18610.0822, rtol=0.01)
        assert np.allclose(np.mean(np.asarray(info["kla"][EVALUATION, 4])) * 24.0, 144.1219, rtol=0.01)
