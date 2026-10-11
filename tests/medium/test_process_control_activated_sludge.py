from collections.abc import Iterator, Mapping

import jax
import jax.numpy as jnp
import pytest
from process_control.fields import Phase
from process_control.flowsheet import FlowsheetReport
from process_control.integration_records import AdvanceStatus
from process_control.kinetics import ZERO_CELSIUS
from process_control.plants.activated_sludge import BSM1_LAYOUT, activated_sludge_flowsheet, bsm1_open_loop_state
from process_control.scenarios.tabular_influent import bsm1_influent_profiles
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA, BSM1_PRESSURE_PA

# Alex et al. (2008) open-loop operating point, converted to hours.
OPEN_LOOP_SIGNALS = {
    "kla_1": 0.0,
    "kla_2": 0.0,
    "kla_3": 240.0 / 24.0,
    "kla_4": 240.0 / 24.0,
    "kla_5": 84.0 / 24.0,
    "q_a": 55338.0 / 24.0,
    "q_r": 18446.0 / 24.0,
    "q_underflow": (18446.0 + 385.0) / 24.0,
    "temperature": 15.0,
    "temperature_k": 288.15,
}
# Table 6, reactors 1-5 in ASM1 schema order.
TABLE_6 = (
    (30.0, 2.81, 1149.0, 82.1, 2552.0, 148.0, 449.0, 0.00430, 5.37, 7.92, 1.22, 5.28, 4.93),
    (30.0, 1.46, 1149.0, 76.4, 2553.0, 148.0, 450.0, 0.0000631, 3.66, 8.34, 0.882, 5.03, 5.08),
    (30.0, 1.15, 1149.0, 64.9, 2557.0, 149.0, 450.0, 1.72, 6.54, 5.55, 0.829, 4.39, 4.67),
    (30.0, 0.995, 1149.0, 55.7, 2559.0, 150.0, 451.0, 2.43, 9.30, 2.97, 0.767, 3.88, 4.29),
    (30.0, 0.889, 1149.0, 49.3, 2559.0, 150.0, 452.0, 0.491, 10.4, 1.73, 0.688, 3.53, 4.13),
)
# Table 7 layer TSS, bottom layer first.
TABLE_7_TSS = (6394.0, 356.0, 356.0, 356.0, 356.0, 356.0, 69.0, 29.5, 18.1, 12.5)
# Table 8 in ASM1 schema order.
TABLE_8_UNDERFLOW = (30.0, 0.889, 2247.0, 96.4, 5005.0, 293.0, 884.0, 0.491, 10.4, 1.73, 0.688, 6.90, 4.13)
TABLE_8_EFFLUENT = (30.0, 0.889, 4.39, 0.188, 9.78, 0.573, 1.73, 0.491, 10.4, 1.73, 0.688, 0.0135, 4.13)
# Three significant figures, and an absolute floor for the near-zero anoxic oxygen.
RTOL = 0.01
ATOL = 1e-3

FLOWSHEET = activated_sludge_flowsheet(BSM1_LAYOUT)
# RK4 substeps of 22.5 s; the anoxic oxygen balance goes unstable above about 30 s.
STEP_HOURS = 0.25
SUBSTEPS = 40


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _inlets() -> dict[str, Stream]:
    constant = bsm1_influent_profiles()["constant"]
    influent = Stream(
        flow=jnp.array(constant.flow[0]),
        components=jnp.array(constant.components[0]),
        temperature_k=jnp.array(constant.temperature[0] + ZERO_CELSIUS),
        pressure_pa=jnp.array(BSM1_PRESSURE_PA),
        schema=ASM1_SCHEMA,
        phase=Phase.LIQUID,
    )
    return {"influent": influent}


def _signals() -> dict[str, jax.Array]:
    return {name: jnp.array(value) for name, value in OPEN_LOOP_SIGNALS.items()}


def _run(state: Mapping[str, object], days: float) -> Mapping[str, object]:
    steps = round(days * 24.0 / STEP_HOURS)

    def step(_: int, carry: tuple[Mapping[str, object], jax.Array]) -> tuple[Mapping[str, object], jax.Array]:
        result = FLOWSHEET.advance(carry[0], _inlets(), _signals(), jnp.array(STEP_HOURS), SUBSTEPS, FLOWSHEET.initial_context({}))
        return result.state, carry[1] & (result.status == AdvanceStatus.SUCCESS)

    def run(start: Mapping[str, object]) -> tuple[Mapping[str, object], jax.Array]:
        return jax.lax.fori_loop(0, steps, step, (start, jnp.array(True)))

    end, succeeded = jax.jit(run)(state)
    assert bool(succeeded)
    return end


def _array(value: object) -> jax.Array:
    assert isinstance(value, jax.Array)
    return value


@pytest.fixture(scope="module")
def settled(float64: None) -> tuple[Mapping[str, object], FlowsheetReport]:
    state = _run(bsm1_open_loop_state(), days=10.0)
    return state, FLOWSHEET.report(state, _inlets(), _signals())


class TestOpenLoopSteadyState:
    def test_reactors_hold_table_6(self, settled: tuple[Mapping[str, object], FlowsheetReport]) -> None:
        """Ten days at the open-loop operating point leave every reactor at its BSM1 Table 6 composition.

        The reference state is a fixed point of the coupled plant, so a modelling error shows up as drift.
        """
        state, _ = settled
        reactors = jnp.stack([_array(state[name]) for name in BSM1_LAYOUT.reactors])

        assert reactors.shape == (5, 13)
        assert jnp.allclose(reactors, jnp.array(TABLE_6), rtol=RTOL, atol=ATOL)

    def test_settler_holds_table_7(self, settled: tuple[Mapping[str, object], FlowsheetReport]) -> None:
        """Layer solids keep the BSM1 Table 7 profile and every layer carries reactor 5's solubles."""
        state, _ = settled
        layers = _array(state["settler"])

        assert jnp.allclose(ASM1_SCHEMA.tss(layers), jnp.array(TABLE_7_TSS), rtol=RTOL)
        assert jnp.allclose(layers[:, ASM1_SCHEMA.soluble_mask], jnp.array(TABLE_6[-1])[ASM1_SCHEMA.soluble_mask], rtol=RTOL)

    def test_outlets_match_table_8(self, settled: tuple[Mapping[str, object], FlowsheetReport]) -> None:
        """The effluent and waste sludge carry the BSM1 Table 8 compositions at its flows."""
        _, report = settled
        effluent, waste = report.outlets["effluent"], report.outlets["waste"]

        assert jnp.allclose(effluent.components, jnp.array(TABLE_8_EFFLUENT), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(waste.components, jnp.array(TABLE_8_UNDERFLOW), rtol=RTOL)
        assert jnp.allclose(effluent.flow * 24.0, 18061.0)
        assert jnp.allclose(waste.flow * 24.0, 385.0)

    def test_splits_are_feasible(self, settled: tuple[Mapping[str, object], FlowsheetReport]) -> None:
        """At the open-loop flows both splitters honour their draws."""
        _, report = settled

        assert all(float(deficit) == 0.0 for deficit in report.split_deficits.values())
