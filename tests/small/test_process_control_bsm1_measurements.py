import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1_measurements import PlantSnapshot, blanket_height, measurement, stream_qualities, take_snapshot
from process_control.plants.activated_sludge import BSM1_LAYOUT, activated_sludge_flowsheet, bsm1_open_loop_state
from process_control.scenarios.tabular_influent import bsm1_influent_profiles
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Params
from process_control.units.layered_settler import BSM1_SETTLER

THRESHOLD = 1500.0
# Alex et al. (2008) Table 7 layer TSS, bottom first.
BSM1_PROFILE = jnp.array([6394.0, 356.0, 356.0, 356.0, 356.0, 356.0, 69.0, 29.5, 18.1, 12.5])
HANDLES = ("kla_5", "q_a")


class TestBlanketHeight:
    def test_interpolates_the_threshold_crossing(self) -> None:
        """The BSM1 profile crosses 1500 g/m³ between the centres of the two bottom layers."""
        expected = 0.2 + (6394.0 - THRESHOLD) / (6394.0 - 356.0) * 0.4

        assert jnp.allclose(blanket_height(BSM1_PROFILE, BSM1_SETTLER, THRESHOLD), expected)

    def test_rises_continuously_as_a_layer_thickens(self) -> None:
        """Thickening the second layer through the threshold moves the blanket up without jumps."""
        second_layer = jnp.linspace(1000.0, 2000.0, 201)
        heights = jnp.array([blanket_height(BSM1_PROFILE.at[1].set(tss), BSM1_SETTLER, THRESHOLD) for tss in second_layer])

        assert jnp.all(jnp.diff(heights) >= 0.0)
        assert jnp.max(jnp.diff(heights)) < 0.01

    def test_thin_settler_shrinks_toward_the_floor(self) -> None:
        """With no layer at the threshold the blanket scales with the bottom layer's solids."""
        thin = BSM1_PROFILE.at[0].set(750.0)

        assert jnp.allclose(blanket_height(thin, BSM1_SETTLER, THRESHOLD), 0.1)

    def test_full_settler_reports_the_top_layer(self) -> None:
        """A settler thick to the top reports the top layer's centre."""
        full = jnp.full(10, 3000.0)

        assert jnp.allclose(blanket_height(full, BSM1_SETTLER, THRESHOLD), 3.8)


class TestStreamQualities:
    def test_aggregates_follow_the_bsm1_definitions(self) -> None:
        """TSS, COD, BOD5, TKN and total N match the BSM1 effluent formulas by hand."""
        effluent = jnp.array([30.0, 0.889, 4.39, 0.188, 9.78, 0.573, 1.73, 0.491, 10.4, 1.73, 0.688, 0.0135, 4.13])
        qualities = stream_qualities(effluent, ASM1Params(volume=1.0))

        biomass = 9.78 + 0.573
        tkn = 1.73 + 0.688 + 0.0135 + 0.08 * biomass + 0.06 * (1.73 + 4.39)
        assert jnp.allclose(qualities["tss"], 0.75 * (4.39 + 0.188 + biomass + 1.73))
        assert jnp.allclose(qualities["cod"], 30.0 + 0.889 + 4.39 + 0.188 + biomass + 1.73)
        assert jnp.allclose(qualities["bod5"], 0.25 * (0.889 + 0.188 + 0.92 * biomass))
        assert jnp.allclose(qualities["tkn"], tkn)
        assert jnp.allclose(qualities["n_tot"], tkn + 10.4)
        assert jnp.allclose(qualities["s_nh"], 1.73)


@pytest.fixture(scope="module")
def open_loop() -> PlantSnapshot:
    constant = bsm1_influent_profiles()["constant"]
    influent = Stream(flow=jnp.array(constant.flow[0]), components=jnp.array(constant.components[0]), schema=ASM1_SCHEMA)
    signals = {
        **{f"kla_{index}": jnp.array(value) for index, value in enumerate((0.0, 0.0, 10.0, 10.0, 3.5), start=1)},
        "q_a": jnp.array(55338.0 / 24.0),
        "q_r": jnp.array(18446.0 / 24.0),
        "q_underflow": jnp.array(18831.0 / 24.0),
        "temperature": jnp.array(15.0),
    }
    handles = {"kla_5": jnp.array(3.5), "q_a": jnp.array(55338.0 / 24.0)}
    return take_snapshot(
        activated_sludge_flowsheet(BSM1_LAYOUT), BSM1_LAYOUT, bsm1_open_loop_state(), influent, signals, handles, {"q_a": jnp.array(1.0)}
    )


def _measure(quantity: str, snapshot: PlantSnapshot):
    return measurement(quantity, BSM1_LAYOUT, THRESHOLD, HANDLES, ("q_a",))(snapshot)


class TestCatalog:
    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            ("s_o_5", 0.49094),
            ("s_no_2", 3.6619),
            ("tss_1", 3285.188),
            ("mlss", 3269.8246),
            ("effluent_flow", 18061.0 / 24.0),
            ("effluent_tss", 12.4969),
            ("underflow_tss", 6393.9599),
            ("underflow_flow", 18831.0 / 24.0),
            ("influent_s_nh", 31.56),
            ("temperature", 15.0),
        ],
    )
    def test_reads_the_open_loop_steady_state(self, open_loop: PlantSnapshot, quantity: str, expected: float) -> None:
        """Named quantities read the BSM1 open-loop steady state of Alex et al. (2008), Appendix 4."""
        assert jnp.allclose(_measure(quantity, open_loop), expected, rtol=1e-3)

    def test_sludge_age_matches_the_reference(self, open_loop: PlantSnapshot) -> None:
        """Reactor solids over the solids leaving give the 7.3155-day traditional sludge age."""
        assert jnp.allclose(_measure("sludge_age", open_loop), 7.3155 * 24.0, rtol=1e-3)

    def test_reads_realised_handles_and_setpoints(self, open_loop: PlantSnapshot) -> None:
        """Handle and setpoint quantities report the values the snapshot carries."""
        assert jnp.allclose(_measure("kla_5", open_loop), 3.5)
        assert jnp.allclose(_measure("q_a_setpoint", open_loop), 1.0)

    def test_open_loop_splits_are_feasible(self, open_loop: PlantSnapshot) -> None:
        """No splitter is overdrawn at the open-loop flows."""
        assert float(open_loop.split_deficit) == 0.0

    @pytest.mark.parametrize("quantity", ["s_o_6", "kla_3", "q_r_setpoint", "effluent_oxygen"])
    def test_rejects_unknown_quantities(self, quantity: str) -> None:
        """Quantities the plant does not offer are refused, naming the alternatives."""
        with pytest.raises(ValueError, match=rf"unknown plant quantity '{quantity}'.*s_o_5"):
            measurement(quantity, BSM1_LAYOUT, THRESHOLD, HANDLES, ("q_a",))
