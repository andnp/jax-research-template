from dataclasses import dataclass

import jax
import jax.numpy as jnp
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.flowsheet import Node, Port, build_flowsheet
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Kinetics, ASM1Params, bsm1_oxygen_saturation
from process_control.units.bioreactor import AeratedBioreactor, BioreactorParams
from process_control.units.contract import step_unit

TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")

SCHEMA = ComponentSchema(names=("tracer", "oxygen"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER,) * 2)
SATURATION = 10.0


@dataclass(frozen=True)
class InertKinetics:
    """No reactions: isolates transport and aeration."""

    @property
    def schema(self) -> ComponentSchema:
        return SCHEMA

    @property
    def oxygen(self) -> str:
        return "oxygen"

    def at_temperature(self, params: None, temperature: jax.Array) -> None:
        return None

    def rates(self, components: jax.Array, params: None) -> jax.Array:
        return jnp.zeros_like(components)


def _constant_saturation(temperature: jax.Array) -> jax.Array:
    return jnp.full_like(temperature, SATURATION)


INERT = AeratedBioreactor(InertKinetics(), oxygen_saturation=_constant_saturation)
INERT_PARAMS = BioreactorParams(volume=100.0, kinetics=None)


def _feed(flow: float, components: jax.Array, schema: ComponentSchema = SCHEMA) -> dict[str, Stream]:
    return {"in": Stream(flow=jnp.array(flow), components=components, schema=schema)}


def _signals(kla: float | jax.Array, temperature: float = 15.0) -> dict[str, jax.Array]:
    return {"kla": jnp.asarray(kla), "temperature": jnp.array(temperature)}


def _array(value: object) -> jax.Array:
    assert isinstance(value, jax.Array)
    return value


class TestTransport:
    def test_dilution_drives_the_state_toward_the_feed(self) -> None:
        """Without reactions or aeration the state moves toward the feed at rate Q/V."""
        state = jnp.array([2.0, 0.0])

        derivatives = INERT.derivatives(state, _feed(20.0, jnp.array([12.0, 0.0])), _signals(0.0), INERT_PARAMS)

        assert jnp.allclose(derivatives, jnp.array([20.0 / 100.0 * (12.0 - 2.0), 0.0]))

    def test_outlet_passes_the_inlet_flow_with_the_reactor_composition(self) -> None:
        """A constant-volume reactor returns its inflow and discharges what it holds."""
        state = jnp.array([3.0, 4.0])

        flows = INERT.outlet_flows(state, {"in": jnp.array(7.0)}, _signals(1.0), INERT_PARAMS)
        compositions = INERT.outlet_compositions(state, _feed(7.0, jnp.zeros(2)), _signals(1.0), INERT_PARAMS)

        assert jnp.allclose(flows["out"], 7.0)
        assert jnp.allclose(compositions["out"], state)

    def test_projection_clips_negative_concentrations(self) -> None:
        """Concentrations are kept non-negative after each integration substep."""
        projected = INERT.project(jnp.array([-1.0, 2.0]), INERT_PARAMS)

        assert jnp.allclose(projected, jnp.array([0.0, 2.0]))


class TestAeration:
    def test_transfer_is_proportional_to_the_saturation_deficit(self) -> None:
        """At the reference temperature oxygen enters at kLa (S_O,sat − S_O) and nothing else changes."""
        state = jnp.array([1.0, 4.0])

        derivatives = INERT.derivatives(state, _feed(0.0, jnp.zeros(2)), _signals(3.0), INERT_PARAMS)

        assert jnp.allclose(derivatives, jnp.array([0.0, 3.0 * (SATURATION - 4.0)]))

    def test_kla_is_corrected_by_theta_to_the_temperature(self) -> None:
        """kLa scales by θ^(T − T_ref), with the BSM2 θ = 1.024 by default."""
        state = jnp.zeros(2)

        warm = INERT.derivatives(state, _feed(0.0, jnp.zeros(2)), _signals(3.0, temperature=20.0), INERT_PARAMS)

        assert jnp.allclose(warm[1], 3.0 * 1.024**5 * SATURATION)

    def test_fixed_temperature_replaces_the_temperature_input(self) -> None:
        """Without a temperature input the reactor reads only kLa and uses the parameter temperature."""
        reactor = AeratedBioreactor(InertKinetics(), temperature_input=False, oxygen_saturation=_constant_saturation)
        params = BioreactorParams(volume=100.0, kinetics=None, temperature=25.0)

        derivatives = reactor.derivatives(jnp.zeros(2), _feed(0.0, jnp.zeros(2)), {"kla": jnp.array(1.0)}, params)

        assert reactor.input_ports == ("kla",)
        assert INERT.input_ports == ("kla", "temperature")
        assert jnp.allclose(derivatives[1], 1.024**10 * SATURATION)

    def test_bsm1_saturation_is_eight_at_fifteen_degrees(self) -> None:
        """The default saturation model reproduces the BSM1 reference S_O,sat and falls as water warms."""
        saturation = bsm1_oxygen_saturation(jnp.array([15.0, 25.0]))

        assert jnp.allclose(saturation[0], 8.0, atol=1e-3)
        assert saturation[1] < saturation[0]


class TestASM1Conservation:
    """In a closed, unaerated ASM1 batch the reactions conserve COD and nitrogen.

    Without nitrate or autotrophs there is no denitrification, so no nitrogen or COD leaves as N₂.
    COD counts dissolved oxygen as negative COD.
    """

    REACTOR = AeratedBioreactor(ASM1Kinetics())
    PARAMS = BioreactorParams(volume=1000.0, kinetics=ASM1Params())
    STATE = jnp.array([30.0, 40.0, 50.0, 120.0, 2000.0, 0.0, 400.0, 6.0, 0.0, 20.0, 5.0, 8.0, 5.0])

    def _derivatives(self, temperature: float) -> jax.Array:
        batch = _feed(0.0, jnp.zeros(13), schema=ASM1_SCHEMA)
        return self.REACTOR.derivatives(self.STATE, batch, _signals(0.0, temperature), self.PARAMS)

    def test_cod_is_conserved(self) -> None:
        """Organic COD gained or lost is matched by oxygen consumed, at any temperature."""
        for temperature in (10.0, 15.0, 20.0):
            derivatives = self._derivatives(temperature)
            organic_cod = derivatives[:7].sum()
            assert organic_cod < 0.0
            assert jnp.allclose(organic_cod, derivatives[7], rtol=1e-4)

    def test_nitrogen_is_conserved(self) -> None:
        """Nitrogen moves between ammonia, organic N and biomass without loss."""
        asm1 = self.PARAMS.kinetics
        nitrogen = jnp.array(
            [0.0, 0.0, asm1.i_xp, 0.0, asm1.i_xb, asm1.i_xb, asm1.i_xp, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        )

        derivatives = self._derivatives(15.0)

        assert jnp.allclose(derivatives @ nitrogen, 0.0, atol=1e-5)


class TestStandaloneIntegration:
    def test_washout_follows_the_exponential_solution(self) -> None:
        """A tracer step relaxes as c_in (1 − e^(−Qt/V)) and unaerated oxygen is washed out by oxygen-free feed."""
        state = step_unit(INERT, jnp.array([0.0, 4.0]), _feed(25.0, jnp.array([8.0, 0.0])), _signals(0.0), INERT_PARAMS, jnp.array(2.0), 40)

        decay = jnp.exp(-25.0 / 100.0 * 2.0)
        assert jnp.allclose(state, jnp.array([8.0 * (1.0 - decay), 4.0 * decay]), rtol=1e-5)

    def test_aeration_without_flow_approaches_saturation(self) -> None:
        """A closed aerated tank fills with oxygen as S_O,sat (1 − e^(−kLa t))."""
        state = step_unit(INERT, jnp.zeros(2), _feed(0.0, jnp.zeros(2)), _signals(1.5), INERT_PARAMS, jnp.array(1.0), 40)

        assert jnp.allclose(state[1], SATURATION * (1.0 - jnp.exp(-1.5)), rtol=1e-5)

    def test_vmap_over_kla(self) -> None:
        """A batch of aeration rates integrates in one jitted call, each matching its unbatched run."""

        def aerate(kla: jax.Array) -> jax.Array:
            return step_unit(INERT, jnp.zeros(2), _feed(0.0, jnp.zeros(2)), _signals(kla), INERT_PARAMS, jnp.array(1.0), 40)

        batched = jax.jit(jax.vmap(aerate))(jnp.array([0.0, 1.5, 3.0]))

        assert batched.shape == (3, 2)
        assert jnp.allclose(batched[0], 0.0)
        assert jnp.allclose(batched[1], aerate(jnp.array(1.5)), rtol=1e-6)


class TestFlowsheet:
    def test_reactors_in_series_read_their_routed_kla(self) -> None:
        """Two reactors wired in series pass flow through and read their own routed kLa signals."""
        plant = build_flowsheet(
            nodes={
                "r1": Node(INERT, INERT_PARAMS, inputs={"kla": "kla_1"}),
                "r2": Node(INERT, INERT_PARAMS, inputs={"kla": "kla_2"}),
            },
            connections=[(Port("r1", "out"), Port("r2", "in"))],
            inlets={"influent": Port("r1", "in")},
            outlets={"effluent": Port("r2", "out")},
        )
        state = {"r1": jnp.array([1.0, 2.0]), "r2": jnp.array([3.0, 4.0])}
        signals = {"kla_1": jnp.array(0.0), "kla_2": jnp.array(2.0), "temperature": jnp.array(15.0)}

        influent = {"influent": _feed(10.0, jnp.array([5.0, 0.0]))["in"]}

        derivatives = plant.derivatives(state, influent, signals)
        report = plant.report(state, influent, signals)

        assert jnp.allclose(report.outlets["effluent"].flow, 10.0)
        assert jnp.allclose(report.outlets["effluent"].components, state["r2"])
        assert jnp.allclose(_array(derivatives["r1"]), jnp.array([0.1 * 4.0, 0.1 * -2.0]))
        assert jnp.allclose(_array(derivatives["r2"]), jnp.array([0.1 * -2.0, 0.1 * -2.0 + 2.0 * 6.0]))
