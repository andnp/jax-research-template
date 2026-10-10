import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1 import BSM1Config
from process_control.binding import NamedAccess, Requirement
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.fields import Phase, PrescribedContext
from process_control.flowsheet import Node, Port, build_flowsheet
from process_control.kinetics import KineticSource
from process_control.plants.activated_sludge import BSM1_LAYOUT
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Kinetics, ASM1Params, bsm1_oxygen_saturation
from process_control.units.contract import step_unit, unit_outlets
from process_control.units.exchange import OXYGEN_BASIS, ExchangeSource, OxygenExchange, OxygenExchangeParams
from process_control.units.reactive_cstr import ReactiveCSTR, ReactiveCSTRParams, nonnegative

TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")
CHARGE = ComponentBasis("mol/m3", Quantity.EQUIVALENT, "charge")

SCHEMA = ComponentSchema(names=("tracer", "oxygen"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER, OXYGEN_BASIS))
SIGNED_SCHEMA = ComponentSchema(names=("tracer", "charge"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER, CHARGE))
SATURATION = 10.0
CONTEXT = PrescribedContext(pressure_pa=101325.0)


@dataclass(frozen=True)
class FirstOrderUptake:
    """Consumes oxygen at a first-order rate."""

    @property
    def access(self) -> NamedAccess:
        return NamedAccess(reads=(Requirement("O2", OXYGEN_BASIS),), writes=(Requirement("O2", OXYGEN_BASIS),))

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: float) -> jax.Array:
        return -params * reads


@dataclass(frozen=True)
class TemperatureProbe:
    """Writes the temperature it receives as the tracer's rate."""

    @property
    def access(self) -> NamedAccess:
        return NamedAccess(writes=(Requirement("tracer", TRACER),))

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: None) -> jax.Array:
        return jnp.atleast_1d(temperature_k)


def _constant_saturation(temperature: jax.Array) -> jax.Array:
    return jnp.full_like(temperature, SATURATION)


AERATION = ExchangeSource.bind(OxygenExchange(_constant_saturation), OxygenExchangeParams(), SCHEMA, {"O2": "oxygen"})
INERT = ReactiveCSTR(SCHEMA, (AERATION,), context=CONTEXT)
INERT_PARAMS = ReactiveCSTRParams(volume=100.0)


def _asm1_reactor(schema: ComponentSchema) -> ReactiveCSTR:
    return ReactiveCSTR(
        schema,
        (
            KineticSource.bind(ASM1Kinetics(), ASM1Params(), schema),
            ExchangeSource.bind(OxygenExchange(bsm1_oxygen_saturation), OxygenExchangeParams(), schema, {"O2": "s_o"}),
        ),
        context=CONTEXT,
    )


def _feed(flow: float, components: jax.Array, schema: ComponentSchema = SCHEMA) -> dict[str, Stream]:
    feed = Stream(
        flow=jnp.array(flow),
        components=components,
        temperature_k=jnp.array(288.15),
        pressure_pa=jnp.array(101325.0),
        schema=schema,
        phase=Phase.LIQUID,
    )
    return {"in": feed}


def _signals(kla: float | jax.Array, temperature: float = 15.0) -> dict[str, jax.Array]:
    return {"kla": jnp.asarray(kla), "temperature": jnp.array(temperature), "temperature_k": jnp.array(temperature + 273.15)}


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

        outlet = unit_outlets(INERT, state, _feed(7.0, jnp.zeros(2)), _signals(1.0), INERT_PARAMS)["out"]

        assert jnp.allclose(outlet.flow, 7.0)
        assert jnp.allclose(outlet.components, state)

    def test_signed_equivalents_pass_through_unclipped(self) -> None:
        """By default the reactor projects nothing, so a negative equivalent is held and diluted, not zeroed."""
        reactor = ReactiveCSTR(SIGNED_SCHEMA, temperature_input=False, context=CONTEXT)
        feed = _feed(25.0, jnp.array([0.0, -3.0]), SIGNED_SCHEMA)

        state = step_unit(reactor, jnp.array([1.0, -1.0]), feed, {}, INERT_PARAMS, jnp.array(2.0), 40)

        decay = jnp.exp(-25.0 / 100.0 * 2.0)
        assert jnp.allclose(reactor.project(jnp.array([1.0, -1.0]), INERT_PARAMS), jnp.array([1.0, -1.0]))
        assert jnp.allclose(state[1], -3.0 + 2.0 * decay, rtol=1e-5)

    def test_injected_nonnegative_projection_clips(self) -> None:
        """A plant that admits only non-negative concentrations injects the clipping projection."""
        reactor = ReactiveCSTR(SCHEMA, (AERATION,), projection=nonnegative, context=CONTEXT)

        assert jnp.allclose(reactor.project(jnp.array([-1.0, 2.0]), INERT_PARAMS), jnp.array([0.0, 2.0]))


class TestSources:
    def test_reactor_without_sources_needs_no_kla(self) -> None:
        """A reactor with no oxygen exchange exposes no kLa input and only dilutes."""
        reactor = ReactiveCSTR(SIGNED_SCHEMA, context=CONTEXT)
        state = jnp.array([1.0, -2.0])

        derivatives = reactor.derivatives(state, _feed(10.0, jnp.zeros(2), SIGNED_SCHEMA), {"temperature": jnp.array(15.0)}, INERT_PARAMS)

        assert reactor.input_ports == ("temperature", "temperature_k")
        assert ReactiveCSTR(SIGNED_SCHEMA, temperature_input=False, context=CONTEXT).input_ports == ("temperature_k",)
        assert jnp.allclose(derivatives, 0.1 * -state)

    def test_routing_kla_to_a_reactor_without_aeration_is_rejected(self) -> None:
        """A flowsheet cannot route a kLa signal to a reactor that declares no such input."""
        with pytest.raises(ValueError):
            build_flowsheet(
                nodes={"r": Node(ReactiveCSTR(SIGNED_SCHEMA, context=CONTEXT), INERT_PARAMS, inputs={"kla": "kla_1"})},
                connections=[],
                inlets={"influent": Port("r", "in")},
                outlets={"effluent": Port("r", "out")},
            )

    def test_sources_writing_one_channel_add_on_the_same_state(self) -> None:
        """Uptake and aeration on the oxygen channel both see the current state, and their rates add."""
        uptake = KineticSource.bind(FirstOrderUptake(), 0.5, SCHEMA, {"O2": "oxygen"})
        reactor = ReactiveCSTR(SCHEMA, (uptake, AERATION), context=CONTEXT)
        state = jnp.array([0.0, 4.0])

        derivatives = reactor.derivatives(state, _feed(0.0, jnp.zeros(2)), _signals(3.0), INERT_PARAMS)

        assert jnp.allclose(derivatives, jnp.array([0.0, -0.5 * 4.0 + 3.0 * (SATURATION - 4.0)]))

    def test_source_bound_to_another_schema_is_rejected(self) -> None:
        """A source's indices address the schema it was bound to, so another carrier refuses it."""
        with pytest.raises(ValueError, match="source bound"):
            ReactiveCSTR(SIGNED_SCHEMA, (AERATION,), context=CONTEXT)

    def test_sources_receive_the_temperature_in_kelvin(self) -> None:
        """The °C temperature input, or the fallback parameter, reaches every source in K."""
        probe = KineticSource.bind(TemperatureProbe(), None, SCHEMA)
        fixed = ReactiveCSTR(SCHEMA, (probe,), temperature_input=False, context=CONTEXT)
        batch = _feed(0.0, jnp.zeros(2))

        read = ReactiveCSTR(SCHEMA, (probe,), context=CONTEXT).derivatives(jnp.zeros(2), batch, {"temperature": jnp.array(20.0)}, INERT_PARAMS)
        fallback = fixed.derivatives(jnp.zeros(2), batch, {}, ReactiveCSTRParams(volume=100.0, temperature=25.0))

        assert jnp.allclose(read[0], 293.15)
        assert jnp.allclose(fallback[0], 298.15)

    def test_source_parameters_are_pytree_data(self) -> None:
        """A bound source crosses jit and vmap with its parameters traced and its binding static."""

        def warm_transfer(params: OxygenExchangeParams) -> jax.Array:
            reactor = ReactiveCSTR(SCHEMA, (dataclasses.replace(AERATION, params=params),), context=CONTEXT)
            return reactor.derivatives(jnp.zeros(2), _feed(0.0, jnp.zeros(2)), _signals(1.0, temperature=20.0), INERT_PARAMS)[1]

        thetas = jnp.array([1.0, 1.024])
        transfer = jax.jit(jax.vmap(warm_transfer))(OxygenExchangeParams(kla_theta=thetas, kla_reference_temperature=jnp.full(2, 15.0)))

        assert jnp.allclose(transfer, thetas**5 * SATURATION)


class TestAeration:
    def test_transfer_is_proportional_to_the_saturation_deficit(self) -> None:
        """At the reference temperature oxygen enters at kLa (S_O,sat − S_O) and nothing else changes."""
        state = jnp.array([1.0, 4.0])

        derivatives = INERT.derivatives(state, _feed(0.0, jnp.zeros(2)), _signals(3.0), INERT_PARAMS)

        assert jnp.allclose(derivatives, jnp.array([0.0, 3.0 * (SATURATION - 4.0)]))

    def test_kla_is_corrected_by_theta_to_the_temperature(self) -> None:
        """kLa scales by θ^(T − T_ref), with the BSM2 θ = 1.024 by default."""
        warm = INERT.derivatives(jnp.zeros(2), _feed(0.0, jnp.zeros(2)), _signals(3.0, temperature=20.0), INERT_PARAMS)

        assert jnp.allclose(warm[1], 3.0 * 1.024**5 * SATURATION)

    def test_fixed_temperature_replaces_the_temperature_input(self) -> None:
        """Without a temperature input the sources use the parameter temperature; only the outlet context reads `temperature_k`."""
        reactor = ReactiveCSTR(SCHEMA, (AERATION,), temperature_input=False, context=CONTEXT)
        params = ReactiveCSTRParams(volume=100.0, temperature=25.0)

        derivatives = reactor.derivatives(jnp.zeros(2), _feed(0.0, jnp.zeros(2)), {"kla": jnp.array(1.0)}, params)

        assert reactor.input_ports == ("kla", "temperature_k")
        assert INERT.input_ports == ("kla", "temperature", "temperature_k")
        assert jnp.allclose(derivatives[1], 1.024**10 * SATURATION)

    def test_bsm1_saturation_is_eight_at_fifteen_degrees(self) -> None:
        """The BSM1 saturation policy reproduces the reference S_O,sat and falls as water warms."""
        saturation = bsm1_oxygen_saturation(jnp.array([15.0, 25.0]))

        assert jnp.allclose(saturation[0], 8.0, atol=1e-3)
        assert saturation[1] < saturation[0]


class TestCarrierExtension:
    """ASM1 binds by channel name, so a reordered carrier with an extra channel leaves its rates unchanged."""

    STATE = (30.0, 2.8, 1149.0, 82.0, 2551.0, 148.0, 448.0, 0.5, 5.4, 7.9, 1.2, 5.3, 4.9)
    FEED = (30.0, 69.5, 51.2, 202.3, 28.2, 0.0, 0.0, 0.0, 0.0, 31.6, 6.9, 10.6, 7.0)

    def test_reordered_carrier_with_a_tracer_keeps_asm1_rates_and_only_dilutes_the_tracer(self) -> None:
        """Each ASM1 channel's derivative matches the canonical reactor; the appended tracer only dilutes."""
        order = (12, 7, 0, 9, 3, 11, 1, 5, 10, 2, 8, 4, 6)
        extended = ComponentSchema(
            names=(*(ASM1_SCHEMA.names[i] for i in order), "tracer"),
            particulate=(*(ASM1_SCHEMA.particulate[i] for i in order), False),
            tss_factor=(*(ASM1_SCHEMA.tss_factor[i] for i in order), 0.0),
            bases=(*(ASM1_SCHEMA.bases[i] for i in order), TRACER),
        )
        with jax.enable_x64(True):
            state, feed = jnp.array(self.STATE), jnp.array(self.FEED)
            permute = jnp.array(order)
            canonical = _asm1_reactor(ASM1_SCHEMA).derivatives(state, _feed(80.0, feed, ASM1_SCHEMA), _signals(5.0, 12.0), ReactiveCSTRParams(1000.0))

            moved = _asm1_reactor(extended).derivatives(
                jnp.append(state[permute], 3.0), _feed(80.0, jnp.append(feed[permute], 7.0), extended), _signals(5.0, 12.0), ReactiveCSTRParams(1000.0)
            )

            assert jnp.allclose(moved[:13], canonical[permute], rtol=1e-12)
            assert jnp.allclose(moved[13], 80.0 / 1000.0 * (7.0 - 3.0))

    def test_bsm1_benchmark_rejects_a_noncanonical_schema(self) -> None:
        """BSM1 observations read ASM1 channels by position, so its benchmark refuses any other carrier."""
        reordered = dataclasses.replace(ASM1_SCHEMA, names=tuple(reversed(ASM1_SCHEMA.names)))

        with pytest.raises(ValueError, match="ASM1_SCHEMA"):
            BSM1Config(layout=dataclasses.replace(BSM1_LAYOUT, schema=reordered))


class TestASM1Conservation:
    """In a closed, unaerated ASM1 batch the reactions conserve COD and nitrogen.

    Without nitrate or autotrophs there is no denitrification, so no nitrogen or COD leaves as N₂.
    COD counts dissolved oxygen as negative COD.
    """

    REACTOR = _asm1_reactor(ASM1_SCHEMA)
    PARAMS = ReactiveCSTRParams(volume=1000.0)
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
        asm1 = ASM1Params()
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
        signals = {"kla_1": jnp.array(0.0), "kla_2": jnp.array(2.0), "temperature": jnp.array(15.0), "temperature_k": jnp.array(288.15)}

        influent = {"influent": _feed(10.0, jnp.array([5.0, 0.0]))["in"]}

        derivatives = plant.derivatives(state, influent, signals)
        report = plant.report(state, influent, signals)

        assert jnp.allclose(report.outlets["effluent"].flow, 10.0)
        assert jnp.allclose(report.outlets["effluent"].components, state["r2"])
        assert jnp.allclose(_array(derivatives["r1"]), jnp.array([0.1 * 4.0, 0.1 * -2.0]))
        assert jnp.allclose(_array(derivatives["r2"]), jnp.array([0.1 * -2.0, 0.1 * -2.0 + 2.0 * 6.0]))
