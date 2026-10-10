from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.fields import EvaluationGroup, GroupResult, LocalField, Phase, PortSpec, PrescribedContext, StreamField
from process_control.flowsheet import Flowsheet, Mixer, Node, Port, Splitter, build_flowsheet
from process_control.streams import Stream
from process_control.units.contract import step_unit

TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")

SCHEMA = ComponentSchema(names=("c",), particulate=(False,), tss_factor=(0.0,), bases=(TRACER,))
LIQUID = PortSpec(SCHEMA, Phase.LIQUID)
CONTEXT = PrescribedContext(pressure_pa=101325.0)
TEMPERATURE_K = 288.15
FEED_FLOW = 1.0
FEED_CONCENTRATION = 10.0
RECYCLE_FLOW = 2.0
DECAY_RATE = 0.5


def _holding_tank_groups(feed: str) -> tuple[EvaluationGroup, ...]:
    """Outlet flow follows the feed, composition is the state, temperature and pressure are forwarded from the feed."""
    context = (StreamField.TEMPERATURE, StreamField.PRESSURE)
    return (
        EvaluationGroup("flow", outputs=(LocalField("out", StreamField.FLOW),), reads=(LocalField(feed, StreamField.FLOW),)),
        EvaluationGroup("composition", outputs=(LocalField("out", StreamField.COMPONENTS),), reads=(), reads_state=True),
        EvaluationGroup(
            "context", outputs=tuple(LocalField("out", kind) for kind in context), reads=tuple(LocalField(feed, kind) for kind in context)
        ),
    )


def _evaluate_holding_tank(group_name: str, feed: str, state: jax.Array, reads: Mapping[LocalField, jax.Array]) -> GroupResult:
    if group_name == "composition":
        return GroupResult((state,))
    group = {group.name: group for group in _holding_tank_groups(feed)}[group_name]
    return GroupResult(tuple(reads[read] for read in group.reads))


@dataclass(frozen=True)
class DecayParams:
    volume: float
    rate: float


@dataclass(frozen=True)
class DecayTank:
    """Constant-volume CSTR with first-order decay: outlet flow passes through, composition is the state."""

    input_ports: tuple[str, ...] = ()

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]:
        return {"in": LIQUID}

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]:
        return {"out": LIQUID}

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]:
        return _holding_tank_groups("in")

    @property
    def stateful(self) -> bool:
        return True

    def evaluate_group(
        self, group_name: str, state: jax.Array, reads: Mapping[LocalField, jax.Array], inputs: Mapping[str, jax.Array], params: DecayParams
    ) -> GroupResult:
        return _evaluate_holding_tank(group_name, "in", state, reads)

    def derivatives(
        self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: DecayParams
    ) -> jax.Array:
        feed = inlets["in"]
        return feed.flow / params.volume * (feed.components - state) - params.rate * state

    def project(self, state: jax.Array, params: DecayParams) -> jax.Array:
        return jnp.maximum(state, 0.0)


@dataclass(frozen=True)
class SampledTank:
    """Holding tank whose outlet follows its feed; a sample inlet feeds its state but no outlet."""

    input_ports: tuple[str, ...] = ()

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]:
        return {"feed": LIQUID, "sample": LIQUID}

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]:
        return {"out": LIQUID}

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]:
        return _holding_tank_groups("feed")

    @property
    def stateful(self) -> bool:
        return True

    def evaluate_group(
        self, group_name: str, state: jax.Array, reads: Mapping[LocalField, jax.Array], inputs: Mapping[str, jax.Array], params: None
    ) -> GroupResult:
        return _evaluate_holding_tank(group_name, "feed", state, reads)

    def derivatives(self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: None) -> jax.Array:
        return inlets["sample"].flow * (inlets["sample"].components - state)

    def project(self, state: jax.Array, params: None) -> jax.Array:
        return state


def _liquid(flow: float, components: list[float], schema: ComponentSchema = SCHEMA) -> Stream:
    return Stream(
        flow=jnp.array(flow),
        components=jnp.array(components),
        temperature_k=jnp.array(TEMPERATURE_K),
        pressure_pa=jnp.array(101325.0),
        schema=schema,
        phase=Phase.LIQUID,
    )


TANK_PARAMS = DecayParams(volume=1.0, rate=DECAY_RATE)


def _feed(flow: float = FEED_FLOW) -> dict[str, Stream]:
    return {"influent": _liquid(flow, [FEED_CONCENTRATION])}


def _inputs(recycle: float = RECYCLE_FLOW) -> dict[str, jax.Array]:
    return {"q_recycle": jnp.array(recycle), "temperature_k": jnp.array(TEMPERATURE_K)}


def _recycle_plant() -> Flowsheet:
    return build_flowsheet(
        nodes={
            "mix": Node(Mixer(LIQUID, ("feed", "recycle"), CONTEXT), None),
            "t1": Node(DecayTank(), TANK_PARAMS),
            "t2": Node(DecayTank(), TANK_PARAMS),
            "split": Node(Splitter(LIQUID, draws=("recycle",), remainder="effluent"), None, inputs={"recycle": "q_recycle"}),
        },
        connections=[
            (Port("mix", "out"), Port("t1", "in")),
            (Port("t1", "out"), Port("t2", "in")),
            (Port("t2", "out"), Port("split", "in")),
            (Port("split", "recycle"), Port("mix", "recycle")),
        ],
        inlets={"influent": Port("mix", "feed")},
        outlets={"effluent": Port("split", "effluent")},
    )


def _steady_state() -> dict[str, jax.Array]:
    # Tank balances Q (c_in - c) = k V c with c_in of tank 1 the flow-weighted feed and recycle.
    through_flow = FEED_FLOW + RECYCLE_FLOW
    damping = 1.0 + DECAY_RATE * TANK_PARAMS.volume / through_flow
    c1 = FEED_FLOW * FEED_CONCENTRATION / (damping * through_flow - RECYCLE_FLOW / damping)
    return {"t1": jnp.array([c1]), "t2": jnp.array([c1 / damping])}


def _empty_state() -> dict[str, jax.Array]:
    return {"t1": jnp.zeros(1), "t2": jnp.zeros(1)}


def _array(value: object) -> jax.Array:
    assert isinstance(value, jax.Array)
    return value


def _settle(plant: Flowsheet, inputs: dict[str, jax.Array]) -> Mapping[str, object]:
    def advance(_: int, state: Mapping[str, object]) -> Mapping[str, object]:
        return plant.step(state, _feed(), inputs, jnp.array(1.0), 10)

    return jax.lax.fori_loop(0, 100, advance, _empty_state())


class TestJointIntegration:
    def test_recycle_steady_state_has_zero_derivatives(self) -> None:
        """The analytic recycle steady state is a fixed point of the joint ODE.

        A one-step recycle lag would leave a residual here, so this pins down the coupling.
        """
        derivatives = _recycle_plant().derivatives(_steady_state(), _feed(), _inputs())

        assert jnp.allclose(_array(derivatives["t1"]), 0.0, atol=1e-5)
        assert jnp.allclose(_array(derivatives["t2"]), 0.0, atol=1e-5)

    def test_integration_converges_to_the_analytic_steady_state(self) -> None:
        """Stepping from empty tanks reaches the recycle steady state."""
        plant = _recycle_plant()

        state = jax.jit(_settle, static_argnums=0)(plant, _inputs())

        expected = _steady_state()
        assert jnp.allclose(_array(state["t1"]), expected["t1"], rtol=1e-4)
        assert jnp.allclose(_array(state["t2"]), expected["t2"], rtol=1e-4)

    def test_steady_state_balances_mass(self) -> None:
        """At steady state the feed load equals the effluent load plus what decays in the tanks."""
        state = _steady_state()

        report = _recycle_plant().report(state, _feed(), _inputs())

        effluent = report.outlets["effluent"]
        decayed = DECAY_RATE * TANK_PARAMS.volume * (state["t1"] + state["t2"])
        assert jnp.allclose(FEED_FLOW * FEED_CONCENTRATION, effluent.flow * effluent.components + decayed)

    def test_vmap_over_input_signals(self) -> None:
        """A batch of recycle flows integrates in one call, each matching its own unbatched run."""
        plant = _recycle_plant()
        recycles = jnp.array([0.5, 2.0, 4.0])

        def settle(recycle: jax.Array) -> Mapping[str, object]:
            return _settle(plant, {**_inputs(), "q_recycle": recycle})

        batched = _array(jax.jit(jax.vmap(settle))(recycles)["t2"])

        assert batched.shape == (3, 1)
        single = _array(_settle(plant, _inputs(4.0))["t2"])
        assert jnp.allclose(batched[2], single, rtol=1e-5)

    def test_rejects_state_for_the_wrong_nodes(self) -> None:
        """Junctions carry no state, and every stateful node must be present."""
        with pytest.raises(ValueError, match="exactly the nodes"):
            _recycle_plant().derivatives({"t1": jnp.zeros(1)}, _feed(), _inputs())


class TestFlowConservation:
    def test_effluent_matches_influent_flow(self) -> None:
        """A constant-volume plant returns exactly the flow it receives, whatever circulates inside."""
        report = _recycle_plant().report(_steady_state(), _feed(), _inputs())

        assert jnp.allclose(report.outlets["effluent"].flow, FEED_FLOW)
        assert jnp.allclose(report.streams[Port("mix", "out")].flow, FEED_FLOW + RECYCLE_FLOW)
        assert jnp.allclose(report.split_deficits["split"], 0.0)

    def test_infeasible_draw_reports_a_deficit(self) -> None:
        """A draw larger than the inflow is honoured, the remainder clamps at zero and the shortfall is reported."""
        plant = build_flowsheet(
            nodes={"split": Node(Splitter(LIQUID, draws=("waste",), remainder="rest"), None)},
            connections=[],
            inlets={"influent": Port("split", "in")},
            outlets={"waste": Port("split", "waste"), "rest": Port("split", "rest")},
        )

        report = plant.report({}, _feed(flow=1.0), {"waste": jnp.array(1.5)})

        assert jnp.allclose(report.outlets["waste"].flow, 1.5)
        assert jnp.allclose(report.outlets["rest"].flow, 0.0)
        assert jnp.allclose(report.split_deficits["split"], 0.5)

    def test_negative_draw_is_refused(self) -> None:
        """A negative draw cannot push flow back into the splitter; it is clamped and reported as a deficit."""
        plant = build_flowsheet(
            nodes={"split": Node(Splitter(LIQUID, draws=("waste",), remainder="rest"), None)},
            connections=[],
            inlets={"influent": Port("split", "in")},
            outlets={"waste": Port("split", "waste"), "rest": Port("split", "rest")},
        )

        report = plant.report({}, _feed(flow=1.0), {"waste": jnp.array(-1.0)})

        assert jnp.allclose(report.outlets["waste"].flow, 0.0)
        assert jnp.allclose(report.outlets["rest"].flow, 1.0)
        assert jnp.allclose(report.split_deficits["split"], 1.0)


class TestStandaloneUnit:
    def test_step_unit_follows_the_exponential_solution(self) -> None:
        """A single tank with a constant feed relaxes exponentially to Q c_in / (Q + k V)."""
        feed = {"in": _liquid(2.0, [4.0])}
        rate = 2.0 / TANK_PARAMS.volume + DECAY_RATE
        steady = 2.0 * 4.0 / (2.0 + DECAY_RATE * TANK_PARAMS.volume)

        state = step_unit(DecayTank(), jnp.zeros(1), feed, {}, TANK_PARAMS, jnp.array(1.0), 20)

        assert jnp.allclose(state, steady * (1.0 - jnp.exp(-rate)), rtol=1e-5)


class TestBuildValidation:
    def test_cycle_through_an_unread_inlet_is_not_algebraic(self) -> None:
        """Only the inlets an outlet declares count: recycling into an inlet the outlet never reads is no loop."""
        plant = build_flowsheet(
            nodes={
                "tank": Node(SampledTank(), None),
                "split": Node(Splitter(LIQUID, draws=("effluent",), remainder="sample"), None),
            },
            connections=[
                (Port("tank", "out"), Port("split", "in")),
                (Port("split", "sample"), Port("tank", "sample")),
            ],
            inlets={"influent": Port("tank", "feed")},
            outlets={"effluent": Port("split", "effluent")},
        )

        report = plant.report({"tank": jnp.array([3.0])}, _feed(flow=2.0), {"effluent": jnp.array(1.5)})

        assert jnp.allclose(report.streams[Port("split", "sample")].flow, 0.5)
        assert jnp.allclose(report.outlets["effluent"].components, 3.0)

    def test_connection_between_reordered_schemas_is_rejected(self) -> None:
        """Equal-length schemas in different orders would silently swap components."""
        forward = ComponentSchema(names=("a", "b"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER,) * 2)
        backward = ComponentSchema(names=("b", "a"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER,) * 2)

        with pytest.raises(ValueError, match=r"schema mismatch: first\.out carries \('a', 'b'\) but second\.x expects \('b', 'a'\)"):
            build_flowsheet(
                nodes={
                    "first": Node(Mixer(PortSpec(forward, Phase.LIQUID), ("x",), CONTEXT), None),
                    "second": Node(Mixer(PortSpec(backward, Phase.LIQUID), ("x",), CONTEXT), None),
                },
                connections=[(Port("first", "out"), Port("second", "x"))],
                inlets={"influent": Port("first", "x")},
                outlets={"effluent": Port("second", "out")},
            )

    def test_external_inlet_with_another_schema_is_rejected(self) -> None:
        """An external stream must match the schema its inlet port expects."""
        other = ComponentSchema(names=("d",), particulate=(False,), tss_factor=(0.0,), bases=(TRACER,))
        influent = {"influent": _liquid(1.0, [1.0], schema=other)}

        with pytest.raises(ValueError, match="external inlet 'influent' carries"):
            _recycle_plant().report(_steady_state(), influent, _inputs())

    def test_flow_loop_through_a_tank_is_algebraic(self) -> None:
        """A tank breaks a composition loop but passes flow straight through, so recycling a remainder is a flow loop."""
        loop = r"cycle: mix\.flow reads split\.back\.flow .* split\.remainder_flow reads t1\.out\.flow .* t1\.flow reads mix\.out\.flow"
        with pytest.raises(ValueError, match=loop):
            build_flowsheet(
                nodes={
                    "mix": Node(Mixer(LIQUID, ("feed", "back"), CONTEXT), None),
                    "t1": Node(DecayTank(), TANK_PARAMS),
                    "split": Node(Splitter(LIQUID, draws=("out",), remainder="back"), None),
                },
                connections=[
                    (Port("mix", "out"), Port("t1", "in")),
                    (Port("t1", "out"), Port("split", "in")),
                    (Port("split", "back"), Port("mix", "back")),
                ],
                inlets={"influent": Port("mix", "feed")},
                outlets={"effluent": Port("split", "out")},
            )

    def test_dangling_inlet_is_rejected(self) -> None:
        """An inlet nothing feeds would be evaluated from nothing."""
        with pytest.raises(ValueError, match=r"unconnected inlets: t1\.in"):
            build_flowsheet(
                nodes={"t1": Node(DecayTank(), TANK_PARAMS)},
                connections=[],
                inlets={},
                outlets={"effluent": Port("t1", "out")},
            )

    def test_unconsumed_outlet_is_rejected(self) -> None:
        """An outlet that goes nowhere would silently drop its flow."""
        with pytest.raises(ValueError, match=r"unconnected outlets: t1\.out"):
            build_flowsheet(
                nodes={"t1": Node(DecayTank(), TANK_PARAMS)},
                connections=[],
                inlets={"influent": Port("t1", "in")},
                outlets={},
            )

    def test_unknown_port_is_rejected(self) -> None:
        """A misspelt port names the ports that do exist."""
        with pytest.raises(ValueError, match="has no outlet 'outlet'"):
            build_flowsheet(
                nodes={"t1": Node(DecayTank(), TANK_PARAMS)},
                connections=[],
                inlets={"influent": Port("t1", "in")},
                outlets={"effluent": Port("t1", "outlet")},
            )

    def test_double_feed_is_rejected(self) -> None:
        """Two streams into one inlet port would need an explicit mixer."""
        with pytest.raises(ValueError, match="connected twice"):
            build_flowsheet(
                nodes={"t1": Node(DecayTank(), TANK_PARAMS)},
                connections=[],
                inlets={"a": Port("t1", "in"), "b": Port("t1", "in")},
                outlets={"effluent": Port("t1", "out")},
            )

    def test_unknown_input_route_is_rejected(self) -> None:
        """Routing a signal to a port the unit does not read is a wiring mistake."""
        with pytest.raises(ValueError, match="no input ports"):
            build_flowsheet(
                nodes={"t1": Node(DecayTank(), TANK_PARAMS, inputs={"kla": "kla_1"})},
                connections=[],
                inlets={"influent": Port("t1", "in")},
                outlets={"effluent": Port("t1", "out")},
            )
