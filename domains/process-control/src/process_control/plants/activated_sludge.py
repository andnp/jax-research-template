"""Activated-sludge plants: a train of aerated reactors with an internal recycle ahead of a layered settler.

The influent, internal recycle and return sludge mix into the first reactor. After the last reactor a
splitter draws the internal recycle (`q_a`) and sends the rest to the settler, whose underflow
(`q_underflow`) splits into the return sludge (`q_r`) and the waste sludge. Every reactor reads its own
`kla_<i>` signal, counted from 1, and the shared `temperature` signal (°C). Every reactor, the inlet mixer
and the settler leave at the layout's prescribed context, whose temperature is the `temperature_k` signal
(K). Flows are in m³/h and kLa in h⁻¹.

The BSM1 layout follows Alex et al. (2008), Sec. 2:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.components import ComponentSchema
from process_control.fields import Phase, PortSpec, PrescribedContext
from process_control.flowsheet import Flowsheet, Mixer, Node, Port, Splitter, build_flowsheet
from process_control.kinetics import KineticsModel, KineticSource
from process_control.units.asm1 import ASM1_SCHEMA, BSM1_PRESSURE_PA, ASM1Kinetics, ASM1Params, bsm1_oxygen_saturation
from process_control.units.exchange import ExchangeSource, OxygenExchange, OxygenExchangeParams
from process_control.units.layered_settler import BSM1_SETTLER, LayeredSettler, ParticulateOutlets, SettlerParams
from process_control.units.reactive_cstr import ReactiveCSTR, ReactiveCSTRParams, nonnegative, unchanged


@dataclass(frozen=True)
class ActivatedSludgeLayout[P]:
    """Reactor volumes in flow order (m³), the carrier schema, what every reactor shares, and the secondary settler.

    Each reactor runs the kinetics and the aeration, whose oxygen binds to `oxygen_channel`, and
    `projection` keeps each integration substep's reactor state admissible. `context` sets the outlet
    temperature and pressure of every unit that does not forward its inlet's.
    """

    volumes: tuple[float, ...]
    schema: ComponentSchema
    kinetics: KineticsModel[P]
    kinetics_params: P
    aeration: OxygenExchange
    oxygen_channel: str
    aeration_params: OxygenExchangeParams
    context: PrescribedContext
    projection: Callable[[jax.Array], jax.Array] = unchanged
    settler: SettlerParams = BSM1_SETTLER
    particulates: ParticulateOutlets = ParticulateOutlets.FEED_FRACTIONS

    def __post_init__(self) -> None:
        if not self.volumes or any(volume <= 0 for volume in self.volumes):
            raise ValueError(f"an activated-sludge plant needs at least one reactor of positive volume, got {self.volumes}")

    @property
    def reactors(self) -> tuple[str, ...]:
        return tuple(f"reactor_{index}" for index in range(1, len(self.volumes) + 1))

    @property
    def kla_signals(self) -> tuple[str, ...]:
        return tuple(f"kla_{index}" for index in range(1, len(self.volumes) + 1))


BSM1_LAYOUT = ActivatedSludgeLayout(
    volumes=(1000.0, 1000.0, 1333.0, 1333.0, 1333.0),
    schema=ASM1_SCHEMA,
    kinetics=ASM1Kinetics(),
    kinetics_params=ASM1Params(),
    aeration=OxygenExchange(bsm1_oxygen_saturation),
    oxygen_channel="s_o",
    aeration_params=OxygenExchangeParams(kla_theta=1.024, kla_reference_temperature=15.0),
    context=PrescribedContext(pressure_pa=BSM1_PRESSURE_PA),
    projection=nonnegative,
)


def activated_sludge_flowsheet[P](layout: ActivatedSludgeLayout[P]) -> Flowsheet:
    """The plant with external inlet `influent` and external outlets `effluent` and `waste`."""
    schema = layout.schema
    sources = (
        KineticSource.bind(layout.kinetics, layout.kinetics_params, schema),
        ExchangeSource.bind(layout.aeration, layout.aeration_params, schema, {"O2": layout.oxygen_channel}),
    )
    port = PortSpec(schema, Phase.LIQUID)
    reactor = ReactiveCSTR(schema, sources, projection=layout.projection, context=layout.context)
    nodes: dict[str, Node] = {"inlet": Node(Mixer(port, ("influent", "internal_recycle", "return_sludge"), layout.context), None)}
    for name, signal, volume in zip(layout.reactors, layout.kla_signals, layout.volumes, strict=True):
        nodes[name] = Node(reactor, ReactiveCSTRParams(volume=volume), {"kla": signal})
    nodes["recycle_split"] = Node(Splitter(port, ("internal_recycle",), "settler_feed"), None, {"internal_recycle": "q_a"})
    nodes["settler"] = Node(LayeredSettler(schema, layout.particulates, context=layout.context), layout.settler)
    nodes["sludge_split"] = Node(Splitter(port, ("return_sludge",), "waste"), None, {"return_sludge": "q_r"})

    train = [Port("inlet", "out"), *(Port(name, "out") for name in layout.reactors)]
    connections = [
        *((outlet, Port(name, "in")) for outlet, name in zip(train, layout.reactors, strict=False)),
        (train[-1], Port("recycle_split", "in")),
        (Port("recycle_split", "internal_recycle"), Port("inlet", "internal_recycle")),
        (Port("recycle_split", "settler_feed"), Port("settler", "feed")),
        (Port("settler", "underflow"), Port("sludge_split", "in")),
        (Port("sludge_split", "return_sludge"), Port("inlet", "return_sludge")),
    ]
    return build_flowsheet(
        nodes,
        connections,
        inlets={"influent": Port("inlet", "influent")},
        outlets={"effluent": Port("settler", "effluent"), "waste": Port("sludge_split", "waste")},
    )


def settled_state[P](layout: ActivatedSludgeLayout[P], reactors: jax.Array, layer_tss: jax.Array) -> dict[str, jax.Array]:
    """Flowsheet state from reactor compositions in flow order and settler layer TSS bottom first.

    Settler layers carry the last reactor's solubles and its particulate fractions of TSS, as the BSM1
    settler outlets do.
    """
    schema = layout.schema
    feed = reactors[-1]
    fractions = jnp.where(schema.particulate_mask, feed / schema.tss(feed), 0.0)
    settler = jnp.where(schema.particulate_mask, layer_tss[:, None] * fractions, feed)
    return {**dict(zip(layout.reactors, reactors, strict=True)), "settler": settler}


# Alex et al. (2008) open-loop steady state (Appendix 4): reactors 1-5, then the settler layer TSS
# bottom first, at kLa 0/0/240/240/84 d⁻¹, Qa 55338, Qr 18446 and Qw 385 m³/d.
BSM1_OPEN_LOOP_REACTORS = (
    (30.0, 2.8082, 1149.1183, 82.1349, 2551.7631, 148.3886, 448.8459, 0.0042984, 5.3699, 7.9179, 1.2166, 5.2849, 4.9277),
    (30.0, 1.4588, 1149.1182, 76.3862, 2553.3824, 148.3083, 449.5167, 6.3132e-05, 3.6619, 8.3445, 0.88207, 5.0291, 5.0802),
    (30.0, 1.1495, 1149.1182, 64.8549, 2557.1288, 148.9404, 450.4123, 1.7184, 6.5408, 5.548, 0.82889, 4.3924, 4.6748),
    (30.0, 0.99532, 1149.1182, 55.694, 2559.18, 149.5262, 451.3087, 2.4289, 9.299, 2.9674, 0.76679, 3.879, 4.2935),
    (30.0, 0.88949, 1149.1182, 49.3056, 2559.341, 149.7963, 452.2051, 0.49094, 10.4152, 1.7334, 0.68828, 3.5272, 4.1256),
)
BSM1_OPEN_LOOP_LAYER_TSS = (6393.9599, 356.0738, 356.0738, 356.0738, 356.0738, 356.0738, 68.9779, 29.5402, 18.1132, 12.4969)


def bsm1_open_loop_state() -> dict[str, jax.Array]:
    """The BSM1 plant at its open-loop steady state."""
    return settled_state(BSM1_LAYOUT, jnp.array(BSM1_OPEN_LOOP_REACTORS), jnp.array(BSM1_OPEN_LOOP_LAYER_TSS))
