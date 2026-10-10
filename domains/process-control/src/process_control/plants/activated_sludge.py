"""Activated-sludge plants: a train of aerated reactors with an internal recycle ahead of a layered settler.

The influent, internal recycle and return sludge mix into the first reactor. After the last reactor a
splitter draws the internal recycle (`q_a`) and sends the rest to the settler, whose underflow
(`q_underflow`) splits into the return sludge (`q_r`) and the waste sludge. Every reactor reads its own
`kla_<i>` signal, counted from 1, and the shared `temperature` signal. Flows are in m³/h and kLa in h⁻¹.

The BSM1 layout follows Alex et al. (2008), Sec. 2:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.flowsheet import Flowsheet, Mixer, Node, Port, Splitter, build_flowsheet
from process_control.kinetics import KineticsModel
from process_control.units.asm1 import ASM1Kinetics, ASM1Params
from process_control.units.bioreactor import AeratedBioreactor, BioreactorParams
from process_control.units.layered_settler import BSM1_SETTLER, LayeredSettler, ParticulateOutlets, SettlerParams


@dataclass(frozen=True)
class ActivatedSludgeLayout[P]:
    """Reactor volumes in flow order (m³), the kinetics every reactor shares, and the secondary settler."""

    volumes: tuple[float, ...]
    kinetics: KineticsModel[P]
    kinetics_params: P
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
    kinetics=ASM1Kinetics(),
    # ASM1Kinetics never reads the volume; each reactor's volume lives in its BioreactorParams.
    kinetics_params=ASM1Params(volume=math.nan),
)


def activated_sludge_flowsheet[P](layout: ActivatedSludgeLayout[P]) -> Flowsheet:
    """The plant with external inlet `influent` and external outlets `effluent` and `waste`."""
    schema = layout.kinetics.schema
    reactor = AeratedBioreactor(layout.kinetics)
    nodes: dict[str, Node] = {"inlet": Node(Mixer(schema, ("influent", "internal_recycle", "return_sludge")), None)}
    for name, signal, volume in zip(layout.reactors, layout.kla_signals, layout.volumes, strict=True):
        nodes[name] = Node(reactor, BioreactorParams(volume=volume, kinetics=layout.kinetics_params), {"kla": signal})
    nodes["recycle_split"] = Node(Splitter(schema, ("internal_recycle",), "settler_feed"), None, {"internal_recycle": "q_a"})
    nodes["settler"] = Node(LayeredSettler(schema, layout.particulates), layout.settler)
    nodes["sludge_split"] = Node(Splitter(schema, ("return_sludge",), "waste"), None, {"return_sludge": "q_r"})

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
    schema = layout.kinetics.schema
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
