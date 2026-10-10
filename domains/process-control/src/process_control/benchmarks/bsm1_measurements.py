"""Quantities measurable on the BSM1 activated-sludge plant, looked up by name.

Concentrations are in g/m³ (alkalinity in mol/m³), flows in m³/h, kLa in h⁻¹, heights in m, sludge age in
h and temperature in °C. Stream qualities follow Alex et al. (2008), Sec. 6:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control.flowsheet import Flowsheet, Port
from process_control.plants.activated_sludge import ActivatedSludgeLayout
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Params, ASM1State, compute_bod5, compute_cod, compute_tkn, compute_tn
from process_control.units.layered_settler import SettlerParams


class PlantSnapshot(NamedTuple):
    """The plant at one instant: reactors in flow order, settler layers bottom first, and its streams and handles."""

    reactors: jax.Array
    settler: jax.Array
    influent: Stream
    effluent: Stream
    underflow: Stream
    waste: Stream
    temperature: jax.Array
    handles: Mapping[str, jax.Array]
    setpoints: Mapping[str, jax.Array]
    split_deficit: jax.Array


type Measure = Callable[[PlantSnapshot], jax.Array]


def take_snapshot[P](
    flowsheet: Flowsheet,
    layout: ActivatedSludgeLayout[P],
    plant: Mapping[str, jax.Array],
    influent: Stream,
    signals: Mapping[str, jax.Array],
    handles: Mapping[str, jax.Array],
    setpoints: Mapping[str, jax.Array],
) -> PlantSnapshot:
    report = flowsheet.report(plant, {"influent": influent}, signals)
    return PlantSnapshot(
        reactors=jnp.stack([plant[name] for name in layout.reactors]),
        settler=plant["settler"],
        influent=influent,
        effluent=report.outlets["effluent"],
        underflow=report.streams[Port("settler", "underflow")],
        waste=report.outlets["waste"],
        temperature=signals["temperature"],
        handles=handles,
        setpoints=setpoints,
        split_deficit=sum(report.split_deficits.values(), jnp.zeros(())),
    )


def asm1_state(components: jax.Array) -> ASM1State:
    """ASM1 fields from component vectors whose last axis follows the ASM1 schema."""
    return ASM1State(**{name: components[..., index] for index, name in enumerate(ASM1_SCHEMA.names)})


def stream_qualities(components: jax.Array, params: ASM1Params) -> dict[str, jax.Array]:
    """Every ASM1 component plus the BSM1 aggregate qualities of a stream."""
    state = asm1_state(components)
    return {
        **{name: components[..., index] for index, name in enumerate(ASM1_SCHEMA.names)},
        "tss": ASM1_SCHEMA.tss(components),
        "cod": compute_cod(state),
        "bod5": compute_bod5(state, params),
        "tkn": compute_tkn(state, params),
        "n_tot": compute_tn(state, params),
    }


def blanket_height(layer_tss: jax.Array, params: SettlerParams, threshold: float) -> jax.Array:
    """Height above the floor where solids fall below `threshold`, interpolated linearly between layer centres.

    The blanket top is the crossing above the highest layer still at the threshold; with no layer that
    thick it shrinks toward the floor in proportion to the bottom layer's solids.
    """
    layer_height = params.height / params.n_layers
    centres = (jnp.arange(params.n_layers) + 0.5) * layer_height
    thick = layer_tss >= threshold
    top = params.n_layers - 1 - jnp.argmax(thick[::-1])
    above = jnp.minimum(top + 1, params.n_layers - 1)
    drop = layer_tss[top] - layer_tss[above]
    crossing = centres[top] + jnp.where(drop > 0.0, (layer_tss[top] - threshold) / jnp.where(drop > 0.0, drop, 1.0), 0.0) * layer_height
    floor = centres[0] * layer_tss[0] / threshold
    return jnp.where(jnp.any(thick), jnp.minimum(crossing, centres[-1]), floor)


def measurements(
    layout: ActivatedSludgeLayout[ASM1Params], blanket_threshold: float, handles: tuple[str, ...], setpoints: tuple[str, ...]
) -> dict[str, Measure]:
    """Every named quantity the plant offers.

    Reactor quantities are `<quality>_<i>` with reactors counted from 1; stream quantities are
    `influent_`, `effluent_` or `underflow_` followed by a quality or `flow`. Qualities are the ASM1
    components and `tss`, `cod`, `bod5`, `tkn` and `n_tot`. `mlss` is the last reactor's TSS,
    `blanket_height` the settler's solids blanket, `sludge_age` the reactors' solids over the solids
    leaving in the waste and effluent, `temperature` the reactors' temperature, each handle name its
    realised value, and `<handle>_setpoint` the setpoint of a supervised controller.
    """
    params = layout.kinetics_params
    volumes = jnp.array(layout.volumes)

    def reactor(index: int, quality: str) -> Measure:
        return lambda snapshot: stream_qualities(snapshot.reactors[index], params)[quality]

    def stream(pick: Callable[[PlantSnapshot], Stream], quality: str) -> Measure:
        return lambda snapshot: stream_qualities(pick(snapshot).components, params)[quality]

    def flow(pick: Callable[[PlantSnapshot], Stream]) -> Measure:
        return lambda snapshot: pick(snapshot).flow

    def handle(name: str) -> Measure:
        return lambda snapshot: snapshot.handles[name]

    def setpoint(name: str) -> Measure:
        return lambda snapshot: snapshot.setpoints[name]

    def sludge_age(snapshot: PlantSnapshot) -> jax.Array:
        held = jnp.sum(ASM1_SCHEMA.tss(snapshot.reactors) * volumes)
        leaving = snapshot.waste.flow * ASM1_SCHEMA.tss(snapshot.waste.components) + snapshot.effluent.flow * ASM1_SCHEMA.tss(snapshot.effluent.components)
        return held / jnp.maximum(leaving, 1e-9)

    qualities = (*ASM1_SCHEMA.names, "tss", "cod", "bod5", "tkn", "n_tot")
    streams: dict[str, Callable[[PlantSnapshot], Stream]] = {
        "influent": lambda snapshot: snapshot.influent,
        "effluent": lambda snapshot: snapshot.effluent,
        "underflow": lambda snapshot: snapshot.underflow,
    }
    catalog: dict[str, Measure] = {
        f"{quality}_{index + 1}": reactor(index, quality) for index in range(len(layout.volumes)) for quality in qualities
    }
    for prefix, pick in streams.items():
        catalog[f"{prefix}_flow"] = flow(pick)
        catalog.update({f"{prefix}_{quality}": stream(pick, quality) for quality in qualities})
    catalog.update({name: handle(name) for name in handles})
    catalog.update({f"{name}_setpoint": setpoint(name) for name in setpoints})
    catalog["mlss"] = lambda snapshot: ASM1_SCHEMA.tss(snapshot.reactors[-1])
    catalog["blanket_height"] = lambda snapshot: blanket_height(ASM1_SCHEMA.tss(snapshot.settler), layout.settler, blanket_threshold)
    catalog["sludge_age"] = sludge_age
    catalog["temperature"] = lambda snapshot: snapshot.temperature
    return catalog


def measurement(
    quantity: str, layout: ActivatedSludgeLayout[ASM1Params], blanket_threshold: float, handles: tuple[str, ...], setpoints: tuple[str, ...]
) -> Measure:
    catalog = measurements(layout, blanket_threshold, handles, setpoints)
    if quantity not in catalog:
        raise ValueError(f"unknown plant quantity {quantity!r}; available: {sorted(catalog)}")
    return catalog[quantity]
