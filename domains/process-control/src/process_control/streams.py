import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from process_control.components import ComponentSchema
from process_control.fields import Phase, StreamField


@dataclass(frozen=True)
class Stream:
    """A flow with components ordered by a static schema, at an absolute temperature (K) and pressure (Pa).

    A liquid flow is m³/h with concentrations in the schema's bases; a gas flow is total mol/h with the
    complete mixture's mole fractions.
    """

    flow: jax.Array
    components: jax.Array
    temperature_k: jax.Array
    pressure_pa: jax.Array
    schema: ComponentSchema
    phase: Phase

    def value(self, field: StreamField) -> jax.Array:
        match field:
            case StreamField.FLOW:
                return self.flow
            case StreamField.COMPONENTS:
                return self.components
            case StreamField.TEMPERATURE:
                return self.temperature_k
            case StreamField.PRESSURE:
                return self.pressure_pa


jax.tree_util.register_dataclass(
    Stream, data_fields=["flow", "components", "temperature_k", "pressure_pa"], meta_fields=["schema", "phase"]
)


def mix_components(phase: Phase, flows: Sequence[jax.Array], components: Sequence[jax.Array]) -> jax.Array:
    """Composition of a same-phase mixture: flow-weighted concentrations for liquids, summed molar amounts for gases.

    At zero total flow there is no material to weigh, so either phase takes the unweighted mean of its inlet compositions.
    """
    stacked_flows = jnp.stack(flows)
    total = stacked_flows.sum()
    flowing = total > 0
    if phase is Phase.GAS:
        stacked = jnp.stack(components)
        return jnp.where(flowing, stacked_flows @ stacked / jnp.where(flowing, total, 1.0), stacked.mean(axis=0))
    weights = jnp.where(flowing, stacked_flows / jnp.where(flowing, total, 1.0), 1.0 / len(flows))
    return weights @ jnp.stack(components)


def mix(*streams: Stream, temperature_k: jax.Array, pressure_pa: jax.Array) -> Stream:
    """Same-phase, same-schema mixture at a prescribed temperature and pressure."""
    if not streams:
        raise ValueError("mix needs at least one stream")
    schema, phase = streams[0].schema, streams[0].phase
    if any(stream.phase is not phase for stream in streams):
        raise ValueError("cannot mix streams of different phases")
    if any(stream.schema != schema for stream in streams):
        raise ValueError("cannot mix streams with different component schemas")

    flows = [stream.flow for stream in streams]
    return Stream(
        flow=jnp.stack(flows).sum(),
        components=mix_components(phase, flows, [stream.components for stream in streams]),
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        schema=schema,
        phase=phase,
    )


def split(
    stream: Stream,
    remainder: str,
    *,
    flows: Mapping[str, ArrayLike] | None = None,
    fractions: Mapping[str, ArrayLike] | None = None,
) -> dict[str, Stream]:
    """Divide a stream into outlets of its composition, temperature and pressure.

    Absolute flows are drawn first, then fractions of the inlet flow, each limited to the flow
    still available; `remainder` takes the rest, so the outlet flows always sum to the inlet flow.
    """
    requests = [
        *(flows or {}).items(),
        *((name, jnp.asarray(fraction) * stream.flow) for name, fraction in (fractions or {}).items()),
    ]
    names = [name for name, _ in requests] + [remainder]
    if len(set(names)) != len(names):
        raise ValueError(f"split outlet names must be unique: {names}")

    available = stream.flow
    outlet_flows: dict[str, jax.Array] = {}
    for name, requested in requests:
        drawn = jnp.clip(requested, 0.0, available)
        outlet_flows[name] = drawn
        available = available - drawn
    outlet_flows[remainder] = available

    return {name: dataclasses.replace(stream, flow=flow) for name, flow in outlet_flows.items()}
