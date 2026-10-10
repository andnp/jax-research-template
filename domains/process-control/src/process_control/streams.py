from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from process_control.components import ComponentSchema


@dataclass(frozen=True)
class Stream:
    """A flow carrying concentrations ordered by a static component schema."""

    flow: jax.Array
    components: jax.Array
    schema: ComponentSchema


jax.tree_util.register_dataclass(Stream, data_fields=["flow", "components"], meta_fields=["schema"])


def mix(*streams: Stream) -> Stream:
    """Flow-weighted mixture; at zero total flow the concentrations are averaged unweighted."""
    if not streams:
        raise ValueError("mix needs at least one stream")
    schema = streams[0].schema
    if any(stream.schema != schema for stream in streams):
        raise ValueError("cannot mix streams with different component schemas")

    flows = jnp.stack([stream.flow for stream in streams])
    total = flows.sum()
    flowing = total > 0
    weights = jnp.where(flowing, flows / jnp.where(flowing, total, 1.0), 1.0 / len(streams))
    components = weights @ jnp.stack([stream.components for stream in streams])
    return Stream(flow=total, components=components, schema=schema)


def split(
    stream: Stream,
    remainder: str,
    *,
    flows: Mapping[str, ArrayLike] | None = None,
    fractions: Mapping[str, ArrayLike] | None = None,
) -> dict[str, Stream]:
    """Divide a stream into outlets of its composition.

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

    return {name: Stream(flow=flow, components=stream.components, schema=stream.schema) for name, flow in outlet_flows.items()}
