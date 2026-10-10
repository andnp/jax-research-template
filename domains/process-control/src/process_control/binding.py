from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.components import ComponentBasis, ComponentSchema


@dataclass(frozen=True)
class Requirement:
    """A module-local channel name and the basis its values must have."""

    name: str
    basis: ComponentBasis


@dataclass(frozen=True)
class NamedAccess:
    """The local channels a module reads and the local channels it writes source rates to, in declared order."""

    reads: tuple[Requirement, ...] = ()
    writes: tuple[Requirement, ...] = ()

    def __post_init__(self) -> None:
        for role, requirements in (("reads", self.reads), ("writes", self.writes)):
            names = [requirement.name for requirement in requirements]
            if len(set(names)) != len(names):
                raise ValueError(f"{role} must not repeat a local name: {names}")

        written = {requirement.name: requirement.basis for requirement in self.writes}
        for requirement in self.reads:
            if written.get(requirement.name, requirement.basis) != requirement.basis:
                raise ValueError(f"{requirement.name!r} is read and written with different bases")


@dataclass(frozen=True)
class Binding:
    """Static carrier-channel indices for a module's declared reads and writes, in declared order."""

    read_index: tuple[int, ...]
    write_index: tuple[int, ...]

    @classmethod
    def resolve(cls, access: NamedAccess, schema: ComponentSchema, channel_map: Mapping[str, str] | None = None) -> "Binding":
        """Bind local names to carrier channels of `schema`, by identity unless `channel_map` renames them.

        Bases must match exactly; no unit conversion is inferred.
        """

        def indices(requirements: tuple[Requirement, ...]) -> tuple[int, ...]:
            return tuple(_channel_index(requirement, schema, channel_map) for requirement in requirements)

        write_index = indices(access.writes)
        if len(set(write_index)) != len(write_index):
            raise ValueError(f"distinct local writes map to the same channel: {[schema.names[i] for i in write_index]}")
        return cls(indices(access.reads), write_index)

    def gather(self, components: jax.Array) -> jax.Array:
        """The declared reads along the last axis."""
        return components[..., jnp.array(self.read_index, dtype=int)]

    def scatter_add(self, target: jax.Array, values: jax.Array) -> jax.Array:
        """A copy of `target` with `values`, one per declared write along the last axis, added to the bound channels."""
        return target.at[..., jnp.array(self.write_index, dtype=int)].add(values)


def _channel_index(requirement: Requirement, schema: ComponentSchema, channel_map: Mapping[str, str] | None) -> int:
    if channel_map is None:
        channel = requirement.name
    elif requirement.name in channel_map:
        channel = channel_map[requirement.name]
    else:
        raise ValueError(f"no channel mapped for local name {requirement.name!r}")
    if channel not in schema.names:
        raise ValueError(f"channel {channel!r} for local name {requirement.name!r} is not in the schema {schema.names}")

    index = schema.index(channel)
    if schema.bases[index] != requirement.basis:
        raise ValueError(f"channel {channel!r} has basis {schema.bases[index]}, but {requirement.name!r} requires {requirement.basis}")
    return index
