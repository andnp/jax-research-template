from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from process_control.events import EventPoint, EventSpec
from process_control.integration_policy import IntegrationPolicy
from process_control.integration_records import IntegrationContext


@dataclass(frozen=True)
class RegisteredEvent:
    node: str
    spec: EventSpec
    local_modes: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.node, self.spec.name)

    @property
    def written(self) -> frozenset[str]:
        return frozenset(f"{self.node}.{mode}" for mode in self.spec.sets_modes)

    def local_view(self, context: IntegrationContext) -> dict[str, jax.Array]:
        return {mode: context.modes[f"{self.node}.{mode}"] for mode in self.local_modes}


@dataclass(frozen=True)
class EventRegistry:
    """Events in deterministic node-then-declaration order; that order breaks ties and nothing more."""

    events: tuple[RegisteredEvent, ...]

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(event.key for event in self.events)

    def active(self, context: IntegrationContext) -> jax.Array:
        flags = [jnp.asarray(event.spec.active(event.local_view(context)), dtype=bool) for event in self.events]
        return jnp.stack(flags) if flags else jnp.zeros((0,), dtype=bool)

    def surfaces(self, point: EventPoint) -> jax.Array:
        values = [
            jnp.asarray(
                event.spec.surface(
                    point.state[event.node],
                    point.params[event.node],
                    event.local_view(point.context),
                    point.inputs[event.node],
                    point.fields[event.node],
                ),
                dtype=float,
            )
            for event in self.events
        ]
        return jnp.stack(values) if values else jnp.zeros((0,))

    def conflicts(self, selected: jax.Array) -> jax.Array:
        """True when two distinct selected events write the same mode, so their transitions cannot both be honoured."""
        count = len(self.events)
        shared = np.array([[i != j and bool(a.written & b.written) for j, b in enumerate(self.events)] for i, a in enumerate(self.events)], dtype=bool)
        return jnp.any(selected[:, None] & selected[None, :] & shared.reshape(count, count))

    def apply(self, index: jax.Array, context: IntegrationContext) -> IntegrationContext:
        """Apply the transition of event `index`; modes keep their dtypes."""
        if not self.events:
            return context
        return jax.lax.switch(index, [_transition(event) for event in self.events], context)


def _transition(event: RegisteredEvent) -> Callable[[IntegrationContext], IntegrationContext]:
    def run(context: IntegrationContext) -> IntegrationContext:
        updated = event.spec.transition(event.local_view(context))
        if set(updated) != set(event.spec.sets_modes):
            raise ValueError(f"event {event.key} must set exactly {sorted(event.spec.sets_modes)}, set {sorted(updated)}")
        modes = dict(context.modes)
        for mode, value in updated.items():
            key = f"{event.node}.{mode}"
            modes[key] = jnp.asarray(value, dtype=modes[key].dtype)
        return IntegrationContext(modes)

    return run


def compile_registry(policies: Mapping[str, IntegrationPolicy]) -> EventRegistry:
    """Collect events from node policies in the mapping's order, then each policy's declaration order."""
    events = tuple(RegisteredEvent(node, spec, tuple(policy.modes)) for node, policy in policies.items() for spec in policy.events)
    return EventRegistry(events)
