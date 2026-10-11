from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum

import jax

from process_control.fields import LocalField
from process_control.integration_records import IntegrationContext

type Modes = Mapping[str, jax.Array]


class CrossingDirection(IntEnum):
    DOWN = -1
    BOTH = 0
    UP = 1


class Continuation(IntEnum):
    """What an event allows after it fires: apply its transition and go on, or end the advance."""

    TRANSITION = 0
    TERMINAL = 1


@dataclass(frozen=True)
class EventTolerance:
    """Narrowest time bracket (h) and largest surface residual (in the surface's units) that count as located."""

    time_h: float
    surface: float


@dataclass(frozen=True)
class EventSpec[S, P]:
    """A scalar surface whose directional zero crossing changes a node's discrete modes.

    `surface` reads the node's state and parameters, its local modes, its routed inputs and the resolved
    fields named in `stage_reads`. It is evaluated only while `active` holds for the local modes. `transition`
    returns new values for exactly the modes in `sets_modes`; it never changes continuous state.
    """

    name: str
    surface: Callable[[S, P, Modes, Modes, Mapping[LocalField, jax.Array]], jax.Array]
    surface_units: str
    direction: CrossingDirection
    active: Callable[[Modes], jax.Array]
    tolerance: EventTolerance
    transition: Callable[[Modes], Modes]
    sets_modes: tuple[str, ...]
    stage_reads: tuple[LocalField, ...] = ()
    continuation: Continuation = Continuation.TRANSITION


@dataclass(frozen=True)
class EventPoint:
    """Everything an event surface may read, keyed by node."""

    state: Mapping[str, object]
    params: Mapping[str, object]
    context: IntegrationContext
    inputs: Mapping[str, Modes]
    fields: Mapping[str, Mapping[LocalField, jax.Array]]
