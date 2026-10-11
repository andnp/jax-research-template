from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax

from process_control.events import EventSpec
from process_control.integration_records import CheckResult, check_result


@dataclass(frozen=True)
class ValidityRequirement:
    """A group diagnostic that must be truthy for the group's outputs to be used downstream."""

    group: str
    diagnostic: str


@dataclass(frozen=True)
class IntegrationPolicy[S, P]:
    """Static numerical declarations of one node.

    `check_state` returns one failure flag per name in `check_names`, in that order, before any group is
    evaluated. `continuation` names the checks bounding a domain an event locates: a location trial may
    cross them, since the physics stays evaluable just beyond, but an accepted state may not. `witness`
    maps a state failing any other check to a domain-safe one evaluated in its place, solely so masked
    lanes stay finite. `modes` maps each persistent mode name to the input port that carries it.
    `finite_rhs` asks for the post-derivative finiteness check. `events` are the node's surfaces that change `modes`.
    """

    check_names: tuple[str, ...]
    check_state: Callable[[S, Mapping[str, jax.Array], P, Mapping[str, jax.Array]], CheckResult]
    validity: tuple[ValidityRequirement, ...] = ()
    modes: Mapping[str, str] = field(default_factory=dict)
    finite_rhs: bool = True
    events: tuple[EventSpec[S, P], ...] = ()
    continuation: tuple[str, ...] = ()
    witness: Callable[[S, P], S] | None = None

    def __post_init__(self) -> None:
        if len(set(self.check_names)) != len(self.check_names):
            raise ValueError(f"check names must be unique: {self.check_names}")
        unknown = sorted(set(self.continuation) - set(self.check_names))
        if unknown:
            raise ValueError(f"continuation names undeclared checks {unknown}")
        owners = list(self.modes.values())
        if len(set(owners)) != len(owners):
            raise ValueError(f"each mode must own its own input port: {dict(self.modes)}")
        names = [event.name for event in self.events]
        if len(set(names)) != len(names):
            raise ValueError(f"event names must be unique: {names}")
        for event in self.events:
            undeclared = sorted(set(event.sets_modes) - set(self.modes))
            if undeclared:
                raise ValueError(f"event {event.name!r} sets undeclared modes {undeclared}")

    def check(self, state: S, inputs: Mapping[str, jax.Array], params: P, modes: Mapping[str, jax.Array]) -> CheckResult:
        result = self.check_state(state, inputs, params, modes)
        if result.failed.shape != (len(self.check_names),) or result.reasons.shape != result.failed.shape:
            raise ValueError(f"check_state must return one flag per declared check {self.check_names}, got shape {result.failed.shape}")
        return result


def legacy_policy[S, P]() -> IntegrationPolicy[S, P]:
    """The policy of a unit with no declared numerical domain: no state checks and no modes."""

    def no_checks(state: S, inputs: Mapping[str, jax.Array], params: P, modes: Mapping[str, jax.Array]) -> CheckResult:
        return check_result(())

    return IntegrationPolicy((), no_checks)
