from dataclasses import dataclass

from process_control.components import ComponentBasis


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
