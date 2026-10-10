from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from process_control.chemistry.species import AnalyticalPool, Registry

type Coefficient = int | Fraction


@dataclass(frozen=True)
class Reaction:
    """A balanced molecular reaction with exact signed coefficients, reactants negative.

    `normalization` states what one unit of extent means, for example one mol of S(-II) consumed.
    `elementary` is false for a reduced overall pathway, which is an endpoint rather than a mechanism.
    """

    name: str
    terms: tuple[tuple[str, Coefficient], ...]
    normalization: str
    reversible: bool = False
    elementary: bool = False
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.terms:
            raise ValueError(f"{self.name}: a reaction needs at least one term")

        exact = []
        for species, coefficient in self.terms:
            if isinstance(coefficient, bool) or not isinstance(coefficient, int | Fraction):
                raise ValueError(f"{self.name}: coefficient of {species} must be an exact int or Fraction, got {coefficient!r}")
            if coefficient == 0:
                raise ValueError(f"{self.name}: coefficient of {species} is zero")
            exact.append((species, Fraction(coefficient)))

        names = [species for species, _ in exact]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{self.name}: duplicate terms {duplicates}")
        object.__setattr__(self, "terms", tuple(exact))

    def coefficient(self, species: str):
        return dict(self.terms).get(species, Fraction(0))


def validate_reaction(registry: Registry, reaction: Reaction):
    """Require every species to be a registry molecule and every element and the charge to balance."""
    element_sums: dict[str, Fraction] = {}
    charge = Fraction(0)
    for name, coefficient in reaction.terms:
        try:
            form = registry.get_form(name)
        except KeyError:
            raise ValueError(f"{reaction.name}: unknown species {name!r}") from None
        if isinstance(form, AnalyticalPool):
            raise ValueError(f"{reaction.name}: {name!r} is an analytical pool with no molecular formula")

        charge += coefficient * form.charge
        for element, count in form.elements:
            element_sums[element] = element_sums.get(element, Fraction(0)) + coefficient * count

    unbalanced = {element: str(total) for element, total in element_sums.items() if total}
    if unbalanced:
        raise ValueError(f"{reaction.name}: elements do not balance: {unbalanced}")
    if charge:
        raise ValueError(f"{reaction.name}: net charge {charge} does not balance")


@dataclass(frozen=True, eq=False)
class ReactionSet:
    """Validated reactions compiled once to a float matrix over their species.

    stoichiometry has shape (len(species), len(reactions)): an extent vector xi gives
    species amount changes `stoichiometry @ xi`.
    """

    reactions: tuple[Reaction, ...]
    species: tuple[str, ...]
    stoichiometry: np.ndarray


def compile_reactions(registry: Registry, reactions: tuple[Reaction, ...]):
    """Validate every reaction against the registry and compile their molecular stoichiometry."""
    names = [reaction.name for reaction in reactions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate reaction names: {duplicates}")
    for reaction in reactions:
        validate_reaction(registry, reaction)

    species = tuple(dict.fromkeys(name for reaction in reactions for name, _ in reaction.terms))
    matrix = np.array(
        [[float(reaction.coefficient(name)) for reaction in reactions] for name in species],
        dtype=np.float64,
    ).reshape(len(species), len(reactions))
    return ReactionSet(reactions, species, matrix)
