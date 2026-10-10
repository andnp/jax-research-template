import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from process_control.chemistry.reactions import ReactionSet
from process_control.chemistry.speciation import SpeciationSystem, make_speciation_system
from process_control.chemistry.species import Registry, Species


@dataclass(frozen=True)
class FamilyTotal:
    """One acid-base family total counting a conserved element, in mol of that element per m3.

    Every form of the registry family contributes its count of the element. `extra_members`
    names further molecular species folded into the same total, such as CO2(aq) into a
    carbonate total, so reactions may be written with real molecules.
    """

    name: str
    family: str
    element: str
    extra_members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Retained:
    """A species or solid carried as its own coordinate, `per_mol` coordinate units per mol of species.

    A charged retained species is a strong ion: its charge enters the strong-ion balance.
    """

    name: str
    species: str
    per_mol: float = 1.0

    def __post_init__(self):
        if not (math.isfinite(self.per_mol) and self.per_mol > 0.0):
            raise ValueError(f"{self.name}: per_mol must be finite and positive")


@dataclass(frozen=True)
class Inert:
    """A transported coordinate deliberately excluded from this chemistry."""

    name: str


type Row = FamilyTotal | Retained | Inert


@dataclass(frozen=True, eq=False)
class AqueousBasis:
    """Every chemistry-active transported coordinate of a dilute aqueous model, in mol/m3.

    Eliminated species (typically H+, OH- and H2O) have no transported row: acid-base
    equilibrium and charge balance recover them. They still appear in validated reactions.
    Build with `make_aqueous_basis`.
    """

    registry: Registry
    rows: tuple[Row, ...]
    eliminated: tuple[str, ...]
    names: tuple[str, ...]
    species_rows: dict[str, tuple[int, float]]
    family_rows: tuple[int, ...]
    strong_weights: np.ndarray
    speciation: SpeciationSystem

    def index(self, name: str):
        return self.names.index(name)

    def projection(self, species: tuple[str, ...]):
        """Matrix P of shape (rows, species) from molecular amounts to basis amounts."""
        matrix = np.zeros((len(self.rows), len(species)))
        for column, name in enumerate(species):
            if name in self.eliminated:
                continue
            if name not in self.species_rows:
                raise ValueError(f"{name!r} has neither a retained row nor an eliminated role")
            row, weight = self.species_rows[name]
            matrix[row, column] = weight
        return matrix

    def family_totals(self, concentrations: jax.Array):
        """Family totals in speciation-system order from basis concentrations (..., rows)."""
        return concentrations[..., jnp.array(self.family_rows, dtype=int)]

    def strong_charge(self, concentrations: jax.Array):
        """Net strong-ion charge in eq/m3, positive for cation excess."""
        return concentrations @ jnp.asarray(self.strong_weights, dtype=concentrations.dtype)


def make_aqueous_basis(registry: Registry, rows: tuple[Row, ...], eliminated: tuple[str, ...] = ()):
    """Validate a basis declaration and compile its static maps."""
    names = tuple(row.name for row in rows)
    if len(set(names)) != len(names):
        raise ValueError(f"basis row names must be unique: {names}")

    species_rows: dict[str, tuple[int, float]] = {}

    def claim(species: str, row: int, weight: float):
        if species in species_rows or species in eliminated:
            raise ValueError(f"species {species!r} is assigned to more than one role")
        species_rows[species] = (row, weight)

    for name in eliminated:
        registry.get_species(name)
    if len(set(eliminated)) != len(eliminated):
        raise ValueError(f"duplicate eliminated species: {eliminated}")

    family_rows: list[int] = []
    family_names: list[str] = []
    strong_weights = np.zeros(len(rows))
    for index, row in enumerate(rows):
        match row:
            case FamilyTotal():
                family = registry.get_family(row.family)
                forms = [registry.get_form(name) for name in (*family.forms, *row.extra_members)]
                counts = {form.count(row.element) for form in forms}
                if len(counts) != 1 or 0 in counts:
                    raise ValueError(
                        f"{row.name}: every member of {row.family} must carry the same nonzero {row.element} count"
                    )
                (count,) = counts
                for form in forms:
                    if isinstance(form, Species):
                        claim(form.name, index, float(count))
                family_rows.append(index)
                family_names.append(row.family)
            case Retained():
                species = registry.get_species(row.species)
                claim(row.species, index, row.per_mol)
                strong_weights[index] = species.charge / row.per_mol
            case Inert():
                pass

    return AqueousBasis(
        registry=registry,
        rows=rows,
        eliminated=eliminated,
        names=names,
        species_rows=species_rows,
        family_rows=tuple(family_rows),
        strong_weights=strong_weights,
        speciation=make_speciation_system(registry, tuple(family_names)),
    )


def compile_basis_stoichiometry(basis: AqueousBasis, reactions: ReactionSet):
    """S = P N: basis amount change per unit extent, shape (rows, reactions)."""
    return basis.projection(reactions.species) @ reactions.stoichiometry


def compile_transfer(basis: AqueousBasis, molecules: tuple[str, ...]):
    """B = P T: basis amount change per mol of each transferred molecule, shape (rows, molecules)."""
    return basis.projection(molecules)
