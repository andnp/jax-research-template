from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from process_control.binding import NamedAccess, Requirement
from process_control.chemistry.basis import AqueousBasis, FamilyTotal, Inert, Retained, Row
from process_control.chemistry.reactions import ReactionSet
from process_control.chemistry.speciation import (
    STANDARD_CONCENTRATION_MOL_PER_M3,
    SpeciationResult,
    evaluate_constants,
    solve_ideal_charge_balance,
)
from process_control.components import ComponentBasis, Quantity

SECONDS_PER_HOUR = 3600.0
_PROTON, _HYDROXIDE = "H+", "OH-"


class ReactionRateLaw[P](Protocol):
    """An extent rate over declared molecular concentrations, in mol reaction/m3/s."""

    @property
    def reads(self) -> tuple[str, ...]:
        """Species names, resolved from the speciation forms, retained rows, H+ and OH-, all in mol/m3."""
        ...

    def extent_rate(self, concentrations: jax.Array, temperature_k: jax.Array, params: P) -> jax.Array:
        """Rate from the declared reads along the last axis, in `reads` order, at a temperature in K."""
        ...


@dataclass(frozen=True)
class ValidityPolicy:
    """Which speciation validity flags a result needs before its rates are trusted.

    Convergence and valid inputs are always required. Admitting an extrapolated policy is a
    named choice that stays visible in the network's provenance.
    """

    name: str
    require_constants_in_domain: bool = True
    require_model_ph_in_range: bool = True


STRICT = ValidityPolicy("strict")
TEMPERATURE_EXTRAPOLATION = ValidityPolicy("temperature-extrapolation", require_constants_in_domain=False)


class NetworkEvaluation(NamedTuple):
    """Rates per hour over the network's writes, extent rates, the speciation and a usability flag.

    Unusable evaluations carry NaN rates and extents, never zero.
    """

    rates_per_hour: jax.Array
    extent_rates: jax.Array
    speciation: SpeciationResult
    usable: jax.Array


def row_component_basis(basis: AqueousBasis, row: Row):
    """The component basis a row's mol/m3 values carry, so a carrier channel can be bound exactly."""
    match row:
        case FamilyTotal():
            return ComponentBasis("mol/m3", Quantity.ELEMENT, row.element)
        case Retained() if row.per_mol == 1.0:
            return ComponentBasis("mol/m3", Quantity.SPECIES, row.species)
        case Retained():
            elements = basis.registry.get_species(row.species).elements
            if len(elements) == 1 and elements[0][1] == row.per_mol:
                return ComponentBasis("mol/m3", Quantity.ELEMENT, elements[0][0])
            raise ValueError(f"{row.name}: cannot infer what {row.per_mol} coordinate units per mol of {row.species} count")
        case Inert():
            raise ValueError(f"{row.name}: inert rows are not read by the chemistry")


@dataclass(frozen=True, eq=False)
class AqueousReactionNetwork[P]:
    """Slow aqueous reactions over a basis with one speciation solve per evaluation; a `KineticsModel`.

    Reads every chemistry-active basis row, because the pH that rate laws see depends on all
    admitted buffers and ions. Writes the rows the reactions change. Build with `make_aqueous_reaction_network`.
    """

    basis: AqueousBasis
    reactions: ReactionSet
    laws: tuple[ReactionRateLaw[P], ...]
    policy: ValidityPolicy
    access: NamedAccess
    read_rows: tuple[int, ...]
    write_stoichiometry: np.ndarray
    molecule_names: tuple[str, ...]
    law_reads: tuple[tuple[int, ...], ...]
    retained: tuple[tuple[str, int, float], ...]

    def evaluate(self, reads: jax.Array, temperature_k: jax.Array, params: P):
        """Rates, extents and speciation at one state; leading batch axes broadcast."""
        concentrations = jnp.zeros((*reads.shape[:-1], len(self.basis.rows)), dtype=reads.dtype)
        concentrations = concentrations.at[..., jnp.array(self.read_rows, dtype=int)].set(reads)

        t = jnp.asarray(temperature_k)
        constants = evaluate_constants(self.basis.speciation, t)
        speciation = solve_ideal_charge_balance(
            self.basis.speciation,
            self.basis.family_totals(concentrations),
            self.basis.strong_charge(concentrations),
            constants,
        )

        usable = speciation.converged & speciation.inputs_valid
        if self.policy.require_constants_in_domain:
            usable = usable & speciation.constants_in_domain
        if self.policy.require_model_ph_in_range:
            usable = usable & speciation.model_ph_in_range

        molecules = self._molecules(concentrations, speciation, constants.log_kw)
        extents = jnp.stack(
            [law.extent_rate(molecules[..., jnp.array(idx, dtype=int)], t, params) for law, idx in zip(self.laws, self.law_reads, strict=True)],
            axis=-1,
        )
        extents = jnp.where(usable[..., None], extents, jnp.nan)

        rates = SECONDS_PER_HOUR * extents @ jnp.asarray(self.write_stoichiometry, dtype=extents.dtype).T
        return NetworkEvaluation(rates, extents, speciation, usable)

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: P):
        return self.evaluate(reads, temperature_k, params).rates_per_hour

    def _molecules(self, concentrations: jax.Array, speciation: SpeciationResult, log_kw: jax.Array):
        c0 = STANDARD_CONCENTRATION_MOL_PER_M3
        h = c0 * 10.0 ** (-speciation.ph)
        oh = c0 * 10.0 ** (log_kw + speciation.ph)
        values = dict(zip(self.basis.speciation.form_names, jnp.moveaxis(speciation.forms_mol_per_m3, -1, 0), strict=True))
        for species, row, per_mol in self.retained:
            values[species] = concentrations[..., row] / per_mol
        values.setdefault(_PROTON, h)
        values.setdefault(_HYDROXIDE, oh)
        return jnp.stack([values[name] for name in self.molecule_names], axis=-1)


def make_aqueous_reaction_network[P](
    basis: AqueousBasis,
    reactions: ReactionSet,
    laws: Sequence[ReactionRateLaw[P]],
    policy: ValidityPolicy = STRICT,
):
    """Validate that every reaction has a law and every law read is available, then compile the static layout."""
    if len(laws) != len(reactions.reactions):
        raise ValueError(f"need one rate law per reaction: {len(reactions.reactions)} reactions, {len(laws)} laws")

    active = [(i, row) for i, row in enumerate(basis.rows) if not isinstance(row, Inert)]
    read_rows = tuple(i for i, _ in active)
    stoichiometry = basis.projection(reactions.species) @ reactions.stoichiometry
    write_rows = [i for i in read_rows if np.any(stoichiometry[i])]
    inert_writes = [basis.names[i] for i, row in enumerate(basis.rows) if isinstance(row, Inert) and np.any(stoichiometry[i])]
    if inert_writes:
        raise ValueError(f"reactions change inert rows {inert_writes}")

    retained = tuple((row.species, i, row.per_mol) for i, row in active if isinstance(row, Retained))
    retained_species = tuple(species for species, _, _ in retained)
    molecule_names = tuple(dict.fromkeys((*basis.speciation.form_names, *retained_species, _PROTON, _HYDROXIDE)))

    law_reads = []
    for reaction, law in zip(reactions.reactions, laws, strict=True):
        missing = [name for name in law.reads if name not in molecule_names]
        if missing:
            raise ValueError(f"{reaction.name}: rate law reads {missing}, which the basis cannot supply")
        law_reads.append(tuple(molecule_names.index(name) for name in law.reads))

    access = NamedAccess(
        reads=tuple(Requirement(basis.names[i], row_component_basis(basis, row)) for i, row in active),
        writes=tuple(Requirement(basis.names[i], row_component_basis(basis, basis.rows[i])) for i in write_rows),
    )
    return AqueousReactionNetwork(
        basis=basis,
        reactions=reactions,
        laws=tuple(laws),
        policy=policy,
        access=access,
        read_rows=read_rows,
        write_stoichiometry=stoichiometry[write_rows],
        molecule_names=molecule_names,
        law_reads=tuple(law_reads),
        retained=retained,
    )
