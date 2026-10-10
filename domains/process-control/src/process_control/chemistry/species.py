import math
from collections import Counter
from dataclasses import dataclass

from process_control.chemistry.thermo import EquilibriumConstant


def _require_unique(kind: str, names: tuple[str, ...]):
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {kind} names: {duplicates}")


@dataclass(frozen=True)
class Species:
    """A real chemical species with a molecular formula."""

    name: str
    charge: int
    molar_mass_g_per_mol: float
    elements: tuple[tuple[str, int], ...]
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise ValueError(f"{self.name}: charge must be an integer, got {self.charge!r}")
        if not (math.isfinite(self.molar_mass_g_per_mol) and self.molar_mass_g_per_mol > 0.0):
            raise ValueError(f"{self.name}: molar mass must be finite and positive")
        if not self.elements:
            raise ValueError(f"{self.name}: at least one element is required")
        _require_unique(f"{self.name} element", tuple(symbol for symbol, _ in self.elements))
        for symbol, count in self.elements:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{self.name}: element {symbol} needs a positive integer count")

    def count(self, element: str):
        return dict(self.elements).get(element, 0)


@dataclass(frozen=True)
class AnalyticalPool:
    """A named analytical form that conserves one element but is not a single molecule.

    Carbonate CO2* = CO2(aq) + H2CO3 is the motivating case: it has no exact
    formula or molar mass, so it must not be modeled as a fake molecule.
    """

    name: str
    charge: int
    conserved_element: str
    conserved_count: int = 1
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise ValueError(f"{self.name}: charge must be an integer, got {self.charge!r}")
        if self.conserved_count <= 0:
            raise ValueError(f"{self.name}: conserved_count must be positive")

    def count(self, element: str):
        return self.conserved_count if element == self.conserved_element else 0


@dataclass(frozen=True)
class AcidBaseFamily:
    """An ordered acid-base ladder from fully protonated to fully deprotonated.

    log_ka[i] is log10 of the dissociation constant of forms[i] -> forms[i + 1] + H+.
    model_ph_max flags the pH above which a truncated ladder stops being valid.
    """

    name: str
    forms: tuple[str, ...]
    log_ka: tuple[EquilibriumConstant, ...]
    sources: tuple[str, ...]
    model_ph_max: float | None = None

    def __post_init__(self):
        if len(self.forms) < 2:
            raise ValueError(f"{self.name}: a family needs at least two forms")
        if len(self.log_ka) != len(self.forms) - 1:
            raise ValueError(
                f"{self.name}: {len(self.forms)} forms need {len(self.forms) - 1} constants, got {len(self.log_ka)}"
            )
        _require_unique(f"{self.name} form", self.forms)
        if not self.sources:
            raise ValueError(f"{self.name}: at least one source is required")


@dataclass(frozen=True)
class HenrySolubility:
    """Henry solubility Hcp(T) = hcp_ref exp[b (1/T - 1/t_ref)] of a molecular species.

    hcp_ref is in mol/(m3 Pa) and b_kelvin is d ln Hcp / d(1/T). The guard range is
    a design envelope, not necessarily the published fit range.
    """

    species: str
    hcp_ref_mol_per_m3_pa: float
    b_kelvin: float
    guard_t_min_k: float
    guard_t_max_k: float
    sources: tuple[str, ...]
    t_ref_k: float = 298.15

    def __post_init__(self):
        if not (math.isfinite(self.hcp_ref_mol_per_m3_pa) and self.hcp_ref_mol_per_m3_pa > 0.0):
            raise ValueError(f"{self.species}: Hcp_ref must be finite and positive")
        if not math.isfinite(self.b_kelvin):
            raise ValueError(f"{self.species}: B must be finite")
        if not 0.0 < self.guard_t_min_k < self.guard_t_max_k:
            raise ValueError(f"{self.species}: need 0 < guard_t_min_k < guard_t_max_k")
        if not self.t_ref_k > 0.0:
            raise ValueError(f"{self.species}: t_ref_k must be positive")
        if not self.sources:
            raise ValueError(f"{self.species}: at least one source is required")


type Form = Species | AnalyticalPool


@dataclass(frozen=True)
class Registry:
    """Immutable chemistry data with separate typed namespaces.

    Species and analytical pools share one form namespace because families
    reference both by name. Water autoionization is a standalone constant,
    not an acid-base family.
    """

    species: tuple[Species, ...]
    pools: tuple[AnalyticalPool, ...]
    families: tuple[AcidBaseFamily, ...]
    henry: tuple[HenrySolubility, ...]
    water_autoionization: EquilibriumConstant

    def __post_init__(self):
        _require_unique("species/pool", tuple(f.name for f in (*self.species, *self.pools)))
        _require_unique("family", tuple(f.name for f in self.families))
        _require_unique("Henry species", tuple(h.species for h in self.henry))

        for family in self.families:
            self._validate_family(family)
        for record in self.henry:
            self._validate_henry(record)

    def get_form(self, name: str):
        for form in (*self.species, *self.pools):
            if form.name == name:
                return form
        raise KeyError(f"unknown species or pool {name!r}")

    def get_species(self, name: str):
        for species in self.species:
            if species.name == name:
                return species
        raise KeyError(f"unknown species {name!r}")

    def get_family(self, name: str):
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(f"unknown family {name!r}")

    def get_henry(self, species_name: str):
        for record in self.henry:
            if record.species == species_name:
                return record
        raise KeyError(f"no Henry record for {species_name!r}")

    def _validate_family(self, family: AcidBaseFamily):
        forms = [self._lookup_form(family.name, name) for name in family.forms]

        for upper, lower in zip(forms, forms[1:], strict=False):
            if upper.charge - lower.charge != 1:
                raise ValueError(
                    f"{family.name}: {upper.name} -> {lower.name} must lose exactly one unit of charge"
                )

        pools = [form for form in forms if isinstance(form, AnalyticalPool)]
        if pools:
            self._validate_pool_family(family.name, forms, pools[0])
            return
        for upper, lower in zip(forms, forms[1:], strict=False):
            if not isinstance(upper, Species) or not isinstance(lower, Species):
                continue
            expected = dict(upper.elements)
            expected["H"] = expected.get("H", 0) - 1
            if {k: v for k, v in expected.items() if v} != dict(lower.elements):
                raise ValueError(f"{family.name}: {upper.name} -> {lower.name} must differ by exactly one H")

    def _validate_pool_family(self, family_name: str, forms: list[Form], pool: AnalyticalPool):
        for form in forms:
            if form.count(pool.conserved_element) != pool.conserved_count:
                raise ValueError(
                    f"{family_name}: {form.name} must carry {pool.conserved_count} {pool.conserved_element}"
                )

    def _lookup_form(self, owner: str, name: str):
        try:
            return self.get_form(name)
        except KeyError:
            raise ValueError(f"{owner}: unknown species or pool {name!r}") from None

    def _validate_henry(self, record: HenrySolubility):
        try:
            species = self.get_species(record.species)
        except KeyError:
            raise ValueError(f"Henry record references unknown species {record.species!r}") from None
        if species.charge != 0:
            raise ValueError(f"Henry record {record.species} must be a neutral molecular species")
