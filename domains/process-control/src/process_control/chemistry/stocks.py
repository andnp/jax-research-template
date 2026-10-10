from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from process_control.components import ComponentSchema
from process_control.fields import Phase
from process_control.streams import Stream

# Available chlorine is the Cl2-equivalent oxidizing capacity of the hypochlorite, so one mol of
# free chlorine (OCl-) corresponds to one mol of Cl2 = 70.90 g, not to 35.45 g of chlorine atoms.
# https://files.engineering.com/download.aspx?file=sodium_hypochlorite.pdf&folder=c789cf96-921c-48b4-8714-6f49ab9cee11
# (OxyChem sodium hypochlorite handbook: terminology, manufacturing and excess caustic).
CL2_MOLAR_MASS_G_PER_MOL = 70.90
NAOH_MOLAR_MASS_G_PER_MOL = 40.00

_LITERS_PER_M3 = 1000.0


@dataclass(frozen=True)
class StockChannels:
    """Carrier channel names for the roles a stock may populate; None means the carrier has no such channel."""

    sodium: str
    chloride: str | None = None
    chlorate: str | None = None
    free_chlorine: str | None = None
    carbonate: str | None = None


@dataclass(frozen=True)
class StockComposition:
    """A delivered chemical stock as carrier concentrations in mol/m3, with its physical context.

    The numerical values are parameter leaves; channel names, assay convention and provenance are
    static. Channels absent from `names` are zero in any carrier the stock is fed into.
    """

    names: tuple[str, ...]
    concentrations_mol_per_m3: jax.Array
    density_kg_per_m3: jax.Array
    temperature_k: jax.Array
    pressure_pa: jax.Array
    assay_convention: str
    source: str
    date: str

    def components(self, schema: ComponentSchema):
        """Concentrations over `schema`, zero for channels the stock does not carry."""
        for name in self.names:
            if name not in schema.names:
                raise ValueError(f"stock channel {name!r} is not in the carrier schema {schema.names}")
            if schema.bases[schema.index(name)].unit != "mol/m3":
                raise ValueError(f"stock channel {name!r} needs a mol/m3 carrier channel")

        index = jnp.array([schema.index(name) for name in self.names], dtype=int)
        return jnp.zeros(len(schema.names), dtype=self.concentrations_mol_per_m3.dtype).at[index].set(
            self.concentrations_mol_per_m3
        )

    def feed_stream(self, schema: ComponentSchema, volumetric_flow_l_per_h: ArrayLike):
        """The stock delivered at a volumetric rate as a liquid stream, flow in m3/h."""
        return Stream(
            flow=jnp.asarray(volumetric_flow_l_per_h) / _LITERS_PER_M3,
            components=self.components(schema),
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_pa,
            schema=schema,
            phase=Phase.LIQUID,
        )

    def amount_rates_mol_per_h(self, schema: ComponentSchema, volumetric_flow_l_per_h: ArrayLike):
        """Delivered amount rate of every carrier channel in mol/h: litres per hour times mol/L."""
        return jnp.asarray(volumetric_flow_l_per_h) * self.components(schema) / _LITERS_PER_M3


jax.tree_util.register_dataclass(
    StockComposition,
    data_fields=["concentrations_mol_per_m3", "density_kg_per_m3", "temperature_k", "pressure_pa"],
    meta_fields=["names", "assay_convention", "source", "date"],
)


def _stock(
    channels: dict[str, ArrayLike],
    density_kg_per_l: ArrayLike,
    temperature_k: ArrayLike,
    pressure_pa: ArrayLike,
    assay_convention: str,
    source: str,
    date: str,
):
    names = tuple(channels)
    mol_per_m3 = _LITERS_PER_M3 * jnp.stack([jnp.asarray(value, dtype=float) for value in channels.values()])
    return StockComposition(
        names=names,
        concentrations_mol_per_m3=mol_per_m3,
        density_kg_per_m3=_LITERS_PER_M3 * jnp.asarray(density_kg_per_l, dtype=float),
        temperature_k=jnp.asarray(temperature_k, dtype=float),
        pressure_pa=jnp.asarray(pressure_pa, dtype=float),
        assay_convention=assay_convention,
        source=source,
        date=date,
    )


def _require(role: str, name: str | None):
    if name is None:
        raise ValueError(f"the carrier needs a {role} channel for this stock")
    return name


def hypochlorite_from_assay(
    *,
    channels: StockChannels,
    available_cl2_g_per_l: ArrayLike,
    density_kg_per_l: ArrayLike,
    temperature_k: ArrayLike,
    pressure_pa: ArrayLike,
    source: str,
    date: str,
    excess_naoh_g_per_l: ArrayLike = 0.0,
    chloride_mol_per_l: ArrayLike | None = None,
    chlorate_mol_per_l: ArrayLike = 0.0,
    carbonate_mol_per_l: ArrayLike | None = None,
    sodium_mol_per_l: ArrayLike | None = None,
):
    """Sodium hypochlorite from a delivered assay, with free chlorine counted as mol Cl2-equivalent.

    Free chlorine is available Cl2 divided by 70.90 g/mol. Chloride defaults to the fresh-manufacture
    prior of one mol per mol free chlorine (Cl2 + 2 NaOH -> NaOCl + NaCl + H2O), which aging breaks;
    pass a measured value for aged stock. Sodium defaults to the salt recipe's balance
    c_F + chloride + chlorate + NaOH + 2 carbonate (Na2CO3 convention); a measured sodium wins.
    """
    free_chlorine = jnp.asarray(available_cl2_g_per_l, dtype=float) / CL2_MOLAR_MASS_G_PER_MOL
    chloride = free_chlorine if chloride_mol_per_l is None else jnp.asarray(chloride_mol_per_l, dtype=float)
    chlorate = jnp.asarray(chlorate_mol_per_l, dtype=float)
    base = jnp.asarray(excess_naoh_g_per_l, dtype=float) / NAOH_MOLAR_MASS_G_PER_MOL
    carbonate = jnp.asarray(0.0 if carbonate_mol_per_l is None else carbonate_mol_per_l, dtype=float)
    sodium = (
        free_chlorine + chloride + chlorate + base + 2.0 * carbonate
        if sodium_mol_per_l is None
        else jnp.asarray(sodium_mol_per_l, dtype=float)
    )

    values: dict[str, ArrayLike] = {
        channels.sodium: sodium,
        _require("free chlorine", channels.free_chlorine): free_chlorine,
        _require("chloride", channels.chloride): chloride,
        _require("chlorate", channels.chlorate): chlorate,
    }
    if carbonate_mol_per_l is not None:
        values[_require("carbonate", channels.carbonate)] = carbonate
    return _stock(
        values, density_kg_per_l, temperature_k, pressure_pa, "available Cl2 g/L, Cl2-equivalent free chlorine", source, date
    )


def caustic_from_mass_fraction(
    *,
    channels: StockChannels,
    mass_fraction: ArrayLike,
    density_kg_per_l: ArrayLike,
    temperature_k: ArrayLike,
    pressure_pa: ArrayLike,
    source: str,
    date: str,
    carbonate_mol_per_l: ArrayLike | None = None,
    chloride_mol_per_l: ArrayLike | None = None,
):
    """Sodium hydroxide of NaOH mass fraction w at density rho (kg/L), c = 1000 rho w / 40.00 mol/L.

    The density must belong to this concentration and temperature. The sodium channel adds the
    declared carbonate (2 Na each) and sodium chloride impurities; it carries no free chlorine.
    """
    naoh = _LITERS_PER_M3 * jnp.asarray(density_kg_per_l, dtype=float) * jnp.asarray(mass_fraction, dtype=float) / NAOH_MOLAR_MASS_G_PER_MOL
    carbonate = jnp.asarray(0.0 if carbonate_mol_per_l is None else carbonate_mol_per_l, dtype=float)
    chloride = jnp.asarray(0.0 if chloride_mol_per_l is None else chloride_mol_per_l, dtype=float)

    values: dict[str, ArrayLike] = {channels.sodium: naoh + 2.0 * carbonate + chloride}
    if carbonate_mol_per_l is not None:
        values[_require("carbonate", channels.carbonate)] = carbonate
    if chloride_mol_per_l is not None:
        values[_require("chloride", channels.chloride)] = chloride
    return _stock(values, density_kg_per_l, temperature_k, pressure_pa, "NaOH mass fraction and density", source, date)
