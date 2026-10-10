import math

from process_control.chemistry.species import (
    AcidBaseFamily,
    AnalyticalPool,
    HenrySolubility,
    Registry,
    Species,
)
from process_control.chemistry.thermo import AnalyticLogK, EquilibriumConstant

# All acid constants are stored as log10 of the dissociation constant, HA <-> A- + H+,
# on the molar scale. Sources quote pKa (or an association constant), so the
# coefficients below are the negated source expression.
#
# Standard atomic weights (conventional values): https://iupac.qmul.ac.uk/AtWt/
_ATOMIC_WEIGHTS = "IUPAC standard atomic weights, https://iupac.qmul.ac.uk/AtWt/"

_NBS_1949 = "Bates & Pinching 1949, NBS RP1982, https://nvlpubs.nist.gov/nistpubs/jres/42/jresv42n5p419_A1b.pdf"

# Harned & Owen evaluation, Bates & Pinching 1949 eq. 21:
# pKw = 4470.99/T - 6.0875 + 0.017060 T.
# Temperature-only fit; it ignores the density dependence of modern Kw models.
WATER_AUTOIONIZATION = EquilibriumConstant(
    AnalyticLogK(6.0875, -0.017060, -4470.99),
    "log10 Kw, H2O <-> H+ + OH-",
    (f"{_NBS_1949} eq. 21",),
    273.15,
    323.15,
)

# Plummer & Busenberg 1982, https://doi.org/10.1016/0016-7037(82)90056-4
# CO2* = CO2(aq) + H2CO3, so Ka1 is the effective first dissociation constant.
_PLUMMER_BUSENBERG = "Plummer & Busenberg 1982, https://doi.org/10.1016/0016-7037(82)90056-4"
_CARBONATE_KA1 = EquilibriumConstant(
    AnalyticLogK(-356.3094, -0.06091964, 21834.37, 126.8339, -1684915.0),
    "log10 Ka1, CO2* <-> HCO3- + H+",
    (_PLUMMER_BUSENBERG,),
    273.15,
    363.15,
)
_CARBONATE_KA2 = EquilibriumConstant(
    AnalyticLogK(-107.8871, -0.03252849, 5151.79, 38.92561, -563713.9),
    "log10 Ka2, HCO3- <-> CO3-- + H+",
    (_PLUMMER_BUSENBERG,),
    273.15,
    363.15,
)

# Hershey, Plese & Millero 1988, https://doi.org/10.1016/0016-7037(88)90183-4, as eq. 4 of
# Millero, Plese & Fernandez 1988, https://doi.org/10.4319/lo.1988.33.2.0269:
# pKa1 = -98.080 + 5765.4/T + 15.0455 ln T. The ln T term is converted to log10 T.
_SULFIDE_KA1 = EquilibriumConstant(
    AnalyticLogK(98.080, 0.0, -5765.4, -15.0455 * math.log(10.0)),
    "log10 Ka1, H2S <-> HS- + H+",
    (
        "Hershey, Plese & Millero 1988, https://doi.org/10.1016/0016-7037(88)90183-4",
        "Millero, Plese & Fernandez 1988 eq. 4, https://doi.org/10.4319/lo.1988.33.2.0269",
    ),
    273.15,
    318.15,
)
# pK2 of H2S exceeds 17 (Ellis & Giggenbach 1971, https://doi.org/10.1016/0016-7037(71)90036-6;
# Schoonen & Barnes 1988, https://doi.org/10.1016/0016-7037(88)90326-2), so S-- is omitted
# and the truncated ladder is only meaningful below this pH.
_SULFIDE_MODEL_PH_MAX = 12.0

# Morris 1966 eq. 13, https://doi.org/10.1021/j100884a007: pKa = 3000/T - 10.0686 + 0.0253 T.
_HOCL_KA = EquilibriumConstant(
    AnalyticLogK(10.0686, -0.0253, -3000.0),
    "log10 Ka, HOCl <-> OCl- + H+",
    ("Morris 1966 eq. 13, https://doi.org/10.1021/j100884a007",),
    273.15,
    308.15,
)

# Bates & Pinching 1949 eq. 20: pKa = 2835.76/T - 0.6322 + 0.001225 T.
_AMMONIUM_KA = EquilibriumConstant(
    AnalyticLogK(0.6322, -0.001225, -2835.76),
    "log10 Ka, NH4+ <-> NH3 + H+",
    (f"{_NBS_1949} eq. 20",),
    273.15,
    323.15,
)

# The PHREEQC/WATEQ expression is for association, SO4-- + H+ -> HSO4-; reversing it
# negates the coefficients: logKa = 56.889 - 0.006473 T - 2307.9/T - 19.8858 log10 T.
# This is a database evaluation rather than a primary measurement, and its range here
# is provisional: the database does not state a fit range.
_BISULFATE_KA = EquilibriumConstant(
    AnalyticLogK(56.889, -0.006473, -2307.9, -19.8858),
    "log10 Ka, HSO4- <-> SO4-- + H+ (provisional range)",
    (
        "USGS PP1728 Table 1-1, https://pubs.usgs.gov/pp/1728/pdf/p1728_508.pdf",
        "phreeqc.dat, https://github.com/usgs-coupled/phreeqc3/blob/master/database/phreeqc.dat",
    ),
    273.15,
    323.15,
)

_SPECIES = (
    Species("H+", 1, 1.008, (("H", 1),), (_ATOMIC_WEIGHTS,)),
    Species("OH-", -1, 17.007, (("O", 1), ("H", 1)), (_ATOMIC_WEIGHTS,)),
    Species("Na+", 1, 22.990, (("Na", 1),), (_ATOMIC_WEIGHTS,)),
    Species("Cl-", -1, 35.45, (("Cl", 1),), (_ATOMIC_WEIGHTS,)),
    Species("HCO3-", -1, 61.017, (("H", 1), ("C", 1), ("O", 3)), (_ATOMIC_WEIGHTS,)),
    Species("CO3--", -2, 60.009, (("C", 1), ("O", 3)), (_ATOMIC_WEIGHTS,)),
    Species("H2S", 0, 34.08, (("H", 2), ("S", 1)), (_ATOMIC_WEIGHTS,)),
    Species("HS-", -1, 33.07, (("H", 1), ("S", 1)), (_ATOMIC_WEIGHTS,)),
    Species("HOCl", 0, 52.46, (("H", 1), ("O", 1), ("Cl", 1)), (_ATOMIC_WEIGHTS,)),
    Species("OCl-", -1, 51.45, (("O", 1), ("Cl", 1)), (_ATOMIC_WEIGHTS,)),
    Species("NH4+", 1, 18.039, (("N", 1), ("H", 4)), (_ATOMIC_WEIGHTS,)),
    Species("NH3", 0, 17.031, (("N", 1), ("H", 3)), (_ATOMIC_WEIGHTS,)),
    Species("HSO4-", -1, 97.07, (("H", 1), ("S", 1), ("O", 4)), (_ATOMIC_WEIGHTS,)),
    Species("SO4--", -2, 96.06, (("S", 1), ("O", 4)), (_ATOMIC_WEIGHTS,)),
    Species("CO2", 0, 44.009, (("C", 1), ("O", 2)), (_ATOMIC_WEIGHTS,)),
    Species("O2", 0, 31.998, (("O", 2),), (_ATOMIC_WEIGHTS,)),
    Species("CH4", 0, 16.043, (("C", 1), ("H", 4)), (_ATOMIC_WEIGHTS,)),
)

_POOLS = (
    AnalyticalPool(
        "CO2_star",
        0,
        "C",
        sources=("CO2(aq) + H2CO3 pool convention of Plummer & Busenberg 1982",),
    ),
)

_FAMILIES = (
    AcidBaseFamily(
        "carbonate",
        ("CO2_star", "HCO3-", "CO3--"),
        (_CARBONATE_KA1, _CARBONATE_KA2),
        (_PLUMMER_BUSENBERG,),
    ),
    AcidBaseFamily("sulfide", ("H2S", "HS-"), (_SULFIDE_KA1,), _SULFIDE_KA1.sources, _SULFIDE_MODEL_PH_MAX),
    AcidBaseFamily("hypochlorous", ("HOCl", "OCl-"), (_HOCL_KA,), _HOCL_KA.sources),
    AcidBaseFamily("ammonium", ("NH4+", "NH3"), (_AMMONIUM_KA,), _AMMONIUM_KA.sources),
    AcidBaseFamily("bisulfate", ("HSO4-", "SO4--"), (_BISULFATE_KA,), _BISULFATE_KA.sources),
)

# Sander 2023 compilation v5.0.0, https://doi.org/10.5194/acp-23-10901-2023; each row is the
# Burkholder et al. 2019 assessed value at 298.15 K, with Hcp in mol/(m3 Pa) and B = d ln Hcp / d(1/T).
# The guard range is a conservative design envelope for the two-parameter fit; the
# published validity range of each row has not been verified.
_SANDER = "Sander 2023 (Henry's Law Constants v5.0.0), https://doi.org/10.5194/acp-23-10901-2023"
_GUARD_K = (278.15, 308.15)


def _henry(species: str, hcp_ref: float, b_kelvin: float, url: str):
    return HenrySolubility(
        species, hcp_ref, b_kelvin, *_GUARD_K, (_SANDER, f"Burkholder et al. 2019 row, {url}")
    )


# Applies to molecular CO2(aq); using it for the CO2* pool assumes CO2* ~ CO2(aq),
# which holds only at low temperature where H2CO3 is a negligible fraction.
_HENRY = (
    _henry("H2S", 1.0e-3, 2100.0, "https://www.henrys-law.org/henry/casrn/7783-06-4"),
    _henry("CO2", 3.4e-4, 2300.0, "https://henrys-law.org/henry/casrn/124-38-9"),
    _henry("O2", 1.3e-5, 1500.0, "https://www.henrys-law.org/henry/casrn/7782-44-7"),
    _henry("NH3", 5.9e-1, 4200.0, "https://www.henrys-law.org/henry/casrn/7664-41-7"),
    _henry("CH4", 1.4e-5, 1600.0, "https://www.henrys-law.org/henry/casrn/74-82-8"),
)

DEFAULT_REGISTRY = Registry(_SPECIES, _POOLS, _FAMILIES, _HENRY, WATER_AUTOIONIZATION)
