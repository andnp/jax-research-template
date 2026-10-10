import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

# CODATA 2018 molar gas constant, J/(mol K): https://physics.nist.gov/cgi-bin/cuu/Value?r
GAS_CONSTANT_J_PER_MOL_K = 8.314462618

_LN10 = math.log(10.0)


class Evaluation(NamedTuple):
    """A property value with its diagnostics; extrapolated values are finite but not in_domain."""

    value: jax.Array
    numerically_valid: jax.Array
    in_domain: jax.Array


@dataclass(frozen=True)
class AnalyticLogK:
    """log10 K(T) = a1 + a2 T + a3/T + a4 log10(T) + a5/T^2 + a6 T^2, T in kelvin.

    The form and coefficient order follow the PHREEQC convention
    (https://water.usgs.gov/water-resources/software/PHREEQC/documentation/phreeqc3-html/phreeqc3-50.htm).
    Coefficients describe one reaction direction; reversing it negates all six.
    """

    a1: float
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    a6: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(c) for c in self.coefficients):
            raise ValueError(f"AnalyticLogK coefficients must be finite, got {self.coefficients}")

    @property
    def coefficients(self):
        return (self.a1, self.a2, self.a3, self.a4, self.a5, self.a6)

    def __call__(self, temperature_k: jax.Array):
        t = jnp.asarray(temperature_k)
        return (
            self.a1
            + self.a2 * t
            + self.a3 / t
            + self.a4 * jnp.log10(t)
            + self.a5 / t**2
            + self.a6 * t**2
        )


def van_t_hoff(log10_k_ref: float, delta_h_j_per_mol: float, t_ref_k: float = 298.15):
    """Constant-enthalpy log K, with the enthalpy taken in the same direction as K."""
    if not t_ref_k > 0.0:
        raise ValueError(f"t_ref_k must be positive, got {t_ref_k}")
    a3 = -delta_h_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * _LN10)
    return AnalyticLogK(a1=log10_k_ref - a3 / t_ref_k, a3=a3)


@dataclass(frozen=True)
class EquilibriumConstant:
    """A sourced log10 K expression, the quantity it describes, and its fitted temperature range."""

    expression: AnalyticLogK
    quantity: str
    sources: tuple[str, ...]
    t_min_k: float
    t_max_k: float

    def __post_init__(self):
        if not 0.0 < self.t_min_k < self.t_max_k:
            raise ValueError(f"need 0 < t_min_k < t_max_k, got ({self.t_min_k}, {self.t_max_k})")
        if not self.sources:
            raise ValueError(f"{self.quantity}: at least one source is required")

    def evaluate(self, temperature_k: jax.Array):
        """Evaluate log10 K without clamping temperature into the fitted range."""
        t = jnp.asarray(temperature_k)
        value = self.expression(t)
        numerically_valid = (t > 0.0) & jnp.isfinite(t) & jnp.isfinite(value)
        in_range = (t >= self.t_min_k) & (t <= self.t_max_k)
        return Evaluation(value, numerically_valid, numerically_valid & in_range)
