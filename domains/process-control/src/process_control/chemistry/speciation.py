import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control.chemistry.species import AcidBaseFamily, Registry
from process_control.chemistry.thermo import EquilibriumConstant

# The model is ideal: activity coefficients are all one, constants are treated as
# molar-scale values (1 mol/kg solvent ~ 1 mol/L, an error of about one percent at 50 C),
# and pH means -log10 of the H+ concentration in mol/L. Ionic strength is not computed,
# so applicability of the ideal assumption to a given solution is unknown, not verified.
IONIC_STRENGTH_APPLICABILITY = "unknown"

# Ideal molar standard state in mol/m3, converting the mol/L scale of pH and the constants.
STANDARD_CONCENTRATION_MOL_PER_M3 = 1000.0

_LN10 = math.log(10.0)


@dataclass(frozen=True)
class SpeciationSystem:
    """Static layout of independent acid-base families for one charge-balance problem.

    Dynamic arrays use the trailing axis of this layout: totals are (..., len(families)),
    constants (..., n_constants) and forms (..., len(form_names)), families in the order given.
    """

    families: tuple[AcidBaseFamily, ...]
    water: EquilibriumConstant
    form_names: tuple[str, ...]
    form_charges: tuple[int, ...]
    form_offsets: tuple[int, ...]
    constant_offsets: tuple[int, ...]

    @property
    def n_constants(self):
        return self.constant_offsets[-1]


def make_speciation_system(registry: Registry, family_names: tuple[str, ...]):
    """Build a layout from registry families; each form may belong to only one family."""
    families = tuple(registry.get_family(name) for name in family_names)
    form_names = tuple(form for family in families for form in family.forms)
    if len(set(form_names)) != len(form_names):
        raise ValueError("families overlap: a form would be counted in two independent totals")

    form_offsets = [0]
    constant_offsets = [0]
    for family in families:
        form_offsets.append(form_offsets[-1] + len(family.forms))
        constant_offsets.append(constant_offsets[-1] + len(family.log_ka))

    return SpeciationSystem(
        families=families,
        water=registry.water_autoionization,
        form_names=form_names,
        form_charges=tuple(registry.get_form(name).charge for name in form_names),
        form_offsets=tuple(form_offsets),
        constant_offsets=tuple(constant_offsets),
    )


class EvaluatedConstants(NamedTuple):
    """log10 Ka in system order and log10 Kw, with diagnostics aggregated over all constants."""

    log_ka: jax.Array
    log_kw: jax.Array
    numerically_valid: jax.Array
    temperature_in_domain: jax.Array


def evaluate_constants(system: SpeciationSystem, temperature_k: jax.Array):
    """Evaluate every constant at the given temperature without clamping it."""
    t = jnp.asarray(temperature_k)
    water = system.water.evaluate(t)
    evaluations = [constant.evaluate(t) for family in system.families for constant in family.log_ka]

    if evaluations:
        log_ka = jnp.stack([e.value for e in evaluations], axis=-1)
    else:
        log_ka = jnp.zeros((*t.shape, 0), dtype=water.value.dtype)

    valid = water.numerically_valid
    in_domain = water.in_domain
    for e in evaluations:
        valid = valid & e.numerically_valid
        in_domain = in_domain & e.in_domain
    return EvaluatedConstants(log_ka, water.value, valid, in_domain)


def fractions(log_ka: jax.Array, ph: jax.Array):
    """Equilibrium fractions of one family's forms, summing to one along the last axis.

    log_ka holds log10 Ka for each dissociation step, most protonated first. Weights are
    normalized in log space so extreme pH never overflows.
    """
    ph = jnp.asarray(ph)[..., None]
    steps = jnp.arange(1, log_ka.shape[-1] + 1, dtype=log_ka.dtype)
    log_weights = steps * ph + jnp.cumsum(log_ka, axis=-1)
    log_weights = jnp.concatenate([jnp.zeros_like(log_weights[..., :1]), log_weights], axis=-1)
    return jax.nn.softmax(log_weights * _LN10, axis=-1)


@dataclass(frozen=True)
class SolveControls:
    """Numerical controls; search bounds are numerical limits, not model validity limits."""

    ph_bounds: tuple[float, float] = (-3.0, 17.0)
    max_iterations: int = 60
    abs_tol_eq_per_m3: float = 1e-8
    rel_tol: float = 1e-4
    ph_tol: float = 1e-4


DEFAULT_CONTROLS = SolveControls()


class SpeciationResult(NamedTuple):
    """Solution and statuses; ph and forms are NaN when inputs are invalid.

    converged needs both a small scaled charge residual and a small pH error estimate.
    constants_in_domain concerns the fitted temperature ranges, independent of convergence.
    """

    ph: jax.Array
    forms_mol_per_m3: jax.Array
    residual_eq_per_m3: jax.Array
    ph_error_estimate: jax.Array
    converged: jax.Array
    bracket_ok: jax.Array
    inputs_valid: jax.Array
    constants_in_domain: jax.Array
    model_ph_in_range: jax.Array


def _family_log_ka(system: SpeciationSystem, log_ka: jax.Array, index: int):
    start, stop = system.constant_offsets[index : index + 2]
    return log_ka[..., start:stop]


def _balance(
    system: SpeciationSystem,
    totals: jax.Array,
    strong_charge: jax.Array,
    log_ka: jax.Array,
    log_kw: jax.Array,
    ph: jax.Array,
):
    """Charge residual (eq/m3), its pH derivative, and a magnitude scale for tolerances."""
    c0 = STANDARD_CONCENTRATION_MOL_PER_M3
    h = c0 * 10.0 ** (-ph)
    oh = c0 * 10.0 ** (log_kw + ph)

    residual = strong_charge + h - oh
    curvature = h + oh
    scale = jnp.abs(strong_charge) + h + oh

    for i in range(len(system.families)):
        start, stop = system.form_offsets[i : i + 2]
        z = jnp.asarray(system.form_charges[start:stop], dtype=ph.dtype)
        alpha = fractions(_family_log_ka(system, log_ka, i), ph)
        mean_z = jnp.sum(alpha * z, axis=-1)
        total = totals[..., i]

        residual = residual + total * mean_z
        curvature = curvature + total * (jnp.sum(alpha * z**2, axis=-1) - mean_z**2)
        scale = scale + total * jnp.sum(alpha * jnp.abs(z), axis=-1)

    return residual, -_LN10 * curvature, scale


def solve_ideal_charge_balance(
    system: SpeciationSystem,
    totals_mol_per_m3: jax.Array,
    strong_ion_charge_eq_per_m3: jax.Array,
    constants: EvaluatedConstants,
    controls: SolveControls = DEFAULT_CONTROLS,
):
    """Solve sum(z c) = 0 for pH with H+ and OH- from Kw and each family in ideal equilibrium.

    strong_ion_charge_eq_per_m3 is positive for net cation excess. The residual is monotone
    decreasing in pH, so a safeguarded Newton iteration keeps a sign bracket and falls back to
    bisection; the iteration count is fixed so the solver is jit- and vmap-safe. Leading batch
    axes of totals, strong charge and constants broadcast; totals' last axis is the family axis.
    """
    totals = jnp.asarray(totals_mol_per_m3)
    if totals.shape[-1] != len(system.families):
        raise ValueError(f"totals last axis must have {len(system.families)} families, got {totals.shape[-1]}")

    batch = jnp.broadcast_shapes(
        totals.shape[:-1],
        jnp.shape(strong_ion_charge_eq_per_m3),
        constants.log_kw.shape,
        constants.log_ka.shape[:-1],
    )
    totals = jnp.broadcast_to(totals, (*batch, len(system.families)))
    strong_charge = jnp.broadcast_to(jnp.asarray(strong_ion_charge_eq_per_m3, dtype=totals.dtype), batch)
    log_kw = jnp.broadcast_to(constants.log_kw, batch).astype(totals.dtype)
    log_ka = jnp.broadcast_to(constants.log_ka, (*batch, system.n_constants)).astype(totals.dtype)

    inputs_valid = (
        jnp.all(jnp.isfinite(totals) & (totals >= 0.0), axis=-1)
        & jnp.isfinite(strong_charge)
        & jnp.broadcast_to(constants.numerically_valid, batch)
        & jnp.isfinite(log_kw)
        & jnp.all(jnp.isfinite(log_ka), axis=-1)
    )
    # Invalid lanes solve a harmless pure-water problem so they cannot poison the batch.
    totals = jnp.where(inputs_valid[..., None], totals, 0.0)
    strong_charge = jnp.where(inputs_valid, strong_charge, 0.0)
    log_kw = jnp.where(inputs_valid, log_kw, -14.0)
    log_ka = jnp.where(inputs_valid[..., None], log_ka, 0.0)

    def balance(ph: jax.Array):
        return _balance(system, totals, strong_charge, log_ka, log_kw, ph)

    def tolerance(scale: jax.Array):
        return controls.abs_tol_eq_per_m3 + controls.rel_tol * scale

    ph_min = jnp.full(batch, controls.ph_bounds[0], dtype=totals.dtype)
    ph_max = jnp.full(batch, controls.ph_bounds[1], dtype=totals.dtype)
    bracket_ok = (balance(ph_min)[0] > 0.0) & (balance(ph_max)[0] < 0.0)

    def step(_: int, state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        ph, lo, hi, last_step = state
        residual, slope, scale = balance(ph)

        lo = jnp.where(residual > 0.0, ph, lo)
        hi = jnp.where(residual > 0.0, hi, ph)

        newton = ph - residual / slope
        newton_ok = (
            jnp.isfinite(newton)
            & (newton > lo)
            & (newton < hi)
            & (jnp.abs(2.0 * residual) <= jnp.abs(last_step * slope))
        )
        proposal = jnp.where(newton_ok, newton, 0.5 * (lo + hi))

        settled = (jnp.abs(residual) <= tolerance(scale)) & (jnp.abs(residual / slope) <= controls.ph_tol)
        proposal = jnp.where(settled, ph, proposal)
        return proposal, lo, hi, jnp.where(settled, last_step, proposal - ph)

    ph0 = jnp.clip(-0.5 * log_kw, ph_min, ph_max)
    ph, _, _, _ = jax.lax.fori_loop(0, controls.max_iterations, step, (ph0, ph_min, ph_max, ph_max - ph_min))

    residual, slope, scale = balance(ph)
    ph_error = jnp.abs(residual / slope)
    converged = (
        inputs_valid
        & bracket_ok
        & (jnp.abs(residual) <= tolerance(scale))
        & (ph_error <= controls.ph_tol)
        & jnp.isfinite(ph)
    )

    forms = [
        totals[..., i, None] * fractions(_family_log_ka(system, log_ka, i), ph) for i in range(len(system.families))
    ]
    forms_flat = jnp.concatenate(forms, axis=-1) if forms else jnp.zeros((*batch, 0), dtype=totals.dtype)

    ceilings = [(i, f.model_ph_max) for i, f in enumerate(system.families) if f.model_ph_max is not None]
    model_ph_in_range = jnp.ones(batch, dtype=bool)
    for i, ceiling in ceilings:
        model_ph_in_range = model_ph_in_range & ((totals[..., i] <= 0.0) | (ph <= ceiling))

    nan = jnp.asarray(jnp.nan, dtype=totals.dtype)
    return SpeciationResult(
        ph=jnp.where(inputs_valid, ph, nan),
        forms_mol_per_m3=jnp.where(inputs_valid[..., None], forms_flat, nan),
        residual_eq_per_m3=jnp.where(inputs_valid, residual, nan),
        ph_error_estimate=jnp.where(inputs_valid, ph_error, nan),
        converged=converged,
        bracket_ok=bracket_ok,
        inputs_valid=inputs_valid,
        constants_in_domain=jnp.broadcast_to(constants.temperature_in_domain, batch),
        model_ph_in_range=model_ph_in_range,
    )
