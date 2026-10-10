# ADR 018: Fast Chemistry and Conservation in Process Simulation

## Status
Proposed. Amends ADR 015's projection and integration assumptions for chemistry-bearing
units.

## Context
Process chemistry spans timescales many orders of magnitude apart:

- **Acid-base equilibria** relax in microseconds in bulk liquid.
- **Fast bimolecular oxidation.** Azizi et al. (2015), *Competitive kinetics study of
  sulfide oxidation by chlorine using sulfite as reference compound*, report an HS-/HOCl
  rate constant of about 1.2e9 L mol^-1 s^-1 at 25 °C, from experiments over pH 6–12
  (https://doi.org/10.1016/j.cherd.2014.07.023). It is a molecular-species coefficient,
  not one that multiplies total sulfide by total free chlorine. The same study identifies
  acid-base kinetic limitations.
- **Inventories and biological populations** change over minutes to days.

The current scrubber sump fails in two ways:

- **Integration and clipping.** It integrates an empirical oxidant–sulfide rate with
  explicit RK4 sub-steps and clips concentrations afterward, rather than the cited
  elementary rate. Its intermediate RK stages can become inadmissible, and independent
  clipping of reactive inventories breaks balanced updates. Missing product inventories
  already prevent complete elemental accounting.
- **pH.** Its pH is a sigmoid of a lumped alkalinity rather than a charge balance.

Any replacement must separate those existing defects from the stiffness that elementary
fast kinetics would add if they were integrated explicitly.

## Decision
- **Each chemistry-bearing right-hand-side evaluation assembles four quantities.**
  Coupled local closures may solve them jointly.
  1. Concentrations recovered from admissible inventories and volume.
  2. Bulk speciation and pH, where they apply (see below).
  3. Matched interfacial fluxes and reaction extents from transfer and film closures.
  4. Slow source rates, evaluated on the current admissible state and its speciated or
     fast-reduced representation, together with hydraulic and thermal fluxes.

  All of these are combined into the joint derivative. Evaluating a flux never mutates
  an inventory or advances time. Explicitly serial algebraic units, such as a dosing
  mixer upstream of a column, transform their outlets according to the dependency graph
  of ADR 017. Non-reactive liquid networks do not solve pH.
- **Bulk equilibrium is algebraic within a stated regime.** Where proton transfer is fast
  relative to the modeled transport and reaction timescales, bulk speciation and pH are
  solved from independent analytical totals and charge constraints.
  - Each closure states its validity regime and activity convention. pH is
    hydrogen-ion activity, so an ideal dilute approximation must be declared as such.
  - A film or finite-kinetics model keeps kinetically relevant acid-base steps wherever
    equilibrium would overpredict reaction or transfer.
- **The equilibrium solver is bounded and honest.**
  - The initial solver supports declared aqueous acid-base families, under an explicit
    activity model and a stated temperature and concentration domain.
  - Each solve returns speciation, a scaled charge residual, convergence status, and
    validity diagnostics. Unsupported chemistry or a failed solve never silently supplies
    pH or rates as though it had converged.
  - Complexation, precipitation, non-ideal activities, and finite acid-base kinetics
    need explicit closure extensions.
  - Calibration checks derivative validity, not only convergence.
- **Film chemistry belongs to the transfer closure.** A film closure returns the net flux
  at each phase and bulk boundary, plus balanced reaction extents. It distinguishes
  unreacted solute reaching the bulk from solute converted in the film; both may be
  non-zero. Products go to the physically appropriate phase or boundary, and no converted
  amount is counted again as unreacted loading. Unmodeled product retention, solids
  formation, or volatile product release needs an explicit assumption and accounting
  boundary.
- **Instantaneous bulk reactions use invariants.**
  - An effectively instantaneous irreversible bulk reaction is represented through
    conserved reaction invariants with a constrained algebraic closure, including a
    conservative initialization onto the fast manifold. It is never resolved by explicit
    RK4 with a large rate constant, and never by subtracting `min(a, b)` inside a
    right-hand side.
  - Finite stiff dynamic kinetics would need a coupled stiff or implicit method for the
    joint flowsheet integration. Until that extension exists, they are unsupported.
  - Stepping a stiff unit on its own with frozen neighbouring inlets is not an
    alternative. It reintroduces the operator splitting and recycle lag that ADR 015
    rejected.
  - A stiff local algebraic solve inside a transfer closure is distinct from integrating
    a dynamic unit, and is allowed.
- **Admissibility comes from the integration policy, not clipping.**
  - The integration policy preserves declared inventory balances and admissibility,
    within stated numerical tolerances and operating domains.
  - Non-negative boundary rates alone do not make RK intermediate stages admissible or a
    finite step feasible.
  - Depletion and capacity limits need conservative realized fluxes, event-aware or
    suitably constrained integration, and integrated discharge accounting.
  - A projection may only enforce invariants it provably preserves, and must report
    infeasible states. Volume or temperature clipping must never create or remove
    unreported matter or energy.
  - Signed quantities such as equivalents are never clipped at zero.
- **Conservation is a tested contract.**
  - For molecular chemistry, tests verify elementally and charge-balanced reaction
    stoichiometry, and flux and inventory balances across phases and boundaries.
  - Oxidant or electron-equivalent accounting includes consumption and the specified
    product pathways. Free oxidant and alkalinity are not presumed conserved.
  - COD and proxy channels have their own declared accounting contracts.
  - Tests cover reagent depletion, zero flow, dilution, discharge, transfer in both
    directions, and extension or reordering of untouched channels.

## Consequences
- **Pros:** reducing validated fast chemistry removes its explicit stiffness. Matched
  fluxes and stoichiometry give conservative equations by construction. pH responds to
  the actual buffer composition.
- **Cons:** the remaining dynamics and algebraic solves still need timestep and
  convergence validation. The chosen numerical method and boundary handling must
  preserve balances to stated tolerances. An algebraic solve inside each chemistry
  right-hand side adds cost.
