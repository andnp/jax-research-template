# ADR 017: Phase-Aware Streams and Field-Dependency Flowsheet Resolution

## Status
Accepted. Amends the outlet-resolution part of ADR 015 (the two-pass scheduler and
outlet contract). Keeps ADR 015's joint plant dynamics and its cycle rules for
holdups and pumps.

## Context
ADR 015 resolves a flowsheet in two passes: every outlet flow from inlet flows, then
every outlet composition from inlet streams. That fits the existing liquid-network
models, whose volumetric outlet flows do not depend on inlet composition. Two cases break
it:

- **More general liquid hydraulics.** Outlet flow can depend on composition through
  density, viscosity, reaction volume change, or phase change.
- **Gas-liquid equipment.** In a packed absorber, the gas outlet's total molar flow
  changes by the net interphase transfer and any gas-phase reaction. That depends on
  inlet composition, liquid chemistry, and temperature, and can go either way:
  absorption lowers it, while stripping or evaporation raises it.

Streams also carry no phase, temperature, or pressure. Henry's-law equilibrium, gas molar
volume, and temperature-dependent chemistry therefore have nowhere to read them from.
Freezing gas flow, or lagging it by a step, would hide the limitation rather than resolve
it.

## Decision
- **One stream carrier with phase metadata.** `Stream` gains a static phase (liquid or
  gas) and numerical temperature (K) and pressure (Pa, absolute) leaves.
  - A liquid flow is m3/h, with channel concentrations in their declared bases
    (ADR 016).
  - Connecting ports of different phases is a build error.
- **Complete gas closure.**
  - A gas flow is total mol/h. Its molecular mole fractions describe the complete
    modeled gas mixture, including an explicit inert carrier and water vapor where
    relevant, and sum to one whenever flow is positive.
  - Boundaries declare dry or wet composition and actual or standard volumetric
    conventions, and convert explicitly to this representation. A boundary given by
    carrier molar flow derives total flow from the carrier's fraction.
  - Gas outlets are assembled from every component's molar flow, untouched components
    included, before renormalization.
  - At zero flow, composition is a bounded placeholder carrying zero material. Transfer
    and reaction closures declare what they do with a stagnant or absent phase, and never
    invent removal.
- **Thermal and pressure context is declared, not averaged by default.**
  - Liquid mixing is flow-weighted on concentrations, and gas mixing is on molar amounts.
  - Temperature comes from a declared sensible-heat or enthalpy closure with stated
    reference and heat-capacity assumptions.
  - Pressure comes from an explicit boundary or a hydraulic/pressure-drop closure, and is
    never flow-weighted like a concentration.
  - An explicitly prescribed isothermal or constant-pressure context remains valid, so
    existing liquid benchmarks need not solve energy or pressure dynamics. Splitters
    preserve the context unless their model says otherwise.
  - Proxy channels such as turbidity keep their declared empirical transport and never
    enter elemental or charge balances.
- **Field-dependency resolution.**
  - A unit declares evaluation groups. Each group owns a set of output fields, possibly
    spanning several outlets, and declares exactly the inlet fields (`flow`,
    `components`, `temperature`, `pressure` of named ports) that those outputs need.
  - The builder compiles one static dependency graph over fields and schedules groups
    atomically. It adds no undeclared dependency by requiring a whole inlet or outlet
    stream. State-derived fields stay separate from inlet-derived ones: a reactor's outlet
    composition from state, its outlet flow from inlet flow.
  - Coupled fields may share one solve, so a column computes its gas and liquid outlets
    together.
  - Group results are cached only within a single resolution call. They are recomputed
    for every change of state, input, or parameter, so no cache survives an integration
    stage or a calibration parameter update. External outlets for reporting are resolved
    by the same mechanism outside right-hand-side evaluation.
- **Cycles.**
  - State-dependent holdup compositions break composition recycles, and prescribed pump
    flows break flow recycles, as in ADR 015.
  - A remaining algebraic cycle is a build error unless a unit owns and solves it
    internally. For example, a counter-current column's boundary-value problem is solved
    inside the column, never as a global flowsheet loop.
  - Adding pressure fields does not make pressure-driven hydraulic networks acyclic.
    Those need an explicit coupled hydraulic unit.
- **One engine.** No separate gas-flowsheet runner exists, and no compatibility scheduler
  remains after cutover.
  - Behavior-preserving dependency and thermodynamic primitives may land first, as
    smaller commits.
  - The final scheduler-contract replacement converts every unit, mixer, splitter, and
    caller declaration in one independently valid change.

## Consequences
- **Pros:** gas-liquid contactors, strippers, aerated tanks with off-gas, and multiphase
  plants compose in the same coupled ODE as liquid networks. Thermodynamic context
  travels with the stream.
- **Cons:** the cutover commit is necessarily large. Gas closure and thermal context add
  boundary data that liquid-only plants previously never had to supply.
