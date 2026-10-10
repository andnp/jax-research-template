# ADR 016: Named Component Access, Channel Bases, and a Species Registry

## Status
Accepted. Amends the schema and stream metadata in ADR 015, while keeping its static
schemas, pure JIT/vmap functions, and joint plant dynamics.

## Context
ADR 015 made plants continuous-time flowsheets over a static `ComponentSchema`. The
toolkit's goal is to assemble a simulator for a new plant, such as one taken from a
customer P&ID, from reusable physics, chemistry, and stoichiometry building blocks. Three
properties of the current design work against that goal:

- **Physics owns the whole vector.** A kinetics model's local species list becomes the
  stream schema, so adding one component to a plant means changing every model that
  touches the stream. `KineticsModel` requires an `oxygen` property and a full `schema`,
  which imposes an aeration assumption on every reaction system, including chlorine or
  sulfide chemistry with no oxygen at all.
- **Units are implicit.** ASM1 channels are mg COD/L, mg N/L, mg O2/L, and mol HCO3-/m3
  in the same vector. Nothing records this, so chemistry that needs molar amounts,
  charge, or elemental balances cannot convert safely.
- **Chemical constants are encoded several times.** The anaerobic digester, the
  neutralization charge solver, and the scrubber's contactor and sump each encode
  dissociation constants and temperature or property conversions, with different values
  and under different reduced speciation approximations.

## Decision
- **Physics modules declare scoped requirements.** A physics module declares named read
  and write requirements, each with its expected basis and the local phase, port, or
  inventory it refers to. A single-phase kinetics module may use a local namespace. A
  multiphase module distinguishes, for example, gas-inlet H2S, liquid-inlet total
  dissolved sulfide, and each affected outlet or inventory. At assembly a binder maps
  these requirements onto carrier channels and static index tuples, validates bases and
  phase roles, and compiles gather and scatter operations. Missing or mismatched
  requirements fail at build time, never during tracing. Channel identity is mapped
  explicitly to registry identity at assembly (ASM1's `s_o` binds to molecular O2), with
  no global aliases.
- **Modules see only what they declare.** A module receives its declared component
  inputs plus explicitly declared non-component context: phase flow, temperature,
  pressure, and parameters. It never sees, names, or types any other channel.
  - Kinetics returns source rates in a declared basis and time unit.
  - Transfer and film modules return matched phase fluxes and stoichiometric reaction
    extents.
  - Algebraic outlet transformations return declared amount changes.
  The carrier layer owns volume multiplication, unit conversion, transport, and phase
  closure. It assembles full outlets and inventories, preserving every untouched channel.
- **Pass-through means generic transport, not frozen concentration.** Untouched channels
  are still mixed, split, diluted, accumulated in holdups, and carried out with
  overflow, according to their transport classification (soluble, particulate, gas). A
  gas inert keeps its molar flow while its mole fraction changes as other species
  transfer.
- **Writers on one channel.** Source rates evaluated on the same state add in compatible
  bases. Adding rates does not make a finite step feasible when several reactions compete
  for one reagent. Integration, or a coupled reaction-extent closure, enforces shared
  availability, without order-dependent clipping and without allocating the same
  inventory to every module. Serial algebraic transformations are declared explicitly,
  never resolved by last-writer-wins.
- **Each channel declares its basis.** A schema records, per channel, its unit, its
  quantity (molecular species, elemental total, COD, equivalents, proxy), and its
  reference (for example `S`, `N`, `Cl2`, `COD`). Connections require equal schemas,
  including bases. A unit conversion is a visible operation, never an implicit cast.
- **A species registry holds chemistry.**
  - Molecular species records carry charge, molar mass, elemental composition,
    acid-base family membership, Henry's-law parameters, and literature sources.
  - Family records define what an analytical total is made of, and its
    temperature-dependent equilibrium relationships. Dynamic total values live in
    streams or state, never in the registry.
  - Registry data is static and sourced. Calibratable numbers are explicit JAX
    parameter leaves, never mutations of a global registry.
  - Non-molecular channels (COD fractions, turbidity, chlorine demand) get no invented
    molar mass or charge.
- **One independent basis per family.** An assembly transports or integrates each
  acid-base family through one independent basis, such as total dissolved sulfide.
  Equilibrium species are reconstructed from that basis and are not separate inventories
  of the same family. Strong-ion charge, a stated alkalinity convention, and family
  totals together give a non-redundant charge closure.
- **Chemistry never silently ignores a channel it needs.** A chemistry consumer such as
  the charge balance receives reduced quantities compiled from registry metadata, for
  example strong-ion charge and family totals. Participation depends on the relevant
  chemistry, not just net charge: neutral CO2 and HOCl affect pH. A channel absent from
  the registry is transported opaquely. A chemistry assembly must either establish that
  channel's contribution, mark it explicitly inert, or reject the configuration. It never
  defaults to zero charge.
- **`KineticsModel` becomes general.** It exposes its access declaration and
  `rates(reads, temperature_k, params)`, and drops `oxygen` and `schema`. Oxygen transfer
  is a separate transfer model declaring O2. ASM1's oxygen channel is bound to it at
  assembly.
- **Migration is canonical.** Consumers move in independently valid milestones. Each
  superseded owned contract or implementation is replaced together with all of its
  callers, leaving no compatibility wrappers or parallel paths.
  - Identical property formulas, such as Henry's-law temperature correction, become
    canonical helpers.
  - Distinct reduced chemistry policies, such as the digester's carbonate approximation,
    remain explicitly named models. They are not treated as duplicates of the general
    solver.
  - Converting the digester's full state is a separate decision from harvesting its
    Henry helper.

## Consequences
- **Pros:** adding a carrier channel leaves kernels whose declared access is unaffected
  untouched. Sourced chemical constants have one home. Unit errors become build errors.
- **Cons:** adding a channel still updates schemas, inventory initialization, property
  reducers, transport metadata, boundary data, and observations. A reactive addition
  such as CO2 may need an extended chemistry closure, because opaque transport does not
  make a new chemical inert. Every schema constructor must state bases. The ASM, BSM1,
  ASM2d, contact-basin, and neutralization consumers must be migrated.
