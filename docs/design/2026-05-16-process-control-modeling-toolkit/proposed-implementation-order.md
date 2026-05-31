# Proposed Implementation Order

## Purpose

This appendix proposes an implementation order for the process-control toolkit that prioritizes shared leverage rather than local completeness.

The guiding question is:

> which runtime pieces and module families should be built first because they unlock multiple benchmarks and future expansions?

The answer should reflect both architectural dependency and strategic value.

## Planning principles

The order below follows a few principles.

### Shared runtime before domain specialization

Core contracts and execution semantics should come before specialized domain modules, otherwise each domain will quietly invent its own local framework.

### Small benchmarks before high-fidelity domains

Early benchmarks should validate the runtime, observation contracts, and module seams without requiring the hardest chemistry or biology immediately.

### Shared modules before single-process luxuries

Prefer module families that unlock several future benchmarks:

- sensors
- actuators
- transport/state contracts
- uncertainty layers
- scheduler/runtime

### Strategically relevant domains should still influence order

Because membrane/fouling, coag/floc, and H2S scrubber concepts map to active plant optimization work, they should shape the expansion order once the core runtime exists.

## Recommended implementation phases

## Phase 0: Architecture and runtime spine

Build the minimum runtime needed to prevent every benchmark from becoming its own bespoke framework.

### Deliverables

- structured runtime transport/state type
- narrow module protocol pattern
- scheduler / execution-phase semantics
- signal / observation builder contracts
- core disturbance and uncertainty interfaces
- module packaging conventions and tiny-file layout

### Why first

Everything else depends on this. If it is not stabilized early, later benchmarks will encode incompatible assumptions about timing, state shape, and observability.

## Phase 1: First small benchmark and core reusable modules

Build one small benchmark that is rich enough to validate the architecture but small enough to keep implementation focused.

### Recommended first benchmark

- chlorine residual control

### Core reusable modules to build in this phase

- source / influent profile modules
- mixed tank and plug-flow / contact-basin units
- dose pump actuator
- flow sensor
- residual analyzer
- quality surrogate sensor
- simple demand-consumption block
- disturbance scheduler
- PI controller baseline
- observation-profile machinery

### Why this phase matters

This establishes:

- the runtime works
- instrumentation contracts work
- comparator baselines work
- observation asymmetry works
- disturbances and dynamics noise have a home

## Phase 2: Shared expansion modules with broad leverage

Before jumping to highly specific domain benchmarks, build a few module families that support many future processes.

### Highest-leverage shared additions

#### 1. Thermal / heat-transfer modules

Adds:

- heat-exchange block
- heater/chiller actuator
- thermal inventory state

Unlocks:

- temperature-control loops
- exchanger outlet control
- utility and recirculation benchmarks

#### 2. Pressure / pump-curve modules

Adds:

- hydraulic head or pressure block
- pump-curve actuator
- valve-Cv-like behavior

Unlocks:

- pump stations
- pressure-zone benchmarks
- lift and transfer systems

#### 3. Richer sensor/actuator realism library

Adds:

- sample-and-hold analyzers
- drift/dropout patterns
- actuator slew limits and derating
- realized-vs-requested actuation contracts

Unlocks:

- better realism across nearly every domain

### Why these go here

They multiply future benchmark options and improve realism across the board instead of helping only one domain.

## Phase 3: Strategic domain-expansion wave

Once the runtime and shared reusable module families are working, prioritize domain-specific expansions that map to active plant optimization relevance.

### Priority 1: membrane / fouling

Build:

- membrane separation unit
- fouling accumulation / recovery block
- TMP or pressure-drop behavior
- cleaning or derating scenarios

Why first in the domain wave:

- directly relevant to industrially realistic optimization problems
- high-value simulator testbed opportunity
- strong RL/control benchmark shape

### Priority 2: coagulation / flocculation

Build:

- coagulation dose-response block
- floc formation / capture block
- solids-sensitive disturbance patterns
- downstream turbidity-impact coupling

Why next:

- also directly relevant to industrially realistic optimization problems
- strong feed-forward sensing story
- naturally leverages transport and sensing architecture

### Priority 3: H2S scrubber

Build:

- gas influent source
- gas-liquid contactor
- recirculating liquid inventory
- oxidant-consumption block
- caustic / pH block
- makeup / refresh logic

Why here:

- strategically important
- richly multivariable
- slightly more domain-specific module cluster than membrane/coag additions

## Phase 4: Wastewater-style reduced-order benchmark

After the runtime and several chemically meaningful domain modules exist, build a reduced-order wastewater benchmark inspired by BSM1.

### Why not earlier

This benchmark is valuable, but it is more likely to stress:

- richer composition/state schemas
- biological reduced-order modeling
- clarifier behavior
- multiple coupled baselines

That makes it a better Phase 4 target than a first proving ground.

## Phase 5: Higher-fidelity module ladders

Once multiple benchmarks exist, start upgrading selected modules from L1 to L2 and L3 fidelity where the extra realism is worth the effort.

Likely candidates:

- chlorine transport / demand
- membrane fouling
- coag/floc capture dynamics
- H2S scrubber chemistry
- aeration / oxygen-transfer modules

## Summary table

| Phase | Focus | Why now |
|------|-------|---------|
| 0 | Runtime spine | prevents fragmented local frameworks |
| 1 | First small benchmark (chlorine) | validates contracts end-to-end |
| 2 | Shared high-leverage modules | unlocks many future domains |
| 3 | Strategic domain expansions | aligns toolkit with active plant priorities |
| 4 | Reduced-order wastewater benchmark | builds after runtime and chemistry maturity |
| 5 | Fidelity ladders | upgrades realism once the toolkit proves useful |

## If the team must be even more aggressive

If roadmap pressure forces a narrower order, the most defensible compressed sequence is:

1. runtime spine
2. chlorine benchmark
3. membrane/fouling
4. coag/floc
5. H2S scrubber

That sequence keeps the architecture honest while aligning quickly with high-value real-plant domains.

## Recommendation

The best implementation order is not the one that reaches the fanciest benchmark fastest.

It is the one that:

- stabilizes the runtime early
- proves the architecture on a compact first benchmark
- then turns quickly toward strategically relevant domain modules

On that basis, the recommended order is:

- runtime spine first
- chlorine benchmark second
- shared realism modules next
- membrane/fouling, coag/floc, and H2S scrubber as the first major domain-expansion wave

That gives the toolkit both architectural integrity and immediate business-facing value.
