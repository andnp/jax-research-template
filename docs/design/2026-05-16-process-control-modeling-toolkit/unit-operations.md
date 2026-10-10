# Unit Operations

## Purpose

Unit operations are the plant-shaped building blocks of the toolkit. They carry most of the domain realism while staying far simpler than a digital twin.

## What this module family is

A unit operation is a dynamic process object that transforms incoming material streams into outgoing material streams while maintaining internal state.

Examples include:

- continuously stirred tank reactor (CSTR)
- plug-flow segment or pipe reach
- contact basin
- mixer
- splitter
- clarifier
- recycle junction
- equalization tank
- first-order holding volume with delay

## Contract

Dynamic units follow the continuous-time contract of [ADR 015](../../adrs/015-continuous-time-flowsheets.md), implemented as the `ContinuousUnit` protocol in [`units/contract.py`](../../../domains/process-control/src/process_control/units/contract.py). A unit declares:

- `inlet_ports`: each inlet's `ComponentSchema`
- `outlet_ports`: each outlet's `Outlet(schema, flow_from, composition_from)`, naming the inlet ports its flow and its composition depend on algebraically
- `input_ports`: signal inputs such as `kla` or `q_underflow`
- `derivatives(state, inlets, inputs, params)`
- `outlet_flows(state, inlet_flows, inputs, params)` and `outlet_compositions(state, inlets, inputs, params)`
- `project(state, params)`: projection onto physical bounds (e.g. non-negative concentrations)

The flow and composition dependency sets are separate and name specific inlets. A constant-volume reactor passes its inlet flow straight through (`flow_from={"in"}`) while its composition comes from state (`composition_from=∅`), so it breaks composition loops. An outlet must not read an inlet it does not declare: the flowsheet evaluates it with placeholders there.

Units no longer own their stepping. `step_unit` integrates one unit standalone with `rk4_substeps`; plants integrate all units jointly (see [scheduler](scheduler-and-execution-phase-contract.md), phase 5). Units not yet migrated, such as the other benchmarks' units, keep their sequential `step`-style contract until moved.

## Recommended first-generation unit types

### CSTR / mixed tank

**What it is:** a perfectly mixed volume where incoming material is instantaneously homogenized.

**Why it matters:** it is the simplest realistic inventory model and appears everywhere in process systems.

**Principles to capture:**

- inventory accumulation
- dilution and washout
- simple reaction or consumption in a bulk volume
- overflow / level constraints when needed

**Ignored realism:**

- dead zones
- stratification
- short-circuiting
- geometry-specific mixing behavior

### Plug-flow segment / pipe reach

**What it is:** a delay-dominated transport element where material moves downstream with limited back-mixing.

**Why it matters:** many control problems are dominated by transport delay rather than by equilibrium behavior.

**Principles to capture:**

- residence-time delay
- downstream propagation of disturbances
- optional consumption or decay during transport

**Ignored realism:**

- detailed axial dispersion in v1 unless needed
- full hydraulic transients

### Contact basin

**What it is:** a multi-segment plug-flow-style unit for reaction during transport, especially appropriate for disinfection examples.

**Why it matters:** it maps directly onto chlorine residual and contact-time problems.

**Principles to capture:**

- compartmental delay
- consumption / residual evolution across the basin
- sparse internal sensing points

**Ignored realism:**

- baffling geometry details beyond what affects effective delay

### Aerated bioreactor (implemented)

`AeratedBioreactor` in [`units/bioreactor.py`](../../../domains/process-control/src/process_control/units/bioreactor.py) is a constant-volume CSTR whose reaction terms come from a pluggable `KineticsModel` ([`kinetics.py`](../../../domains/process-control/src/process_control/kinetics.py)): a component schema, the name of the oxygen component, `at_temperature`, and `rates` per hour. ASM1 is the BSM1 kinetics. Aeration transfers kLa (corrected by 1.024^(T-15)) times the oxygen deficit into the oxygen component. Inputs are `kla` and optionally `temperature`.

### Mixer

**What it is:** a unit that combines multiple incoming streams into one outgoing stream.

**Why it matters:** many disturbances and reagent injections are best represented as stream combination rather than as direct state mutation.

**Implemented:** `Mixer` in [`flowsheet.py`](../../../domains/process-control/src/process_control/flowsheet.py) is a stateless junction (flow-weighted `mix`) whose outlet depends on all inlets for both flow and composition.

**Principles to capture:**

- flow-weighted mixing
- composition reconciliation

**Ignored realism:**

- transient mixing zones
- incomplete local mixing unless explicitly needed

### Splitter / junction

**What it is:** a routing block that divides or recombines material flows.

**Why it matters:** recycle loops, bypasses, and distribution branches are common in process systems.

**Implemented:** `Splitter` in [`flowsheet.py`](../../../domains/process-control/src/process_control/flowsheet.py) is a stateless junction. Named draw outlets take input-signal flows, clamped at zero but not capped by the inlet flow (a cap would turn every recycle into an algebraic loop); the remainder outlet takes the rest, clamped at zero, and any infeasible split is reported as a deficit.

**Principles to capture:**

- conservation of flow and tracked components
- configurable split ratios or control-driven routing

**Ignored realism:**

- pressure-network detail in v1

### Clarifier / separation block

**What it is:** a simplified solids/liquid separation unit. Implemented for wastewater as `LayeredSettler` ([`units/layered_settler.py`](../../../domains/process-control/src/process_control/units/layered_settler.py)): a one-dimensional Takacs layered settler (BSM1: 10 layers, feed at layer 5) that carries every component per layer, so solubles move with the bulk flow and particulates also settle by their share of layer TSS. It has a `feed` inlet, `effluent` and `underflow` outlets, and a `q_underflow` input; the underflow is capped by the feed so outlets always balance it. `ParticulateOutlets` selects how outlets compose particulates: `FEED_FRACTIONS` (BSM1 default; outlets depend on the feed composition) or `LAYER_COMPOSITION` (outlets carry the layer composition, delayed by retention).

**Why it matters:** this is required for wastewater-style examples and other separation problems.

**Principles to capture:**

- phase or component separation behavior adequate for control benchmarking
- sensitivity to hydraulic loading where it matters

**Ignored realism:**

- full settling physics if a reduced-order model is enough for controller evaluation

## Realistic principles to capture

- inventory and delay matter more than geometric detail
- simple conservation should hold where it is central to the control task
- units should expose diagnostics only through configured ports, not arbitrary internal state access
- each unit should be swappable so alternative process topologies can be tested cleanly

## Realism we are intentionally ignoring

- CFD-level hydraulics
- plant-design-grade geometry modeling
- every secondary chemical or biological mechanism
- all maintenance and operator intervention behaviors inside the unit itself

## Recommended first-generation priorities

1. mixed tank
2. plug-flow segment
3. contact basin
4. mixer and splitter
5. clarifier / separation block
6. recycle link / junction helper

That set is enough to express both the current WTP and much of BSM1-like structure.
