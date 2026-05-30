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

Every unit operation should define:

- input material ports
- output material ports
- optional signal ports for diagnostics or control-relevant latent summaries
- internal state
- parameters
- `reset(...)`
- `step(dt, inputs, control_inputs)` or an equivalent explicit update contract

A unit operation may own its own numerical stepping details, but it should present a stable external interface.

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

### Mixer

**What it is:** a unit that combines multiple incoming streams into one outgoing stream.

**Why it matters:** many disturbances and reagent injections are best represented as stream combination rather than as direct state mutation.

**Principles to capture:**

- flow-weighted mixing
- composition reconciliation

**Ignored realism:**

- transient mixing zones
- incomplete local mixing unless explicitly needed

### Splitter / junction

**What it is:** a routing block that divides or recombines material flows.

**Why it matters:** recycle loops, bypasses, and distribution branches are common in process systems.

**Principles to capture:**

- conservation of flow and tracked components
- configurable split ratios or control-driven routing

**Ignored realism:**

- pressure-network detail in v1

### Clarifier / separation block

**What it is:** a simplified solids/liquid separation unit.

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
