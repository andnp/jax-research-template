# Software Interface: Shared Transport, Minimal Protocols, and JAX-Native Execution

## Purpose

This appendix defines a software-architecture direction for the toolkit runtime.

The proposed approach is:

- one structured, authoritative latent transport/state type for runtime execution
- very small module-facing `Protocol` contracts that describe only the fields a module needs
- one file per sensor, actuator, unit, disturbance, or chemistry block
- a design that can remain compatible with end-to-end JAX execution and JIT compilation

This is intentionally opinionated because interface drift here would affect every other module family.

## Core idea

The system should have a single shared latent transport/state representation, but modules should not be forced to depend on the full surface area of that representation.

Instead:

- the runtime has one structured transport object
- each module declares a minimal `Protocol` describing the subset it reads and writes
- tests can use tiny transport fixtures that satisfy only that protocol
- the full runtime can evolve internally without forcing broad module rewrites

That is a strong fit for both modularity and testability.

## Why this makes sense

This pattern gives four major benefits.

### 1. A single authoritative transport contract

Disturbances, unit operations, chemistry blocks, and sensors all operate on the same latent process language.

That means a storm can add:

- flow
- suspended particulates
- ammonia

and any downstream module that already understands those fields can respond naturally.

### 2. Minimal coupling at module boundaries

A turbidity sensor should not need to know about every wastewater species. A chlorine-demand block should not need to know about every hydraulic diagnostic field.

Minimal `Protocol`s let each module state only what it cares about.

### 3. Much easier testing

This is one of the best parts of the idea.

If a module depends only on a tiny protocol, its tests can use tiny fixtures and tiny fake transports instead of constructing the whole world.

### 4. Easier later rearchitecture

If the full transport representation changes internally, many modules can remain stable so long as the protocol they rely on is still satisfied.

## Recommended software split

There should be three distinct layers.

### 1. Runtime transport/state type

This is the actual latent process object used in execution.

It should be:

- structured
- explicit
- stable enough to version
- suitable for JAX pytree semantics if JAX-native execution is adopted

### 2. Module-facing protocols

Each module defines or depends on a tiny protocol expressing only the fields it needs.

Examples:

- `SupportsFlow`
- `SupportsResidual`
- `SupportsAmmonia`
- `SupportsSuspendedSolids`
- `SupportsTemperature`
- `SupportsAnalyzerSignal`

The important principle is that modules speak narrow interfaces even when runtime carries a richer object.

### 3. Observation / signal contracts

Sensors and controllers should operate on signal/measurement contracts, not directly on the latent transport object.

This preserves the architecture:

- latent transport is the process truth
- sensors expose selected signals
- observations assemble selected signals for controllers or RL agents

## Transport representation guidance

The runtime transport object should be **single and structured**, but not a giant untyped bag of arbitrary keys.

Conceptually it should separate at least:

- hydraulics
- species or compositional fields
- bulk properties
- optional metadata / quality / bookkeeping fields

This helps distinguish things like:

- flow rate
- conserved or transported composition fields
- pH or temperature-like bulk properties
- non-physical metadata that should not be treated like transport

### Default representation recommendation

The default architectural recommendation is:

- use **dataclass pytrees with named fields** for the runtime transport/state shape
- keep the top-level structure explicit and readable
- allow compact fixed-shape nested substructures only where they materially improve ergonomics or JAX execution

This recommendation exists because the toolkit is a software system as much as a simulation system. Named dataclass pytrees provide better readability, editor support, refactoring behavior, protocol clarity, and debugging ergonomics than a more opaque array-first design.

### Recommended shape

Prefer a structure like:

- top-level `Transport`
- named sub-dataclasses such as `Hydraulics`, `Composition`, `BulkProperties`, and optional metadata/diagnostic groups

rather than a single flat dataclass with every possible field at top level.

This preserves readability while avoiding an unmanageably large monolith type.

### Where hybrid nested structures are acceptable

Hybrid nested compact structures are acceptable when all of the following are true:

- the field family is naturally vector-like
- the shape is stable under JAX execution
- named-per-field representation would become unwieldy
- the nested structure remains wrapped in a clear named dataclass boundary

Examples might include richer composition families or other grouped state blocks that benefit from compact internal representation.

## Pass-through semantics

To make schema evolution work, modules need a strong default policy.

### Recommended default

Modules should:

- read what they declare
- modify only what they own
- preserve/pass through all other transport fields by default

This is what allows additive evolution like:

- a disturbance starts modifying `ammonia`
- existing transport units carry it forward unchanged
- a later chemistry block begins consuming it

without earlier modules silently deleting it.

### Recommended declaration model

Each module should make it easy to tell:

- which fields it reads
- which fields it modifies
- which fields it derives
- which fields it leaves untouched

This can be documentation, static typing, light metadata, or all three.

## Tiny-file organization

The preference for many small files is sound and should be reflected directly in package layout.

### Recommended rule

Each concrete module type gets its own file.

Examples:

- one sensor per file
- one actuator per file
- one unit operation per file
- one disturbance family per file
- one chemistry block per file

This keeps files focused and lowers the cognitive cost of editing, testing, and replacing modules.

### Recommended package shape

A plausible structure is:

```text
process_toolkit/
  transport/
    runtime_types.py
    protocols/
      flow.py
      residual.py
      ammonia.py
      solids.py
      temperature.py
  units/
    cstr.py
    plug_flow_segment.py
    contact_basin.py
    mixer.py
    splitter.py
    clarifier.py
  sensors/
    flow_sensor.py
    residual_analyzer.py
    turbidity_sensor.py
    ph_sensor.py
    temperature_sensor.py
    soft_sensor.py
  actuators/
    dose_pump.py
    valve.py
    recycle_pump.py
    blower.py
  chemistry/
    first_order_decay.py
    demand_consumption.py
    oxygen_transfer.py
  disturbances/
    storm_event.py
    diurnal_profile.py
    load_slug.py
    actuator_derating.py
  runtime/
    scheduler.py
    assembly.py
    observation_builders.py
```

The exact names can change. The important point is that the module taxonomy should be visible in the filesystem, not hidden inside giant files.

## JAX-native execution: does it make sense?

Yes — with the right constraints, it makes a lot of sense.

In fact, if the toolkit is meant to support RL experimentation seriously, a JAX-native core could be a major advantage.

## Why a JAX-native runtime is attractive

### 1. End-to-end JIT compilation

A full plant step could be compiled into one optimized executable path, which matters for:

- fast rollout generation
- large-scale policy training
- batched simulation
- repeated evaluation across seeds or scenarios

### 2. Vectorization across environments

JAX makes it natural to batch over:

- seeds
- scenario variants
- controller candidates
- policy rollouts

This is very attractive for RL and component testing.

### 3. Functional state updates

JAX encourages pure-function state evolution, which is actually a good fit for simulation clarity when done well.

### 4. Optional differentiability where useful

Not every block needs to be differentiable, but a JAX-native design at least keeps the door open for:

- sensitivity analysis
- differentiable surrogates
- gradient-based calibration of selected modules

## What a JAX-native design would require

This is where the spice needs a fire extinguisher nearby.

### Requirement 1: static structure

The transport/state structure must be shape-stable inside compiled execution.

That means:

- no arbitrary field creation during a `jit`ted step
- no dynamic topology rewiring inside the compiled function
- no Python-object mutation hiding inside modules

### Requirement 2: pytree-friendly runtime types

The concrete runtime transport/state must be representable as a pytree.

That strongly suggests:

- frozen dataclasses, namedtuples, or another explicit tree structure
- arrays with stable shapes
- careful treatment of optional fields

### Requirement 3: functional modules

Modules should look more like:

```text
new_state, outputs = module(params, state, inputs, rng)
```

than like large mutable Python objects with hidden side effects.

### Requirement 4: compile-time-fixed process graph

Assembly can happen in Python, but once a plant is built, its step graph should be fixed for compiled execution.

That suggests a split between:

- **build-time graph assembly** in Python
- **run-time compiled stepping** in JAX

## The biggest architectural tension

The biggest tension is between:

- a rich, extensible structured transport contract
- and JAX’s preference for static shapes and explicit trees

This does **not** kill the idea.

It just means the runtime transport representation should be more disciplined than a free-form Python dictionary.

## Best compromise

The strongest design is probably:

- a single structured runtime transport pytree based on named dataclass pytrees
- narrow module `Protocol`s for authoring, static typing, and testing
- array-backed or dataclass-backed fixed-shape fields under the hood
- Python used for assembly and configuration
- JAX used for execution

That gives you both:

- ergonomic module development
- and a credible path to end-to-end JIT compilation

## What I would avoid

Avoid these if JAX-native execution is a real goal:

- runtime-growing dictionaries of species
- module logic that depends on Python reflection inside the hot path
- deep inheritance trees with hidden mutable state
- dynamic graph editing during a rollout
- per-module ad hoc output shapes

## Recommendation

At a high level, the proposed direction is excellent:

- one structured authoritative transport type
- narrow per-module protocols
- many small files
- JAX-native runtime ambition

My recommendation would be:

1. make the **runtime transport type** a named dataclass pytree and keep it explicitly JAX-compatible from day one
2. use **minimal protocols** for module boundaries and tests
3. reserve hybrid nested compact substructures for field families that become clearly unwieldy as pure named-per-field dataclasses
4. keep **assembly/configuration in Python**
5. keep **stepping/runtime in pure JAX-friendly functions**
6. require every module to preserve unknown transport fields unless it explicitly owns them

That would produce a toolkit that is modular, testable, extensible, and actually plausible to JIT end-to-end instead of merely daydreaming about it.

## Remaining design question

If the JAX-native direction is serious, the next major representation question is no longer the top-level default; it is the boundary for hybrid nested compact structures:

> which transport substructures, if any, should move from named-per-field dataclasses to compact fixed-shape grouped representations,
> and what criteria justify that move?

That question likely deserves its own small follow-on design note or explicit decision record once the first concrete benchmark implementation starts to stress the representation.