# ADR 015: Continuous-Time Flowsheets for Process Control

## Status
Accepted. Schema/stream metadata amended by ADR 016; outlet resolution amended by ADR 017.

## Context
The BSM1 family in `process_control` deviates from the benchmark as specified in
Alex et al. (2008), *Benchmark Simulation Model no. 1 (BSM1)*,
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf. Plants used a perfect
clarifier, stepped units one after another (operator splitting, so every recycle lags by
one step), and each variant (`bsm1`, `bsm1_recycle`, `bsm1_takacs`, `bsm1_lt`, …) is a
bespoke module that re-wires the same pieces. The toolkit is meant to be composable:
new plants should be assembled from existing units, not copied. BSM1 is the natural
source of reusable units because its models and evaluation protocol are accepted by the
wastewater community.

## Decision
Plants are continuous-time flowsheets assembled from generic pieces:

- **Component schemas and streams.** A `ComponentSchema` is a static, hashable
  description of a component vector: ordered names, which components are particulate,
  and each component's COD-to-TSS factor. A `Stream` is a pytree of `flow` and a
  `components` vector; its schema is static metadata, not a leaf.
- **Stream operations.** Flow-weighted mixing (safe at zero total flow) and splitting by
  fractions or absolute flows with a remainder, conserving flow, for any schema.
- **Unit contract.** A dynamic unit provides `derivatives`, a projection onto physical
  bounds, and its outlets in two parts: `outlet_flows` from inlet flows and
  `outlet_compositions` from inlet streams. Each inlet port declares the component
  schema it expects, and each outlet declares which inlet ports its flow and its
  composition depend on algebraically. The two sets are separate because a
  constant-volume reactor passes its inlet flow straight through while its composition
  comes from state, and they name specific inlets so an inlet an outlet never reads
  cannot close a false loop. A single unit can be integrated standalone with
  `rk4_substeps`.
- **Flowsheet builder.** A declarative graph of named unit nodes and stateless
  junctions (mixers, splitters), connections including recycles, external inlets and
  outlets, and named input signals routed to nodes. Flows are resolved for the whole
  graph first, then compositions, each in a static topological order. Splitter draws
  are set by input signals, clamped at zero, and not capped by the inlet flow, since a
  cap would make every recycle a genuine algebraic loop. Only the remainder is clamped
  at zero, and an infeasible split is reported rather than silently losing mass.
  Building validates the graph in plain Python (unknown nodes or ports, unconnected
  inlets, schema mismatches, algebraic loops in either pass) and yields pure, jit- and vmap-able functions: joint `derivatives` over a pytree of node states,
  a `step` that integrates the whole plant as one coupled ODE (no operator splitting, no
  recycle lag), and the external outlet streams for a state.

BSM1 is then a configuration over BSM1 units: a bioreactor with pluggable kinetics and
kLa temperature correction 1.024^(T−15), a generalized layered Takács settler that
carries solubles, sensor classes A–D, first-order actuators, PI controllers with
back-calculation anti-windup, and a tabular influent scenario with the BSM1 dry, rain
and storm profiles and per-episode randomization. BSM1 variants become registry presets
of one configurable benchmark factory; `bsm1_takacs` folds into `bsm1`. The BSM1 family
requires float64 (`jax_enable_x64`): in float32 the slow particulate states stop moving
after about 17 simulated days because their per-step increments fall below resolution.

## Consequences
- **Pros:** one implementation per unit; recycles are solved jointly, matching the
  reference model; new plants are configurations rather than modules.
- **Cons:** other benchmarks keep sequential stepping until migrated. Existing BSM1
  experiment results come from different dynamics and are re-planned rather than
  compared against new runs.
