# Validation, Schema Governance, Assembly, and Module Certification

## Purpose

This appendix defines the non-negotiable framework needed to keep the process-control toolkit usable, trustworthy, and maintainable as it grows.

The topics covered here are:

1. validation and calibration strategy
2. units, naming, and schema governance
3. plant assembly and topology specification
4. module certification and invariant testing

These topics are not optional polish. They are what prevent a modular toolkit from turning into a large collection of incompatible components with unclear scientific status.

## 1. Validation and calibration strategy

### Why this matters

The toolkit should not treat “module exists” as equivalent to “module is good enough.”

Each module and benchmark needs a path to demonstrate that it is:

- behaviorally plausible
- internally consistent
- useful for its intended benchmark purpose

### Validation layers

#### Module-level validation

Every module should define expected qualitative and quantitative behaviors.

Examples:

- a tank inventory responds correctly to step inflow changes
- a bleach-consumption block reduces oxidant state under load
- a sensor lag model produces the expected sample-and-hold dynamics
- a fouling block drifts in the right direction under sustained stress

#### Benchmark-level validation

Each benchmark should have a small suite of reference scenarios that test whether the assembled process behaves plausibly.

Examples:

- load step response
- flow surge response
- reagent underfeed / overfeed scenarios
- sensor-failure scenarios
- warmup and recovery behavior

#### Reference-behavior validation

Where appropriate, modules or benchmarks should be compared against:

- accepted reference models
- site-inspired operating traces
- engineering expectations from domain SMEs
- previously approved benchmark versions

### Calibration strategy

Calibration does not need to be fully automated in v1, but the toolkit should support it.

That implies:

- explicit parameter sets
- clean separation between structure and parameterization
- reproducible scenario packs
- easy replay against reference traces

### Recommended artifact types

- parameter pack
- scenario pack
- validation scenario set
- golden output trace or behavior envelope

## 2. Units, naming, and schema governance

### Why this matters

Once modules are reused across domains, ambiguity in names and units becomes a serious risk.

Without explicit governance, the toolkit will eventually accumulate bugs caused by:

- concentration vs load confusion
- inconsistent naming for the same concept
- mixed physical and normalized quantities
- signal names that drift across benchmarks

### Units policy

The toolkit should make units explicit for:

- latent transport fields
- sensor signals
- actuator signals
- benchmark metrics

The design does not require a heavy runtime unit library on day one, but it does require an explicit canonical unit convention and metadata discipline.

### Naming policy

The toolkit should use canonical names for:

- shared transport fields
- benchmark signal names
- common species or process properties
- observation-profile features where stability matters

Where aliases exist, one canonical name should still be the source of truth.

### Schema governance

The toolkit should define:

- how transport schemas evolve
- how observation profiles are versioned
- how new fields are introduced
- how deprecated fields are handled

The most important rule is that schema evolution should be additive and reviewable, not informal and silent.

## 3. Plant assembly and topology specification

### Why this matters

The toolkit has a clear conceptual graph model, but that is not enough by itself. It also needs a repeatable way to describe a plant instance.

### Assembly questions the toolkit must answer

- how are modules instantiated?
- how are ports wired?
- how are units named?
- how are scenario modules attached to the process graph?
- how are sensors attached to latent state seams?
- how are observation builders tied to the instrumentation graph?

### Recommended direction

The toolkit should support a clear plant-definition artifact, whether code-first, config-first, or hybrid.

Whatever the exact representation, it should define:

- module instances
- port connections
- parameter packs
- scenario attachments
- sensor placement
- observation profiles
- baseline controller attachments

### Why this matters for review

Plant assembly should be inspectable without reading arbitrary execution code. A reviewer should be able to answer:

- what units exist?
- how are they connected?
- where are the sensors?
- what are the actuators?
- what disturbances are present?

without reverse-engineering the runtime.

## 4. Module certification and invariant testing

### Why this matters

Composable systems depend on trust in individual parts.

If modules are swappable, each module should have a standard way to demonstrate that it behaves safely and consistently under expected conditions.

### Certification goals

Every benchmark-ready module should show evidence that it:

- respects declared bounds
- preserves required invariants
- composes correctly with the runtime
- handles missing or irrelevant fields according to its contract
- behaves plausibly under reference perturbations

### Common invariant examples

- nonnegative inventories remain nonnegative unless the model explicitly allows otherwise
- untouched transport fields are preserved across pass-through modules
- sensor outputs respect valid ranges and update cadence
- actuator outputs respect saturation and rate-limit rules
- uncertainty injection preserves physical constraints

### Certification artifact idea

Each module family should eventually have a lightweight certification checklist or test bundle.

For example:

- interface contract satisfied
- invariant tests passed
- reference scenario behavior approved
- declared maturity level recorded

That makes module maturity explicit rather than assumed.

## Recommended implementation order for these governance topics

1. units/naming/schema conventions
2. plant assembly representation
3. module invariant and certification testing pattern
4. calibration and validation workflow scaffolding

This order helps because schema and assembly decisions shape nearly everything else.

## Recommendation

These topics should be treated as part of the toolkit core, not as follow-on documentation.

In practical terms:

- no shared module should ship without a clear units and naming story
- no benchmark should ship without an inspectable plant assembly definition
- no benchmark-ready module should exist without invariant tests
- no scientifically motivated module should be considered mature without some validation story, even if calibration remains approximate

This is what will make the toolkit not just powerful, but reliable enough for repeated internal use.