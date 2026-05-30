# Process Control Modeling Toolkit

| Field | Value |
|-------|-------|
| Status | Draft |
| Owner | Andy |
| Reviewers | TBD |
| Created | 2026-05-16 |
| Last Updated | 2026-05-16 |
| Related Docs | [Appendix pages](2026-05-16-process-control-modeling-toolkit/README.md) |

## Executive summary

This document proposes a shared **process control modeling toolkit** that decomposes process environments into reusable, configurable modules. The starting point for the design is observing that existing process simulators mix at least three concerns — physical chemistry, scenario generation, and env contracts — in ways that make independent improvement difficult.

The toolkit is explicitly **not** a digital twin platform. Its purpose is to support control and RL research with simulators that are realistic enough to expose the right closed-loop difficulty: delay, nonlinearity, disturbance structure, actuator constraints, sensor imperfections, and partial observability. The architecture therefore prioritizes composition, observability contracts, and scenario coverage over first-principles completeness.

## Context

Current environment development mixes at least three concerns:

- **process physics and chemistry**
- **instrumentation and observability**
- **benchmark / control wrapper semantics**

That coupling makes each new environment expensive to build and hard to compare. A process engineer can usually describe a plant in modular terms — tanks, pipes, recirculation loops, analyzers, pumps, disturbances, operator modes — but the current env structure does not consistently mirror that vocabulary.

This matters for RL in particular because benchmark quality depends on the distinction between:

- the **true latent process state**
- the **instrumented process state** available to a baseline controller
- the **augmented observation set** available to an RL controller

Without explicit boundaries, it is easy either to under-instrument the simulator until the task is dominated by hidden-state luck, or to over-expose simulator internals until the RL policy gets an unrealistic advantage.

The existing code already shows both the opportunity and the problem:

- well-structured environments separate reusable pieces such as influent models, flow models, clarifier logic, and controllers.
- compact environments can be clean and small, but risk packaging forcing, latent demand, plug-flow dynamics, sensing, disturbances, and comparator control in a single file.

The proposed toolkit turns these patterns into an explicit shared architecture.

## Boundary

This document owns:

- the architecture of a reusable process control modeling toolkit
- the core modeling concepts: streams, ports, unit operations, sensors, actuators, controllers, scenario modules, and observation builders
- the contract boundaries between plant state, instrumentation, controllers, and gym-style env wrappers
- the recommended realism principles for benchmarking control and RL algorithms
- the strategy for building a new toolkit alongside existing envs while using those envs as design references rather than migration targets

This document does **not** own:

- a full digital-twin program
- calibration workflows against specific facilities
- a commitment to a single domain-specific process model family
- reward design for every future benchmark
- deployment or online control integration contracts

## Goals and scope

By the end of this work, we will be able to:

- Build new process simulators by composing reusable modules instead of writing env-specific monoliths
- Express realistic process structure using domain-familiar concepts such as tanks, contact basins, recycles, sensors, pumps, and disturbance schedules
- Separate true latent state from measured state so benchmark observability is intentional and reviewable
- Reuse sensor, actuator, controller, and disturbance modules across multiple process families
- Configure multiple observability profiles for the same physical plant so PID, heuristic, and RL baselines can be compared fairly
- Model the control-relevant realism that matters most for benchmark quality: delays, nonlinear demand, saturation, rate limits, noise, drift, and operating-mode changes
- Evolve current `wtp` and `bsm1` envs toward a shared toolkit without requiring a one-shot rewrite

## Non-goals and out-of-scope

This document does not cover:

- high-fidelity CFD, multiphase transport, or plant-grade hydraulic simulation
- chemistry fidelity required for regulatory or design sign-off
- parameter estimation, system identification, or full calibration tooling
- native support for every industrial process family on day one
- replacing domain-specific detailed models where they already exist and are appropriate
- pretraining-quality simulators meant to stand in for production plants

## Key decisions

- The toolkit will be **composition-first**, with process graphs assembled from reusable modules rather than inheritance-heavy environment hierarchies.
- The architecture will treat **true plant state**, **instrumented measurements**, and **agent observations** as separate layers.
- The primary reusable primitives will be **streams**, **unit operations**, **sensors**, **actuators**, **controllers**, and **scenario events**.
- Benchmark realism will focus on **control-relevant realism** rather than digital-twin completeness.
- The new toolkit will be built as a separate architecture, borrowing selectively from `wtp` and `bsm1` where useful but not preserving 1:1 correspondence with either environment.
- The toolkit will support multiple observation contracts for the same plant so comparator baselines and RL policies can see different subsets of instrumentation intentionally rather than accidentally.

## Decision matrix

| Option | Benefits | Costs / risks | Decision |
|--------|----------|---------------|----------|
| Keep building environment-specific monoliths | Lowest short-term design overhead; easy to prototype one-off ideas | Reuse stays low, benchmark observability stays implicit, cross-env comparisons remain inconsistent | Rejected |
| Build a generic state-transition framework with minimal process semantics | Very flexible on paper; low domain assumptions | Modules become abstract and hard to reason about; process realism must be reinvented per env | Rejected |
| Build a domain-familiar process toolkit with explicit unit ops, sensors, actuators, and contracts | Reuse across domains, clear benchmark semantics, process engineers can reason about modules directly | Requires up-front interface design and disciplined boundary selection for a greenfield build | Selected |
| Pursue a digital-twin architecture from the outset | Maximum realism potential | High complexity, high calibration burden, mismatched to benchmark purpose | Rejected |

## Solution

### Architecture overview

The toolkit models a process simulation as a **directed graph of modules** connected by typed ports.

At a high level:

1. **Sources and scenarios** create or modify incoming conditions.
2. **Unit operations** transform material streams and maintain internal state.
3. **Sensors** measure selected latent states through realistic measurement contracts.
4. **Actuators** transform requested control actions into actual manipulated inputs subject to constraints.
5. **Baseline controllers or agent adapters** consume measured signals and produce requested actions.
6. The **env wrapper** defines the action space, observation space, reward, reset semantics, and baseline comparators.

The toolkit should make the physical plant, the instrumentation layer, and the benchmark wrapper composable but separate.

### Core layers

#### 1. Process layer

The process layer owns latent plant state and material transformation. It contains:

- streams and signal carriers
- unit operations
- chemistry / reaction blocks
- hydraulic and transport delays
- operating inventories and internal states

This layer should not know about RL, reward, or gym contracts.

#### 2. Instrumentation layer

The instrumentation layer converts latent plant state into imperfect measurements.

It owns:

- sample times
- noise
- lag
- bias and drift
- clipping and quantization
- dropout and stuck-signal behavior
- soft-sensor estimates derived from other measurements

This separation is critical. A plant state variable only becomes available to a controller when a sensor or estimator explicitly exposes it.

#### 3. Control layer

The control layer contains baseline controllers, supervisory logic, and controller-facing adapter contracts.

It owns:

- PID / PI / feed-forward blocks
- cascade and override structures
- actuator request generation
- supervisory enable/disable logic
- safety interlocks and fallback policies

The toolkit should allow the same plant to be driven by different controllers without changing the plant definition. Learned policy training code still belongs in the benchmark / env layer; the toolkit only needs the controller-facing contracts that let those policies act on the same plant and instrumentation seams as the baselines.

#### 4. Benchmark / env layer

The benchmark layer adapts the plant to RL evaluation.

It owns:

- action and observation spaces
- reward definitions
- reset policy and episode semantics
- comparator baselines
- logging and metric emission
- training vs evaluation observation variants

This is where experiment-specific choices belong. The plant and instrumentation layers should remain reusable across many env wrappers.

### Canonical module taxonomy

The first-class reusable module families are:

- **Streams and ports**: typed carriers for material and signal flow
- **Unit operations**: tanks, contact basins, plug-flow segments, splitters, mixers, clarifiers, recycle links
- **Sensors**: residual, flow, level, pressure, quality surrogates, soft sensors
- **Actuators**: dose pumps, control valves, blowers, recycle pumps, variable-speed drives
- **Chemistry and process blocks**: first-order decay, demand consumption, mass transfer, temperature correction, simple reaction networks
- **Controllers**: PI/PID, feed-forward, cascade, selectors, interlocks
- **Scenarios**: influent profiles, disturbances, failures, operator mode schedules
- **Observation builders**: named bundles that expose selected measurements to different agents

The toolkit should prefer a small number of strong abstractions over a large number of partially overlapping ones.

### Contracts and interfaces

#### Time and execution semantics

The toolkit needs one explicit simulation-phase contract so delay, sampling, and actuation semantics stay comparable across environments.

A first-generation default should be:

1. scenario modules update exogenous conditions for the current interval
2. controllers or agent adapters emit requested commands at their configured sample times
3. actuators map those requests to realized plant inputs subject to limits and lag
4. the material-process graph advances latent plant state over $dt$
5. sensors sample or update held values according to their own cadences
6. observation builders assemble consumer-specific views from the measurement bus

Two rules matter more than the exact implementation detail:

- the material graph should execute in a deterministic order within each phase
- multi-rate sensors, controllers, and actuators should use explicit sample-and-hold semantics rather than env-specific ad hoc timing

If the toolkit later needs more advanced solvers or substepping, they should refine this phase contract rather than replace it per environment.

#### Streams and ports

A stream contract should represent the physical quantities that move between units. The exact implementation can evolve, but the conceptual contract should support:

- flow rate
- a named composition vector or species map
- optional scalar properties such as temperature, pH, or density
- timestamp or simulation time metadata when useful

A separate signal-port contract should represent control and measurement signals. Signal ports are not material streams; they are values such as valve command, analyzer reading, tank level, or supervisor mode.

#### Unit operation contract

Every unit operation should define:

- required input ports
- produced output ports
- internal dynamic state
- tunable parameters
- a step/update method over $dt$
- an initialization/reset contract

Unit operations should be deterministic given state, inputs, parameters, and RNG-injected disturbances routed through explicit scenario modules.

#### Sensor contract

Every sensor should define:

- what latent variable or derived quantity it measures
- its sample period
- its latency / lag behavior
- its noise / bias / drift model
- failure modes such as dropout, freezing, or clipping
- the units and valid range of its emitted signal

A sensor should not mutate the plant. It observes the plant and emits a measurement signal.

#### Actuator contract

Every actuator should define:

- requested command input
- realized physical output
- saturation and ramp-rate rules
- optional deadband, backlash, stiction, or delay
- failure and override modes

This keeps the distinction between controller intent and plant manipulation explicit.

#### Observation-builder contract

An observation builder assembles a named measurement bundle for a specific consumer.

Examples:

- PID baseline observation
- RL observation package with extra analyzers
- supervisor observation package with alarm and health tags

This layer is the clean answer to the recurring “what should RL be allowed to see?” question. The answer becomes a reviewable contract rather than an ad hoc decision hidden inside one env file.

### Recommended realism principles

The toolkit should capture realism where it most affects closed-loop behavior.

#### Realism to capture deliberately

- transport and residence-time delay
- mass balance at the fidelity needed for control behavior
- actuator saturation and slew limits
- sensor noise, lag, and occasional failure
- exogenous disturbances with time structure
- operating-mode switches
- nonlinear gain changes where they materially affect control difficulty
- partial observability and instrumentation asymmetry

#### Realism to ignore intentionally in the first generations

- geometry-level detail that does not change closed-loop difficulty
- chemistry side reactions with no control consequence in the benchmark
- spatial resolution beyond what the chosen sensor layout can identify
- calibration-quality parameter precision
- maintenance workflows and operator HMI behavior unless they affect the control task directly

This principle keeps the toolkit usable and prevents it from becoming an unfocused digital-twin program.

### Recommended module graph pattern

A typical process env should be assembled as:

1. **Scenario sources** define influent loading, diurnal patterns, faults, and events.
2. **Hydraulic routing modules** move material through the process graph.
3. **Unit operations and chemistry blocks** update latent process state.
4. **Sensors** generate a measurement bus from selected latent states.
5. **Observation builders** derive one or more controller-visible views of that bus.
6. **Controllers and agent adapters** emit requested actuator commands.
7. **Actuators and interlocks** map requests to realized plant inputs.
8. **Benchmark wrapper** computes reward, metrics, and episode progression.

### Relationship to current environments

#### WTP / chlorine dosing

The current WTP env is a useful reference for the kinds of pieces the new toolkit should support:

- diurnal / drift influent and demand profile source
- raw-water quality latent process block
- disturbance scheduler
- plug-flow contact basin or contact pipe
- residual analyzers at selected locations
- flow sensor
- optional level / detention-time proxy sensor
- dose pump actuator
- PI baseline controller
- observation builder for PID and RL variants

The toolkit does **not** need to preserve the existing WTP environment structure or behavior one-for-one. The value of WTP here is as a reference problem that highlights the need for modular delay, demand, sensing, and comparator-control seams.

#### BSM1

`bsm1` already contains separable modeling ideas that should inform the common toolkit:

- influent sources and flow models
- tank models
- clarifier model
- sensor models
- baseline / direct-action / setpoint controllers

The toolkit does **not** need to recreate BSM1 module-for-module. The goal is to learn from its existing decomposition and reuse only the abstractions that strengthen the new architecture.

### RL/ML considerations

The toolkit exists to support benchmark design, not only process simulation. That creates several RL-specific requirements.

#### Observation asymmetry must be intentional

The simulator should be able to support multiple instrumentation profiles for the same latent plant. This allows meaningful questions such as:

- Can RL outperform PID if it sees upstream demand proxies and mid-basin residuals?
- How much value does a residence-time estimator add?
- When does extra instrumentation matter more than a stronger controller class?

#### Hidden state should stay hidden unless instrumented

The architecture must make it hard to leak simulator internals accidentally. For example, a plug-flow model may maintain many internal segment states, but the agent should only see those states if a configured sensor or estimator exposes them.

#### Benchmark comparators should share the same plant

PID, heuristic, and RL policies should operate on the same latent plant and disturbance schedule whenever possible. Differences in performance should then trace back to controller class, observation set, and reward objective rather than to separate plant implementations.

#### Scenario diversity is more valuable than excessive physical detail

For RL evaluation, a moderately realistic plant with rich disturbance regimes, sensor imperfections, and operating-mode shifts is usually more valuable than a highly detailed plant that only operates in one narrow regime.

### Experimentation

This design benefits from early experiments that reduce architecture risk before implementation hardens around the wrong abstractions.

#### Experiment 1: WTP extraction pilot

- **Hypothesis**: building a new small chlorine-style benchmark from source, basin, sensor, actuator, and controller modules will validate the proposed boundaries without requiring 1:1 correspondence to the existing WTP env.
- **Success criteria**: the new benchmark expresses the same class of control problem, supports at least two observation profiles, and is simpler to reason about than the monolithic reference env.
- **Decision unlocked**: whether the proposed module boundaries are practical for small process environments.

#### Experiment 2: Observation profile comparison

- **Hypothesis**: explicit observation builders improve benchmark clarity and enable fairer PID vs RL comparisons.
- **Success criteria**: the same WTP plant can run at least three reviewed observability packages (PID baseline, realistic RL, rich RL) with no plant-code changes.
- **Decision unlocked**: whether observation builders should be a first-class contract in v1.

#### Experiment 3: BSM1 contract alignment

- **Hypothesis**: BSM1-inspired abstractions can inform shared toolkit contracts without forcing the new system to mirror the legacy BSM1 implementation.
- **Success criteria**: at least one wastewater-style subsystem can be expressed cleanly with the new toolkit vocabulary while remaining understandable in process terms.
- **Decision unlocked**: whether the toolkit can truly span multiple process families.

## Milestones

1. Define the toolkit vocabulary and contracts for streams, ports, unit ops, sensors, actuators, controllers, scenarios, and observation builders.
2. Build a new small chlorine-style benchmark from toolkit modules, using the current WTP env only as inspiration and reference.
3. Add multiple reviewed observation profiles to that benchmark, including sparse intermediate residual sensing and delay proxies.
4. Express at least one wastewater-style subsystem with the new toolkit vocabulary, informed by BSM1 patterns but not constrained by BSM1 structure.
5. Create a scenario library for disturbances, sensor faults, and operating-mode schedules.
6. Build at least one additional process simulator from toolkit components to validate reuse beyond the first benchmark.
7. Publish a tech-spec style reference for implemented toolkit contracts once the design stabilizes.

## Conditions for done

### Functionality

- At least two process benchmarks built or re-expressed with the new toolkit share a common set of toolkit contracts
- The same plant definition can be paired with multiple observation profiles without plant-code edits
- Sensors, actuators, controllers, and disturbance schedules are reusable across at least two process families or benchmark variants
- WTP supports sparse internal sensing through configured analyzers rather than latent-state leakage

### Testing

- New toolkit benchmarks preserve the intended control-problem shape and observability semantics documented in their benchmark contracts
- Module-level tests cover representative unit ops, sensors, actuators, and scenario modules
- Observation builders are tested for contract stability and correct feature exposure
- Failure-mode tests exist for at least one sensor dropout and one actuator saturation scenario

### Documentation

- Main architecture doc is ratified
- Appendix pages describe the initial module families and recommended realism principles
- Future env authors have a documented recipe for composing a new process model from toolkit blocks

### Operations / observability

- Toolkit modules emit stable named metrics for latent state, sensor outputs, and actuator realizations where appropriate
- Comparator controllers and RL policies can be run against the same seeded scenario schedule for evaluation
- Observation contracts are named and versioned so benchmark comparisons remain reproducible

## Alternatives considered

| Alternative | Why not selected |
|-------------|------------------|
| Keep a small number of hand-built envs and document their differences informally | This leaves benchmark semantics implicit and reuse low. The same design questions about observability, disturbances, and controllers will keep being solved repeatedly. |
| Build a single ultra-generic simulation kernel with plugin equations only | This is flexible but loses process vocabulary. Engineers end up recreating tanks, sensors, and pumps informally on top of a low-level state machine. |
| Standardize only the gym wrapper layer and leave internals unconstrained | This fails to deliver the reuse that motivates the toolkit and does not help with realism or instrumentation consistency. |
| Focus only on one domain, such as water treatment, before generalizing | This may produce a strong local design but risks encoding domain-specific assumptions too deeply. The toolkit should still start from current domains, but its contracts should be process-family-agnostic where possible. |

## Open questions

- **Merge blocker**: what is the minimum common stream/state contract that can span WTP-style simple chemistry and BSM1-style richer composition without becoming unwieldy?
- **Follow-up**: where should the boundary sit between the default named dataclass-pytree representation and any hybrid nested compact substructures introduced for JAX ergonomics or grouped composition fields?
- **Follow-up**: how much shared time-integration machinery is worth centralizing versus letting complex units own their own stepping internals?
- **Follow-up**: should soft sensors live in the instrumentation layer, the scenario layer, or a separate estimation layer?
- **Follow-up**: what naming and versioning scheme should observation profiles use so benchmark papers and internal experiment logs stay comparable over time?

## Appendix map

Detailed appendix pages live in [`docs/design/2026-05-16-process-control-modeling-toolkit/`](2026-05-16-process-control-modeling-toolkit/README.md):

- [Streams, state, and ports](2026-05-16-process-control-modeling-toolkit/process-state-streams-and-ports.md)
- [Unit operations](2026-05-16-process-control-modeling-toolkit/unit-operations.md)
- [Sensors](2026-05-16-process-control-modeling-toolkit/sensors.md)
- [Actuators](2026-05-16-process-control-modeling-toolkit/actuators.md)
- [Chemistry and process blocks](2026-05-16-process-control-modeling-toolkit/chemistry-and-process-blocks.md)
- [Controllers and supervisory logic](2026-05-16-process-control-modeling-toolkit/controllers-and-supervisory-logic.md)
- [Disturbances and operating scenarios](2026-05-16-process-control-modeling-toolkit/disturbances-and-operating-scenarios.md)
- [Observations, benchmark contracts, and comparator design](2026-05-16-process-control-modeling-toolkit/observations-benchmarks-and-env-contracts.md)
- [Benchmark catalog](2026-05-16-process-control-modeling-toolkit/benchmark-catalog.md)
- [Uncertainty layers: disturbances, dynamics, and sensors](2026-05-16-process-control-modeling-toolkit/uncertainty-layers-disturbances-dynamics-and-sensors.md)
- [Software interface: shared transport, minimal protocols, and JAX-native execution](2026-05-16-process-control-modeling-toolkit/software-interface-shared-transport-and-jax-native-architecture.md)
- [H2S scrubber benchmark concept](2026-05-16-process-control-modeling-toolkit/h2s-scrubber-benchmark-concept.md)
- [Proposed implementation order](2026-05-16-process-control-modeling-toolkit/proposed-implementation-order.md)
- [Realism and operations extensions](2026-05-16-process-control-modeling-toolkit/realism-and-operations-extensions.md)
- [Validation, schema governance, assembly, and module certification](2026-05-16-process-control-modeling-toolkit/validation-schema-assembly-and-module-certification.md)
