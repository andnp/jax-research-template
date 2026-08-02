# Controls and Online-RL Improvement Roadmap

## Purpose

This document reviews the process-control toolkit as an internal test and research platform for a continual, online reinforcement-learning controller: a general-purpose control block intended to improve on PID while learning on an operating process.

The intended use is **not** transfer learning or simulator pretraining. The toolkit should instead help answer:

- Is an online learning algorithm safe and useful during commissioning?
- Does it improve after seeing new operating conditions without forgetting old ones?
- Can it outperform a well-engineered classical control stack for a defensible reason?
- Does it remain stable under delay, saturation, sensor faults, process drift, and operating-mode changes?
- Can an experiment explain why an algorithm succeeded or failed?

This roadmap combines a software review, a controls review, and a process-model review. It is intentionally actionable: each recommendation includes enough scope and acceptance guidance for another agent to implement it.

## Executive assessment

### Is the toolkit useful for this product direction?

**Yes, with an important qualification.** It is already useful as a research sandbox and algorithm smoke-test suite. It is not yet strong enough to serve as convincing validation evidence for a general-purpose online-RL control product.

The foundation is good:

- pure JAX state transitions support fast, repeatable rollouts and parallel experiments
- explicit state and random keys make experiments reproducible
- benchmarks cover several real control structures rather than one domain
- sensors, actuators, disturbances, chemistry, and unit operations are separate modules
- the suite includes delay, nonlinear gain, MIMO coupling, constraints, fouling, seasonal drift, and partial observability
- the wastewater models provide a bridge to recognized process-control benchmark structures

The main limitation is **evidence quality rather than model quantity**. Many models are reduced-order or qualitative, classical comparators are not yet consistently first-class, benchmark contracts vary, and the suite does not yet exercise the full lifecycle of a controller that learns online.

The right goal is not to turn every benchmark into a high-fidelity digital twin. The right goal is a **tiered validation suite** in which each benchmark has a declared purpose, known fidelity, calibrated operating envelope, strong baselines, and a reproducible set of control challenges.

## What the toolkit should prove

A useful online-RL validation toolkit should separate six claims that are often mixed together:

1. **Basic control competence:** the controller regulates a stable, well-behaved process.
2. **Structural advantage:** it uses information or objectives that a fixed PID cannot use naturally, such as long histories, multiple measurements, or economic costs.
3. **Safe adaptation:** it improves online without unacceptable excursions during learning.
4. **Nonstationary operation:** it tracks drift, wear, seasonal change, and changing process gain.
5. **Robustness:** it handles sensor faults, saturation, delays, missing data, and unmodeled disturbances.
6. **General control-block behavior:** the same algorithm and mostly the same tuning work across substantially different process classes.

Every experiment should state which claim it tests. A benchmark should not be called successful merely because an RL policy obtains more reward than a weak PID.

## Current benchmark portfolio

The portfolio is directionally strong because it spans distinct control archetypes.

| Benchmark family | Control archetype | Main value for online RL | Current concern |
| --- | --- | --- | --- |
| Chlorine and two-stage chlorine | long, flow-dependent delay with consumption | anticipation, delay handling, upstream context | reduced chemistry and limited hydraulics need declared fidelity |
| pH neutralization | strongly nonlinear process gain | local adaptation and gain scheduling | chemistry is too compressed to represent buffering and titration diversity |
| Equalization tank | inventory and constraint control | forecasting, constraint handling, downstream smoothing | clipping hides overflow and lost mass |
| Membrane fouling | slow degradation and hybrid maintenance | continual adaptation and long-horizon decisions | backwash timing semantics are incomplete |
| H2S scrubber | interacting chemical loops and inventory | supervisory MIMO control over plant-side loops | chemistry needs calibration envelopes and fault scenarios |
| BSM1 variants | biological kinetics, recycle, aeration, clarification | multivariable control, seasonal drift, slow biology | variants duplicate plant logic and need reference validation |
| Takács settler and sludge blanket | distributed separation and washout constraints | latent-state estimation and risk-sensitive control | conservation and scenario certification should be stronger |
| ASM2d N/P benchmarks | coupled biological and chemical objectives | cross-stage coordination and economic control | reduced assembly must be compared against expected process trends |
| Anaerobic digestion | slow kinetics and upset avoidance | safe exploration and delayed economic reward | a simplified ADM1-like model should not be presented as full ADM1 |
| Drinking-water train | cascade interactions and maintenance | plant-wide coordination | each simplified stage compounds model error |
| Dewatering and primary clarification | static/dynamic separation trade-offs | economic optimization and operating-region changes | empirical response surfaces need traceable calibration |
| Reject-water management | periodic load shifting and inventory | temporal scheduling and downstream protection | should include uncertain forecasts and storage constraints |
| Chemical phosphorus dosing | nonlinear dose/cost trade-off | adaptive economic control | instantaneous chemistry omits mixing, flocculation, and settling delays |

### Portfolio recommendation

Keep the breadth, but designate a smaller **gold benchmark set** for serious claims:

1. pH neutralization for nonlinear SISO adaptation
2. chlorine contact basin for transport delay and feed-forward context
3. equalization tank for constraints and disturbance forecasting
4. membrane fouling for slow drift and hybrid decisions
5. H2S scrubber for supervisory MIMO control
6. BSM1-Takács or BSM1-combined for slow biological dynamics and plant-wide interaction

The remaining benchmarks can remain exploratory until they receive calibration and certification artifacts.

## Immediate correctness work

### 1. Prevent NaNs in rain disturbances

`rain_storm` computes a dilution factor as `flow / (flow + added_flow)`. Zero flow and zero disturbance magnitude produce `0 / 0`, which propagates NaNs through composition fields.

Implementation task:

- define the physical result when total flow is zero; retaining the original composition is a reasonable default
- use a guarded division such as `jnp.where(total_flow > epsilon, flow / total_flow, 1.0)`
- decide whether negative flow or negative rain magnitude is invalid and validate it outside JIT execution
- add eager and JIT tests for zero flow, zero magnitude, extreme magnitude, and invalid type IDs
- add a general finite-output property test for all disturbance types

Acceptance criterion: every valid disturbance applied to a valid finite transport produces a finite transport.

### 2. Make disturbance support honest across benchmarks

Several benchmark states contain a `DisturbanceSchedule`, but their step functions never apply it. This currently affects the BSM1 family, pH neutralization, and sludge-blanket variants.

Implementation task:

- inventory every benchmark that stores a schedule
- either apply scheduled disturbances at a clearly documented seam or remove the schedule from that benchmark state
- define what a generic disturbance changes: source flow/composition, equipment parameters, sensor behavior, or actuator availability
- avoid forcing water-quality `Transport` disturbances into gas or biological models where the schema is not appropriate
- add one end-to-end test per supported benchmark showing that an active event changes the expected physical signal

Acceptance criterion: the presence of a disturbance schedule in state always means that scheduling an event has an observable, documented effect.

### 3. Give membrane backwash real duration semantics

`MembraneParams.bw_duration` is currently unused. A trigger removes a fixed fraction of reversible fouling and stops production for one model step, so behavior changes when `dt` changes.

Implementation task:

- add an operating mode or remaining-backwash-time field to membrane state
- make a trigger enter backwash mode for `bw_duration`
- define backwash flow, lost permeate, energy, and reversible-fouling removal over that interval
- decide whether repeated triggers are ignored, extend the wash, or incur an invalid-action penalty
- keep clean-in-place as a separate timed maintenance state rather than an instantaneous threshold reset
- test approximate timestep invariance by running the same physical schedule at two integration steps

Acceptance criterion: the same backwash program produces similar recovery and downtime when simulated with smaller timesteps.

### 4. Preserve mass balance at inventory limits

The generic tank clips level at minimum and maximum bounds. Clipping stabilizes simulation but silently creates or destroys inventory.

Implementation task:

- return actual outlet flow, overflow flow, and unmet requested outflow
- integrate inventory using physically realized flows
- expose overflow, starvation, and constraint status in benchmark `info`
- base rewards and safety metrics on those physical violations
- add a cumulative mass-balance test over arbitrary inflow/outflow sequences

Acceptance criterion: initial inventory plus integrated net realized flow equals final inventory plus reported spill, within numerical tolerance.

### 5. Add PI anti-windup

The PI controller clips integral state and output independently. It can continue integrating while the actuator is saturated and recover slowly after a large upset.

Implementation task:

- support conditional integration or back-calculation anti-windup
- expose both raw and saturated output for diagnostics
- include manual/automatic transfer and integrator tracking if the controller will model plant practice
- add tests for sustained high and low saturation, recovery time, and bumpless transfer

Acceptance criterion: saturation does not drive the integral farther into windup, and transfer between manual and automatic control does not create a large output step.

## Software architecture and code-surface reduction

### 6. Define a benchmark protocol and metadata contract

Benchmark factories currently return reset and step closures with implicit action, observation, unit, and horizon contracts.

Create a lightweight `Benchmark` object or protocol containing:

- `reset(key) -> (state, observation)`
- `step(state, action, key) -> (state, observation, reward, terminated, truncated, info)`
- action and observation specifications with names, shapes, physical units, and valid ranges
- model timestep and control timestep
- default episode or evaluation horizon
- benchmark version and fidelity tier
- declared control challenges and supported disturbances/faults

Keep the JAX execution path as arrays and pytrees; metadata can remain static Python data outside JIT.

Acceptance criterion: one generic certification runner can discover and execute every registered benchmark without benchmark-specific branching.

### 7. Extract the shared BSM plant kernel

The BSM1 variants repeat reactor initialization, recycle hydraulics, clarification, sensor updates, observations, and rewards. This increases the chance that variants differ accidentally rather than scientifically.

Implementation task:

- create reusable reactor-bank, hydraulic-loop, and clarification functions
- keep differences such as seasonal kinetics, recycle actuation, and Takács settling as explicit strategy modules
- create named state groups for reactors, loops, settlers, sensors, and source state
- build observations and rewards outside the physical plant transition
- add equivalence tests showing that the extracted standard configuration reproduces existing traces

Acceptance criterion: each BSM variant mostly declares its differing modules and configuration instead of copying the full plant step.

### 8. Separate plant, instrumentation, controller, and objective

A benchmark should be four composable layers:

1. physical plant transition
2. instrumentation and signal timing
3. actuator/controller interface
4. observation, reward, and evaluation objective

This separation matters because a PID and an RL agent must often receive different derived features while controlling the same physical plant. It also prevents reward changes from altering process equations.

Implementation task:

- make the latent plant step return physical outputs only
- attach sensors to named physical signals
- map normalized actions to actuator requests in one adapter
- construct observations and rewards in versioned adapters
- ensure tests can inspect latent truth without exposing it to the controller

Acceptance criterion: a plant trajectory can be replayed through alternative sensor and objective profiles without changing plant dynamics.

### 9. Standardize reset and step signatures

Some module resets accept an RNG key but ignore it; others require parameters or initial values in different orders.

Implementation task:

- choose consistent conventions for deterministic and stochastic modules
- use keyword-only configuration where argument confusion is likely
- remove unused RNG parameters from purely deterministic low-level modules, or explicitly retain them through a common protocol adapter
- document time units and whether `dt` is static or dynamic
- standardize output order and use small named output dataclasses where tuples become ambiguous

Acceptance criterion: module composition does not require memorizing family-specific argument ordering.

### 10. Curate the public API

Package `__init__.py` files are empty, so callers import internal file paths and there is no declared stable surface.

Implementation task:

- publish stable types and factories from a small number of package namespaces
- keep experimental benchmarks clearly marked as experimental
- add `__all__` and API documentation
- define a deprecation policy for renamed observations, parameters, and benchmark factories
- avoid exporting every internal state helper

Acceptance criterion: common users can construct benchmarks and controllers without importing private implementation modules.

### 11. Centralize observation and action schemas

Benchmarks repeatedly construct normalized arrays inline. Reset and step can therefore drift in ordering or scale.

Implementation task:

- define each feature once with name, source signal, unit, normalization, bounds, and availability
- generate reset and step observations through the same builder
- expose physical and normalized action specifications
- version schemas when feature meaning or order changes
- test that observations match declared shape, order, dtype, and finite bounds

Acceptance criterion: telemetry can always map an array index back to a named physical signal and normalization rule.

## Controls improvements

### 12. Use strong classical comparators

The product claim is “better PID,” so a basic fixed PI is necessary but insufficient. RL should be compared against the strongest reasonable controller for each challenge.

Add comparator families:

- well-tuned PI/PID with filtering, anti-windup, rate limits, and bumpless transfer
- cascade control where secondary measurements exist
- ratio and feed-forward plus feedback trim for dosing applications
- gain-scheduled PID for pH and other nonlinear processes
- Smith predictor or internal-model control for delay-dominated chlorine systems
- override/select logic for hard constraints
- linear MPC for constrained multivariable benchmarks
- simple adaptive control or recursive-estimation plus gain scheduling for drift benchmarks
- rule-based maintenance policies for membrane backwash and cleaning

Each baseline should receive information appropriate to its architecture. Do not give RL privileged measurements unless the experiment explicitly tests the value of richer instrumentation.

Acceptance criterion: every gold benchmark has at least one production-credible classical baseline and one deliberately simple baseline.

### 13. Model the control hierarchy explicitly

A general-purpose RL block is more credible as a supervisory controller than as a direct replacement for every millisecond-scale loop.

Support at least three interface modes:

- **direct:** RL commands the actuator; useful for simple research tasks
- **setpoint:** RL adjusts setpoints while plant-side PI/PID loops handle fast regulation
- **trim:** RL adds a bounded correction to an existing controller or feed-forward calculation

Setpoint and trim modes are especially important for safe online learning because established regulatory loops preserve basic stability.

Implementation task:

- make these modes common actuator adapters rather than benchmark-specific choices
- include rate, magnitude, and dwell-time limits on RL authority
- expose fallback activation and controller handoff events
- test bumpless entry and exit from RL control

Acceptance criterion: the same benchmark can compare direct, setpoint, and trim authority without rewriting its plant.

### 14. Add explicit safe-exploration mechanisms

Online learning is defined by what happens before the agent is competent. Mean reward after convergence is not enough.

Add reusable mechanisms:

- action projection into a safe set
- rate and acceleration limits
- safety interlocks and override selectors
- fallback controller with configurable trip and recovery logic
- constraint budgets and recovery policies
- shadow mode, where the agent proposes actions but does not control the plant
- staged authority, where allowed trim grows only after performance gates are met

Metrics should include interventions, time under fallback, worst excursion, recovery duration, and unsafe learning cost.

Acceptance criterion: every online-learning run reports both performance and the cost/risk incurred while learning.

### 15. Test continual adaptation directly

Current stochastic rollouts are not enough to evaluate continual learning. Add long scenarios with named phases:

1. nominal commissioning
2. changed load or operating point
3. slow equipment or sensor drift
4. abrupt maintenance or replacement
5. return to a prior condition

Measure:

- time and cumulative cost to adapt
- performance during adaptation, not only afterward
- retention when an earlier operating regime returns
- stability of policy updates
- amount of exploratory actuator movement
- sensitivity to update cadence and replay strategy

Acceptance criterion: results distinguish adaptation speed, forgetting, and transient safety.

### 16. Add system-identification experiments

An online controller should be tested for whether it excites the process safely enough to learn useful dynamics.

Implementation task:

- add pseudo-random binary, step, ramp, and multisine commissioning scenarios
- calculate basic excitation and identifiability indicators
- distinguish commanded from realized actuator movement
- test learning when normal operations provide poor excitation
- compare passive learning against bounded active probing

Acceptance criterion: an algorithm cannot appear adaptive merely because the benchmark exposes the latent parameters in its observation.

### 17. Treat delay and sampling as first-class dynamics

Real process loops contain transport delay, analyzer cadence, computation delay, communication jitter, and zero-order holds.

Implementation task:

- separate physical integration, sensor sampling, control execution, and learning-update rates
- support fixed and variable dead time on measurements and actuators
- model timestamped stale measurements and missing packets
- add asynchronous multi-rate analyzers and lab samples
- include delay mismatch between training assumptions and evaluation conditions

Acceptance criterion: benchmarks can state a timing diagram, and changing controller cadence does not silently change physical kinetics.

### 18. Use control-specific evaluation metrics

Reward is an optimization interface, not an adequate evaluation report.

Report at least:

- integral absolute and squared error
- time outside target and compliance bands
- maximum and percentile excursion
- settling and recovery time after disturbances
- actuator travel, reversals, saturation time, and rate-limit activity
- reagent and energy use
- overflow, washout, starvation, and maintenance events
- constraint intervention count
- robustness margins or empirical stability envelope where practical
- online learning regret relative to a fixed safe baseline

Acceptance criterion: two controllers can be compared without inspecting their benchmark-specific rewards.

## Process-model and kinetics improvements

### 19. Establish fidelity tiers

Not every model needs first-principles fidelity, but every model needs an honest label.

Recommended tiers:

- **Tier 0 — contract toy:** tests APIs, JIT behavior, or a single algorithm property
- **Tier 1 — qualitative control model:** has correct signs, delays, constraints, and operating-region behavior
- **Tier 2 — calibrated reduced-order model:** matches reference step responses or operating envelopes
- **Tier 3 — reference benchmark:** reproduces a recognized model or curated plant-like dataset within stated tolerance

Record the tier, calibration source, valid operating envelope, known omissions, and intended claim in benchmark metadata.

Acceptance criterion: no benchmark is used for a stronger scientific claim than its fidelity tier supports.

### 20. Make conservation and dimensions testable

Clipping concentrations or inventories after integration can hide numerical or equation errors.

Implementation task:

- define conserved quantities and expected reaction sinks/sources for each unit
- report correction terms whenever clipping is retained
- add dimensional comments or metadata to every state, parameter, input, and output
- add balance residuals to debug telemetry
- run randomized invariant tests across valid operating ranges

Acceptance criterion: water, solids, and tracked species either balance or have an explicit modeled reaction, waste, gas, or correction term.

### 21. Improve numerical integration policy

The toolkit contains Euler-like discrete updates, an upwind contact basin, and RK4 ODE integration. There is no shared policy for timestep choice or stiffness.

Implementation task:

- declare the numerical method and stable timestep range for every dynamic module
- add substepping where the control interval is too coarse for physical integration
- test convergence by halving the physical timestep
- detect non-finite state and excessive balance residuals
- avoid relying on final-state clipping as the primary stability mechanism
- consider positivity-preserving updates for concentrations and inventories

For ASM and anaerobic kinetics, check whether the chosen explicit method remains stable over the entire parameter and temperature envelope.

Acceptance criterion: reference trajectories converge as integration resolution increases, within a model-specific tolerance.

### 22. Strengthen pH chemistry

The current pH benchmark is useful for demonstrating nonlinear gain, but a general process-control test should vary buffering and titration behavior.

Recommended model progression:

- track acid/base equivalents and liquid volume
- include strong-acid/strong-base balance plus configurable buffer capacity
- support at least one weak-acid equilibrium or an empirical titration curve
- make reagent strength and influent alkalinity uncertain
- include mixing and sensor delay separately from chemical equilibrium
- create operating regimes on both sides of the equivalence region

Calibration target: reproduce a family of titration curves and step responses rather than one fixed input/output curve.

### 23. Strengthen chlorine and contact-basin dynamics

The segmented basin is a useful delay model, but it should distinguish hydraulic transport from disinfectant chemistry.

Recommended improvements:

- enforce or substep around the Courant condition instead of silently clipping transported fraction
- allow dispersion or tanks-in-series behavior to vary independently of segment count
- model at least fast and slow chlorine-demand fractions
- include temperature and pH dependence where useful
- distinguish free chlorine, combined chlorine, and ammonia breakpoint behavior for higher-fidelity variants
- report contact-time or CT metrics and minimum-residual constraint windows
- add flow-dependent residence-time validation traces

Keep a simpler Tier-1 version for fast algorithm tests and a calibrated Tier-2 version for delay-control evidence.

### 24. Certify ASM1, ASM2d, and settler behavior

These modules are among the most valuable assets because they create slow, coupled, partially observed dynamics.

Implementation task:

- document exactly which published state variables, processes, and parameter sets are implemented or omitted
- compare steady states and standard disturbance responses with reference outputs
- test elemental or aggregate COD, nitrogen, phosphorus, and solids balances
- test temperature correction independently from plant assembly
- verify recycle-flow accounting and clarifier split consistency
- validate Takács blanket movement and solids washout under hydraulic overload
- create warm-start procedures so online-control tests do not begin from arbitrary transients

Acceptance criterion: each wastewater gold benchmark ships with steady-state initialization, reference scenarios, and balance/error envelopes.

### 25. Clarify anaerobic-digester model identity

The module is named `ADM1Params`, but the benchmark appears reduced-order. That is acceptable if it is explicit.

Implementation task:

- rename or document it as ADM1-inspired unless it implements the full reference state and process set
- identify the modeled carbon, VFA, gas, biomass, temperature, and pH relationships
- add inhibition and washout scenarios only when supported by the equations
- calibrate slow time constants and upset/recovery trends
- expose gas inventory and methane-production accounting clearly

Acceptance criterion: users cannot mistake a qualitative digester model for a validated ADM1 implementation.

### 26. Improve membrane fouling and maintenance physics

Beyond backwash duration, membrane control needs state-dependent resistance and maintenance consequences.

Recommended improvements:

- distinguish reversible, hydraulically irreversible, and chemically recoverable resistance
- make deposition depend on flux as well as feed solids
- represent backwash effectiveness as a function of duration/intensity
- represent cleaning downtime, chemical cost, and incomplete recovery
- include TMP or flux control modes and hard operating limits
- add sensor bias because TMP-derived permeability is often an inferred health signal

Acceptance criterion: fixed-flux operation raises TMP, fixed-TMP operation loses flux, and cleaning decisions produce realistic throughput/health trade-offs.

### 27. Improve coagulation, precipitation, and dewatering modules

Current empirical curves are useful Tier-1 control surfaces, but they need dynamics and calibration to support process claims.

Implementation task:

- separate reaction/destabilization from flocculation and solid separation
- include mixing-energy and residence-time effects where they create a control decision
- allow raw-water NOM, temperature, and pH to shift optimal coagulant dose
- account for chemical sludge production and downstream loading
- make polymer dose and shear interact in dewatering
- calibrate response envelopes against literature or site-inspired traces

Acceptance criterion: optimal dose moves for physically sensible reasons, and upstream chemical choices affect downstream solids and cost accounting.

### 28. Improve H2S scrubber chemistry and hydraulics

The H2S benchmark is well aligned with supervisory online RL because it has coupled loops, latent liquid condition, and an economic/compliance trade-off.

Recommended improvements:

- conserve absorbed sulfur and oxidant equivalents across contactor and sump
- tie absorption efficiency to gas/liquid ratio, pH, loading, and contactor condition
- model bleed and makeup explicitly rather than hiding inventory correction
- distinguish pH/alkalinity from ORP/oxidant state
- add analyzer delay, poisoning, drift, and calibration events
- include pump degradation, nozzle fouling, and loss of chemical strength
- validate load-step breakthrough and recovery behavior

Acceptance criterion: outlet H2S breakthrough can be attributed to loading, liquid chemistry, hydraulics, or instrumentation rather than an opaque response curve.

## Uncertainty, faults, and operating scenarios

### 29. Separate uncertainty layers

Each experiment should distinguish:

- measured external disturbance, such as inlet flow
- unmeasured load disturbance, such as composition change
- parametric uncertainty, such as reaction-rate variation
- structural mismatch, such as omitted side reactions
- sensor noise, bias, lag, dropout, and calibration
- actuator deadband, stiction, hysteresis, saturation, and wear
- operating-mode changes and maintenance events

Do not collapse all uncertainty into Gaussian sensor noise. Online adaptation is most valuable when the source of nonstationarity changes the plant rather than merely corrupting measurements.

### 30. Build scenario packs, not isolated random seeds

Create named, versioned scenario packs:

- nominal daily operation
- load steps and ramps
- extreme but valid envelope corners
- recurring diurnal/weekly patterns
- seasonal drift
- sensor degradation and replacement
- actuator degradation and maintenance
- abrupt feedstock or chemistry change
- communication loss and delayed measurements
- combined fault plus process upset

Each pack should specify what is visible to the controller and what remains latent.

Acceptance criterion: experiment reports identify a scenario version, and the same pack can be replayed across algorithms.

### 31. Avoid unrealistic oracle observations

An RL agent can appear strong if it observes latent state that would not exist in a plant.

Implementation task:

- label signals as measured, calculated, lab-derived, forecast, or latent truth
- define instrumentation profiles separately from the plant
- include sampling, calibration, and inference cost
- use latent truth only for evaluation and debugging unless the experiment explicitly studies a soft sensor
- test sensitivity to removal or degradation of each privileged feature

Acceptance criterion: every observation feature has a credible plant-side origin and timing contract.

## Online-RL experimental protocol

### 32. Evaluate the whole learning curve

For each algorithm, report:

- performance before any updates
- cumulative performance during learning
- worst transient and constraint violations
- time to beat the fallback controller
- steady post-adaptation performance
- performance after returning to a previous regime
- variability across seeds and scenario realizations

Use paired scenarios so algorithms see the same disturbance realization where practical.

### 33. Distinguish controller state from learner state

The runtime should separate:

- fast controller state, such as recurrent memory or integrator state
- replay or adaptation memory
- model/optimizer parameters
- normalization statistics
- safety-supervisor state

Resetting an episode should not silently reset all continual-learning state. The experiment API should say exactly which state persists through shutdowns, maintenance, or operating-mode changes.

### 34. Add reproducible deployment events

Simulate operational events that matter to a product:

- cold start with a safe fallback
- shadow evaluation
- authority enable and disable
- controller software update
- rollback to prior parameters
- sensor recalibration
- process shutdown and warm restart
- loss and restoration of learner persistence

Acceptance criterion: deployment lifecycle behavior can be tested without embedding special cases in individual benchmarks.

## Validation and certification

### 35. Add a common module certification suite

Every benchmark-ready module should pass applicable checks:

- eager/JIT equivalence
- deterministic replay for a fixed key
- `vmap` compatibility where promised
- valid shapes and dtypes
- finite outputs over the declared envelope
- nonnegative inventories and concentrations where required
- conservation residual within tolerance
- actuator bounds, rate limits, and deadband behavior
- sensor cadence, lag, dropout, and range behavior
- timestep-convergence envelope
- reference step-response envelope

Certification results should be recorded with the model version rather than assumed from ordinary unit tests.

### 36. Add benchmark dossiers

Each gold benchmark should include a short dossier with:

- process diagram and control hierarchy
- state and equation summary
- units and parameter sources
- actuator and sensor placement
- operating envelope
- known omissions
- fidelity tier
- warm-start procedure
- reference scenarios and expected trends
- baseline controllers and tuning method
- evaluation metrics
- claims the benchmark may and may not support

Acceptance criterion: a reviewer can understand the experiment without reverse-engineering the benchmark step function.

## Suggested implementation sequence

### Phase 1: Trustworthy runtime

1. fix zero-flow disturbance NaNs
2. remove or activate dead disturbance schedules
3. implement mass-balanced tank limits
4. implement timed membrane backwash
5. add PI anti-windup
6. add finite-output and conservation property tests

### Phase 2: Common experimental contract

1. define benchmark metadata and action/observation schemas
2. separate plant, sensors, controller adapter, reward, and metrics
3. add common timing and multi-rate scheduling
4. curate the public API
5. build the generic certification runner

### Phase 3: Credible controls comparison

1. add production-style PID and feed-forward/cascade baselines
2. add delay compensation and gain scheduling where appropriate
3. add constrained MPC to selected MIMO benchmarks
4. add direct, setpoint, and trim RL interfaces
5. add fallback, shadow, and staged-authority modes

### Phase 4: Gold benchmark calibration

1. pH neutralization
2. chlorine contact basin
3. equalization tank
4. membrane fouling
5. H2S scrubber
6. BSM1-Takács or BSM1-combined

For each benchmark, create a dossier, fidelity declaration, reference traces, strong baselines, and scenario packs before moving to the next.

### Phase 5: Continual-learning research suite

1. long phase-based nonstationarity scenarios
2. adaptation, forgetting, and unsafe-learning metrics
3. bounded system-identification experiments
4. deployment lifecycle and rollback tests
5. multi-seed paired statistical reports

## Definition of success

The toolkit is ready to support a serious “better PID” research claim when:

- six gold benchmarks represent genuinely different control archetypes
- each has an honest fidelity tier and validated operating envelope
- each has strong, appropriately structured classical baselines
- plant truth is separated from realistic instrumentation
- online learning is evaluated from first control action, not only after convergence
- safety interventions and adaptation cost are reported
- results survive timestep, seed, disturbance, sensor, and parameter variation
- the same RL system works through direct, setpoint, or trim adapters without benchmark-specific algorithm changes
- benchmark dossiers make successful results understandable and reproducible

At that point, the toolkit would be more than a collection of simulators. It would be a controlled experimental instrument for determining when continual online RL adds value, when conventional control remains preferable, and what deployment safeguards a general-purpose learning control block requires.
