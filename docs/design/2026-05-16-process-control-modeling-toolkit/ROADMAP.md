# Process-Control Toolkit Roadmap

## Purpose

This is the top-level table of contents and task map for developing the process-control toolkit into:

- a trustworthy internal testbed for continual online RL
- a credible comparison framework for classical and learning control
- a publishable process-control Gym for RL research

Detailed design and implementation notes remain in the linked documents. This page records the current priorities and miscellaneous work that does not yet have a more specific home.

## Current direction

The toolkit should prioritize depth over adding more benchmarks. The first goal is a complete vertical slice for chlorine, followed by pH neutralization and equalization control.

A complete environment is composed as:

```text
Environment = (model, scenario, instrumentation profile, control task)
```

A complete experiment adds a controller and evaluation protocol:

```text
EnvironmentSpec + ControllerSpec + EvaluationProtocol = Experiment
```

Training reward must always use signals available through the active instrumentation profile. Latent simulator truth is reserved for independent evaluation, diagnostics, and model validation.

## Roadmap summary

### 1. Make the runtime trustworthy

- Fix known numerical and behavioral correctness issues.
- Add conservation, finite-output, and timestep-convergence checks.
- Make disturbances, maintenance, limits, and failure behavior explicit.
- Define fidelity tiers and honest operating envelopes.

Detailed tasks: [Controls and online-RL improvement roadmap](controls-and-online-rl-improvement-roadmap.md#immediate-correctness-work)

### 2. Define the public environment contract

- Implement the model, scenario, instrumentation-profile, and control-task composition.
- Separate physical plant state from observations, reward, and evaluation metrics.
- Version action and observation schemas.
- Standardize reset, timing, persistence, and termination semantics.
- Curate a stable public API.

Design source: [Observations, benchmark contracts, and comparator design](observations-benchmarks-and-env-contracts.md)

Architecture tasks: [Controls and online-RL improvement roadmap](controls-and-online-rl-improvement-roadmap.md#software-architecture-and-code-surface-reduction)

### 3. Build scenario packs and instrumentation profiles

- Add nominal, disturbance, drift, fault, seasonal, and stress scenario packs.
- Define ideal, standard, rich, and degraded instrumentation profiles.
- Give every signal units, cadence, delay, noise, bias, and provenance.
- Separate measured disturbances, forecasts, soft sensors, and latent truth.
- Use independent random streams for process, sensor, fault, and controller randomness.

Design sources:

- [Disturbances and operating scenarios](disturbances-and-operating-scenarios.md)
- [Sensors](sensors.md)
- [Uncertainty layers](uncertainty-layers-disturbances-dynamics-and-sensors.md)
- [Environment contracts](observations-benchmarks-and-env-contracts.md#complete-environment-specification)

### 4. Establish strong control baselines

- Implement production-quality PI/PID with anti-windup and realistic mode handling.
- Add feed-forward, cascade, ratio, override, gain scheduling, and dead-time compensation.
- Generate safe commissioning datasets.
- Identify models from data rather than using simulator parameters.
- Implement constrained, offset-free linear MPC.
- Add online identification and adaptive MPC.
- Keep oracle controllers clearly labeled as ceilings.

Detailed guide: [Control baselines and fair comparisons](control-baselines-and-fair-comparisons.md)

Task list: [Control-baseline implementation tasks](control-baseline-implementation-tasks.md)

### 5. Certify gold benchmarks

Prioritize:

1. chlorine contact basin
2. pH neutralization
3. equalization tank
4. membrane fouling
5. H2S scrubber
6. one BSM1/Takács wastewater benchmark

Each gold benchmark needs:

- model and control dossier
- fidelity declaration and operating envelope
- reference steady-state and response traces
- conservation and numerical checks
- scenario packs and instrumentation profiles
- standard-practice, predictive, adaptive, and oracle baselines
- independent physical evaluation metrics

Detailed tasks: [Gold benchmark calibration](controls-and-online-rl-improvement-roadmap.md#phase-4-gold-benchmark-calibration)

Benchmark catalog: [Benchmark catalog](benchmark-catalog.md)

Wastewater details: [Wastewater RL environment roadmap](wastewater-rl-environment-roadmap.md)

### 6. Build the continual-learning evaluation protocol

- Score behavior from the first control action, not only after convergence.
- Measure adaptation cost, safety interventions, forgetting, and recovery.
- Separate plant, controller-memory, learner, optimizer, replay, and normalization state.
- Add shadow, trim, setpoint, staged-authority, fallback, and rollback workflows.
- Record exactly what persists across episodes, regimes, maintenance, and restarts.

Detailed tasks: [Online-RL experimental protocol](controls-and-online-rl-improvement-roadmap.md#online-rl-experimental-protocol)

### 7. Prepare the public Gym release

- Register stable, versioned environment IDs.
- Publish certified benchmark dossiers and baseline result cards.
- Add environment conformance and controls-certification tests.
- Write reproducible examples for PID, MPC, offline RL, and online RL.
- Separate experimental benchmarks from certified public benchmarks.
- Preserve old environment versions when behavior or schemas change.
- Publish development and locked evaluation scenario packs.

Validation source: [Validation, schema governance, assembly, and module certification](validation-schema-assembly-and-module-certification.md)

## First vertical slice: chlorine

- [ ] Fix zero-flow disturbance NaNs and verify all disturbance paths.
- [ ] Validate basin delay, flow dependence, demand response, and timestep sensitivity.
- [ ] Define standard and degraded instrumentation profiles.
- [ ] Define nominal, demand-change, flow-surge, drift, and sensor-fault scenario packs.
- [ ] Define direct, setpoint, and bounded-trim tasks.
- [ ] Enforce instrumentation-only reward inputs.
- [ ] Implement flow-paced feed-forward plus robust PI trim.
- [ ] Tune PI from identified response data using IMC/lambda tuning.
- [ ] Refine PID/feed-forward parameters on development scenarios only.
- [ ] Generate safe closed-loop identification data.
- [ ] Fit and validate first-order-plus-dead-time, ARX, and state-space models.
- [ ] Implement identified offset-free MPC and adaptive MPC.
- [ ] Add a bounded online-RL controller using the same information and authority.
- [ ] Produce a reference comparison report with physical and learning metrics.

## Miscellaneous task list

### Environment and provenance

- [ ] Create immutable specifications for model, scenario, instrumentation, and control task.
- [ ] Validate every reward input against the active instrumentation profile.
- [ ] Give evaluation metrics a separate latent-truth access path.
- [ ] Assign separate seeds to process, sensor, fault, scenario, and controller randomness.
- [ ] Write a machine-readable run manifest containing all component versions and persistence rules.
- [ ] Record whether learning remains enabled during evaluation.

### PID tuning

- [ ] Add bounded step, PRBS, and multisine response-test utilities.
- [ ] Fit first-order-plus-dead-time models for ordinary SISO loops.
- [ ] Add an integrating-process identification method for tank level.
- [ ] Implement conservative IMC/lambda PI/PID tuning.
- [ ] Identify feed-forward relationships separately from feedback tuning.
- [ ] Add bounded numerical refinement from the engineering tuning rather than blind gain sweeps.
- [ ] Lock tuning before running final challenge scenarios.
- [ ] Publish both robust and optimized PID variants where useful.

### Evaluation quality

- [ ] Keep training reward separate from published physical metrics.
- [ ] Report constraint violations, actuator movement, economics, and safety interventions.
- [ ] Use paired scenario realizations across controllers.
- [ ] Report commissioning data, excitation, tuning trials, and compute cost.
- [ ] Distinguish plant-realistic, data-driven, and oracle information tiers.
- [ ] Test nearby structural variants to detect simulator exploitation.

### Documentation and governance

- [ ] Add a benchmark maturity label: experimental, candidate, certified, or reference.
- [ ] Define versioning rules for plant dynamics, observations, rewards, and scenario distributions.
- [ ] Add a controls-facing dossier template.
- [ ] Add a baseline result-card template.
- [ ] Record supported and unsupported scientific claims for every certified benchmark.
- [ ] Identify tasks duplicated across roadmap documents and replace duplicates with links.

## Supporting design index

The complete appendix collection remains available in the [Process Control Modeling Toolkit Appendices](README.md).
