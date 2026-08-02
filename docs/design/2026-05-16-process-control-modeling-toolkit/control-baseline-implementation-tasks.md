# Control-Baseline Implementation Tasks

## Purpose

This appendix turns [Control baselines and fair comparisons for online RL](control-baselines-and-fair-comparisons.md) into an ordered implementation backlog. Keep this page short; design explanations belong in the main guide.

## Phase 1: Experiment contract

- [ ] Add controller metadata: information tier, required sensors, offline data, online updates, and simulator knowledge.
- [ ] Define shared physical action bounds, rate limits, control interval, safety overrides, and fallback behavior.
- [ ] Separate controller-development, identification, validation, and locked challenge scenarios.
- [ ] Add common physical metrics for tracking, constraints, actuator movement, economics, and adaptation.
- [ ] Store controller preparation cost: samples, excitation, tuning trials, and compute.

Acceptance: reports can compare complete commissioning workflows without benchmark-specific interpretation.

## Phase 2: Production-quality PID toolkit

- [ ] Add filtered PI/PID with direct and reverse action.
- [ ] Add conditional-integration and back-calculation anti-windup.
- [ ] Add asymmetric output-rate limits and actuator saturation tracking.
- [ ] Add setpoint weighting, setpoint ramps, and derivative-on-measurement.
- [ ] Add manual/automatic modes, external-reset tracking, and bumpless transfer.
- [ ] Expose raw output, realized output, saturation, integral, error, and mode diagnostics.
- [ ] Add conservative IMC/lambda tuning and objective-based tuning utilities.

Acceptance: saturation recovery and mode transfer pass focused response tests.

## Phase 3: Standard control structures

- [ ] Add feed-forward plus feedback trim.
- [ ] Add cascade-controller composition.
- [ ] Add ratio control.
- [ ] Add high/low selectors and override control.
- [ ] Add split-range output mapping.
- [ ] Add static MIMO decoupling for multiloop PI.
- [ ] Add interpolated gain scheduling with bumpless region changes.
- [ ] Add Smith-predictor/dead-time baseline for chlorine.
- [ ] Add versioned rule-policy support for maintenance and scheduling.

Acceptance: each gold benchmark has a credible standard-practice baseline using only realistic measurements.

## Phase 4: Commissioning-data generation

- [ ] Add bounded step, ramp, PRBS/generalized-binary, and multisine excitation.
- [ ] Allow excitation around a safe incumbent controller rather than requiring open loop.
- [ ] Record commanded and realized actuator values.
- [ ] Enforce safety, quality, dwell-time, and excitation-cost limits.
- [ ] Generate time-separated identification, validation, and challenge datasets.
- [ ] Store signals, units, timing, scenario version, preprocessing, and seed.

Acceptance: datasets are reproducible and cover declared operating regions without violating the commissioning safety budget.

## Phase 5: System identification

- [ ] Fit first-order-plus-dead-time models for SISO loops.
- [ ] Fit FIR and ARX models.
- [ ] Fit MIMO state-space models using subspace identification.
- [ ] Estimate delays and compare candidate model orders.
- [ ] Add local-model or LPV identification for pH and other nonlinear regimes.
- [ ] Validate one-step, multistep, and free-run predictions.
- [ ] Add residual autocorrelation and residual/input-correlation diagnostics.
- [ ] Record operating envelope, uncertainty, and known failure regions.
- [ ] Persist model and validation artifacts with dataset provenance.

Acceptance: every MPC model can be regenerated from stored data and passes a benchmark-specific validation envelope.

## Phase 6: State estimation

- [ ] Add linear Kalman filtering for identified state-space models.
- [ ] Add disturbance-state estimation for offset-free control.
- [ ] Add one nonlinear estimator or moving-horizon estimator only after linear coverage works.
- [ ] Add soft-sensor interfaces with explicit input signals and sampling cadence.
- [ ] Compare raw-measurement and observer-assisted PID baselines.

Acceptance: estimator performance is evaluated against latent simulator truth without exposing that truth to the controller.

## Phase 7: Linear MPC

- [ ] Implement receding-horizon linear MPC.
- [ ] Support action, action-rate, and output constraints.
- [ ] Support soft constraints with reported slack variables.
- [ ] Add measured-disturbance feed-forward.
- [ ] Add offset-free state/disturbance estimation.
- [ ] Record solver status, solve time, infeasibility, and fallback events.
- [ ] Add tracking and economic objectives.
- [ ] Add oracle-model mode, clearly labeled as an upper bound.

Acceptance: identified MPC controls pH, chlorine, and equalization under unseen scenarios without simulator parameters.

## Phase 8: Adaptive classical control

- [ ] Add recursive least squares with bounded parameters and configurable forgetting.
- [ ] Track excitation/identifiability and pause unsafe updates.
- [ ] Validate candidate models over a rolling window.
- [ ] Gate model promotion using prediction, stability, and safety checks.
- [ ] Add gradual or bumpless MPC model handoff.
- [ ] Add fallback on estimator, model, or optimizer failure.
- [ ] Add model-bank selection for distinct operating regimes.

Acceptance: adaptive MPC reports adaptation speed, transient cost, update rejection, and retention when a prior regime returns.

## Phase 9: Gold-benchmark stacks

- [ ] Chlorine: flow pacing, PI trim, Smith predictor, identified MPC, adaptive MPC, oracle ceiling.
- [ ] pH: local PI, gain-scheduled PI, feed-forward trim, LPV MPC, adaptive local models, oracle ceiling.
- [ ] Equalization: fixed flow, PI with overrides, rules, forecast MPC, adaptive MPC, perfect-forecast ceiling.
- [ ] Membrane: periodic rules, threshold rules, regulatory PID, economic MPC, adaptive fouling model, oracle scheduler.
- [ ] H2S: cascade PI, gas-load feed-forward, decoupled PI, MIMO economic MPC, adaptive/model-bank MPC.
- [ ] BSM1: local DO PI, supervisory ammonia control, recycle control, reduced-order MIMO MPC, adaptive/scheduled MPC.

Acceptance: every stack uses the same sensor, authority, safety, data, and tuning-budget declarations within an information tier.

## Phase 10: Research comparators

- [ ] Add robust or scenario MPC to one uncertainty-focused benchmark.
- [ ] Add data-enabled predictive control to one MIMO benchmark.
- [ ] Add mixed-integer or hybrid MPC only for a maintenance benchmark where discrete decisions matter.
- [ ] Compare passive online identification with bounded active probing.

Acceptance: specialized baselines are added only when they test a stated RL advantage.

## Phase 11: Reporting

- [ ] Generate the baseline ladder table automatically.
- [ ] Report identification and tuning data separately from evaluation data.
- [ ] Report preparation cost and unsafe commissioning cost.
- [ ] Report reward plus physical control, safety, economic, and learning metrics.
- [ ] Separate plant-realistic, data-driven, and oracle results.
- [ ] Add paired-seed comparisons on common disturbance realizations.
- [ ] Record omissions and failed controller/model configurations.

Acceptance: a reader can tell whether RL gains came from prediction, estimation, adaptation, constraints, privileged information, or reward design.

## First milestone

Complete Phases 1–8 for chlorine, pH, and equalization before expanding the baseline stack to the remaining benchmarks.
