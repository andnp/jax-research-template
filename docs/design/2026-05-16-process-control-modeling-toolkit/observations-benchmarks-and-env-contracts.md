# Observations, Benchmark Contracts, and Comparator Design

## Purpose

This page describes how the toolkit should expose process models to controllers and RL agents without confusing plant definition with benchmark definition.

## What this module family is

Observation and benchmark contracts define:

- what a controller can see
- what action it can request
- what reward or evaluation metrics apply
- how resets, seeds, and scenarios are handled
- how comparator baselines and RL policies share the same plant

## Core principle

A plant model should be reusable across multiple benchmark contracts.

The same latent plant may support:

- a minimal PID view
- a more instrumented RL view
- a diagnostic operator view
- an ablation view with sensor failures enabled

## Observation-builder contract

An observation builder should define:

- which sensor signals are included
- how signals are transformed or normalized
- whether histories, traces, or temporal summaries are included
- naming and schema stability
- which consumer the bundle is intended for

Examples:

- `pid_baseline_v1`
- `rl_realistic_v1`
- `rl_rich_instrumentation_v1`

Stable names matter for reproducibility.

## Comparator design principles

### Same plant, different observability

The cleanest benchmark comparison is usually:

- same latent plant
- same disturbance schedule
- same action constraints
- different controller classes and observation sets

This lets reviewers reason about whether RL is winning because it is better informed, better optimized, or both.

### Avoid simulator leakage

Do not expose latent states simply because they exist. Expose them only when a sensor, estimator, or diagnostic contract justifies them.

### Allow honest asymmetry

It is acceptable — often desirable — for RL to see more than PID if the extra information is realistic instrumentation.

Examples of honest asymmetry:

- RL gets mid-basin residual and a quality surrogate; PID gets only outlet residual
- RL gets a residence-time estimator; PID stays feedback-only
- RL gets sensor quality flags; PID ignores them

Examples of dishonest asymmetry:

- RL gets true hidden demand
- RL gets all plug-flow segment states
- RL gets future disturbance schedules in online observations

## Recommended benchmark contract fields

Each env wrapper should define or version:

- plant model name and version
- scenario profile name and seed
- observation profile name and version
- controller policy class
- action contract and constraints
- reward or score definition
- reset / warm-start semantics
- logged metrics and comparator outputs

## Realistic principles to capture

- the observation profile is a first-class benchmark object
- comparison baselines should be explicit about what they are allowed to see
- warm-start versus cold-start semantics should be configurable and documented
- historical traces and feature engineering should remain downstream of raw measurement contracts, not substitutes for them

## Realism we are intentionally ignoring

- enterprise historian or SCADA integration detail in the benchmark wrapper
- fully generic offline dataset export contracts in v1
- cross-plant fleet benchmark orchestration in the first implementation

## Recommended first-generation benchmark profiles

### Minimal baseline profile

A sparse controller view approximating a typical PID or PI installation.

### Realistic RL profile

A modestly richer set of realistic measurements, such as upstream quality surrogates, one extra analyzer, or a detention-time proxy.

### Rich instrumentation profile

A still-defensible but more aggressively instrumented view for understanding upper-bound value from better sensing.

These three profiles are enough to make the toolkit useful immediately for comparative control studies.
