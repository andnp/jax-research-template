# Realism and Operations Extensions

## Purpose

This appendix captures realism layers that sit around the core process model and make the toolkit more useful as an internal control and RL testbed.

These extensions are not the first thing required to make a simulator run. They are the things that make a simulator feel operationally believable and support higher-value experiments.

The selected extension families in scope here are:

1. measurement quality, calibration, and periodic lab tests
2. multi-rate timing and reporting
3. equipment degradation and health state
4. soft sensors and estimators
5. cost, waste, and reagent accounting
6. delayed labels and delayed truth

## 1. Measurement quality, calibration, and periodic lab tests

### What this adds

This extension family represents the reality that plants often rely on a mixture of:

- fast online analyzers
- noisy or drifting sensors
- infrequent but higher-trust lab measurements
- calibration and reconciliation logic

### Representative modules

- periodic grab-sample / lab-measurement source
- lab turnaround delay
- online-analyzer drift and bias model
- calibration event or bias-reset event
- measurement quality flags
- online/lab reconciliation block

### Why it matters

This creates realistic information asymmetry between:

- the true plant state
- what the controller sees right now
- what operations learns later from the lab

It also supports realistic studies of:

- trust in noisy analyzers
- calibration strategy
- control under uncertain measurement quality

## 2. Multi-rate timing and reporting

### What this adds

Not everything in a plant updates at the same cadence.

Examples:

- process state evolves continuously or at a fast simulation step
- online sensors update every few seconds or minutes
- lab values arrive every few hours
- control loops execute on their own cadence
- historians or reports may aggregate more slowly still

### Representative modules or runtime features

- explicit per-module sample periods
- sample-and-hold signal behavior
- asynchronous controller update schedules
- delayed report streams
- timestamped measurement events

### Why it matters

This makes the simulator much more believable and prevents unrealistic assumptions about synchronous perfect updates.

## 3. Equipment degradation and health state

### What this adds

This extension family represents slow nonstationarity and degradation in equipment or process capability.

Examples:

- pump efficiency loss
- sensor aging
- fouling accumulation
- declining transfer efficiency
- valve stiction growth
- reagent quality drift

### Representative modules

- slow-drift health-state block
- fouling or efficiency-loss block
- actuator derating model
- sensor-health degradation model

### Why it matters

Many real control problems are hard not because of one disturbance event, but because the plant gradually becomes different over time.

This is a major realism win and also a strong source of benchmark value.

## 4. Soft sensors and estimators

### What this adds

Many practically important variables are not measured directly. They are estimated from other signals.

Examples:

- residence-time estimate
- demand estimate
- blanket-height estimate
- fouling-state estimate
- chemistry-state estimate

### Representative modules

- deterministic soft-sensor block
- observer / estimator module
- bias- or uncertainty-aware estimate wrapper
- estimate quality / confidence signal

### Why it matters

This provides a realistic way to increase controller observability without leaking hidden simulator internals directly.

It also supports realistic internal debates like:

- should the controller trust the soft sensor?
- what happens when the estimator drifts?
- how much value does a better estimator provide?

## 5. Cost, waste, and reagent accounting

### What this adds

Many real optimization problems are not just about hitting a quality target. They also trade off:

- reagent spend
- water or makeup usage
- energy usage
- waste or disposal burden
- cleaning and regeneration burden

### Representative modules or accounting hooks

- reagent-consumption accounting
- energy-consumption accounting
- water / makeup accounting
- waste-stream or disposal accounting
- maintenance or regeneration cost hooks

### Why it matters

This lets benchmarks express realistic optimization questions instead of only setpoint tracking questions.

Examples:

- how much extra bleach is worth spending to protect compliance margin?
- when is makeup flow worth the water cost?
- when should a controller trade throughput against fouling burden?

## 6. Delayed labels and delayed truth

### What this adds

In many operational settings, the best available “truth” does not arrive in real time.

Examples:

- lab values arrive later
- compliance measurements are reviewed after the fact
- human-labeled events are only known much later
- offline quality assurance resolves which reading was trustworthy

### Representative modules or interfaces

- delayed-truth stream
- label-availability schedule
- posthoc evaluation signal
- delayed-annotation channel

### Why it matters

This extension family is especially useful if the toolkit should support not only online control simulation but also learning/evaluation workflows where the controller and the analyst do not have the same information at the same time.

## Why these extensions are valuable together

These extensions create a simulator that is not just physically plausible, but operationally plausible.

Together they let the toolkit model situations like:

- a drifting online analyzer is partially corrected by sparse lab updates
- an estimator fills in a missing or low-quality signal
- the process and its sensors evolve at different rates
- equipment gradually degrades
- the controller balances quality against reagent and water cost
- the best available truth arrives only later for evaluation or retraining

That is much closer to how real systems are operated and improved.

## Suggested implementation order within this extension set

1. measurement quality, calibration, and periodic lab tests
2. multi-rate timing and reporting
3. soft sensors and estimators
4. equipment degradation and health state
5. cost, waste, and reagent accounting
6. delayed labels and delayed truth

This order reflects both realism payoff and dependency structure.

## Recommendation

These extension families should be treated as first-class additions to the toolkit roadmap, not as afterthoughts.

In particular:

- periodic lab tests plus flaky online analyzers
- multi-rate timing
- soft sensors

are likely to produce a very large realism gain for relatively modest architectural complexity.