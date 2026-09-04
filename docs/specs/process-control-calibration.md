# Process-control simulator calibration

## Purpose

Calibrate a parameterized simulator against a reference process trace while
preserving the source plant's supervisory-control semantics. Calibration is a
consumer of trace replay and comparison; it does not load storage formats or
define benchmark-specific signal mappings.

## Calibration problem

A calibration problem contains:

- an ordered parameter space with one finite lower and upper bound per
  parameter;
- one or more chronological replay cases;
- a named comparison specification for each case;
- a residual scale and weight for each compared signal;
- an injected simulator evaluator that maps a parameter vector and replay case
  to a candidate trace.

The parameter vector is an array in the exact order declared by the parameter
space. Parameter names and metadata remain outside the numeric payload. Every
parameter has an initial value inside its bounds and a strictly positive scale
used to interpret normalized changes.

The evaluator must be deterministic for a fixed parameter vector and case. A
calibration run must identify its algorithm, seed, parameter space, comparison
specifications, and input-data provenance.

## Supervisory semantics

Replay cases pass requested supervisory controls to the simulator as actions.
Measured actuator outputs, when available, are diagnostic signals and may be
comparison targets, but they are never replay actions. A calibration result is
invalid if it obtains its apparent fit by bypassing the simulator's inner
controller.

## Cases and chronology

Cases are contiguous, fixed-grid trace windows. A calibration dataset should
normally be partitioned chronologically into training, validation, and locked
test cases:

- training cases determine parameter values;
- validation cases select parameter groups, bounds, weights, and optimizer
  settings;
- test cases are evaluated once after all choices are locked.

Each case may include leading warmup rows. Warmup advances the simulator but
does not contribute residuals. Cases must not cross trace segment boundaries,
and gaps must never be bridged by a replay or residual calculation.

## Residuals and objective

For each named signal pair, valid post-warmup samples contribute:

```text
residual = weight ** 0.5 * (candidate - reference) / scale
```

The residual payload preserves time-major rows, signal columns, validity, and
segment identity. Invalid or non-finite samples contribute no residual. The
objective is the sum of squared valid residuals divided by the valid residual
count, with an explicit undefined flag when no residual is valid.

Aggregate bias, MAE, RMSE, correlation, R², and normalized RMSE remain useful
for reports, but an optimizer should receive residuals rather than only an
aggregate metric. This keeps signal weighting and temporal coverage explicit.

## Parameter groups

Parameter groups are metadata for staged fitting, not hidden optimizer policy.
A caller may select groups explicitly for a calibration stage. A parameter is
eligible only when the simulator evaluator actually consumes it; unused
parameters must not be silently optimized.

Initial staged fitting should separate:

1. inner-loop and actuator dynamics;
2. process and measurement dynamics;
3. a small joint refinement.

Initial conditions are separate from physical parameters whenever possible.
Long warmup or an explicitly estimated initial-state policy must be used rather
than allowing physical parameters to absorb reset-state error.

## Provenance and scope boundary

Calibration artifacts should retain parameter values, bounds, scales, selected
groups, case definitions, comparison specifications, objective settings,
optimizer settings, algorithm version, and data provenance. The core toolkit
must not contain customer data, customer tags, database paths, or
benchmark-specific mappings. Project adapters own those details and may pass
generic traces and identified parameter values into this layer.

## Non-goals

This first contract does not prescribe an optimizer, infer parameter bounds,
fit stochastic noise, search time lags, or automatically choose a model family.
Those decisions belong to a caller with evidence about the simulator and the
available excitation.
