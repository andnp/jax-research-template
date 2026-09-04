# Process-control system-identification traces

## Purpose

Provide a process-control-native representation for system identification,
replay, and calibration. The API must describe process signals without knowing
the database, customer, plant, or storage format that produced them.

## Representations

The toolkit has two representations:

- `SignalSeries` stores one signal at its native timestamps. Native series may
  have different lengths and sampling intervals.
- `UniformTrace` stores aligned struct-of-arrays data on one fixed analysis
  grid. Each signal group is a dense matrix with shape `[time, signal]`.

The numeric fields are array-oriented. Signal names, roles, units, and source
metadata are immutable metadata separate from the numeric matrices.

## Signal roles

Every aligned signal belongs to one role:

- `control`: a command or supervisory setpoint supplied to the process;
- `disturbance`: an exogenous process input not controlled by the experiment;
- `measurement`: a measured process output used for identification or scoring;
- `actuator`: a realized manipulated variable or controller output.

Roles determine safe default alignment policies. A caller may override a policy
only explicitly and the resulting policy must be retained in provenance.

## Time semantics

Times are finite, strictly increasing seconds from an arbitrary epoch. A
`UniformTrace` has one positive `sample_period`; sample `t` represents the
interval `[time[t], time[t] + sample_period)`.

Alignment is causal: a value at grid time `t` may use source observations at or
before the end of the represented interval, but never observations from a
later interval. Controls use right-continuous, last-known semantics by default.

## Missing data and resampling

Resampling is explicit and deterministic. Supported policies include causal
last-value hold, interval aggregation, bounded interpolation, and invalidation.
Long gaps are never filled implicitly. The aligned trace retains validity,
source coverage, and segment identifiers so lagged models cannot cross a gap.

The trace layer owns semantic alignment. A database adapter may perform an
equivalent server-side pre-aggregation for efficiency, but it must expose the
same policy and provenance to the trace layer.

## Invariants

- construction rejects non-finite or non-monotonic timestamps;
- all signal arrays agree with their timestamp lengths and declared widths;
- a uniform trace has a positive finite period and a regular time grid;
- signal metadata lengths match matrix widths;
- validity and provenance are never inferred from filled values;
- fixed-step simulators consume only uniform traces with an explicit gap policy.

## Non-goals

This API does not load databases, infer units from tag names, choose a model
family, tune controller parameters, or provide statistical experiment design.
