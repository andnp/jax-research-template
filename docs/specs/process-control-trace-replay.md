# Process-control trace replay

## Purpose

Replay a canonical process trace through a simulator using the same timing and
supervisory-control semantics used by the source plant. The replay layer is
generic: benchmark-specific disturbance construction and signal naming remain
outside the shared toolkit.

## Replay case

`ReplayCase` is immutable metadata around one `UniformTrace` and contains:

- the trace to replay;
- an optional half-open row range;
- the number of leading warmup rows;
- the random seed;
- names of required control and disturbance signals.

The selected range must be non-empty, must not cross a trace segment boundary,
and must contain valid values for every required signal. A caller must split a
trace at gaps before replaying it. The runner never fills missing values or
silently carries a state across a gap.

Controls retain their trace schema order unless an adapter explicitly maps
named signals to a simulator action vector. In a supervisory benchmark, these
are setpoints. Realized actuator values are not replay actions.

Warmup rows advance the simulator state but are excluded from evaluation. The
warmup count must be smaller than the selected row count.

## Adapter boundary

The runner depends on a small benchmark adapter with two responsibilities:

1. construct a replay environment from the `ReplayCase`;
2. expose `reset(key)` and `step(state, action, key)` functions.

The adapter may turn trace disturbances into a benchmark-specific source
trajectory at construction time. It may also map simulator diagnostics into
named output signals. The shared runner does not know benchmark names,
disturbance types, latent state fields, or customer data formats.

The environment step is expected to return the next state, observation,
reward, termination flag, and diagnostic mapping. The runner preserves the
diagnostic mapping as time-major arrays without interpreting its keys.

## Result representation

`ReplayResult` contains a JAX-compatible time-major payload with:

- replay time;
- requested controls;
- replay disturbances;
- observations;
- rewards;
- termination and truncation flags;
- diagnostic arrays;
- the final simulator state.

The result also retains the source `ReplayCase` metadata and warmup boundary.
Output signal names and units are supplied by a benchmark adapter or a
project-local projection into a `UniformTrace`; they are not inferred from
arbitrary diagnostic keys.

## Execution semantics

- Reset uses exactly the case seed.
- One trace row produces one environment step.
- The requested control row is passed unchanged to the adapter’s action map.
- Randomness is split once per step in a deterministic order.
- The scan stops only after the selected horizon or an environment termination.
- Output arrays preserve the time axis, including rows before termination; a
  separate termination mask identifies the final transition.

The first implementation uses `jax.lax.scan` so the same path is suitable for
single-case and batched execution later. File I/O, pandas, plotting, and
statistical comparison are outside this layer.

## Non-goals

This layer does not fit parameters, select a model family, compare two traces,
or define customer-specific signal mappings. Those consumers operate on the
replay result and remain separate from execution.
