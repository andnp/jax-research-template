# Scheduler and Execution-Phase Contract

## Purpose

This appendix specifies the scheduler — the runtime component that orchestrates module execution within each simulation step. The main design doc describes the six execution phases conceptually. This appendix turns them into concrete design decisions.

## Why the scheduler is critical

Without an explicit scheduler contract:

- each benchmark will invent its own stepping logic
- multi-rate modules will use ad hoc timing that differs across benchmarks
- the execution order will become implicit and hard to audit
- demo mode and training mode may diverge in subtle ways

The scheduler is the single owner of "what runs when" inside a step.

## Execution phases

Each simulation step proceeds through these phases in order:

### Phase 1: Drain external influences

The scheduler drains the influence bus (see [external influence appendix](external-influence-and-demo-wiring.md)) and merges external events with scenario-generated events for this step.

### Phase 2: Scenario and disturbance updates

Scenario modules update exogenous conditions for the current interval: influent profiles advance, disturbance events activate or deactivate, operating-mode schedules apply.

### Phase 3: Controller and agent output

Controllers and agent adapters that are scheduled to execute at this step emit requested actuator commands based on their most recent measurement inputs.

Controllers that are not scheduled to execute this step hold their previous output (sample-and-hold semantics).

### Phase 4: Actuator realization

Actuators map requested commands to realized physical outputs subject to saturation, ramp-rate limits, delay, and fault states.

### Phase 5: Process graph advance

The material-process graph advances latent plant state over $dt$. Unit operations consume incoming streams and actuator outputs, apply chemistry and transport, and produce outgoing streams and updated internal state.

The process graph executes in a **deterministic topological order** within this phase.

### Phase 6: Sensor sampling

Sensors that are scheduled to sample at this step read from latent plant state and update their held measurement values. Sensors not scheduled to sample hold their previous reading.

After sensor sampling, observation builders assemble consumer-specific views from the measurement bus.

## Scheduler ownership

The scheduler is a **runtime object** that the plant assembly function creates. It is not a per-benchmark ad hoc loop.

The scheduler should:

- hold references to all registered modules
- know the topological execution order of the process graph
- track simulation time and step count
- manage per-module sample cadences
- drain the influence bus at step start
- call each module's step/update method in the correct phase and order
- produce the final observation, reward, and done signals (or delegate to the benchmark wrapper)

## Multi-rate module registration

Modules that operate at rates different from the base simulation $dt$ must declare their cadence at registration time.

### Cadence types

- **every step**: the default; module executes every simulation step
- **fixed period**: module executes every $N$ steps (or every $T$ simulation-time units)
- **on-demand**: module executes only when explicitly triggered (e.g., lab sample events)

### Sample-and-hold rule

When a module does not execute at a given step, its most recent output is held constant. This is the **only** valid default behavior for multi-rate modules. No interpolation, no extrapolation.

### Example cadences

| Module type | Typical cadence |
|-------------|----------------|
| Process unit operations | Every step |
| Chemistry blocks | Every step |
| Flow sensor | Every step or every few steps |
| Online analyzer | Every 5–60 steps (minutes-scale) |
| Lab measurement | Every 100–1000 steps (hours-scale) |
| PI controller | Every 1–10 steps |
| Supervisory logic | Every 10–100 steps |
| Scenario modules | Every step (check schedule) |

## Misaligned controller/process timing

When a controller's sample time does not align with the process $dt$:

- the controller sees **stale sensor values** from the last sensor sample
- the controller's output is **held constant** until its next execution
- there is no interpolation or extrapolation in either direction

This is realistic: real controllers see sample-and-hold measurements and hold their output between executions.

## Topological ordering

The process graph within Phase 5 must execute in deterministic topological order. This means:

- upstream units execute before downstream units
- mixers execute after all their input sources
- splitters execute before their downstream consumers
- recycle loops require explicit handling (see below)

### Recycle loop handling

Recycle loops create cycles in the module graph. The scheduler should handle these by:

- using a one-step-delayed recycle value (the recycle stream carries the previous step's output)
- this is physically realistic for most process systems where recycle flows have transport delay

The scheduler should detect cycles during assembly and require explicit recycle-link module placement to break them.

## Scheduler interface sketch

```
class Scheduler:
    def register_module(module, phase, cadence)
    def set_topology(execution_order_within_phase_5)
    def step(dt, influence_bus, rng_key) -> StepResult
    def reset(rng_key) -> InitialState
    def current_time() -> float
    def current_step() -> int
```

The exact API will evolve, but the scheduler must own step orchestration rather than delegating it to benchmark-specific code.

## Relationship to JAX compilation

If the toolkit targets JAX-native execution:

- the scheduler's `step` function must be `jit`-compatible
- module registration and topology happen at **build time** (Python)
- the compiled `step` function is a **fixed graph** at runtime
- multi-rate cadences should be representable as modular arithmetic on the step counter, not as Python conditionals that break tracing

This means cadence checks should look like `jnp.where(step_count % period == 0, ...)` rather than Python `if` statements.

## Relationship to the influence bus

The scheduler owns the influence bus drain. At the start of each step:

1. drain pending events from the bus
2. merge with scenario-generated events
3. apply merged events to the appropriate modules

This ensures external inputs follow the same execution-phase contract as scenario-driven events.

## What the scheduler does not own

- reward computation (belongs to the benchmark wrapper)
- episode termination logic (belongs to the benchmark wrapper)
- observation space / action space definitions (belongs to the benchmark wrapper)
- module implementation details (each module owns its own step logic)

## Recommended Phase 0 deliverable

Build a minimal scheduler prototype that can:

1. register modules with phase and cadence declarations
2. execute the six-phase step sequence in order
3. handle multi-rate sample-and-hold for at least one sensor and one controller
4. drain a simple influence bus
5. work with a trivial two-module process graph (source → tank)

This prototype validates the scheduler contract before any domain-specific modules are built.
