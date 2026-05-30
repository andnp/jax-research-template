# External Influence and Demo Wiring

## Purpose

This appendix defines how the toolkit supports external runtime inputs — disturbance injection, setpoint changes, parameter overrides, and sensor-fault triggers — when the simulator is driven by an external clock or interactive demo rather than a pure training loop.

## Why this matters

Most of the time, the toolkit runs as a closed simulation: scenarios, controllers, and plant modules execute together with no outside intervention. But a strategically important use case is **demo mode**, where the simulation runs on a fixed wall-clock cadence and an operator, presenter, or UI can inject external influences in real time.

The current WTP env already supports this through:

- `schedule_disturbance(magnitude, start_in_steps, duration_steps)` for queued disturbance events
- `apply_runtime_control(tag, value)` for live parameter changes (PID gains, setpoints, sensor failures)
- a FastAPI surface (`wtp/fastapi.py`) that exposes these methods as HTTP endpoints

The toolkit needs to preserve and generalize this capability without coupling every module to a specific API framework.

## Two execution modes

The toolkit should support two distinct execution modes.

### Training mode

- the simulation runs as fast as possible
- scenarios and disturbances are fully pre-scheduled or seed-driven
- no external inputs during a rollout
- this is the default and should have zero overhead from demo machinery

### Demo / interactive mode

- the simulation advances on a fixed clock or on external step triggers
- an external source can inject disturbances, change setpoints, trigger faults, or modify parameters between steps
- the plant, sensors, and controllers see injected influences the same way they see scenario-driven ones

The key design rule is: **demo mode should use the same plant and module graph as training mode**. The only difference is where the exogenous events come from.

## Architecture

### External influence bus

The toolkit should define an **influence bus** — a lightweight channel through which external events reach the simulation. The bus should support at minimum:

- **disturbance injection**: queue a disturbance event (demand slug, flow spike, etc.) for a specific step or step range
- **setpoint override**: change a controller setpoint at the next step
- **parameter override**: change a module parameter (e.g., sensor bias, actuator gain) at the next step
- **fault injection**: trigger a sensor dropout, actuator stiction event, or equipment derating
- **mode change**: switch operating mode (e.g., day/night, maintenance bypass)

The bus collects pending events between steps. At the start of each step, the scheduler drains the bus and routes events to the appropriate modules before the normal execution-phase sequence begins.

### Relationship to scenario modules

In training mode, scenario modules populate the same event types internally. The influence bus is simply an additional event source that is active only when external inputs are enabled.

This means:

- scenario modules and external inputs can coexist
- external inputs do not bypass the normal scheduling or execution-phase contract
- modules do not need to know whether an event came from a scenario or from an external source

### Relationship to the scheduler

The scheduler should:

1. drain the influence bus at the start of each step
2. merge external events with scenario-generated events
3. apply the merged events through the normal execution phases

This keeps the scheduler as the single owner of step orchestration.

### FastAPI / demo surface

The toolkit should **not** embed FastAPI inside the plant or module layer. Instead:

- the influence bus is a plain Python object with typed methods
- a thin FastAPI adapter (or any other API framework) translates HTTP requests into bus events
- the adapter lives in the benchmark / env layer, not in the toolkit core

This preserves the current FastAPI demo capability while keeping the core toolkit framework-agnostic.

### Relationship to `EnvMetricsSink`

The existing `EnvMetricsSink` protocol handles outbound metrics from the simulation. The influence bus handles inbound events to the simulation. These are complementary and should remain separate:

- `EnvMetricsSink`: simulation → external consumers (metrics, logging, dashboards)
- influence bus: external sources → simulation (disturbances, overrides, faults)

The toolkit should support both without coupling them. A demo surface might use both: it reads metrics via the sink and injects events via the bus.

## Contract sketch

The influence bus should support at least:

```
queue_disturbance(target_module, event_type, magnitude, start_step, duration_steps)
queue_setpoint_change(controller_id, new_setpoint, effective_step)
queue_parameter_override(module_id, parameter_name, new_value, effective_step)
queue_fault(module_id, fault_type, duration_steps)
queue_mode_change(mode_name, effective_step)
drain_pending(current_step) -> list of events to apply
```

The exact API will evolve during implementation, but the shape should support targeted delivery to specific modules while keeping the bus itself simple.

## What this enables

With the influence bus pattern:

- demos can inject disturbances and faults in real time without modifying plant code
- the same benchmark can run headless for training and interactive for demos
- multiple demo surfaces (FastAPI, WebSocket, CLI) can coexist by writing to the same bus
- external inputs are auditable and reproducible if the bus is logged

## What this does not cover

- the specific FastAPI route design (that belongs in the benchmark layer)
- UI/HMI design for demo surfaces
- authentication or authorization for demo endpoints
- real-time clock synchronization details

## Recommended implementation approach

1. Define the influence bus interface in Phase 0 alongside the scheduler
2. Wire it into the scheduler's step-start drain logic
3. Build the first FastAPI demo adapter when the chlorine benchmark reaches demo-ready state
4. Port the existing WTP FastAPI patterns to use the bus rather than direct env mutation
