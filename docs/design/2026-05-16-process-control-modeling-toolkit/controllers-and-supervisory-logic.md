# Controllers and Supervisory Logic

## Purpose

The toolkit should support strong baseline comparisons and realistic closed-loop behavior without baking any one controller class into the plant model.

## What this module family is

Controllers consume measurements and produce requested actuator commands. Supervisory logic modifies or overrides those requests based on operating mode, safety, or higher-level objectives.

## Common controller contract

Each controller should define:

- required measurement inputs
- optional setpoints or mode inputs
- internal control state
- output command signals
- reset behavior
- sampling interval or expected call cadence

## Recommended controller types

### PI / PID controller

**What it is:** the standard baseline for many single-loop and cascade tasks.

**Why it matters:** it is the natural comparator for RL in most process-control settings.

**Principles to capture:**

- proportional, integral, and optional derivative action
- anti-windup behavior
- output limits
- optional feed-forward input

### Feed-forward block

**What it is:** adjusts actuator request using measured disturbance proxies.

**Why it matters:** many realistic baselines are stronger than pure feedback PID.

### Cascade controller

**What it is:** an outer loop sets the target for an inner loop.

**Why it matters:** common when a slow quality variable depends on a faster local manipulated variable.

### Override selector / supervisor

**What it is:** picks among candidate control actions based on rules, priorities, or safety states.

**Why it matters:** real plants often have supervisory logic above local loops.

### Safety interlock

**What it is:** clamps or reroutes control when measured conditions violate safety or operating constraints.

**Why it matters:** benchmark plants should be able to model action blocking, fallback modes, and alarm-driven restrictions cleanly.

## Realistic principles to capture

- baseline controllers should consume only the measurements they are allowed to see
- controller state such as integral terms should be explicit and resettable
- requested action and realized action should be distinct when actuator dynamics matter
- supervisory logic should sit above both baseline and learned controllers where possible

## Realism we are intentionally ignoring

- full DCS configuration workflows
- operator faceplates and hand/auto stations unless benchmark-critical
- plant-specific tuning rituals and commissioning sequences in detail

## Recommended first-generation controller set

- PI controller
- PID controller
- feed-forward wrapper
- simple cascade composition
- override selector
- safety clamp / interlock wrapper

These are enough to build meaningful industrial-style comparator baselines without turning the toolkit into a full control-system engineering suite.
