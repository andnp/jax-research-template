# Actuators

## Purpose

Actuators define how controller intent becomes physical manipulation. They are essential because many control failures come from actuator reality rather than controller logic.

## What this module family is

An actuator consumes a requested command and emits a realized manipulated variable subject to physical and operational constraints.

## Common actuator contract

Every actuator should define:

- requested command input
- realized output signal or manipulated stream effect
- saturation limits
- slew / ramp-rate limits
- optional deadband, backlash, stiction, or delay
- failure and override modes
- reset behavior

## Recommended actuator types

### Dose pump

**What it is:** meters a reagent into a process stream.

**Why it matters:** it is the core actuator in chlorine dosing and many chemical treatment loops.

**Realistic principles to capture:**

- output saturation
- finite ramping or update rate
- mismatch between requested and realized dose
- optional minimum on/off or low-end deadband

**Ignored realism:**

- detailed pump wear physics
- tank refill logistics unless benchmark-critical

### Control valve

**What it is:** modulates flow or split ratio.

**Why it matters:** many hydraulic control problems are expressed through valve position rather than direct flow command.

**Realistic principles to capture:**

- nonlinear valve gain if it matters to control difficulty
- travel limits and rate limits
- occasional stiction or hysteresis

### Variable-speed pump / recycle pump

**What it is:** changes flow through a branch or recycle loop.

**Why it matters:** common in wastewater and recirculating treatment processes.

**Realistic principles to capture:**

- finite speed changes
- flow saturation
- optional efficiency or gain changes across operating regions

### Blower / aeration actuator

**What it is:** injects air or oxygen-transfer capacity into a unit.

**Why it matters:** needed for BSM1-style aeration control.

**Realistic principles to capture:**

- saturation and minimum operating region
- lag between requested and realized transfer effect
- optional energy penalty signals

## Realistic principles to capture across all actuators

- controllers should not directly overwrite latent plant state
- realized actuation should be observable separately from requested action when it matters
- actuator failure modes should be explicit and schedulable
- rate limits often matter as much as absolute limits for RL difficulty

## Realism we are intentionally ignoring

- maintenance work-order detail
- electrical starter and drive internals
- detailed equipment health prognosis in v1

## Recommended first-generation actuator set

- dose pump
- valve
- variable-speed pump
- blower / aeration actuator
- a generic saturation + rate-limit wrapper usable around simpler actuators
