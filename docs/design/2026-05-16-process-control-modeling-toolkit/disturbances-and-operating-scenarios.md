# Disturbances and Operating Scenarios

## Purpose

A process benchmark becomes valuable when it exposes controllers to structured variation, not just one nominal operating point.

## What this module family is

Scenario modules define exogenous conditions and event schedules that modify the plant or instrumentation over time.

Examples include:

- influent loading profiles
- diurnal flow variation
- storm or upset events
- scheduled reagent-demand slugs
- sensor dropout events
- actuator faults
- operating-mode schedules
- setpoint changes

## Common contract

A scenario module should define:

- what it modifies
- when it activates
- stochastic vs deterministic behavior
- parameterization and seed behavior
- whether it is visible to the controller directly, indirectly, or not at all

## Recommended scenario types

### Influent source profile

**What it is:** a source of incoming flow and composition.

**Examples:**

- steady-state profile
- recorded time series
- mixed seasonal library
- procedurally generated diurnal profile

### Disturbance event

**What it is:** a localized or sustained perturbation.

**Examples:**

- chlorine-demand slug
- sudden ammonia increase
- storm inflow event
- temperature shift
- sensor calibration jump

### Failure event

**What it is:** a sensor or actuator abnormal condition.

**Examples:**

- dropout
- frozen sensor
- biased analyzer
- valve sticking
- pump saturation degradation

### Operating-mode schedule

**What it is:** a structured change in plant objectives or configuration.

**Examples:**

- setpoint schedule
- daytime / nighttime operating mode
- operator-imposed throughput band
- maintenance bypass state

## Realistic principles to capture

- disturbances should have temporal structure and finite duration when appropriate
- not every disturbance should be directly observed; many should be inferred through sensors
- repeated scenario families should vary in amplitude, timing, and shape to prevent overfitting to one canned event
- seeds should make scenarios reproducible for fair controller comparison

## Realism we are intentionally ignoring

- every possible plant upset category in v1
- organization-specific maintenance and staffing workflows
- market, weather, or network models unless they directly affect the control task

## Recommended first-generation scenario library

- deterministic diurnal profiles
- stochastic drift processes
- impulse and sustained disturbance events
- setpoint schedules
- sensor dropout / frozen-signal events
- actuator saturation or gain-loss events

This library will produce far more benchmark value than an ultra-detailed nominal plant with no scenario diversity.
