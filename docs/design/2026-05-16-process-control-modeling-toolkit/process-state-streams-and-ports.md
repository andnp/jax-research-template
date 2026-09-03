# Streams, State, and Ports

## Purpose

The toolkit needs a common language for what moves through a process and what modules are allowed to exchange. This page defines the conceptual contract for **material streams**, **signal ports**, and **internal state**.

## What this module family is

A process simulator usually has two kinds of things moving around:

- **material**: water, sludge, air, dissolved species, particulate species, heat, inventory
- **signals**: analyzer values, control requests, valve positions, setpoints, alarm states

Those should not share the same contract. A material stream obeys conservation and transport logic. A signal does not.

## Contract

### Material stream contract

A material stream should support at least:

- flow rate or transport magnitude
- one or more conserved or approximately conserved components
- optional scalar properties such as temperature, pH, conductivity, or density
- valid units and ranges
- time metadata when delayed transport matters

A first-generation implementation can represent composition either as:

- a fixed-order array plus schema metadata, or
- a named mapping with a stable schema wrapper

The contract matters more than the initial storage type.

### Signal port contract

A signal port should support:

- signal name
- units
- value
- valid range and clipping policy
- timestamp / sample time
- quality flags such as good, stale, estimated, bad, or missing

Examples:

- flow measurement
- outlet chlorine residual analyzer
- requested dose setpoint
- realized pump output
- operator mode flag

### Internal state contract

Each module should own its own internal state and avoid exposing it unless an explicit sensor or diagnostic port publishes it.

Examples:

- tank inventory
- plug-flow segment concentrations
- controller integrator state
- sensor lag buffer
- actuator internal travel state

## Realistic principles to capture

- material streams should preserve the physical interpretation of flow and composition
- signals should be allowed to lag, drift, clip, or drop out without altering the latent plant
- internal state should remain hidden unless instrumentation or diagnostics explicitly expose it
- schemas should remain stable enough that observation builders and unit tests can reason about them cleanly

## Realism we are intentionally ignoring

- full thermodynamic property packages
- arbitrary unit-conversion systems in v1
- geometry-resolved spatial fields when a lower-order approximation is adequate
- exact conservation guarantees for every noncritical derived field

## Recommended first-generation scope

Start with a small number of well-supported stream/state families:

- **liquid process stream** for WTP-style flow and dissolved demand/residual components
- **activated-sludge process stream** for BSM1-style biochemical compositions
- **scalar signal port** for all instrumentation and actuation signals

This provides enough structure for reuse without over-generalizing too early.

## Why this matters for RL benchmarking

A stable distinction between latent state and signal ports is what prevents simulator leakage. If a plug-flow reactor stores ten segment states internally, the agent only gets those values if a configured measurement path publishes them.
