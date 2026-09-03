# Sensors

## Purpose

Sensors are the main bridge between true plant state and controller-visible state. They are also one of the most important realism levers for RL benchmarking.

## What this module family is

A sensor reads some aspect of latent plant state and emits a measurement signal. The signal may be delayed, noisy, biased, clipped, stale, missing, or estimated.

## Common sensor contract

Every sensor should define:

- measured quantity
- units and normal range
- sample period
- latency or lag model
- noise model
- bias / drift behavior
- dropout / failure behavior
- quality flags
- reset behavior

The output should be a signal, not a direct pointer into latent state.

## Recommended sensor types

### Flow sensor

**What it is:** measures volumetric or mass flow through a stream.

**Why it matters:** flow affects residence time, dilution, loading, and actuator effectiveness.

**Contract notes:**

- sampled scalar output
- optional low-pass lag
- optional pulse / smoothing behavior
- optional sensor failure or frozen reading

**Realistic principles to capture:**

- bias, lag, and occasional dropout matter more than exotic failure models
- measured flow may differ from true flow transiently

**Ignored realism:**

- detailed meter installation effects
- full meter diagnostics in v1

### Residual or concentration analyzer

**What it is:** measures a dissolved species or quality metric such as chlorine residual, ammonia, nitrate, or dissolved oxygen.

**Why it matters:** many closed-loop tasks revolve around these analyzers.

**Contract notes:**

- sample-and-hold behavior is common
- analyzer lag and transport delay are often important
- clipping to feasible ranges is realistic

**Realistic principles to capture:**

- analyzers are often slower and noisier than operators wish
- different analyzers can live at inlet, mid-basin, and outlet locations
- sparse intermediate analyzers are usually more realistic than full latent-state exposure

**Ignored realism:**

- reagent consumption and maintenance procedures in detail
- chemistry-specific calibration workflow unless benchmark-critical

### Level sensor

**What it is:** measures inventory or liquid level in a vessel or basin.

**Why it matters:** level is often a strong proxy for hydraulic state, residence time, or impending constraints.

**Realistic principles to capture:**

- noise is often less important than lag and occasional bias
- derived detention-time or occupancy proxies may be built from level plus flow

### Quality surrogate sensor

**What it is:** measures a leading indicator of downstream demand or process load, such as turbidity, UV254, TOC proxy, conductivity, or a raw-water-quality composite.

**Why it matters:** these are often the most valuable predictive features for RL because they move before the main controlled variable responds.

**Realistic principles to capture:**

- correlations with true demand can drift over time
- a surrogate should be informative but imperfect
- noise and bias should preserve partial observability rather than collapse it

**Ignored realism:**

- laboratory workflows and sample handling
- cross-interference models unless they change benchmark conclusions

### Temperature and pH sensors

**What they are:** scalar chemistry context sensors.

**Why they matter:** reaction rates and effective process gain often depend on them.

**Realistic principles to capture:**

- relatively smooth evolution
- occasional calibration offset
- process-family-specific importance

### Soft sensor / estimator

**What it is:** an inferred variable computed from other measurements rather than measured directly.

**Examples:**

- estimated chlorine demand
- estimated residence time
- inferred sludge blanket height
- inferred fouling index

**Why it matters:** many real plants rely on estimators where direct sensing is sparse or expensive.

**Realistic principles to capture:**

- estimator error and bias should be explicit
- the estimator should only use allowed upstream measurements
- a soft sensor can legitimately give RL an edge if the plant could plausibly compute it too

## Sensor realism for benchmark design

### Realistic ways to help RL

- give RL one or two additional sparse analyzers
- expose leading indicators such as quality surrogates
- expose residence-time proxies derived from level and flow
- include sensor-health or quality flags where realistic

### Unrealistic ways to help RL

- expose every internal compartment concentration directly
- expose true demand or disturbance magnitude when the plant would not know it
- leak future signals or post-processed hindsight features into online observations

## Recommended first-generation sensor set

Across process families, the initial reusable set should include:

- flow sensor
- residual / concentration analyzer
- level sensor
- quality surrogate sensor
- temperature sensor
- pH sensor
- soft sensor base class or contract

These cover most of the realistic observability gains that matter for control benchmarking.
