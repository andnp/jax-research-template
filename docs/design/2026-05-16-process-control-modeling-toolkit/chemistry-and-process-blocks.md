# Chemistry and Process Blocks

## Purpose

Chemistry and process blocks add the nonlinear, delayed, load-dependent behavior that makes a simulator useful for control research.

## What this module family is

These modules describe transformations that may live inside unit operations or between transport steps.

Examples include:

- first-order decay
- demand consumption
- oxidation or disinfection proxy kinetics
- oxygen transfer or mass-transfer approximations
- Monod-style biological reactions
- temperature correction factors

## Common contract

Each process block should define:

- required state variables or components
- required parameters
- update contribution over $dt$
- valid operating assumptions
- failure or clipping behavior if negative concentrations would otherwise arise

## Recommended first-generation blocks

### First-order decay

**What it is:** a simple exponential loss model.

**Why it matters:** a good first approximation for many fading or consumption-like behaviors.

**Principles to capture:**

- magnitude scales with current amount
- temperature dependence can optionally modify rate
- easy to interpret and calibrate roughly

**Ignored realism:**

- multi-path competing reaction networks

### Demand consumption block

**What it is:** couples a reagent or beneficial species to an opposing demand term.

**Why it matters:** this is the core shape of chlorine residual problems and many limited-reagent control tasks.

**Principles to capture:**

- consumption should depend on both available reagent and demand
- disturbance slugs should propagate through this interaction
- the system can show strong gain changes under different loading conditions

**Ignored realism:**

- full chlorine chemistry or breakpoint chlorination detail unless required

### Temperature correction

**What it is:** modifies reaction or transfer rates based on temperature.

**Why it matters:** often enough to make summer and winter control meaningfully different.

### Mass-transfer / aeration proxy

**What it is:** approximates transfer of a species between phases or from aeration to liquid.

**Why it matters:** necessary for BSM1-like dissolved oxygen dynamics.

**Principles to capture:**

- diminishing returns near saturation
- actuator-dependent transfer intensity
- time-scale differences relative to biological consumption

### Reduced biological reaction block

**What it is:** a simplified kinetics block sufficient for benchmark behavior without full scientific-model fidelity.

**Why it matters:** allows reuse beyond chemistry-only examples.

**Principles to capture:**

- load dependence
- saturation effects
- plausible timescales
- interactions between limiting species where they affect control difficulty

## Realistic principles to capture

- reduced-order nonlinearities should preserve the main control consequences even if they omit many side reactions
- timescale separation matters: transport, sensing, actuation, and chemistry should not all collapse into the same step response
- state updates should remain numerically stable under expected benchmark conditions

## Realism we are intentionally ignoring

- regulatory-grade kinetic fidelity
- full equilibrium chemistry packages
- exhaustive species tracking where only a small subset drives control difficulty

## Recommended first-generation priority

1. first-order decay
2. demand consumption
3. temperature correction
4. aeration / transfer proxy
5. one reduced biological block sufficient for BSM1-family reuse
