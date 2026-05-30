# H2S Scrubber Benchmark Concept

## Purpose

This appendix sketches a reduced-order H2S scrubber benchmark for the process-control toolkit.

The benchmark is motivated by real operational relevance: bleach dosing, caustic dosing, and makeup-flow control in an H2S scrubbing system create a multi-input process with meaningful chemistry, inventory memory, and cost/compliance tradeoffs.

The goal is **not** to build a full chemical-engineering scrubber design package. The goal is to define a scientifically plausible, control-relevant benchmark that can later support higher-fidelity modules incrementally.

## Why this is a strong benchmark candidate

An H2S scrubber benchmark would be valuable because it combines:

- multiple manipulated variables
- coupled chemistry and hydraulic effects
- latent liquid-condition state
- compliance-like output objectives
- feed and load disturbances
- realistic sensor and actuator imperfections

That makes it a good fit for both classical control comparisons and RL benchmarking.

## Core process picture

At reduced order, the process can be thought of as:

1. a gas stream arrives with variable gas flow and H2S loading
2. the gas contacts a recirculating scrubbing liquid
3. the scrubbing liquid condition determines how effectively H2S is absorbed and neutralized / oxidized
4. bleach, caustic, and makeup flow maintain the chemistry and inventory of the scrubbing liquid
5. outlet H2S, liquid chemistry state, and operating costs determine performance

This is already enough to create a rich benchmark, even without detailed packed-column or full speciation modeling.

## Candidate module breakdown

### 1. Gas influent source

**Role:** generates gas flow and H2S loading entering the scrubber.

**Likely outputs:**

- gas flow rate
- inlet H2S concentration or load
- optional temperature or humidity surrogate if ever needed

**Disturbances:**

- step changes in H2S load
- gas-flow surges
- diurnal or schedule-driven loading variation

### 2. Gas-liquid contactor / scrubber tower

**Role:** represents transfer and removal of H2S from the gas stream into or through the scrubbing liquid.

**Reduced-order responsibility:**

- map gas-side loading and liquid-side condition to outlet H2S
- capture contact efficiency or transfer limitation
- expose sensitivity to gas flow and liquid condition

**What it does not need initially:**

- full packed-tower hydrodynamics
- geometry-resolved spatial fields
- high-fidelity two-phase transport

### 3. Recirculating sump / liquid inventory

**Role:** stores the liquid-phase condition that gives the system memory.

**Likely latent state:**

- liquid volume or level
- oxidant availability surrogate
- alkalinity / pH-driving state
- accumulated reaction burden or spent-chemistry state

**Why it matters:**

This is the main latent state that prevents the benchmark from becoming purely instantaneous.

### 4. Bleach-consumption / oxidation block

**Role:** captures how oxidant availability changes with H2S loading and bleach dosing.

**Reduced-order responsibility:**

- bleach dosing raises oxidant availability
- sulfur load consumes available oxidant
- performance degrades when oxidant inventory is low

**Why it matters:**

This is a key coupling point between load disturbances and one of the main manipulated variables.

### 5. Caustic / pH block

**Role:** captures the part of chemistry state maintained by caustic dosing and relevant to scrubber effectiveness.

**Reduced-order responsibility:**

- caustic raises or restores the pH/alkalinity-like state
- scrubber effectiveness depends in part on that state
- chemistry condition drifts under load and dilution

**Why it matters:**

This lets the benchmark represent the real intuition that bleach and caustic are not interchangeable knobs.

### 6. Makeup / refresh logic

**Role:** represents liquid refresh, dilution, and inventory maintenance.

**Reduced-order responsibility:**

- makeup changes inventory and dilution state
- refresh can recover liquid condition or prevent drift
- makeup may trade off chemistry stability against operating cost or waste

**Why it matters:**

This creates a genuine third manipulated variable with a different control role from reagent dosing.

## Candidate sensors

A realistic benchmark could use some subset of:

- inlet H2S analyzer
- outlet H2S analyzer
- pH
- ORP or oxidation surrogate
- recirculation flow
- makeup flow
- sump level
- optional conductivity or liquid-quality surrogate

## Candidate actuators

The obvious actuators are:

- bleach dose pump
- caustic dose pump
- makeup-flow valve or pump

Optional future actuators might include:

- bleed flow
- recirculation flow setpoint
- gas bypass or loading redistribution logic if the process supports it

## Baseline control shape

A realistic baseline would probably be:

- multiple PI loops
- rule-based overrides or selectors
- operator-style heuristics for coordinating bleach, caustic, and makeup

That makes the benchmark particularly interesting because a learned controller is not competing against a toy single-loop baseline.

## Why this process is hard for classical control

Several factors make the system nontrivial:

- multiple manipulated variables with coupled effects
- changing gas flow and H2S load
- latent liquid-condition state
- competing objectives: outlet quality, reagent cost, chemistry stability, and inventory management
- imperfect sensing and analyzer lag

This is exactly the kind of problem where a richer state representation and better coordination could matter.

## Good RL edge opportunities

The benchmark supports realistic ways to give RL an information advantage without cheating.

Examples:

- better use of inlet-gas sensing and trend context
- ORP or oxidation-state context that a simpler baseline might ignore
- richer history features for changing load regimes
- coordinated action selection across bleach, caustic, and makeup flow

These are believable advantages because they come from instrumentation and coordination, not from leaking hidden future information.

## Reduced-order chemistry ideas

For a first benchmark version, a good reduced-order chemistry model probably does **not** need full detailed speciation.

Instead, it can represent:

- incoming sulfur burden
- an oxidant-availability state
- a pH/alkalinity-effectiveness state
- a liquid-refresh / dilution effect
- a scrubber effectiveness function that maps those states to outlet H2S removal

That is often enough to create realistic control interactions.

## Fidelity ladder

### L1: benchmark-simple scrubber

**Characteristics:**

- gas load in
- simple removal-efficiency relationship
- bleach and caustic each maintain one latent liquid-condition state
- makeup manages dilution / inventory
- basic sensor lag and actuator limits

**Use:**

- prove out architecture
- compare multiloop PID vs RL / coordinated control

### L2: physically informed reduced-order scrubber

**Characteristics:**

- stronger gas-liquid transfer dependence
- better coupling between oxidant state, pH-like state, and removal effectiveness
- more realistic inventory / refresh effects
- more realistic analyzer contracts

**Use:**

- testbed for controller ideas closer to plant behavior
- better study of sensing and coordination value

### L3: higher-fidelity research scrubber

**Characteristics:**

- more explicit chemistry structure
- more careful transfer assumptions
- richer liquid-condition state and byproduct memory
- tighter calibration and validation expectations

**Use:**

- deeper scientific or plant-specific studies

## Recommended first implementation scope

For the first benchmark version, I would aim for:

- one gas influent source
- one reduced gas-liquid contactor block
- one recirculating sump state
- one oxidant-consumption block
- one caustic / pH block
- one makeup / dilution block
- realistic analyzer lag and actuator saturation

That is enough to create a useful and strategically relevant benchmark without overcommitting to chemistry fidelity too early.

## Why this should be a priority

This benchmark is not only architecturally interesting. It provides a realistic and structurally distinct multi-loop control problem, which makes it disproportionately valuable as a simulator testbed.

That means even a reduced-order version could pay off quickly by supporting:

- safer controller iteration
- control-policy comparisons
- observability studies
- ablations on sensing, disturbances, and coordination logic

## Open questions

- What is the minimum reduced-order chemistry state that preserves the right coupling between bleach, caustic, makeup, and outlet H2S?
- Which sensor set is realistic enough to represent current plant instrumentation without overfitting the benchmark to one site?
- Should makeup flow be modeled primarily as inventory management, chemistry refresh, or both?
- What disturbances are most representative of real scrubber operation: gas-load spikes, flow swings, chemistry drift, sensor faults, or some combination?