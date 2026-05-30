# Drinking Water Treatment Train Benchmark Concept

## Purpose

This appendix sketches a multi-stage drinking water treatment train benchmark that connects coagulation dosing, membrane filtration, and chlorine disinfection into a single process graph. This benchmark is motivated by the need to couple three control loops — coagulation dosing, membrane backwash, and chlorine dosing — that are individually tractable but become strategically interesting when their interactions are modeled together.

## Why this is a high-priority benchmark

This benchmark combines three control loops that are individually interesting but become strategically valuable when coupled:

- **coagulation dosing** affects downstream particle loading, which affects membrane fouling rate, which affects backwash frequency, which affects effective plant capacity
- **membrane filtration control** (backwash timing, backpulse volume) manages transmembrane pressure (TMP) while maintaining throughput
- **chlorine dosing** must handle variable demand driven in part by upstream treatment effectiveness

The cascade interaction between these stages is where the most interesting control and RL questions live. A coag underdose increases downstream fouling and chlorine demand. A coag overdose wastes reagent and may create downstream issues. The membrane stage acts as both a physical barrier and an information bottleneck — downstream chlorine demand depends on how well coag and filtration removed precursors.

## Core process picture

At reduced order, the treatment train can be thought of as:

```
Raw water → Coag injection → [Flocculation] → Membrane filtration → Chlorine injection → Contact basin → Treated water
     ↑              ↑                                    ↑                    ↑
  Influent      Coag dose                          Backwash/pulse       Chlorine dose
  source        actuator                            actuators            actuator
```

### Stage 1: Coagulation

Raw water with variable turbidity, NOM, and particle loading receives a coagulant dose. The coagulant destabilizes particles, forming floc that can be captured by downstream filtration.

**Key dynamics:**

- dose-response is nonlinear (underdose is ineffective, overdose can restabilize)
- effectiveness depends on raw-water quality (pH, temperature, NOM character)
- the result is not directly measurable in real time — it manifests as downstream filter performance and treated-water quality

### Stage 2: Membrane filtration

Flocculated water passes through membrane filters. Particles and floc are captured, but the membrane fouls progressively. Backwash cycles restore membrane permeability, but each cycle costs throughput and water.

**Key dynamics:**

- TMP rises between backwashes as fouling accumulates
- fouling rate depends on upstream coag effectiveness and influent loading
- backwash timing and backpulse volume trade off membrane health against throughput
- irreversible fouling accumulates slowly, requiring eventual chemical cleaning (CIP)

### Stage 3: Chlorine disinfection

Filtered water receives a chlorine dose and passes through a contact basin. The goal is to maintain target residual at the basin outlet.

**Key dynamics:**

- demand depends on remaining organic load (affected by upstream coag + filtration effectiveness)
- transport delay through the contact basin
- flow-dependent residence time
- demand variability driven by both raw-water quality and upstream treatment performance

## Candidate module breakdown

### Raw-water source

- variable turbidity, NOM, particle loading
- diurnal flow pattern
- seasonal raw-water-quality variation
- storm or upset events

### Coagulation injection module

- mixer / injection point
- coag dose-response block (particle destabilization model)
- residual coagulant tracking (optional)

### Flocculation block (optional)

- time-dependent floc formation
- could be simplified to a delay + transformation for L1

### Membrane filter module

- filtration with progressive fouling (TMP rise)
- backwash events (TMP reduction, throughput cost)
- backpulse volume effect on cleaning effectiveness
- particle/NOM rejection as a function of membrane condition
- irreversible fouling accumulation (slow drift)

### Post-filtration stream

- filtered water with reduced but nonzero particle/NOM load
- the composition here determines downstream chlorine demand

### Chlorine dosing and contact basin

- dose pump actuator
- contact basin (plug-flow or multi-compartment)
- demand-consumption block
- outlet residual as the controlled variable

## Candidate sensors

| Sensor | Location | What it tells the controller |
|--------|----------|------------------------------|
| Raw-water turbidity | Influent | Upstream loading indicator |
| Raw-water UV254 / NOM proxy | Influent | Demand and fouling predictor |
| Temperature | Influent | Rate modifier for all stages |
| pH | Influent / post-coag | Coag effectiveness context |
| Streaming current / zeta potential | Post-coag | Coag effectiveness indicator (rare but valuable) |
| TMP | Membrane | Fouling state indicator |
| Permeate flow | Membrane | Throughput indicator |
| Filtered-water turbidity | Post-membrane | Filtration effectiveness |
| Filtered-water UV254 | Post-membrane | Chlorine demand predictor |
| Outlet chlorine residual | Contact basin outlet | Primary chlorine control variable |
| Intermediate residual | Mid-basin | Early response indicator |
| Flow | Multiple points | Hydraulic state |

## Candidate actuators

| Actuator | Stage | What it controls |
|----------|-------|-----------------|
| Coag dose pump | Coagulation | Coagulant dose rate |
| Backwash trigger | Membrane | When to initiate backwash |
| Backpulse volume | Membrane | How aggressively to clean |
| Chlorine dose pump | Disinfection | Chlorine dose rate |

## Why this is hard for classical control

- **Three coupled control loops** with different timescales
- **Upstream-downstream interaction**: coag dose affects membrane fouling rate AND chlorine demand
- **Delayed feedback**: coag effectiveness is only observable through downstream consequences
- **Mixed objectives**: water quality, membrane health, reagent cost, throughput
- **Nonlinear dose-response** in coag and chlorine stages
- **Slow drift** from irreversible fouling and seasonal raw-water changes

## Good RL edge opportunities

- **Cross-stage coordination**: learning that adjusting coag dose reduces downstream fouling and chlorine demand simultaneously
- **Predictive backwash**: learning optimal backwash timing from TMP trajectory context rather than fixed-threshold rules
- **Demand anticipation**: using upstream quality surrogates (UV254, turbidity) to anticipate chlorine demand before it manifests
- **Multi-objective optimization**: balancing reagent cost, membrane life, and treated-water quality simultaneously
- **Seasonal adaptation**: adjusting strategy as raw-water quality shifts with seasons

## Observation profiles

### Minimal baseline (per-stage PI)

Each stage sees only its own local measurements:
- coag loop: flow, raw-water turbidity, maybe pH
- membrane loop: TMP, permeate flow
- chlorine loop: flow, outlet residual

### Realistic RL profile

Cross-stage information that a plant could plausibly provide:
- all baseline measurements
- raw-water UV254 / NOM proxy
- filtered-water turbidity and UV254
- intermediate chlorine residual
- TMP trend features
- temperature

### Rich instrumentation profile

Additional realistic but expensive sensors:
- streaming current for coag effectiveness
- multiple intermediate residual analyzers
- membrane fouling rate estimator (soft sensor)
- chlorine demand estimator (soft sensor)

## Fidelity ladder

### L1: benchmark-simple DWT train

- reduced-order coag dose-response (dose → particle removal fraction)
- simple TMP-rise fouling model with backwash reset
- standard chlorine demand-consumption
- basic sensor lag and actuator limits

### L2: physically informed DWT train

- dose-response dependent on raw-water quality
- fouling model sensitive to upstream treatment and loading
- demand-consumption coupled to upstream rejection performance
- realistic analyzer contracts and multi-rate timing

### L3: research-grade DWT train

- NOM-character-dependent coag and fouling models
- irreversible fouling accumulation
- temperature-dependent kinetics throughout
- richer membrane state and CIP modeling

## Recommended first implementation scope

Start at L1 with:

- one raw-water source with turbidity and NOM proxy variation
- one coag injection with simplified dose-response
- one membrane filter module with TMP-rise and backwash
- one chlorine contact basin with demand-consumption
- realistic analyzer lag and actuator saturation

This is enough to study the cross-stage interaction that makes this benchmark valuable.

## Relationship to existing benchmarks

This benchmark is a **composition** of pieces that also appear in standalone benchmarks:

- the chlorine stage reuses the standalone chlorine benchmark modules
- the membrane stage shares modules with the standalone membrane/fouling benchmark
- the coag stage shares modules with the standalone coag/floc benchmark

This is exactly the reuse pattern the toolkit is designed to enable.

## Open questions

- What is the minimum reduced-order coag dose-response model that preserves the right coupling to downstream stages?
- Should backwash be modeled as a discrete event (instantaneous TMP reset) or as a time-consuming process that affects throughput during execution?
- How much of the coag → membrane → chlorine interaction should be modeled through explicit composition tracking vs simplified transfer functions between stages?
- What disturbance scenarios are most representative of real drinking water treatment variation: seasonal, storm, source-switch, or some combination?
