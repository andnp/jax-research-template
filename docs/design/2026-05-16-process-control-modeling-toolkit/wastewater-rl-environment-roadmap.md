# Wastewater RL Environment Roadmap

## Purpose

This appendix maps the ASM/BSM model family into focused RL environments with low-dimensional action spaces (1–6D). Each environment isolates a single control loop or tightly coupled set of control knobs from a larger plant layout.

The guiding principle is **"inspired by BSM, enriched by the toolkit."** Standard BSM specifications define the biological models and plant layouts, but the toolkit adds sensor realism, disturbance richness, and instrumentation flexibility that the original benchmarks lack.

## Background: The ASM/BSM Family

### Activated Sludge Models (ASM)

Biokinetic models of increasing complexity from the IWA:

| Model | Year | State Variables | Key Additions |
|-------|------|----------------|---------------|
| **ASM1** | 1987 | 13 | Carbon oxidation, nitrification, denitrification. The foundational model. |
| **ASM2** | 1995 | 19 | Biological phosphorus removal (PAOs), fermentation |
| **ASM2d** | 1999 | 19 | Extends ASM2 with denitrifying PAOs |
| **ASM3** | 2000 | 13 | Rethinks ASM1 — adds internal cell storage, removes direct hydrolysis-to-growth coupling |

### Benchmark Simulation Models (BSM)

Standardized plant layouts + influent patterns + evaluation criteria:

| Benchmark | Bio Model | Plant Layout | Key Focus |
|-----------|----------|--------------|-----------|
| **BSM1** | ASM1 | 5-reactor AS + 10-layer Takács clarifier | DO and nitrate control |
| **BSM1-LT** | ASM1 | Same as BSM1 | Long-term (1-year) evaluation with temperature variation |
| **BSM2** | ASM1 | Full WWTP: primary clarifier → AS → secondary clarifier → anaerobic digester → dewatering → reject water | Whole-plant control |
| **BSM2-P** | ASM2d | BSM2 layout | Biological + chemical phosphorus removal |
| **BSM-MBR** | ASM1/ASM2d | MBR replacing secondary clarifier | Membrane fouling, dual-purpose aeration |

### Other Key Models

- **Takács settler**: 10-layer 1D flux model for secondary clarification. Standard in BSM1/BSM2.
- **ADM1**: Anaerobic Digestion Model No. 1. 32 state variables, models biogas production. Used in BSM2.
- **Temperature models**: Arrhenius temperature dependence for ASM kinetics. Used in BSM1-LT.

---

## Environment Roadmap

### Tier 1: BSM1-Derived (we have the biological core)

#### BSM1 DO Control ✅ *Implemented*
- **Action**: 2D `[kla_34, kla_5]`
- **Observation**: 6D — effluent NH₄/NO₃, DO in R3/R5, NO₃ in R2, normalised flow
- **What's interesting**: The classic BSM1 aeration control problem. Trade off energy (aeration) against effluent quality (ammonia).
- **Enrichment opportunities**: Add sensor noise/lag to DO and NH₄ measurements; add occasional sensor dropout; vary influent composition (not just flow).

#### BSM1 Nitrate Recycle Control ✅ *Implemented*
- **Action**: 2D `[Q_a, Q_rs]` (internal recycle ratio, return sludge ratio)
- **Observation**: 6D — NO₃ R2, effluent NH₄/NO₃, flow, Q_a, Q_rs
- **What's interesting**: Different energy-quality trade-off from DO control. Recycle rates affect denitrification efficiency.
- **Implementation**: Q_a via DosingSystem (reverse-acting PI on NO₃ R2), Q_rs via RampLimitedActuator.

#### BSM1 Combined Control ✅ *Implemented*
- **Action**: 4D `[kla_34, kla_5, Q_a, Q_rs]`
- **Observation**: 11D — DO R3/R5, effluent NH₄/NO₃, NO₃ R2, flow, influent NH₄, aeration power, dq/dt, Q_a, Q_rs
- **What's interesting**: Full BSM1 control problem with coupled aeration and pumping trade-offs.
- **Implementation**: Composes DosingSystem loops (kla, Q_a) + RampLimitedActuator (Q_rs). Asymmetric blower ramp + VFD startup delay.

### Tier 2: New Unit Models Required

#### Sludge Blanket Control (Takács Clarifier)
- **New model needed**: `units/takacs_settler.py` — 10-layer 1D settler with double-exponential flux
- **Action**: 1–2D `[Q_rs, Q_w]` (return sludge rate, waste sludge rate)
- **Observation**: ~4D — sludge blanket height, effluent TSS, underflow TSS, inflow
- **What's interesting**: Integrating dynamics with lag. Failure mode = solids washout during storms. SRT management.
- **Enrichment**: Add blanket-height sensor with noise and sampling delay; storm-event disturbances.

#### Chemical Phosphorus Dosing ✅ *Implemented*
- **Module**: `chemistry/precipitation.py` — FeCl₃ precipitation with Monod-type saturation
- **Action**: 1D `[FeCl₃_dose]` (mg-Fe/L)
- **Observation**: 5D — effluent PO₄, flow, dose, influent PO₄, dq/dt
- **What's interesting**: Simple process with clear cost-quality trade-off. Monod-like kinetics.
- **Implementation**: Diurnal influent PO₄ via SinusoidalChannel, DosingSystem loop (PO₄ sensor → PI → pump), instantaneous precipitation. Reward penalises effluent violation² + normalised chemical cost.

### Tier 3: BSM2 Components (each a standalone environment)

#### Primary Clarifier
- **Action**: 1D `[Q_w_primary]` (primary sludge wastage rate)
- **Observation**: ~3D — influent TSS, effluent TSS, sludge inventory
- **What's interesting**: Controls how much organic load goes to activated sludge vs. digester. Simple but affects whole-plant energy balance.

#### Anaerobic Digester (ADM1-inspired)
- **New model needed**: `units/adm1.py` — simplified ADM1 or reduced-order digester model
- **Action**: 1–2D `[Q_feed, T_digester]` (sludge feed rate, temperature setpoint)
- **Observation**: ~4D — biogas flow rate, digester pH, VFA concentration, gas composition
- **What's interesting**: Biogas optimisation. ADM1 has 32 internal states but the control interface is small. Failure = digester upset (pH crash, foaming).

#### Reject Water Management
- **Action**: 1–2D `[Q_reject, return_timing]`
- **Observation**: ~4D — reject water NH₄ load, main-plant NH₄, time-of-day, influent flow
- **What's interesting**: High-NH₄ reject water from dewatering must be returned to plant headworks. Timing matters — returning during low-load periods avoids ammonia spikes.

#### Dewatering Control
- **Action**: 1–2D `[polymer_dose, belt_speed]`
- **Observation**: ~3D — cake dryness, filtrate quality, throughput
- **What's interesting**: Optimise polymer use vs. cake quality for disposal.

### Tier 4: Phosphorus Track (ASM2d)

#### Bio-P Anaerobic Zone
- **New model needed**: `units/asm2d.py` — ASM2d with PAO kinetics
- **Action**: 1–2D `[Q_recycle, carbon_dose]`
- **Observation**: ~4D — VFA, PO₄, NO₃ intrusion indicator, flow
- **What's interesting**: Keep the anaerobic zone truly anaerobic with enough VFA for PAOs. NO₃ intrusion from internal recycle destroys bio-P performance.

#### Combined N+P Control
- **Action**: 3–4D `[kla, Q_a, FeCl₃_dose, carbon_dose]`
- **Observation**: ~6D
- **What's interesting**: Hardest variant — simultaneous nitrogen and phosphorus removal with conflicting requirements (carbon for denitrification vs. carbon for PAOs).

### Tier 5: MBR Track

#### Membrane Fouling Control
- **New model needed**: `units/membrane.py` — fouling/permeability dynamics
- **Action**: 2–3D `[air_scour_rate, flux_setpoint, backwash_interval]`
- **Observation**: ~4D — TMP, permeability, MLSS, permeate flow
- **What's interesting**: Classic degradation-control problem. Prevent irreversible fouling while maximising throughput.

---

## Realistic Instrumentation and Observation Design

### The "wrong sensor" principle

Standard BSM specifications give the controller idealised, noise-free readings of exactly the process variables that appear in the kinetic equations. A real plant looks nothing like this. Real plants have dozens of sensors already installed — flow meters, turbidity probes, conductivity cells, temperature transmitters, ORP electrodes, energy meters — most of which a controls engineer would dismiss as "the wrong sensor" for a given control loop.

RL's strength is not that it gets *better* sensors. It is that it can exploit *existing* sensors that a traditional controller would ignore because they don't directly measure the controlled variable.

**Example:** A chlorine dosing process is controlled to maintain a target chlorine residual. The relevant chemistry depends on organic demand, not turbidity. A controls engineer might reasonably decline to use an inlet turbidity sensor because "turbidity doesn't affect chlorine demand directly." But turbidity correlates with storm runoff events, which *do* cause large demand swings. An RL agent that receives the turbidity reading can learn this correlation and begin adjusting dose *before* the residual analyzer detects a deviation — something a PID loop on residual alone cannot do.

This is not about giving RL an unfair advantage. It is about modelling the realistic sensor landscape of a modern plant and letting RL demonstrate whether it can extract value from the signals that are already there.

### Design principle: model what exists, don't curate for RL

When building environments, the guiding question is not "what sensors would help RL?" but rather "what sensors already exist at a typical plant running this process?" If a sensor exists in the real world, include it in the observation space. If a traditional controller wouldn't use it, that's precisely what makes the benchmark interesting.

Concretely:
- **Include sensors that correlate with, but don't directly measure, the controlled variable.** Turbidity for chlorine demand. Conductivity for pH buffering capacity. Temperature for biological kinetics. ORP for redox state in anaerobic zones.
- **Include sensors at positions a PID loop wouldn't wire to.** Upstream quality sensors, intermediate reactor probes, energy meters on blowers. These are commonly installed for monitoring or regulatory reporting but ignored by single-loop controllers.
- **Model realistic sensor imperfections.** Noise, sampling delay, drift between calibrations, occasional dropout. These are not enrichments — they are realism. Standard BSM's noise-free sensors are the unrealistic case.

### Sensor realism (matching real-world conditions)
- **Measurement noise**: Gaussian noise on analyzer outputs (NH₄, NO₃, DO, TSS, PO₄). Already implemented for DO and pH sensors.
- **Sampling delay**: Analyzers like NH₄ and PO₄ have 5–15 minute measurement cycles. Model as zero-order hold with configurable sample interval.
- **Sensor drift**: Slow bias drift between calibrations. Already supported in the sensor framework.
- **Sensor dropout**: Occasional missing readings (return last-known-good).

### Richer influent dynamics (matching real-world conditions)
- **Composition variation**: BSM1 fixes influent composition and varies only flow. Real influent has correlated flow–composition dynamics (storms dilute concentrations; industrial discharges spike specific components).
- **Weekend/seasonal patterns**: BSM1-LT adds these. Straightforward extension of the diurnal source model.
- **Storm events**: Short, high-intensity flow spikes with diluted concentrations and potential hydraulic overload.

### Realistic actuator constraints
- **Ramp-rate limits**: Already implemented via DosePumpState. Apply to blowers, pumps, and valves.
- **Discrete actuator states**: Some actuators (e.g., backwash valves) are on/off rather than continuous.
- **Actuator wear costs**: Penalise excessive switching or high ramp rates in the reward to encourage smoother, equipment-friendly control.

### Observation profiles
Each environment should support at least two observation profiles:
1. **Baseline**: The sensors a well-tuned PID or rule-based controller would traditionally use for this process. Matches standard BSM instrumentation where applicable.
2. **Full-plant**: All sensors that would realistically be installed and transmitting data at a modern plant — including the ones a controls engineer might not wire into the control loop. This is not a privileged view; it is the realistic view.

---

## Implementation Status

| Environment | Status | Module | Notes |
|-------------|--------|--------|-------|
| Chlorine residual control | ✅ Done | `benchmarks/chlorine.py` | Single contact basin, 1D action |
| Two-stage chlorine | ✅ Done | `benchmarks/chlorine_two_stage.py` | Two basins in series |
| pH neutralisation | ✅ Done | `benchmarks/ph_neutralization.py` | CSTR with titration |
| Equalization tank | ✅ Done | `benchmarks/equalization_tank.py` | Level control |
| BSM1 DO control (13-var ASM1) | ✅ Done | `benchmarks/bsm1.py` | 5 reactors, perfect settler, 2D action |
| BSM1 nitrate recycle | ✅ Done | `benchmarks/bsm1_recycle.py` | 2D action [Q_a, Q_rs] ratios |
| BSM1 combined | ✅ Done | `benchmarks/bsm1_combined.py` | 4D action [kla_34, kla_5, Q_a, Q_rs], 11D obs |
| H₂S scrubber | ✅ Done | `benchmarks/h2s_scrubber.py` | 3-loop supervisory control, 3D action |
| Takács clarifier | ✅ Done | `units/takacs_settler.py` | 10-layer 1D settler model |
| Sludge blanket control | ✅ Done | `benchmarks/sludge_blanket.py` | 1D action [Q_u], 4D obs |
| Chemical P dosing | ✅ Done | `benchmarks/chem_p_dosing.py` | FeCl₃ precipitation, 1D action |
| BSM1-LT (seasonal) | ✅ Done | `benchmarks/bsm1_lt.py` | Arrhenius kinetics, 10D obs |
| Primary clarifier | ✅ Done | `benchmarks/primary_clarifier.py` | Gravity settling, 1D action [Q_w] |
| Dewatering control | ✅ Done | `benchmarks/dewatering.py` | Belt press, 2D action [polymer, speed] |
| BSM1 + Takács settler | ✅ Done | `benchmarks/bsm1_takacs.py` | Realistic sludge dynamics, 3D action [kla_34, kla_5, Q_w] |
| Reject water management | ✅ Done | `benchmarks/reject_water.py` | Timing-dependent NH₄ return, 2D action |
| Membrane fouling control | ✅ Done | `benchmarks/membrane_fouling.py` | TMP, backwash, air scour, 3D action |
| Anaerobic digester | ✅ Done | `benchmarks/anaerobic_digester.py` | Reduced-order ADM1, 2D action [Q_feed, T] |
| Drinking water train | ✅ Done | `benchmarks/drinking_water_train.py` | Coag→membrane→Cl₂ cascade, 3D action |
| Bio-P (ASM2d) | ✅ Done | `benchmarks/bio_p.py` | PAO kinetics, 2D action [Q_recycle, carbon] |
| Combined N+P | ✅ Done | `benchmarks/combined_np.py` | ASM2d + FeCl₃, 4D action — hardest variant |

### New unit models (this batch)

| Unit model | Module | Notes |
|-----------|--------|-------|
| Primary clarifier | `units/primary_clarifier.py` | Monod SLR-dependent removal, sludge inventory |
| Dewatering (belt press) | `units/dewatering.py` | Polymer dose-response, belt speed trade-off |
| Membrane filtration | `units/membrane.py` | Resistance-in-series, reversible + irreversible fouling |
| Coagulation chemistry | `chemistry/coagulation.py` | Peaked dose-response with re-stabilisation |
| Anaerobic digester (ADM1) | `units/anaerobic_digester.py` | 8-state reduced model, gas transfer, pH |
| ASM2d (Bio-P extension) | `units/asm2d.py` | 17-state, PAO kinetics (PHA/PP storage/release) |

---

## Recommended Build Order

~~All planned items are now implemented.~~

Completed build order:
1. ~~**BSM1 nitrate recycle**~~ ✅
2. ~~**Takács settler** → **sludge blanket control**~~ ✅
3. ~~**Chemical P dosing**~~ ✅
4. ~~**Anaerobic digester**~~ ✅
5. ~~**BSM1-LT dynamics**~~ ✅
6. ~~**ASM2d** → **bio-P environments**~~ ✅
7. ~~**MBR / membrane fouling**~~ ✅
