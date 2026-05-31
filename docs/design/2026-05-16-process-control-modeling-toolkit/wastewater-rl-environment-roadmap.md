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

#### BSM1 Nitrate Recycle Control
- **Action**: 1–2D `[Q_a]` or `[Q_a, Q_rs]` (internal recycle, return sludge)
- **Observation**: ~4D — NO₃ in R2, effluent NH₄, flow, pumping energy proxy
- **What's interesting**: Different energy-quality trade-off from DO control. Recycle rates affect denitrification efficiency.
- **Implementation**: Trivial extension of existing BSM1 — expose Q_a/Q_rs as actions instead of fixed ratios.

#### BSM1 Combined Control
- **Action**: 3–4D `[kla_34, kla_5, Q_a, Q_rs]`
- **Observation**: ~6D — union of DO and recycle observations
- **What's interesting**: Full BSM1 control problem with coupled aeration and pumping trade-offs.

### Tier 2: New Unit Models Required

#### Sludge Blanket Control (Takács Clarifier)
- **New model needed**: `units/takacs_settler.py` — 10-layer 1D settler with double-exponential flux
- **Action**: 1–2D `[Q_rs, Q_w]` (return sludge rate, waste sludge rate)
- **Observation**: ~4D — sludge blanket height, effluent TSS, underflow TSS, inflow
- **What's interesting**: Integrating dynamics with lag. Failure mode = solids washout during storms. SRT management.
- **Enrichment**: Add blanket-height sensor with noise and sampling delay; storm-event disturbances.

#### Chemical Phosphorus Dosing
- **New model needed**: `chemistry/precipitation.py` — metal-salt (FeCl₃) precipitation of phosphate
- **Action**: 1D `[FeCl₃_dose]`
- **Observation**: ~3D — influent PO₄, effluent PO₄, chemical cost proxy
- **What's interesting**: Simple process with clear cost-quality trade-off. Monod-like kinetics.
- **Enrichment**: Sensor delay on effluent PO₄; varying influent P load.

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
| BSM1 reduced (6-var) | ✅ Done | `benchmarks/bsm1_reduced.py` | 2 reactors, perfect settler |
| BSM1 full (13-var ASM1) | ✅ Done | `benchmarks/bsm1.py` | 5 reactors, perfect settler |
| BSM1 nitrate recycle | ✅ Done | `benchmarks/bsm1_recycle.py` | 2D action [Q_a, Q_rs] ratios |
| BSM1 combined | Planned | — | kla + Q_a + Q_rs |
| Takács clarifier | ✅ Done | `units/takacs_settler.py` | 10-layer 1D settler model |
| Sludge blanket control | ✅ Done | `benchmarks/sludge_blanket.py` | 1D action [Q_u], 4D obs |
| Chemical P dosing | Planned | — | New chemistry model |
| Anaerobic digester | Planned | — | ADM1-inspired |
| Membrane fouling | Planned | — | New unit model |

---

## Recommended Build Order

1. **BSM1 nitrate recycle** — minimal new code, just re-wire existing BSM1 benchmark
2. **Takács settler** → **sludge blanket control** — high-value new unit model, enables realistic BSM1
3. **Chemical P dosing** — simple standalone environment, 1D action
4. **Anaerobic digester** — complex model but clean control interface
5. **BSM1-LT dynamics** — temperature variation + seasonal influent for existing BSM1
6. **ASM2d** → **bio-P environments** — major new biological model
7. **MBR** — membrane dynamics, niche but interesting
