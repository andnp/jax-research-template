# Benchmark Catalog

## Purpose

This appendix turns the toolkit architecture into concrete benchmark candidates. The goal is not to predict every future simulator, but to show the kinds of process-control tasks the module set can express cleanly.

## How to read this catalog

Each benchmark entry lists:

- the process idea
- the main unit-operation modules
- likely sensors
- likely actuators
- typical disturbances or operating modes
- why the benchmark is hard for traditional control
- what kind of extra observability or structure could give RL a legitimate edge

## Benchmark summary table

| Benchmark | Main units | Core sensors | Core actuators | Typical baseline | Legitimate RL edge |
|-----------|------------|--------------|----------------|------------------|--------------------|
| Chlorine residual control | Mixer, contact basin / plug-flow reach | outlet residual, flow, raw-water-quality proxy | dose pump | PI/PID + optional flow pacing | intermediate residuals, residence-time proxy, richer quality surrogates |
| pH neutralization | mixed tank, optional delay volume | pH, flow, reagent flow | acid/base dose pump | PI/PID | upstream quality surrogate, multi-sensor context, mode-aware dosing |
| Flow-paced chemical dosing | source, mixer, transport delay | flow, downstream quality proxy | dose pump | feed-forward + PI trim | better load surrogates, sparse downstream sensing |
| Equalization tank control | source, tank, outlet restriction | level, inflow, outflow | pump or valve | PI/PID or rule-based | storm/event context, richer level-rate and occupancy signals |
| Blend / ratio control | multiple sources, mixer, splitter | composition or surrogate analyzers, flow | valves, pumps | ratio control + PI trim | multi-stream quality inference, delayed downstream quality sensing |
| Contact-time / detention management | basin train, level-sensitive storage | level, flow, sparse quality analyzers | pump, valve, dose pump | rule-based + PI | detention-time estimation, internal analyzers |
| Dissolved oxygen / aeration control | aeration basin, transfer block | DO, flow, load surrogate | blower / aeration actuator | PI/PID | richer loading context, multi-zone sensing, energy-aware control |
| Clarifier / recycle control | clarifier, recycle loop, upstream tank | blanket-height proxy, effluent quality, flow | recycle pump, underflow / waste pump | PI/PID + heuristics | soft sensors, fault-aware sensing, multi-objective control |
| Recirculation / bypass control | splitters, mixers, recycle loop | flow, quality, inventory proxies | valves, pumps | ratio or rule-based control | delayed quality sensing plus state inference |
| H2S scrubber control | gas-liquid contactor, recirculation sump, oxidation / pH block | inlet/outlet H2S, pH, ORP, level, flow | bleach pump, caustic pump, makeup-flow valve/pump | multiloop PI + heuristics | richer chemistry context, disturbance anticipation, coupled MIMO control |
| Wastewater reduced-order benchmark | anoxic/aerobic tanks, clarifier, recycle links | DO, NH, NO, flow, blanket proxy | blowers, recycle pumps, waste pump | multiple PI loops | observation shaping, soft sensors, richer actuator/sensor realism |

## Detailed benchmark entries

### Chlorine residual control

**Process shape:** reagent injection into a delay-dominated transport and consumption system.

**Main modules:**

- influent or raw-water-quality source
- mixer or injection seam
- contact basin or plug-flow reach
- demand-consumption block
- disturbance scheduler

**Typical sensors:**

- outlet residual analyzer
- flow sensor
- raw-water-quality surrogate
- optional intermediate residual analyzers

**Typical actuators:**

- dose pump

**Why PID struggles:**

- long delay
- varying demand
- changing residence time under flow swings

**Good RL edge:**

- one or two sparse intermediate analyzers
- detention-time estimate
- richer upstream demand indicators

### pH neutralization

**Process shape:** reagent addition to maintain pH in a mixed volume under changing influent chemistry.

**Main modules:**

- source profile
- mixed tank
- optional holding volume / delay unit
- simplified neutralization chemistry block

**Typical sensors:**

- pH sensor
- flow sensor
- optional conductivity or quality surrogate

**Typical actuators:**

- acid or caustic dose pump

**Why PID struggles:**

- nonlinear gain near setpoint
- varying influent chemistry
- sensor lag and sampling

**Good RL edge:**

- extra upstream quality measurements
- mode-aware behavior for strong-vs-weak influent changes

### Flow-paced chemical dosing

**Process shape:** additive feed paced against changing flow and process load.

**Main modules:**

- variable source
- mixer
- simple transport / delay block

**Typical sensors:**

- flow
- downstream quality or concentration proxy
- upstream load surrogate

**Typical actuators:**

- dose pump

**Why PID struggles:**

- much of the problem is feed-forward rather than pure feedback
- downstream response is delayed and noisy

**Good RL edge:**

- correlated but imperfect upstream load features
- richer temporal context

### Equalization tank control

**Process shape:** absorb variable inflow while maintaining downstream stability and avoiding inventory constraint violations.

**Main modules:**

- influent profile or storm source
- storage tank
- outlet valve or pump

**Typical sensors:**

- level
- inflow
- outflow

**Typical actuators:**

- pump or control valve

**Why PID struggles:**

- competing objectives: level management vs downstream smoothness
- storms and bursts create forecasting value

**Good RL edge:**

- scenario-aware context
- occupancy / rate-of-change features
- optional weather or schedule signals when realistic

### Blend / ratio control

**Process shape:** combine two or more streams to hit a target property.

**Main modules:**

- multiple source blocks
- mixer
- splitter or recycle if needed

**Typical sensors:**

- flow sensors on each feed
- blended quality analyzer or surrogate

**Typical actuators:**

- valves
- feed pumps

**Why PID struggles:**

- interacting manipulated variables
- composition uncertainty in the incoming streams

**Good RL edge:**

- upstream quality context on each source
- delayed downstream analyzer plus historical traces

### Dissolved oxygen / aeration control

**Process shape:** maintain DO in a tank or basin under changing loading and transfer conditions.

**Main modules:**

- tank or basin
- aeration transfer block
- reduced biological demand block
- load source

**Typical sensors:**

- DO analyzer
- flow
- optional ammonia / nitrate surrogate

**Typical actuators:**

- blower or aeration command

**Why PID struggles:**

- changing process gain with loading and saturation
- energy-performance trade-off
- multi-timescale dynamics

**Good RL edge:**

- richer loading context
- multiple zone analyzers
- explicit energy-aware objective

### Clarifier / recycle control

**Process shape:** control recycle or underflow behavior to balance blanket stability, effluent quality, and throughput.

**Main modules:**

- clarifier or separation block
- upstream tanks
- recycle loop

**Typical sensors:**

- blanket-height proxy or soft sensor
- effluent quality analyzers
- recycle flow

**Typical actuators:**

- recycle pump
- sludge / waste pump

**Why PID struggles:**

- latent settling state is hard to observe directly
- constraints and delays matter heavily

**Good RL edge:**

- soft-sensor estimates
- richer fault-aware observation package

### H2S scrubber control

**Process shape:** absorb and chemically neutralize or oxidize sulfur species in a gas-liquid contactor with recirculating chemistry and makeup management.

**Main modules:**

- gas influent source with H2S loading and gas-flow variation
- gas-liquid contactor or scrubber tower block
- recirculation sump or mixed liquid inventory
- oxidation / bleach-consumption block
- pH / alkalinity or caustic-neutralization block
- makeup / bleed or liquid-refresh logic

**Typical sensors:**

- inlet and outlet H2S analyzers
- pH
- ORP or oxidation surrogate
- recirculation or makeup flow
- sump level

**Typical actuators:**

- bleach dose pump
- caustic dose pump
- makeup-flow valve or pump

**Typical baseline:**

- multiple PI loops plus rules or operator heuristics

**Why PID struggles:**

- multiple manipulated variables with coupled chemistry
- load swings in gas flow and H2S concentration
- competing cost, chemistry, and compliance objectives
- meaningful latent state in scrubbing liquid condition

**Good RL edge:**

- richer chemistry-state context
- better disturbance anticipation from inlet-gas sensing
- explicit multivariable coordination across bleach, caustic, and makeup flow

### Wastewater reduced-order benchmark

**Process shape:** a compact wastewater train with anoxic/aerobic behavior, recycles, and clarification, but intentionally reduced for control benchmarking rather than scientific completeness.

**Main modules:**

- influent source
- anoxic tank
- aerobic tank
- clarifier
- recycle links
- reduced biological reaction blocks

**Typical sensors:**

- dissolved oxygen
- ammonia or ammonium
- nitrate / nitrite surrogate
- flow
- blanket-height proxy

**Typical actuators:**

- blowers
- recycle pumps
- waste pump

**Typical baseline:**

- multiple PI loops with selectors or supervisory rules

**Good RL edge:**

- richer sensing, especially sparse upstream/downstream analyzers and soft sensors
- actuator-realization awareness
- mode-aware control during storms, surges, or equipment degradation

## Could the toolkit build a “better BSM1”?

Yes, but it is important to define **better**.

### Better for control benchmarking

The toolkit should be able to produce a **better control benchmark than legacy BSM1-style envs** in several ways:

- more realistic sensor behavior: lag, sample-and-hold, bias, drift, dropout, soft sensors
- more realistic actuation: saturation, rate limits, stiction, degraded capacity, realized-vs-requested output
- clearer observation contracts: PID, heuristic, and RL views can be separated intentionally
- better disturbance structure: storm events, load surges, equipment faults, operating-mode schedules
- cleaner modular experimentation: swap a clarifier model, add an extra analyzer, test a new recycle topology without rewriting the whole env

In that sense, yes — the toolkit can absolutely support a **better BSM1-style benchmark**.

### Better in scientific or process-model fidelity

That is also possible, but it is **not automatic**.

If the goal is a more scientifically faithful wastewater model, the toolkit would need deliberate investment in:

- richer biological reaction blocks
- stronger stream/state schemas for wastewater composition
- more careful settling and separation dynamics
- better sensor definitions for analyzers used in wastewater practice
- calibration and validation discipline against accepted reference behavior

The toolkit architecture enables that work, but it does not magically provide it.

### Best framing

The strongest near-term target is probably:

> a **reduced-order wastewater benchmark inspired by BSM1** that is better instrumented, better actuated, more scenario-rich, and easier to reason about for RL/control research.

That would be a major win even before chasing “strictly more scientifically accurate BSM1.”

## Recommended first benchmark portfolio

If the toolkit needs a first coherent portfolio rather than an open-ended wish list, a strong sequence would be:

1. chlorine residual control
2. equalization tank control
3. pH neutralization
4. dissolved oxygen / aeration control
5. reduced-order wastewater benchmark inspired by BSM1

That portfolio exercises most of the important module families without forcing the toolkit into premature full-plant fidelity.

## Strategic next expansions

If the toolkit is prioritized based on immediate practical value rather than architectural completeness alone, two module families stand out as especially important next expansions:

### 1. Membrane / fouling benchmarks

Why prioritize them:

- highly relevant to industrially realistic optimization problems
- common across water and advanced-treatment settings
- rich in latent degradation state, competing objectives, and operational tradeoffs
- strong internal-testbed value for controller and RL development

Representative additions:

- membrane separation unit
- fouling accumulation / recovery block
- TMP or pressure-drop model
- cleaning / derating scenario support

Representative benchmark questions:

- how aggressively should the controller push flux under uncertain fouling growth?
- when should the system trade off quality, throughput, and cleaning burden?
- what extra sensing most improves predictive control under degradation?

### 2. Coagulation / flocculation benchmarks

Why prioritize them:

- also directly relevant to industrially realistic optimization problems
- common in drinking-water treatment and pretreatment contexts
- highly sensitive to upstream water-quality variation
- strong fit for observability asymmetry and feed-forward control comparisons

Representative additions:

- coagulation dose-response block
- floc formation / capture block
- settling or downstream turbidity-impact coupling
- raw-water quality and solids-sensitive disturbance scenarios

Representative benchmark questions:

- how should dosing respond to changing raw-water quality and solids loading?
- when does richer upstream sensing outperform purely reactive control?
- what reduced-order floc or capture dynamics are sufficient for realistic optimization behavior?

### Practical prioritization takeaway

If the toolkit must choose only a small number of next module families after the initial benchmark set, membrane/fouling and coagulation/flocculation should move to the front of the line. They are not only architecturally interesting; they also map to industrially relevant optimization problems, which increases the value as a simulator testbed substantially.

An H2S scrubber benchmark is also a strong candidate for this same reason. It likely depends on a somewhat different module cluster — especially gas-liquid contact and reduced oxidation / pH chemistry — but it is strategically attractive because it maps to a realistic multi-loop optimization problem and would exercise genuinely multivariable dosing and inventory control.
