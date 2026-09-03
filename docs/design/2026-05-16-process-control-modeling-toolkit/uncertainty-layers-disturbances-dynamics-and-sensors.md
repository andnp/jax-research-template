# Uncertainty Layers: Disturbances, Dynamics Noise, and Sensor Noise

## Purpose

This appendix defines how the toolkit should represent uncertainty without collapsing every source of mismatch into a single generic noise term.

The central idea is simple:

- some uncertainty belongs in **external disturbances**
- some belongs in the **latent plant dynamics**
- some belongs in **sensors and analyzers**

Keeping those layers separate makes the simulator easier to reason about and keeps benchmark difficulty honest.

## Why this distinction matters

If all uncertainty is modeled as sensor noise, the latent process becomes unrealistically clean and the control problem can collapse into filtering.

If all uncertainty is modeled as process noise, measurements become unrealistically perfect and observability questions disappear.

If every module adds arbitrary jitter everywhere, the simulator becomes noisy without becoming realistic.

The right approach is to model uncertainty at the seam where it physically or operationally belongs.

## Three uncertainty layers

### 1. External disturbances

External disturbances are structured exogenous changes that affect the process from outside the local module state.

Examples:

- storm inflow or throughput surge
- demand slug in raw water
- sudden load change
- setpoint change
- scheduled maintenance mode
- actuator derating event

**Characteristics:**

- often have clear timing
- often have finite duration or scenario structure
- may or may not be directly observed
- should usually be reproducible under a given scenario seed

**Best use:** scenario design and robustness benchmarking

### 2. Dynamics noise / latent-process uncertainty

Dynamics noise represents small uncertainty inside the latent plant evolution itself.

Examples:

- unresolved influent variation between scenario events
- small mixing imperfections
- effective transport variability
- slight variation in decay or transfer coefficients
- slow drift in process gain

**Characteristics:**

- should be bounded
- should usually be slower and more correlated than raw sensor noise
- should preserve physical plausibility and state constraints
- should not duplicate explicitly modeled disturbances

**Best use:** preventing reduced-order models from becoming unrealistically deterministic

### 3. Sensor / analyzer noise

Sensor noise represents imperfect observation of the latent plant.

Examples:

- analyzer noise
- sample-and-hold effects
- low-pass lag
- calibration bias
- drift
- dropout or frozen readings
- quantization or clipping

**Characteristics:**

- acts on measurements, not latent plant state
- can differ across instruments even when they observe related variables
- often has a faster or more jagged character than dynamics noise

**Best use:** observability realism and realistic controller information constraints

## Recommended modeling patterns

### Good patterns for dynamics noise

#### Bounded mean-zero perturbation

Use for:

- small unresolved variability around nominal behavior

Examples:

- small perturbation to an effective reaction coefficient
- bounded perturbation to influent load

#### Clipped random walk or slow drift

Use for:

- fouling-like degradation
- gradual gain changes
- raw-water-quality regime drift

Examples:

- slowly drifting chlorine-demand coefficient
- slowly drifting oxygen-transfer efficiency

#### Colored or correlated noise

Use for:

- smooth wandering processes that should not change abruptly each step

Examples:

- effective residence-time variability
- unresolved loading variability around a diurnal profile

### Good patterns for sensor noise

- white noise plus clipping for simple sensors
- sample-and-hold analyzers with update period and lag
- bias plus slow calibration drift
- explicit dropout or frozen-reading states
- quality flags that indicate stale or estimated measurements

### Good patterns for disturbances

- deterministic scheduled events
- stochastic event libraries with seed control
- amplitude and duration sampling within bounded families
- operating-mode schedules and maintenance states

## Recommended defaults

For first-generation toolkit modules:

- prefer **structured disturbances** over large generic process noise
- prefer **small bounded dynamics noise** over unconstrained IID perturbations
- prefer **explicit sensor contracts** over ad hoc observation jitter in env wrappers

As a rule of thumb:

- disturbances should carry the big story
- dynamics noise should add modest latent uncertainty
- sensor noise should define what the controller actually sees

## Anti-patterns

Avoid the following:

- injecting IID white noise independently into every state every step
- using noise in ways that violate positivity or basic physical bounds
- double-counting the same uncertainty in both the plant and the measurement path without intent
- encoding scenario-level disturbances as unexplained microscopic jitter
- adding measurement noise directly in an observation builder instead of in the sensor contract

## Example: chlorine residual benchmark

### Disturbances

- demand slug events
- diurnal flow shifts
- raw-water-quality step changes

### Dynamics noise

- small bounded variation in demand-consumption gain
- slow drift in latent raw-water-quality baseline
- small correlated transport variability affecting effective residence time

### Sensor noise

- analyzer lag and sample period on residual probes
- flow sensor bias or intermittent dropout
- noisy raw-water-quality surrogate

This split produces a simulator that is both more realistic and easier to interpret during controller debugging.

## Example: wastewater / BSM1-inspired benchmark

### Disturbances

- storm or surge influent profile
- sludge-load event
- setpoint or operating-mode schedule

### Dynamics noise

- slow drift in reduced biological-rate coefficients
- bounded variability in settling or transfer effectiveness
- unresolved short-timescale loading variation

### Sensor noise

- DO analyzer lag and noise
- blanket-height proxy drift
- intermittent analyzer availability on nitrogen surrogates

This structure would usually be more realistic and more benchmark-useful than leaving the biological core deterministic and putting all uncertainty into the measurement layer.

## Practical guidance for implementation

When adding uncertainty to a module, ask three questions:

1. Is this effect an externally meaningful event or regime change?
   - If yes, model it as a disturbance or scenario event.
2. Is this effect genuine latent unpredictability in the reduced-order process?
   - If yes, model it as bounded dynamics noise or parameter drift.
3. Is this effect only about what the controller observes?
   - If yes, model it in the sensor.

If the answer is “a little of each,” split the mechanism across layers intentionally rather than hiding the ambiguity inside one generic noise knob.

## Recommendation

The toolkit should treat uncertainty as a first-class design axis.

In practice that means:

- every benchmark should state its disturbance family
- every module that uses latent uncertainty should declare where that uncertainty enters
- every sensor should own its own measurement imperfections

That structure will make simulators more realistic, benchmark contracts more reviewable, and controller failures much easier to diagnose.