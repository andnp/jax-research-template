# Control Baselines and Fair Comparisons for Online RL

## Purpose

This document explains how to build credible control baselines for evaluating a continual, online reinforcement-learning controller. It assumes the reader has taken an introductory control course but may not have practical experience identifying, tuning, commissioning, or comparing industrial controllers.

The central recommendation is simple:

> Compare complete control workflows, not isolated control equations.

A proportional-integral controller is more than `Kp * error + Ki * integral` when used in a plant. It normally comes with filtering, actuator limits, anti-windup, feed-forward signals, mode switching, alarms, and operator procedures. Likewise, model predictive control is more than an optimizer. Someone must collect data, identify a model, validate it, design an estimator, choose constraints and horizons, and commission the controller safely.

The process-control toolkit should represent those surrounding workflows. Otherwise an RL method may appear better only because its baseline was incomplete.

The shorter implementation checklist is in [Control-baseline implementation tasks](control-baseline-implementation-tasks.md).

## 1. What is a baseline?

A baseline is a reference method used to interpret the performance of a proposed method. Different baselines answer different questions.

- A fixed action answers whether any feedback control is useful.
- A basic PI controller answers whether the problem requires more than simple feedback.
- An engineered PID control structure answers whether RL adds value beyond common practice.
- Model predictive control answers whether RL adds value beyond model-based prediction and constraint handling.
- Adaptive MPC answers whether RL adds value beyond a classical controller that also learns online.
- An oracle controller estimates what could be achieved with unavailable simulator knowledge.

No single baseline answers all of these questions. A good experiment uses a **ladder of baselines** so that performance improvements can be attributed to a specific capability.

## 2. A small control-theory refresher

### 2.1 Process, manipulated variable, disturbance, and controlled variable

Consider a chlorine process:

- the **process** is the contact basin and chlorine chemistry
- the **manipulated variable** is chlorine dose
- the **controlled variable** is outlet chlorine residual
- the **setpoint** is the desired residual
- a **measured disturbance** might be inlet flow
- an **unmeasured disturbance** might be an unexpected change in chlorine demand

The controller observes measurements and selects manipulated variables. It does not normally observe every physical state in the process.

### 2.2 Feedback

Feedback reacts to the difference between a desired value and a measured value:

```text
error = setpoint - measurement
```

Feedback is valuable because it corrects unknown disturbances and model errors. Its limitation is that it usually reacts only after a disturbance has affected the measurement.

### 2.3 Feed-forward

Feed-forward acts on a measured disturbance before the controlled variable moves. In chlorine dosing, a common idea is to increase dose when water flow increases, even before the outlet analyzer detects a residual change.

Feed-forward needs a relationship between the measured disturbance and the required control action. Feedback trim is often added to correct errors in that relationship.

### 2.4 Dynamics, gain, and dead time

Three basic properties matter when designing a controller:

- **gain:** how much the output eventually changes for a given input change
- **time constant:** how quickly the output approaches its new value
- **dead time:** how long it takes before the output begins responding

A first-order-plus-dead-time model is often written conceptually as:

```text
output_change(s) / input_change(s) = gain * exp(-dead_time * s) / (time_constant * s + 1)
```

This simple model is widely useful even when the underlying process has many states.

### 2.5 Constraints

Real control actions and process variables have limits:

- pumps have minimum and maximum output
- valves have finite travel speed
- tanks cannot exceed their physical volume
- emissions and effluent quality have compliance limits
- rapid actuator motion may damage equipment

A credible comparison must give every controller the same physical limits and must report violations separately from average performance.

## 3. The information-tier rule

Simulator knowledge creates a major fairness problem. The code knows every equation, state, and parameter, but a real controller would not.

Classify each controller into one of three information tiers.

### 3.1 Plant-realistic tier

The controller is designed from information plausibly available during normal engineering and commissioning:

- process diagrams and equipment limits
- controller input/output tags
- historical operating data
- safe step or bump tests
- normal sensor measurements
- engineering knowledge of signs and rough timescales

This tier best represents deployable standard practice.

### 3.2 Data-driven tier

The controller receives the same measurements and data that an online RL system receives, but it may fit an explicit model or observer from those data.

Examples include:

- system identification followed by MPC
- recursive least squares followed by adaptive MPC
- a learned soft sensor followed by PID
- data-enabled predictive control

This tier provides the fairest advanced comparison with online RL.

### 3.3 Oracle tier

The controller may use simulator equations, exact parameters, or latent state that would not be measured in a plant.

Oracle methods are useful as ceilings and debugging tools. They should never be presented as deployable baselines or mixed into the headline plant-realistic comparison.

### 3.4 Required metadata

Every experimental result should identify:

- information tier
- available measurements
- historical data provided
- commissioning excitation allowed
- simulator knowledge used
- controller tuning budget
- online updates allowed

## 4. Baseline ladder

The recommended ladder moves from minimal reference policies to strong adaptive control.

### 4.1 Sanity-reference policies

These are not serious competitors, but they catch experimental mistakes.

Include:

- zero or minimum action
- nominal fixed action
- random bounded action
- simple open-loop schedule
- a do-nothing maintenance policy where relevant

If a sophisticated controller cannot reliably beat these, the benchmark, reward, or implementation likely has a problem.

### 4.2 Production-quality PI and PID

#### Proportional control

Proportional control moves the actuator in proportion to error:

```text
u = bias + Kp * error
```

It responds immediately but often leaves a steady offset when a constant disturbance is present.

#### Integral control

Integral action accumulates error:

```text
integral_next = integral + error * dt
u = bias + Kp * error + Ki * integral
```

Integral action removes steady offset, but it can make a loop oscillatory or slow to recover after saturation.

#### Derivative control

Derivative action responds to how quickly the measurement or error changes. It can add damping but is sensitive to noise. Practical derivative terms therefore use filtering and are often applied to the measurement rather than the setpoint error.

#### Features needed for a credible implementation

A serious PID baseline should support:

- output minimum and maximum
- asymmetric rate limits
- anti-windup
- derivative filtering
- setpoint weighting
- manual and automatic modes
- bumpless transfer between modes
- external-reset or output tracking when another controller overrides it
- setpoint ramps
- measurement filtering
- direct and reverse action
- useful diagnostic outputs

#### Anti-windup

Suppose a valve is fully open but positive error remains. A naive integral term continues growing even though the actuator cannot respond. When the error finally reverses, the large stored integral delays recovery. This is called **integral windup**.

Common anti-windup methods are:

- **conditional integration:** stop integrating when saturated in the direction that worsens saturation
- **back-calculation:** drive the integral state toward the value consistent with the saturated output

#### Tuning variants

Use at least two defensible tunings:

- a conservative IMC/lambda-style tuning emphasizing robustness
- a performance-oriented tuning optimized under the same evaluation objective

The purpose is not to search indefinitely for the best PID. It is to show that the conclusion is not caused by one poor tuning.

### 4.3 Common structures built around PID

Industrial control performance often comes from structure rather than a more complicated individual controller.

#### Feed-forward plus feedback trim

Feed-forward estimates the action needed for a measured load. A PI loop adds a smaller correction based on residual error.

Example:

```text
chlorine_dose = flow_paced_dose(flow) + PI(outlet_residual_error)
```

This should be a primary baseline for chemical dosing.

#### Cascade control

An outer controller adjusts the setpoint of a faster inner controller. For example, an ammonia controller may adjust a dissolved-oxygen setpoint, while a fast DO loop commands aeration.

Cascade control works well when the inner loop is substantially faster and rejects disturbances before they reach the outer variable.

#### Ratio control

Ratio control maintains one flow or dose in proportion to another. It is common for chemical-to-water ratios and blending applications.

#### Override or selector control

Several controllers calculate requests, and a high or low selector chooses the safest request. An ordinary level controller might command a pump until a high-pressure constraint controller takes over.

This is important when comparing RL with safety-constrained control. The baseline should not be denied normal protective logic.

#### Split-range control

One controller output drives different actuators over different ranges, such as acid dosing below a neutral point and caustic dosing above it.

#### Decoupled multiloop PI

For interacting variables, several PI loops can be combined with a static decoupling matrix. This is simpler than MPC and is common enough to be a meaningful MIMO baseline.

#### Gain scheduling

Different controller gains are used in different operating regions. pH control is a classic example because process gain changes sharply near the equivalence region.

Gain schedules may be built from commissioning tests rather than simulator equations. Interpolation and bumpless switching between regions should be tested.

#### Smith predictor and dead-time compensation

A Smith predictor uses a model to estimate the process response before delayed feedback arrives. It can greatly improve delay-dominated loops when the model is adequate.

It is an important chlorine baseline because otherwise RL may be credited merely for using history to handle delay.

### 4.4 Rule-based and heuristic baselines

Rules are often the real incumbent for scheduling and maintenance problems.

Examples include:

- backwash when TMP exceeds a threshold
- use a minimum dwell time between washes
- reduce membrane flux at high feed solids
- increase equalization discharge before a forecast storm
- hold extra chlorine residual during high-demand periods
- limit H2S scrubber load when ORP is low

Rules should be based on realistic available measurements and tuned with the same development data as other controllers.

### 4.5 Model predictive control

MPC repeatedly predicts future process behavior and solves an optimization problem.

At each control step it:

1. estimates the current process state or disturbance
2. predicts future outputs for candidate action sequences
3. scores tracking error, actuator movement, and economic cost
4. enforces constraints
5. applies only the first action
6. repeats when the next measurement arrives

This repeated replanning is called a **receding horizon**.

#### Why MPC is a strong baseline

MPC naturally handles:

- coupled inputs and outputs
- known delays
- actuator and process constraints
- measured-disturbance forecasts
- trade-offs between tracking and actuator movement
- multistep planning

These overlap heavily with claimed RL advantages.

#### Core MPC design choices

- **prediction horizon:** how far into the future the controller predicts
- **control horizon:** how many future action changes it chooses independently
- **output weights:** relative importance of controlled variables
- **move weights:** penalty on changing actuators
- **constraints:** hard or soft bounds on actions, rates, and outputs
- **slack cost:** penalty paid when a soft constraint is violated
- **state estimator:** method used to infer unmeasured state
- **disturbance model:** method used to remove steady prediction offset

#### Offset-free control

Even a good identified model will be imperfect. Without correction, MPC may settle with a steady error. Offset-free MPC augments the model with a constant or slowly changing disturbance estimate so predictions align with measurements.

This is an essential feature, not optional polish.

#### Economic MPC

Tracking MPC follows setpoints. Economic MPC directly optimizes operating value, such as:

- chemical cost
- energy
- throughput
- equipment wear
- compliance risk
- maintenance cost

If RL is trained on an economic reward, compare it with economic MPC where feasible. Comparing economic RL only with tracking PID can exaggerate the value of RL.

## 5. System identification before MPC

### 5.1 Why pretend the simulator model is unknown?

That is the correct experimental choice. In a plant, exact equations and parameters are rarely available in a form suitable for controller design. Even when a mechanistic model exists, parameters drift and some dynamics are omitted.

The baseline should therefore learn its prediction model from data unless it is explicitly labeled oracle MPC.

### 5.2 The identification problem

System identification estimates a dynamic relationship between recorded inputs and outputs.

Conceptually:

```text
past inputs + past outputs + current inputs -> predicted future outputs
```

The goal is not necessarily to recover the true physical parameters. The goal is to obtain a model accurate enough for prediction and control over the intended operating envelope.

### 5.3 Start under a safe incumbent controller

Identification data should normally be collected while a safe controller is operating. Open-loop tests may be unsafe or impractical for integrating, unstable, delayed, or tightly constrained processes.

Closed-loop data create statistical complications because controller actions are correlated with disturbances and noise. Identification methods and validation must account for this rather than assuming inputs are independent random signals.

### 5.4 Excitation

A model cannot learn dynamics that the data never reveal. **Excitation** means varying inputs enough to expose their effects.

Useful excitation signals include:

- bounded steps
- ramps
- pseudo-random binary sequences
- generalized binary noise with minimum dwell time
- multisines covering selected frequencies
- coordinated multivariable signals designed to avoid unsafe combined moves

Excitation should respect:

- actuator bounds and rate limits
- process safety and product-quality constraints
- normal operating regions
- maximum acceptable commissioning cost
- settling time and dead time

Record both commanded and realized actuator positions. A pump rate limit or valve stiction can make them different.

### 5.5 Operating-region coverage

A model identified around one operating point may fail elsewhere.

Collect data across:

- low, nominal, and high throughput
- different seasonal or temperature conditions
- important chemistry regions
- different inventory or fouling states
- normal equipment configurations

For strongly nonlinear processes, fit local models and schedule between them, or use a nonlinear model class.

### 5.6 Dataset splits

Use time-separated data:

1. **identification set:** fits model parameters
2. **validation set:** selects order and model family
3. **challenge set:** evaluates the final controller and remains unseen during model selection

Randomly shuffling individual timesteps leaks temporal information and should not be the default.

### 5.7 Candidate model families

#### First-order-plus-dead-time models

These are easy to understand and tune. They are strong choices for individual SISO loops.

#### Finite impulse response models

An FIR model represents the output as a weighted sum of past inputs. It is flexible and linear but may require many coefficients for slow processes.

#### ARX models

An autoregressive model with exogenous inputs uses past outputs and inputs. It is computationally convenient and works well as an initial data-driven baseline.

#### ARMAX and output-error models

These model noise and process dynamics differently. They may fit closed-loop and noisy measurements better but are more involved to estimate.

#### State-space models

A state-space model uses a compact latent state:

```text
x_next = A x + B u
y = C x + D u
```

Subspace identification can estimate multivariable state-space models directly from input/output data. This is a practical choice for MIMO MPC.

#### Local and LPV models

Several local linear models can represent a nonlinear process. Their parameters are selected or interpolated using a scheduling variable such as flow, pH region, temperature, or fouling state.

#### Nonlinear black-box models

Neural state-space, recurrent, Hammerstein, or Wiener models can be considered later. They add modeling capacity but also complicate stability, uncertainty, and fair attribution. Start with simpler models and add nonlinearity only when validation shows it is needed.

### 5.8 Model-order and delay selection

A model with too little state misses important dynamics. A model with too much state may fit noise and behave badly outside the dataset.

Select order and delay using:

- validation prediction error
- free-run simulation error
- information criteria where applicable
- residual analysis
- physical plausibility
- closed-loop performance on development scenarios

Do not tune model order on the final challenge scenarios.

### 5.9 Model validation

One-step prediction accuracy is not enough. A model may predict the next sample using the latest measured output but drift badly when rolled forward inside MPC.

Validate:

- one-step prediction
- multistep prediction
- free-run simulation
- step-response gain, direction, delay, and timescale
- stability
- behavior at operating-envelope boundaries
- residual autocorrelation
- correlation between residuals and past inputs
- uncertainty across repeated datasets
- eventual controller performance

Residuals should resemble unpredictable noise. Structure remaining in residuals often means the model missed dynamics or disturbances.

### 5.10 Identification artifacts

Store:

- raw data reference and scenario version
- preprocessing and resampling steps
- signal names and units
- fitted model and parameters
- candidate-model comparison
- validation plots and metrics
- valid operating envelope
- known model failures
- random seed and software version

The identified model should be reproducible from the stored dataset rather than appearing as an unexplained constant in a controller config.

## 6. State estimation and soft sensing

### 6.1 Why an estimator is needed

MPC models often contain states that are not directly measured. An estimator combines predictions with new measurements to infer them.

This is the classical counterpart to a recurrent RL policy learning an internal representation.

### 6.2 Kalman filtering

A Kalman filter estimates the state of a linear system under Gaussian process and measurement noise assumptions. It alternates between:

1. predicting the next state
2. correcting the prediction using measurement error

The process-noise and measurement-noise assumptions control how much the estimator trusts model predictions versus sensors.

### 6.3 Nonlinear estimators

For nonlinear processes, consider:

- extended Kalman filters
- unscented Kalman filters
- moving-horizon estimation

Moving-horizon estimation solves an optimization problem over a recent data window and can enforce state constraints, but costs more computation.

### 6.4 Soft sensors

A soft sensor estimates a hard-to-measure variable from easier measurements. Examples include estimating ammonia load, sludge blanket condition, or membrane health.

A fair comparison might include:

- PID using raw measurements
- PID using the soft sensor
- MPC using the same soft sensor or estimator
- RL using the same raw measurement set

This reveals whether an apparent RL advantage comes from better state inference or better control decisions.

## 7. Adaptive classical control

An online-RL controller learns during operation. Comparing it only against fixed controllers confounds two questions:

1. Is adaptation useful?
2. Is reinforcement learning the best adaptation method?

Adaptive classical controllers help separate them.

### 7.1 Recursive least squares

Recursive least squares updates a linear model as each new data point arrives. A forgetting factor gives recent data more influence, allowing estimates to track drift.

Important concerns include:

- insufficient excitation
- parameter drift caused by noise
- abrupt estimator changes
- maintaining a stable controller while the model changes
- resetting or bounding implausible estimates

### 7.2 Self-tuning regulators

A self-tuning regulator repeatedly estimates a model and derives controller gains from it. It is a direct classical analogue to online adaptation.

### 7.3 Adaptive MPC

A practical adaptive-MPC workflow may:

1. keep control under the current validated model
2. update a candidate model online
3. validate candidate predictions over a rolling window
4. accept the model only when safety and quality gates pass
5. update the MPC gradually or during a controlled handoff

This gated workflow is more credible than replacing the prediction model after every noisy sample.

### 7.4 Model banks and change detection

Instead of continuously changing parameters, maintain several models for known regimes. A detector estimates which regime is active and switches or blends controllers.

This is a strong baseline for:

- seasonal wastewater behavior
- distinct pH titration regions
- clean versus fouled membranes
- different throughput bands
- equipment configurations

### 7.5 Recommended flagship adaptive baseline

The strongest broadly useful comparator is:

> Safe incumbent control, bounded commissioning excitation, online recursive identification, gated model updates, constrained offset-free MPC, and fallback on validation failure.

If RL beats this baseline, the result says more than beating a fixed PID.

## 8. Robust control and uncertainty

### 8.1 Nominal versus robust control

A nominal controller is designed for one estimated model. A robust controller accounts for a family of plausible models or disturbances.

Robustness can be introduced through:

- conservative PID tuning
- gain and phase margin checks
- model ensembles
- constraint tightening
- scenario MPC
- tube MPC
- min-max optimization
- robust loop shaping or `H-infinity` design

### 8.2 Where robust baselines matter most

Use robust baselines selectively where uncertainty is a central claimed RL advantage:

- chlorine demand and dead-time uncertainty
- membrane fouling-rate uncertainty
- H2S load and chemistry uncertainty
- seasonal biological-rate uncertainty

It is unnecessary to build sophisticated robust control for every benchmark initially.

## 9. Data-enabled predictive control

Data-enabled predictive control uses recorded trajectories directly to construct future behavior, rather than first fitting a conventional state-space model. DeePC-style methods are the best-known example.

Why it is relevant:

- it uses input/output data
- it supports multistep prediction and constraints
- it is closer to model-free learning than conventional MPC
- its trajectory dataset can be updated over time

Why it is not the primary baseline:

- it is less established in routine industrial practice than PID or conventional MPC
- noise handling and dataset size require care
- computational requirements can grow with stored trajectories

Treat it as a research comparator after the conventional identification-and-MPC path works.

## 10. Benchmark-specific baseline stacks

### 10.1 Chlorine residual

Minimum credible stack:

1. nominal fixed dose
2. residual PI
3. flow-paced feed-forward plus residual PI trim
4. Smith predictor or IMC dead-time controller
5. identified constrained MPC with flow as a measured disturbance
6. online delay/gain identification plus adaptive MPC
7. oracle MPC ceiling

Key system-identification concerns:

- flow-dependent residence time
- changing chlorine demand
- sparse delayed residual measurements
- distinct fast and slow demand dynamics

### 10.2 pH neutralization

Minimum credible stack:

1. fixed reagent bias
2. PI around one operating region
3. gain-scheduled PI
4. acid/base feed-forward from estimated equivalents plus PI trim
5. local linear or LPV MPC
6. online local-model adaptation
7. nonlinear oracle MPC ceiling

Key concern: a controller trained or identified only near one point may fail when the titration-curve slope changes.

### 10.3 Equalization tank

Minimum credible stack:

1. fixed outlet flow
2. level PI with high/low overrides
3. rule-based smoothing policy
4. constrained MPC using an inflow forecast
5. adaptive forecast/model plus MPC
6. perfect-forecast oracle MPC

Important metrics include overflow, starvation, downstream flow variation, pump travel, and safety interventions—not only level error.

### 10.4 Membrane fouling

Minimum credible stack:

1. fixed flux with periodic backwash
2. TMP or flux PID plus threshold backwash
3. condition-based rules with dwell time
4. estimated-fouling-state economic MPC
5. online fouling-model adaptation plus MPC
6. oracle health-state scheduler

This benchmark mixes continuous actions with discrete maintenance. Rule baselines are especially important.

### 10.5 H2S scrubber

Minimum credible stack:

1. independent pH, ORP, and level PI loops
2. gas-load feed-forward plus PI trim
3. cascade and override control
4. static-decoupled multiloop PI
5. identified MIMO economic MPC
6. online model-bank or adaptive MPC
7. oracle chemistry-state MPC

The baseline should receive realistic analyzer delays and should not observe latent oxidant or alkalinity state unless a soft sensor estimates it.

### 10.6 BSM1 and related wastewater plants

Minimum credible stack:

1. local dissolved-oxygen PI loops
2. ammonia-based supervisory DO setpoint control
3. nitrate-based internal-recycle control
4. rule and override logic for clarifier and recycle constraints
5. identified reduced-order MIMO MPC
6. estimator plus adaptive or scheduled MPC across temperature regimes
7. full-model oracle MPC ceiling

Do not require the identified MPC model to reproduce every ASM state. A smaller input/output model may be better for control.

### 10.7 Sludge blanket and clarification

Minimum credible stack:

1. fixed underflow ratio
2. blanket-height PI
3. PI with effluent-quality override
4. constrained MPC with a blanket-state estimator
5. adaptive settler model plus MPC

### 10.8 Chemical dosing and precipitation

Minimum credible stack:

1. fixed dose
2. flow-paced ratio control
3. downstream-quality PI trim
4. gain-scheduled dose-response controller
5. economic MPC
6. online dose-response identification

## 11. Fair commissioning and tuning budgets

### 11.1 Compare workflows, not just online execution

Controller preparation has costs:

- historical data requirements
- commissioning excitation
- expert engineering time
- offline computation
- number of simulator or plant interactions
- unsafe or off-spec production during tuning

Report these costs. An RL controller that performs well after enormous hidden tuning may be less attractive than a simpler controller with modest commissioning requirements.

### 11.2 Same data principle

Data-driven MPC and RL should receive the same initial historical dataset unless the experiment intentionally studies different data requirements.

If RL begins with no data and learns online, include an MPC or adaptive-control variant that also begins from limited knowledge.

### 11.3 Same authority principle

Give controllers the same:

- action bounds
- action-rate limits
- control interval
- sensor set and cadence
- safety overrides
- fallback controller
- constraint definitions

### 11.4 Same tuning budget principle

Use comparable automated search budgets for:

- PID gains
- MPC weights and horizons
- RL hyperparameters

Also report human-designed structure. A carefully engineered RL reward should not be treated as free while PID feed-forward design is counted as unfair expert knowledge.

### 11.5 Development versus final evaluation

Separate:

- controller development scenarios
- identification and tuning scenarios
- final locked evaluation scenarios

Do not repeatedly inspect final challenge performance while modifying the controller. That turns the challenge set into another tuning set.

## 12. Evaluation metrics

### 12.1 Tracking and regulation

- integral absolute error
- integral squared error
- maximum error
- settling time
- recovery time after disturbance
- time within target band

### 12.2 Constraint and safety performance

- number and duration of constraint violations
- worst violation magnitude
- fallback activations
- safety-supervisor interventions
- overflow, washout, starvation, or breakthrough events

### 12.3 Actuator behavior

- total actuator travel
- action reversals
- time saturated
- rate-limit activity
- starts and stops
- maintenance-trigger count

### 12.4 Economic performance

- reagent use
- energy use
- throughput
- off-spec material
- maintenance and cleaning cost
- equipment-wear proxy

### 12.5 Learning and adaptation

- performance before first update
- cumulative cost during learning
- time to beat the incumbent controller
- adaptation time after a regime change
- retained performance when an old regime returns
- exploratory actuator movement
- unsafe-learning cost

### 12.6 Model and estimator quality

- one-step and multistep prediction error
- simulation error
- residual whiteness diagnostics
- state-estimate error using simulator truth only for evaluation
- uncertainty calibration
- prediction degradation outside the identification region

## 13. Recommended experiment matrix

For each gold benchmark, run at least:

| Family | Controller |
| --- | --- |
| Sanity | nominal fixed action |
| Standard | production-quality PID structure |
| Standard-plus | feed-forward, cascade, scheduling, or rules appropriate to the process |
| Predictive | identified offset-free constrained MPC |
| Adaptive | online identification plus gated adaptive MPC |
| Proposed | online RL with the same measurements and authority |
| Ceiling | oracle predictive controller or oracle state estimator |

Evaluate each controller under:

- nominal operation
- an unseen disturbance realization
- parameter drift
- sensor degradation
- actuator degradation
- operating-point change
- return to a previously seen regime

Not every matrix cell needs to be run during early development. The final gold-benchmark report should make omissions explicit.

## 14. How to interpret outcomes

### RL beats PID but not MPC

The likely value is prediction, constraint handling, or multivariable coordination rather than reinforcement learning itself. This may still support a product, but the product case must address why RL is easier or more robust than MPC.

### RL beats fixed MPC but not adaptive MPC

The advantage likely comes from adaptation. Investigate whether a simpler online model update captures most of the benefit.

### RL beats adaptive MPC only with privileged state

The result is primarily a sensing or state-estimation result. Remove privileged signals or compare against a classical observer.

### RL wins in reward but violates more constraints

The reward or safety specification is inadequate. Physical safety metrics take precedence over aggregate reward.

### Oracle MPC is only slightly better than PID

The benchmark may not contain much exploitable control advantage. It may still be useful as a basic test, but it is unlikely to demonstrate the product's value.

### Every method performs poorly

Possible causes include:

- insufficient actuator authority
- impossible objectives
- poor observations
- numerical artifacts
- incorrect benchmark equations
- unstable or unidentifiable operating regime

Use oracle state and control experiments to diagnose the benchmark before changing learning algorithms.

## 15. Recommended implementation order

### Stage 1: Industrial PI/PID foundation

Implement a reusable controller with anti-windup, filtering, rate limits, mode handling, external tracking, and diagnostics. Add feed-forward, cascade, ratio, override, and gain-schedule composition.

### Stage 2: Commissioning and identification

Implement safe excitation generators, time-series dataset artifacts, preprocessing, ARX identification, first-order-plus-dead-time fitting, subspace state-space identification, and validation reports.

### Stage 3: Constrained MPC

Implement offset-free linear MPC with action, rate, and output constraints. Begin with pH, chlorine, and equalization.

### Stage 4: Adaptive comparison

Implement recursive identification, model validation gates, gradual controller handoff, and fallback behavior.

### Stage 5: Specialized baselines

Add economic MPC, model banks, condition-based maintenance, robust MPC, and data-enabled predictive control only where the benchmark requires them.

## 16. Final recommendation

The baseline program should answer increasingly difficult questions:

1. Can the algorithm beat doing nothing?
2. Can it beat a properly tuned feedback loop?
3. Can it beat the standard industrial structure for this process?
4. Can it beat a controller that identifies and predicts the process from the same data?
5. Can it beat a classical controller that also adapts online?
6. How close does it come to the oracle ceiling?
7. What did it cost and risk while learning?

This ladder will make both positive and negative results useful. A failed RL experiment can reveal that the advantage came from feed-forward information, state estimation, model adaptation, or constraint handling. A successful experiment will be much more convincing because it survived comparison with the methods a knowledgeable controls engineer would actually consider.
