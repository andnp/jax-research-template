# RL Advantage Patterns: When and Why RL Outperforms Classical Control

## Purpose

This appendix catalogs the specific process-control realism features that create legitimate opportunities for RL to outperform PID and other classical control strategies. These patterns should guide benchmark design: a benchmark is valuable to the extent that it exposes one or more of these features clearly enough to measure the RL advantage.

## Why this catalog matters

Demonstrating RL value in process control requires more than showing RL can track a setpoint. PID does that well in simple cases. The case for RL is built by showing advantage in situations where classical control has **structural** limitations — not just tuning limitations.

This catalog serves two purposes:

1. **benchmark design guide**: when designing a new benchmark, check which advantage patterns it exercises
2. **evidence framework**: when reporting results, cite which patterns explain the observed advantage

## Core advantage patterns

### 1. Long transport delay with anticipatory information

**Why PID struggles:** PID is fundamentally reactive. With long delays, it either oscillates or responds too slowly. Adding delay compensation (Smith predictor) helps but requires an accurate process model.

**How RL wins:** RL can learn to use upstream proxy measurements to anticipate downstream effects before they manifest. This is genuine anticipatory control, not just faster reaction.

**Benchmark examples:** chlorine contact basin (outlet residual lags dose changes by minutes), multi-stage treatment train (upstream quality changes propagate through stages).

**Realism features to include:** transport delay proportional to flow-dependent residence time, upstream quality surrogates, sparse intermediate sensors.

### 2. Nonlinear process gain

**Why PID struggles:** PID is tuned for a linear operating region. When the process gain changes substantially (e.g., pH near the equivalence point, coag near optimal dose, aeration near DO saturation), a single set of gains produces poor performance across the full operating range.

**How RL wins:** RL can learn region-aware strategies that effectively adapt controller behavior based on process state, without needing explicit gain scheduling.

**Benchmark examples:** pH neutralization, coag dose-response, aeration near saturation.

**Realism features to include:** operating-region-dependent gain, disturbances that push the process through different gain regions, realistic range of operating conditions.

### 3. Multi-rate information fusion

**Why PID struggles:** PID uses one feedback measurement at its sample rate. It cannot naturally exploit the information structure of having fast proxies, slow analyzers, and infrequent lab values for the same underlying variable.

**How RL wins:** RL can learn to weight multiple information sources appropriately — trusting fast but noisy proxies for responsiveness while using slow but accurate analyzers for bias correction.

**Benchmark examples:** any process with both online analyzers and lab measurements, processes with quality surrogates that move faster than the controlled variable.

**Realism features to include:** multi-rate sensor contracts, sample-and-hold analyzers, periodic lab values, quality surrogate sensors with imperfect correlation.

### 4. Coupled MIMO interactions

**Why PID struggles:** multiple independent PID loops often fight each other when manipulated variables interact. Decoupling compensation helps but requires accurate interaction models. MIMO classical control (MPC) handles this but requires a plant model.

**How RL wins:** RL can learn coordinated multi-actuator strategies directly from interaction experience.

**Benchmark examples:** H2S scrubber (bleach + caustic + makeup), DWT train (coag + backwash + chlorine), blend/ratio control, wastewater (DO + recycle).

**Realism features to include:** multiple actuators affecting shared process state, cross-coupling between control loops, disturbances that require coordinated response.

### 5. Structured disturbances with temporal patterns

**Why PID struggles:** PID is memoryless within a sample — it has no concept of diurnal patterns, recurring events, or temporal structure in disturbances. Feed-forward can help if the right measurement is available, but PID alone cannot exploit predictable patterns.

**How RL wins:** RL with recurrent or history-augmented architectures can learn temporal patterns in disturbances and pre-position control action before the disturbance arrives.

**Benchmark examples:** diurnal flow and demand patterns, recurring storm events, operator shift changes, seasonal raw-water quality cycles.

**Realism features to include:** diurnal profiles, scenario libraries with temporal structure, time-of-day features in observations, seasonal parameter drift.

### 6. Economic and multi-objective optimization

**Why PID struggles:** PID optimizes setpoint tracking (minimize deviation). Real plants balance quality, reagent cost, energy cost, equipment life, throughput, and compliance risk simultaneously. PID cannot express or optimize a composite economic objective.

**How RL wins:** RL can directly optimize a reward function that expresses the plant's actual operating objective — including asymmetric costs, constraint penalties, and economic tradeoffs.

**Benchmark examples:** any benchmark with reagent or energy costs, membrane fouling (cleaning cost vs throughput), chlorine (reagent vs compliance margin), H2S scrubber (bleach cost vs emission penalty).

**Realism features to include:** reagent cost accounting, energy cost signals, maintenance/cleaning cost proxies, asymmetric penalty structures, compliance threshold constraints.

### 7. Partial observability with informative proxies

**Why PID struggles:** PID requires a direct measurement of the controlled variable. When the true quality variable cannot be measured directly or in real time, PID must use a substitute that may not track the true variable well.

**How RL wins:** RL can learn to infer latent process state from correlated but imperfect proxy measurements, effectively learning an implicit observer as part of the control policy.

**Benchmark examples:** coag effectiveness (inferred from downstream turbidity and filter performance), membrane fouling state (inferred from TMP trajectory), demand changes (inferred from quality surrogates).

**Realism features to include:** latent state that is not directly measured, proxy sensors with imperfect correlation, observation asymmetry between baseline and RL profiles.

### 8. Asymmetric risk and constraint satisfaction

**Why PID struggles:** PID treats positive and negative deviations symmetrically. In practice, the cost of under-treatment (compliance violation, safety risk) vastly exceeds the cost of over-treatment (reagent waste). PID must be conservatively tuned to avoid the downside, which wastes resources during normal operation.

**How RL wins:** RL can learn asymmetric strategies that maintain a compliance buffer under uncertainty while reducing conservatism when conditions are favorable. It can directly encode asymmetric penalties in its reward function.

**Benchmark examples:** chlorine (minimum residual compliance), effluent quality limits, pH discharge limits, H2S emission limits.

**Realism features to include:** hard constraint boundaries, asymmetric penalty structures, stochastic disturbances that occasionally threaten constraints, compliance window evaluation.

### 9. Slow drift and nonstationarity

**Why PID struggles:** PID tuning assumes a stationary process. When equipment degrades, fouling accumulates, or raw-water quality shifts seasonally, PID performance degrades until an operator retunes. Adaptive PID exists but adds complexity.

**How RL wins:** RL trained on diverse conditions can generalize across slow drift, and online RL or fine-tuning can adapt to changing plant characteristics.

**Benchmark examples:** membrane fouling (irreversible accumulation), sensor drift, seasonal raw-water quality changes, equipment aging.

**Realism features to include:** slow parameter drift, equipment degradation models, seasonal variation, sensor calibration drift.

### 10. Cross-stage cascade interactions

**Why PID struggles:** each PID loop sees only its own local measurements. In a multi-stage process, upstream control decisions affect downstream difficulty in ways that local loops cannot anticipate or coordinate.

**How RL wins:** RL with cross-stage observation access can learn to coordinate upstream and downstream control for global optimality rather than local stage-by-stage optimization.

**Benchmark examples:** DWT train (coag dose affects membrane fouling AND chlorine demand), wastewater (aeration affects nitrification which affects denitrification).

**Realism features to include:** multi-stage process topology, upstream-downstream coupling through stream composition, cross-stage sensor access in RL observation profile.

## Summary matrix

| Advantage pattern | Key realism feature | Classic control workaround | RL structural advantage |
|------------------|--------------------|-----------------------------|------------------------|
| Transport delay | Flow-dependent residence time | Smith predictor (needs model) | Learns anticipation from proxies |
| Nonlinear gain | Operating-region variation | Gain scheduling (needs mapping) | Learns region-aware behavior |
| Multi-rate information | Mixed sensor cadences | None standard | Learns multi-source fusion |
| MIMO coupling | Interacting actuators | Decoupling or MPC (needs model) | Learns coordination directly |
| Temporal patterns | Structured disturbances | Feed-forward (needs measurement) | Learns temporal structure |
| Economic objectives | Cost accounting | Not expressible in PID | Directly optimizes composite reward |
| Partial observability | Proxy sensors | Limited workarounds | Learns implicit observation |
| Asymmetric risk | Compliance constraints | Conservative tuning (wastes resources) | Learns adaptive conservatism |
| Slow drift | Equipment degradation | Manual retuning | Generalizes or adapts online |
| Cross-stage cascade | Multi-stage process | Independent loops per stage | Coordinates across stages |

## Benchmark design checklist

When designing a new benchmark, check:

- [ ] Which advantage patterns does this benchmark exercise?
- [ ] Are the realism features that create those advantages explicitly included?
- [ ] Does the observation profile give RL the information needed to exploit the advantage?
- [ ] Does the baseline controller represent a reasonable classical approach, not a strawman?
- [ ] Can the advantage be measured and attributed (not just "RL got higher reward")?

A benchmark that exercises three or more advantage patterns simultaneously is likely to produce compelling results. A benchmark that exercises only one pattern should demonstrate that pattern clearly.

## Recommendation

Every benchmark in the catalog should declare which advantage patterns it primarily exercises. This makes the case for RL concrete and measurable rather than vague. It also helps prioritize: benchmarks that exercise patterns not yet demonstrated are more valuable than those that replicate already-proven advantages.
