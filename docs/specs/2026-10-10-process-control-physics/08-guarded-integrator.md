# D2 slices 5–9: one canonical guarded joint RK4

Design only, 10 October 2026. This follows Andy's approved change to [spec 07](07-holdup-fast-closure-manifold.md): **the guarded joint RK4 replaces the current integrator, every caller migrates, and the old public step functions are deleted.** Spec 07's recommendation to retain a separate fixed RK4 path is superseded. There is one RK trial kernel and one advance algorithm, including for BSM1. Static empty guard/event/ledger declarations are capabilities of that algorithm, not a second solver or compatibility implementation.

Reviewed current `integration.py`, `flowsheet.py`, `units/contract.py`, `benchmarks/bsm1.py`, current callers, and spec 07. Repowise returned no current integration target; ragdocs did not retrieve the recent spec, so current files were used directly. D2 slices 1–4 are in progress; proposed names below specify contracts, not APIs already landed.

## 1. Transient stage evaluation

Add a registered pytree `RhsEvaluation` with `derivative`, `streams`, and `diagnostics`. Maps have static keys and stable pytree structure; stream schema/phase remain static metadata. The flowsheet's `evaluate_rhs(state, inlets, inputs)` performs exactly the current work:

1. Call `_resolve` once with the supplied state and inputs.
2. Construct the derivative mapping in the current `stateful_nodes` order, using the same `_inlets_of`, `_inputs_of`, node parameters, and unit `derivatives` calls.
3. Return that derivative and the already resolved streams/diagnostics.

`Flowsheet.derivatives` returns `evaluate_rhs(...).derivative`. Do not reorder source accumulation, node summation, graph evaluation or derivative expressions. `report` still performs a fresh requested-state resolution; it is not the integrator's stage evaluator and stores nothing. The integrator's stage adapter adds typed validity and ledger-rate results to this transient evaluation. Empty ledgers have empty fixed pytrees.

No `HoldupEvaluation` argument is added to `ContinuousUnit.derivatives`. Holdup reconstruction/hydraulics can be recomputed with the same pure helper, as specified in 07. If its accepted ledger needs physical nonfast rates, expose a typed pure unit balance helper used by both the derivative and ledger adapter. This avoids duplicating physical equations without forcing every unit to return an opaque auxiliary object.

Use one public `Flowsheet.advance(..., dt, substeps, context)` returning `AdvanceResult`; delete `Flowsheet.step` in the atomic cutover. `derivatives` remains the ODE contract, not an alternate integrator. A standalone-unit adapter and a derivative-only adapter for existing numerical helpers both call the same `advance_rk4` algorithm. They are boundary adapters, not separate RK implementations.

## 2. Arithmetic that must survive the cutover

The landed paths are `rk4_step` and `rk4_substeps` in `integration.py`, reached from `Flowsheet.step`, `step_unit`, ASM2d's `reactor_step`, and directly from the older `scrubber_sump.step`.

For an accepted uninterrupted segment of duration `h`, retain these operations **in this order**:

```text
k1 = f(s)
s2 = tree_add(s, tree_scale(k1, h * 0.5))
k2 = f(s2)
s3 = tree_add(s, tree_scale(k2, h * 0.5))
k3 = f(s3)
s4 = tree_add(s, tree_scale(k3, h))
k4 = f(s4)
dy_leaf = (a + 2.0 * b + 2.0 * c + d) / 6.0
raw = tree_add(s, tree_scale(dy, h))
```

`tree_scale` is leaf `x * scalar`; `tree_add` is leaf `a + b`. Do not replace `h * 0.5` with division or pre-scaled slopes, introduce an `h/6` weighted sum, reassociate the four-slope addition, concatenate all derivatives into a new vector, or use a new precision policy. The trial kernel returns the candidate and transient stage data; it is the sole RK4 implementation. The old public `rk4_step` and `rk4_substeps` disappear, rather than delegate through retained aliases.

The outer grid remains `sub_dt = dt / substeps`, computed once, and static `lax.fori_loop(0, substeps, ...)`. On a substep with no rejection or event, pass **that exact `sub_dt`** to the kernel. Do not compute the ordinary duration by subtracting accumulated floating times, `dt - elapsed`, or `(i+1)*sub_dt - i*sub_dt`. Those can change the duration by roundoff and break zero diff.

The existing projection is applied **once after each completed nominal substep**, to its RK4 raw endpoint. There is no stage projection. Split/retried segments within a nominal substep do not multiply projection calls. Check the raw endpoint before projection and the projected endpoint afterward against the unit's declared policy. D2 holdup uses identity projection; legacy ASM/settler projections retain their exact expressions and placement. Projection does not repair an invalid stage for a unit with a strict domain guard.

Within split substeps maintain local remaining time; only that path uses remainder arithmetic. Completed time is a result/accounting quantity, never a source for a no-event RK duration. After all nominal substeps succeed, report the requested `dt` as completed (not an accumulated roundoff approximation). Failure reports only accepted progress. An event exactly at an outer boundary updates modes for the next call without repeating a positive-duration interval.

Expression identity alone is not proof of compiled bit identity: new control flow can change compiler optimization. Verify on the same JAX/backend/x64 configuration against the pre-cutover durable baseline. If equality fails, inspect lowered arithmetic and eliminate unnecessary dynamic control on statically empty policies while retaining the same kernel and algorithm. There is no exemption that replaces zero diff with a loose tolerance.

## 3. Domain and validity contracts

### Static explicit policy on nodes

Prefer a separate typed `IntegrationPolicy` attached to `Node` over expanding every unit's physics protocol with numerical machinery. Give each node an explicit policy at construction, with an empty/default legacy domain policy for existing units. No `hasattr`, method probing, or runtime fallback. A policy is static; its arrays/parameters and persistent modes are supplied explicitly.

The policy provides:

- `check_state(state, routed_inputs, params, modes) -> CheckResult`: cheap, side-effect-free, before division/speciation/graph evaluation. Contains fixed-order Boolean flags and integer reason identifiers, not Python exceptions for traced failures.
- Declared group validity requirements, keyed by group and an explicit diagnostic name; include admission/extrapolation policy. Never infer validity by scanning arbitrary diagnostic names.
- A post-RHS check for finite derivatives, relevant context/stream values, and source validity.
- Event declarations and a ledger adapter, described below.
- A static domain-safe evaluation witness for inactive/invalid lanes when required for vectorized execution. This is evaluated only for numerical masking, never accepted as physical state or reported as successful integration.

The flowsheet assembles named check flags in a fixed order and returns the first failing reason deterministically. Build-time errors (bad schema, missing routes, unknown validity diagnostics) remain ordinary exceptions. Runtime flags are arrays.

| Unit family | Declared numerical domain |
|---|---|
| New inventory holdup | Finite wet volume, reconstructable chart, valid product/spectator inventories; no blanket nonnegativity on signed invariants; explicit depletion/capacity events |
| New chemistry/column/manifold | Required context and supported chemistry/closure domain, convergence and admitted applicability flags; a converged but unsupported property solve is not automatically valid |
| Existing ReactiveCSTR / layered settler / ASM benchmark helpers | Preserve their existing constitutive domain and projection semantics; finite checks initially, without a new universal concentration nonnegativity rule |
| Stateless Mixer / Splitter | Their finite flow/context checks; splitter deficit is a diagnostic unless an explicitly chosen policy makes it a failure |

A test already integrates a CSTR with a negative initial carrier entry. Adding unconditional nonnegative-component checks would silently change that contract. Likewise BSM1's projected endpoint policy does not establish that every current RK stage is nonnegative. Stronger biological guards are a separate physics change, not part of an arithmetic-preserving cutover.

### Stateless failures must stop downstream use

Pre-state checks alone cannot detect an invalid manifold or failed algebraic column. Add explicit validity-gating metadata to the compiled stage resolution, using the policy's declared group diagnostic requirements. After each group result, evaluate its declared flags **before downstream groups consume its outputs**. On failure the whole stage is invalid; downstream physics uses only declared domain-safe placeholders or is skipped by scalar `lax.cond`. Do not let a failed column's plausible finite output propagate and only check a report at the end.

`GroupResult` can keep its array-only structure. The resolver's stage-validity adapter knows the declared validity diagnostics; ordinary reporting still exposes them. There is no diagnostic-only group and no cross-stage cache. A derivative-source failure with only an array return is detected by nonfinite rates; richer typed source flags can be exposed through the balance helper. Returning zero reaction on failure is prohibited.

This is a real resolver extension: specify safe placeholders/witnesses for each guarded chemistry boundary. A Boolean `where` around a final result is insufficient to prevent invalid constitutive arithmetic. The safeguards must also satisfy the vectorization requirement below.

## 4. Bounded retry and transactional advance

`AdvanceResult` is a registered fixed-structure pytree:

```text
state               last accepted continuous state
completed_time_h    actual accepted duration; dt on full success
status              integer code
quadratures         accepted integrated rates, static keyed pytree
context             persistent discrete modes and integration context
checks              failing reason/event index and bounded-work counters
```

Statuses minimally distinguish success, invalid initial state, invalid stage/domain, retry budget exhausted, event bracket/location failure, depletion/no admissible continuation, and event budget exhausted. A rejected transient stage flag is diagnostic history, not necessarily the final failure status after a successful retry. No NaN state is returned as successful.

Use statically bounded `lax.fori_loop`/`scan` with an active Boolean carry, not a data-dependent Python loop or an unbounded `while_loop`. Carry includes accepted state/context/quadratures, nominal substep progress, proposed duration, accepted duration, status, and counters. Static budgets limit retries, accepted segments per nominal substep, event-location iterations and zero-time mode transitions.

For each nominal substep:

1. Begin at the last accepted state with duration `sub_dt` and context held fixed.
2. Check state, evaluate four guarded stages using the single kernel, and check raw endpoint. Collect candidate rates but do not accumulate them.
3. If invalid without a valid event bracket, reject everything, halve candidate duration (`h * 0.5`), and retry from the same accepted state. Stop if duration is too small or the static budget is exhausted.
4. If valid and event-free, commit candidate state and its quadratures together. Continue over the local remainder if the nominal interval was split. A deterministic policy may retry the remaining duration at its full size; guards still decide acceptance. No adaptive error estimator is claimed.
5. If an event is isolated, commit only the recomputed accepted event segment and its quadratures, update context, and restart over the remainder.
6. At nominal completion apply the declared projection once; validate and record any projection inventory change separately. A projection failure cannot be reported as completed success. Keep a nominal-entry checkpoint if rollback of that whole nominal substep is needed.

A trial cannot mutate modes, counters used as physical state, or ledgers. Location trials likewise have no effects. Only the final accepted segment contributes. Accepted partial progress may be returned on final failure; distinguish this from rolling back the entire requested advance. A caller must not advance sensors, influent, controller clocks or plant time by the full `dt` after a partial advance.

### JIT, vectorization and differentiation

Shapes, dictionary keys, registry order and loop bounds are static. Per-lane durations, statuses, modes and active masks are arrays. `jit` stages the whole bounded algorithm. `vmap` must produce the same accepted result/status for each lane as separate scalar advances, including one failing lane alongside a successful one.

Crucially, batched `lax.cond` can become selection and execute both branches. Use sanitized **domain-safe evaluation arguments** for inactive/failed lanes at the boundaries before constitutive calls, then mask their results and discard them. Do not simply compute chemistry on invalid `V` and hope the inactive branch avoids it. This safe evaluation is not a physical repair. Test forward values and derivatives for NaN contamination with heterogeneous lanes. These JAX facts are documented in [lax.cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html).

Static bounded `fori_loop` supports reverse-mode through its scan lowering; dynamic-bound looping does not provide that guarantee. See [lax.fori_loop](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html). Use static budgets for the differentiable path. Away from guards/events, `grad` should match the old RK4 composition. Stop gradients through discrete decisions, masks and bookkeeping. On retried trajectories the derivative is conditional on the selected step schedule; finite-iteration event bisection is not a mathematically correct event-time sensitivity. Do not promise smooth gradients through capacity/interlock timing. Event-aware sensitivity/custom rules are outside D2; report event occurrence so optimization callers can recognize that limitation. Safe placeholders must also keep inactive reverse-mode branches finite.

## 5. Event registry and location

Compile an ordered tuple of `EventSpec` records from node integration policies. Each record declares stable `(node,name)`, scalar surface in stated units, crossing direction, active condition/mode, surface dependency set (state-only or resolved-stage reads/signals), tolerances, allowed continuation, and a pure context transition. Registry order is a deterministic tie breaker, not an invented priority among physically incompatible simultaneous events.

The event surface evaluator receives explicit state, context, input signals, and any declared transient stage fields. Cheap state-only surfaces do not require a graph solve. Events requiring flow fields, such as capacity release when net inlet supply changes sign, declare those dependencies and evaluate the graph without persistent cache. An endpoint surface evaluation may require an extra resolution; do not replace it with the k4-stage fields, since the k4 stage state is not the RK endpoint.

Modes are a fixed mapping/tuple of integer/Boolean array leaves in `IntegrationContext`, outside RK-scaled state. An event's mode names map explicitly to unit `input_ports`, for example `capacity_active`. Validate unique ownership and reject collision with an externally supplied ordinary signal. Merge mode inputs before normal routing. Freeze modes within each smooth segment, return changed context, and pass it into the next advance. Initial context is explicitly constructed/validated from physical initial conditions, not reset on every call.

### Required events

- Holdup capacity entry: `g=V_c-V`, outward/downward crossing; activate overflow constraint.
- Capacity release: unconstrained net balance `b` crossing into negative flow, when active; release constraint. Immediate inward direction at the current boundary releases at zero time.
- Wet-level draw interlock/depletion: `g=V-V_min`; disable only physically declared withdrawals, or return depletion when no inward continuation exists.
- Fast chart `I=0`: both directions, with outgoing direction deciding branch; do not repeatedly trigger merely because the start value is zero.

Transition functions for these D2 events change modes, not continuous inventory. Any future state-reset event must return its explicit material/volume impulse ledger. Numerical snapping to capacity is not such a transfer.

### Location algorithm

Take all trials from the same accepted origin and fixed pre-event mode. Build a bracket `[h_lo,h_hi]` for each candidate crossing using complete joint RK trials. The lower side must be admissible; the upper side must either be admissible or have a declared valid smooth continuation. For inactive capacity, zero-overflow continuation above capacity may be legal for location if all constitutive domains remain valid. An invalid negative-volume state is not a valid upper sign sample. If stage validity fails before a surface sign can be evaluated, reduce the bracket through admissible trials; return bracket failure if it cannot be established. Halving alone is not a root proof.

Use bounded bisection first. Safeguarded secant is optional only if the proposal stays strictly inside the bracket and finite, with bisection fallback. Every candidate re-runs the same RK kernel from the accepted origin; no concentration interpolation, no accepted history, no projection during location. Choose the earliest verified event time across candidates and handle simultaneous events within declared time tolerance. Re-evaluate surfaces after transitions; incompatible transitions fail explicitly.

Require both a narrow time bracket and a surface residual within its physical tolerance. Choose an admissible near-boundary endpoint and accept its computed state without snapping. For capacity, one-sided residual and boundary tolerance permit mode activation within that tolerance; preserve the actual volume/material budget and report the residual. If the tolerance cannot be achieved, fail. The subsequent active capacity law may retain that tiny offset; it must not silently manufacture missing liquid. Make this error visible and bounded by the event tolerance.

Endpoint sign checks cannot discover arbitrary even-numbered crossings inside a step. Declare a monotonic/single-crossing bracket assumption or a maximum sampling interval for each D2 event. Examine available stage surfaces to flag uncertain intervals and subdivide; stage states are warning samples, not exact dense trajectory points. A generic oscillatory event system needs stronger isolation, not a promise hidden in this primitive. Grazing roots use directional conditions; an exact zero with no outgoing crossing is not repeatedly fired. Bound zero-time transition chains to detect chatter/Zeno-like behavior.

## 6. Accepted quadratures and material accounting

`LedgerSpec` declares static names, rate units, and pure rate extraction from current stage state, `RhsEvaluation`, inputs and context. Return a fixed pytree of rates. Examples are per-external-port volume and amount flows, per-node physical nonfast inventory rates, slow reaction extent rates, and explicit transfer/source rates. The RK trial combines each rate leaf in the same expression order `(a + 2*b + 2*c + d)/6`, then multiplies by segment duration. Only commit on acceptance.

Boundary amount flux is `Q*C` in the declared stream basis. Gas mole flow and liquid volume flow cannot be combined without an explicit transfer conversion. Internal connected ports cancel once at a plant boundary; avoid treating a recirculation stock manifold outlet as new external feed. For transformed manifolds, external stock inputs and physical column/sump changes belong to the same balance. Bind chemistry extent ledgers by exact named bases.

Projection accounting is separate: record physical inventory change from pre-projection to post-projection. Legacy concentration clamping may have a nonzero correction and must not be called conservative merely because RK quadratures balance. Preserve existing legacy state results; D2 identity projection contributes zero. Initialization fast extents are a separate one-time record outside step quadratures.

For the fast one-column chart, reconstruct accepted start/end inventories and use

```text
Delta_xi = integral(F_nonfast_Sr) - (n_end_Sr - n_start_Sr)
Delta_n - integral(F_nonfast) = S_fast Delta_xi
```

Verify every row and nonnegative irreversible progress to roundoff-scaled tolerances. Never apply `S_fast Delta_xi` to state again. A stage extent rate is a diagnostic; endpoint-rate times `dt` is not the ledger. Integrated invariant RHS and physical nonfast rates must use the same accepted stages. This makes volume/material audit failures actionable even when plotted concentrations look reasonable.

## 7. Atomic caller migration

Current search found the following direct numerical callers. Repeat the exact-symbol search immediately before cutover because slices 1–4 are landing concurrently. Do not mistake actuator/sensor/environment `.step` methods for RK callers.

| Current call site | Cutover |
|---|---|
| `flowsheet.py`: `Flowsheet.step → rk4_substeps` | Replace with `advance` returning `AdvanceResult`; no old method alias. |
| `units/contract.py`: `step_unit → rk4_substeps` | Replace with standalone `advance_unit` adapter returning the same result; it uses the sole algorithm, holding inlets fixed as before. |
| `benchmarks/bsm1.py:491`, interval integration | Use advance result and empty persistent context; successful plant values, control update order, observation snapshot timing and `config.substeps` unchanged. Thread failure/status explicitly; never advance full-time exogenous state on partial numerical failure. |
| `units/asm2d.py:109`, `reactor_step` | Adapt derivative/project to the canonical algorithm with its existing ten substeps; preserve returned physical state on success and expose/propagate failure into `benchmarks/combined_np.py`, which calls it three times. No private RK loop remains. |
| `units/scrubber_sump.py:286`, old sump helper | Replace the custom Python RK loop with the canonical advance adapter. Preserve `_clamp` placement and account for its current overflow observable through the projection-observer ledger (raw volume excess above `1.5*volume_max`). This historical clipping quantity is not the new conservative weir/capacity overflow. Propagate status through its H2S benchmark caller. Do not conflate this numerical migration with installing the new D2 chemistry sump. |
| `core/tests/medium/test_process_control_{flowsheet,reactive_cstr,layered_settler,activated_sludge}.py` | Migrate `step_unit`/plant-step calls to result API; assert success before consuming state; keep standalone frozen-inlet semantics. |
| `core/tests/large/test_process_control_bsm1_reference.py:190` | Jitted settling loop consumes successful advance state and tracks numerical status; retain 240 substeps/hour and baseline checks. |
| `projects/process-control-baselines/scripts/bsm1_refactor_baseline.py:196` | Migrate open-loop trace; fail visibly on non-success. Preserve trace fields and never regenerate the expected pre-cutover artifact with the new solver as its own oracle. |

`plants/activated_sludge.py` assembles the flowsheet; the searched plant module has no direct old numerical-step invocation. Its policy declarations/layout wiring must still compile and its callers migrate. The BSM1 benchmark's outer `lax.scan` over intervals remains in the same order. Its current records measure `plant_now` before the plant advance; moving them to end-of-interval values would be an unrelated reference-breaking change.

Failure propagation is an explicit integration contract change. Benchmark info should include a numerical status and stop/fail the interval consistently; a partial result is not a valid ordinary reward transition. Within JIT, use array status and a declared failure branch, not Python raising. At host-side baseline generation, raise with status/completed time. Do not add a fallback to the deleted old integrator. Empty mode contexts require no added BSM1 continuous leaves; nonempty contexts are persisted in the owning benchmark state separately from plant ODE state.

The baseline script belongs to a separate project repository while the solver belongs to core. Prepare both changes before cutover; coordinate core and project commits/pointer updates so the checked-out workspace is coherent. Independent histories cannot make one literal cross-repository atomic commit. Do not conceal that boundary or retain a compatibility alias to avoid it.

## 8. Proposed commits for slices 5–9

Sizes below are estimates of changed production lines, excluding focused tests; reconcile with landed slices 1–4 before implementation. Tiny record/adapter changes can be ~20–40 lines. Numerical state-machine changes exceed the tiny-commit target when splitting would create an unusable public integrator. No implementation is authorized by this design alone.

| Commit | Slice / ownership | Estimated size and validity |
|---|---|---|
| A | 5: `RhsEvaluation` and `evaluate_rhs`; derivatives delegates | 30–50 lines, additive, identical values; no stepping change. |
| B | 6: check/result/context records and explicit node integration policy declarations | 50–90 lines; all existing nodes use declared finite/legacy policies, new holdup strict wet/chart policy. Group validity requirements validated at build time. |
| C | 7: ledger rate declarations and pure accepted-quadrature helpers | 40–70 lines; no new public integrator; pure helpers independently verified, no alternate solver path. |
| D | 8: event registry/surface/transition declarations and bounded bracket locator helper | 80–140 lines; locator accepts a trial callback, so it contains no copied RK equations. Tests use real trial evaluations/simple analytic callbacks. |
| E | 9: capacity law/mode routing and wet/fast event policies | 50–100 lines; additive unit capabilities, not yet wired into a running unguarded fixed-level plant. |
| F | **Atomic integrator cutover:** sole guarded trial/advance, stage validity gates, all core caller and required test migrations, deletion of `rk4_step`, `rk4_substeps`, `step_unit`, and `Flowsheet.step` | Approximately 220–400 solver/resolver lines plus 60–140 caller lines; larger coherent contract change. Includes bounded retry/events/transactional ledger and statuses, not a partial replacement waiting for later fixes. Companion project migration prepared and applied with core version update. |
| G | Additional additive validation/performance diagnostics if not required for cutover correctness | Focused behavior/convergence tests, heterogeneous vmap/grad cases and budget measurements. Required zero-diff and domain/event tests are already part of F. |

A–E are static contracts/pure helpers; they do not install a second live integrator. F is the first and only runtime replacement. If the large F becomes hard to review, extract more independently useful pure helpers beforehand; do not split by leaving old and new public solvers live or breaking callers between core commits. Every core commit passes existing checks; F includes required test fixes because this is a contract-breaking API migration.

## 9. Validation gates

- **Arithmetic/regression:** durable pre-cutover float64 BSM1 derivative, open-loop and closed-loop traces; zero max absolute diff under the same configuration/backend. Compare one substep, many substeps, projection-active cases, and outer interval scans. Assert no rejection/event occurs in the reference trajectory. Preserve controller/actuator/influent RNG and measurement timing. Existing ~0.1% published reference reproduction remains an independent check, not the migration tolerance.
- **No hidden second path:** exact-symbol search shows no old imports/definitions/calls, and all production RK stages occur in one kernel. Standalone, ASM2d and old sump adapters exercise that kernel.
- **Guard safety:** wet draining with an excessively large candidate interval never evaluates chemistry on negative volume; reject before division. Invalid initial state returns zero completed time. Test exhausted budgets, nonfinite RHS, a stateless algebraic failure before a downstream unit, and independent successful/failing vmap lanes. No accepted correction from a rejected trial.
- **Analytic volume/material:** constant-composition draining preserves concentration and inventory balance; stop at declared wet boundary or continue under a real draw interlock. Capacity case starts at 9 m³, capacity 10, feed 2 and draw 1 m³/h for two hours: capacity event at one hour, final 10, draw integral 2, overflow integral 1 m³. Check solute removal with unequal feed/initial concentrations, not just constant concentration.
- **Events:** start-on-surface inward/outward/zero directions, release after changing net flow, simultaneous compatible events, incompatible transitions, grazing/no crossing, capacity exactly at nominal boundary, bounded chatter and bracket failure. No volume snap. Event residual/time tolerance tightening should converge and preserve the accepted quadrature budget.
- **Fast chart:** I-crossing under alternating reagent supplies, conserved sulfur/chlorine/sodium, initialization extent once, nonnegative accepted extent, all-row residual. Signed invariant stays signed. Ledger remains correct when trials are rejected or split.
- **Smooth convergence/grad:** analytic decay and variable-volume mixing on successively halved nominal steps exhibit fourth-order global convergence away from events; scalar/vmap/jit results agree. Gradients of no-event trajectories match the old smooth numerical mapping. Event gradients are not certified without a separate sensitivity design.
- **Projection:** legacy final states unchanged and projection correction ledger exposes any inventory change. New holdup identity projection has zero correction and no hidden clamping.

During implementation run repository-wide lint/type checks and focused behavioral tests using the project's environment. This document is a design review; no solver implementation or runtime regression checks were performed here.
