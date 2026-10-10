# Milestone D2: variable holdup, fast invariant closure, and injection manifold

Design only, 10 October 2026. Reviewed against the landed `streams.py`, `fields.py`, `field_graph.py`, `flowsheet.py`, `units/{contract,reactive_cstr,layered_settler}.py`, `kinetics.py`, `binding.py`, and `integration.py`. This makes [spec 03](03-transfer-column-holdup-orp.md), [spec 05](05-milestone-C-engine.md), and [spec 06](06-scrubber-liquid-chemistry.md) concrete after milestone C. D1 chemistry interfaces are being implemented; names below describe proposed responsibilities rather than claim an unverified D1 API.

**Recommendation:** implement a liquid inventory holdup with a static inventory chart, a reusable analytic fast closure, and a stateless amount-flow manifold. Start with a finite weir. Add guarded *joint* RK4 with transactional material quadratures before using the holdup in a running plant. A fixed-level overflow requires event handling and an explicit boundary mode; final-state volume clipping cannot implement it. Keep the current BSM1 integration path and arithmetic unchanged.

## 1. Constraints from the landed engine

The engine already separates output field ownership and inlet reads. A sump can publish composition from state without reading its returning inlet, and a prescribed recirculation flow without reading return flow. These roots break the physical recycle into an acyclic field graph. A fixed-level overflow may read incoming flows without making recirculation flow depend on them, provided the outputs belong to separate evaluation groups.

There are concrete limits:

- `ContinuousUnit.derivatives` receives state, complete resolved inlet streams, routed inputs, and parameters. It does not receive its resolved outlets or a `HoldupEvaluation`.
- `GroupResult` holds array values and array diagnostics. It is not an opaque auxiliary-object channel. Groups must own at least one output; there is no diagnostic-only group. A group's field reads name inlet fields, not another group of the same unit.
- `Flowsheet.derivatives` resolves the graph once and then calls each stateful unit's derivative. `report` resolves it again independently. The current resolver does not preserve arbitrary stage auxiliaries.
- Existing joint RK4 has no intermediate-stage admissibility checks, event handling, or accepted-flux quadratures. Projection runs after an entire RK4 substep. An invalid intermediate volume can already have divided concentrations or entered a chemistry/column calculation before projection runs.
- Every dynamic state leaf is subjected to RK arithmetic. A Boolean boundary mode must not be inserted as an ordinary state leaf and given a nominal zero derivative.

D2 does not require a breaking change to `ContinuousUnit`. Use the layered settler's existing pattern: a cheap pure hydraulic calculation shared by output groups and derivatives. Recomputing this small calculation is preferable to introducing an opaque stage cache. If later profiling proves that expensive coupled chemistry needs a shared evaluation, design that typed extension separately; do not smuggle it into diagnostics or store it on a unit.

## 2. Generic variable-volume liquid holdup

### State, coordinates, and physical domain

Use a continuous pytree state with `volume_m3` and `amount_coordinates`. Its static chart is either identity inventory or a declared fast-invariant chart. With the identity chart, the coordinates are the full transported inventories `n`. With a fast chart, they are `y = U n`; physical inventories are reconstructed at every evaluation.

Inventories have concentration-basis units multiplied by m³. They are mol for the scrubber's mol/m³ carrier; a generic COD carrier instead has COD amount. Do not call every generic coordinate molecular moles or apply molecular element tests to opaque proxies. `V` remains a separate independent state; volume is not one of the chemical invariants.

The unit declares ordered liquid inlet names, prescribed draw outlets, one optional overflow outlet, the carrier schema, a volume/level geometry, sources, chart, and context. All outlets initially carry the same well-mixed liquid composition. Suspended material is advected on that declared assumption. Settling, retained deposits, or selective removal require their own inventory and conservative transfer model.

For the initial implementation the domain is **wet, incompressible, prescribed-temperature liquid**, with `V >= V_min > 0`. `V_min` is a declared physical operating boundary, not an epsilon silently used in division. All volume-changing feeds and removals must be accounted for. Reaction/thermal volume changes are neglected unless an explicit volume source is supplied. This is the dilute-volume approximation already declared in spec 06, not a concentrated-mixture mass/energy model.

```text
C = reconstruct(amount_coordinates) / V
b = sum(Q_in) - sum(Q_draw) + Q_volume_source
Q_out = sum(Q_draw) + Q_overflow
F_nonfast = sum(Q_in C_in) - Q_out C
            + V * concentration_sources(C, temperature_k, inputs)
            + explicitly declared amount_sources
V_dot = b - Q_overflow
identity chart: n_dot = F_nonfast
fast chart:     y_dot = U F_nonfast
```

Time is hours, flows m³/h, and source rates use the landed concentration-per-hour convention. An existing bound `KineticSource.add_to` receives the reconstructed **full physical concentration vector**, never invariant coordinates. Sum its sources on that same physical state and multiply by `V` once. Sources do not independently advance state. Every transported channel participates in inlet/outlet transport; bindings restrict chemistry reads/writes, not transport.

A withdrawal may exceed current incoming flow because stored liquid supplies it. Do not reuse a splitter's feed-flow cap. Do not cap a draw by `V/dt` inside a derivative: that makes the physical law depend on the numerical step size. Explicit pump limits/interlocks determine realized draws; a guarded integrator handles the resulting state boundary.

### Exact evaluation groups

The table describes group ownership, not a requirement for the spelling of the group names. `reads_state` must match the implementation, and every listed signal must be in `input_ports` and the group `input_names`.

| Group | Owned outputs | Inlet reads | State / signals |
|---|---|---|---|
| `draw_flows` | FLOW for each prescribed draw | None | Draw commands; state only if a declared wet-level interlock affects realization |
| `composition` | COMPONENTS for every draw and overflow | None | State, chart, schema; any explicitly requested chemistry diagnostic's temperature signal |
| `weir_flow` | Overflow FLOW | None | Volume, geometry and weir parameters |
| `capacity_overflow` (alternative) | Overflow FLOW | FLOW of every inlet | Volume, draw commands/interlocks, volume-source signal, explicit capacity mode |
| context | TEMPERATURE_K and PRESSURE_PA for every outlet | None | Landed `PrescribedContext`, including `temperature_k` |

Only one overflow-law group exists in a built unit. Composition is state-only even when overflow flow reads inlets. The prescribed recirculation draw must never be bundled into the capacity overflow group. Otherwise a recirculation flow edge through the manifold/column back into the sump recreates a cycle.

`composition` reconstructs inventories and divides by the same volume used by the RHS. Attach chart validity and balance residual diagnostics there if needed; there is no diagnostic-only group. Reconstructing analytical totals does not itself require pH or temperature. If this group additionally solves and reports pH, declare its temperature input and the entire chemical closure read set; do not introduce a hidden context dependency. The RHS obtains its temperature directly from its routed Kelvin signal. The biological Celsius input convention in `ReactiveCSTR` is not the sump's temperature convention.

Use one pure `hydraulics(state, inlet_flows, inputs, params, mode)` calculation in group evaluation and the RHS. It must reproduce the same realized draw and overflow flows. Each group consumes only its declared reads; a weir or draw calculation cannot inspect an undeclared returning inlet. The capacity calculation may sum its declared inlet flow fields.

### Finite weir: recommended first overflow law

With known level geometry `h(V)` and crest elevation `h_c`, a simplified sharp-crested rectangular weir is

```text
Q_overflow [m³/h] = 3600 * (2/3) C_d b_w sqrt(2 g)
                   * max(h(V) - h_c, 0)^(3/2)
```

Use metres and seconds inside this formula; `b_w` is width, not the hydraulic balance `b`. A positive head stores liquid **above** the crest. The crest is not a hard maximum volume. A separate structural capacity, if present, needs a guard or a separately accounted spill law. Geometry and discharge calibration are required; a simplified coefficient does not substitute for the head and installation corrections in standard weir practice. See the primary [USBR Water Measurement Manual, rectangular weirs](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_06.html).

The EPCOR 695 mm close / 600 mm reopen thresholds describe the makeup valve hysteresis. They do not identify a weir crest, vessel capacity, or sump area. Do not derive those unknown parameters from the thresholds. Makeup affects blowdown through the material/volume balance, not by setting an independent arbitrary blowdown command.

### Fixed-level overflow: a different declared law

For a rigid capacity `V_c`, the boundary constraints are

```text
V <= V_c; Q_overflow >= 0; (V_c - V) Q_overflow = 0
below capacity: Q_overflow = 0, V_dot = b
at capacity:   Q_overflow = max(b, 0), V_dot = min(b, 0)
```

Complementarity alone does not choose the overflow rate; the viability condition supplies the last line. For constant capacity, the overflow's solute removal is `Q_overflow * C` at the contemporaneous stage composition. A varying capacity would need its own `V_c_dot` term and is outside the first D2 law.

Treat reaching capacity as an event. At the event, activate the boundary law, then integrate the remaining interval under that law. Release it when the unconstrained net flow points inward. Hold its discrete mode outside the continuous RK state and route it as a declared input, such as `capacity_active`, for the affected node. Preserve that mode in the advance result and across calls. Do not infer an active state from a wide numerical band that begins dumping below capacity, and do not make Boolean modes continuous ODE variables.

An inactive trial may use the smooth zero-overflow continuation solely to locate the capacity event; its beyond-capacity trial state is never accepted. Other constitutive models still need admissible stage states. An event at the current time needs a directional branch decision, not repeated zero-length steps. For the active mode, positive `b` keeps volume on the boundary; negative `b` releases it. A changing `b` may require locating a mode-release event inside the interval.

For a simple check, start at `V=9`, `V_c=10` m³ with 2 m³/h inlet and 1 m³/h draw, no volume source, for two hours. Capacity is reached after one hour; final volume is 10 m³, total draw is 2 m³, and total overflow is 1 m³. Clipping an unconstrained final volume from 11 to 10 gives no corresponding solute discharge and is wrong.

## 3. Reusable fast invariant closure

### Declaration and compilation

A `FastReactionClosure` is a separate physical policy, not a large `KineticsModel` coefficient. Its declaration contains:

1. Balanced D1 reaction records and their projected local stoichiometry `S_fast`, including extent normalization.
2. Explicit named access/binding to the physical carrier, with exact transported bases under ADR 016.
3. A fixed product-yield policy and its applicability/provenance.
4. An invariant chart `U`, reconstruction rule, domain predicates, and directional extent/reconstruction rule.

Validate `U S_fast = 0`, full chart rank, consistent dimensions/bases, and the stated reconstruction domain at build time. Keep chemistry coefficients exact where possible before numerical compilation. Computing a left nullspace does **not** specify a physically unique endpoint. A generic declaration must provide the closure/selection rule; do not promise automatic equilibrium construction from an arbitrary stoichiometric matrix. Start with the analytic one-progress-variable closure below. Multiple reactions competing for a reagent need a joint endpoint policy, not a sequence of independently greedy reaction applications whose result depends on record order.

The local chart replaces only its declared reactive coordinates. Untouched carrier coordinates remain identity inventories, including opaque transport channels. The complete chart therefore has the carrier's full information modulo the explicitly eliminated fast progress coordinate(s). A component's name or elemental reference alone does not infer its family or participation.

### Fixed-yield sulfide/FAC example

Use mol-S extent and a constant sulfate yield `f` in `[0,1]`; set `d=1+3f`. In the physical order `(S_r, F, S_0, S_6, Cl)`,

```text
S_fast = (-1, -d, 1-f, f, d)^T
I   = n_F - d n_Sr
J0  = n_S0 + (1-f) n_Sr
J6  = n_S6 + f n_Sr
KCl = n_Cl + n_F
```

Together with unchanged inventories and `V`, these are the stored coordinates. `I` is allowed to be negative; it is not a concentration subject to nonnegative clipping. Reconstruction is

```text
n_Sr = max(-I/d, 0)
n_F  = max(I, 0)
n_S0 = J0 - (1-f) n_Sr
n_S6 = J6 - f n_Sr
n_Cl = KCl - n_F
```

Reject negative reconstructed products/spectators or an otherwise invalid chart. Do not separately clip them. Sulfur is `J0 + J6`; chlorine in these five coordinates is `KCl`. This is a **declared reduced instantaneous endpoint**, not a claim that all real sulfate production finishes instantaneously. The product-policy limitations and literature in [spec 06 §§4–5](06-scrubber-liquid-chemistry.md) continue to apply. Subsequent S0 oxidation is a separately justified slow source. Acid-base speciation is evaluated from the reconstructed totals as needed; pH is not an extra stored state.

`f` is fixed for a trajectory under this chart. Making it a pH-dependent signal changes `U` and invalidates `y_dot = U F_nonfast`. Such a model needs an explicitly derived history/selection law or a new chart with conservative reinitialization; it is not a parameter update hidden inside reconstruction.

### Initialization and each-stage evaluation

Initialization takes a physical inventory `n_initial`, validates it, computes `y_initial=U n_initial`, and reconstructs `n_relaxed`. The one-time initialization extent is

```text
xi_initial = min(n_initial_Sr, n_initial_F / d)
n_relaxed = n_initial + S_fast xi_initial
```

Verify both formulations agree and report the initial extent separately. This is an initialization/amount transformation, not an RHS subtraction or an impulse to be reapplied each call. Volume is unchanged in the declared dilute closure.

At every joint RK stage:

```text
stage (V,y) -> reconstruct n -> validate -> C=n/V
            -> field graph / outlet streams / sources
            -> physical F_nonfast -> y_dot=U F_nonfast
```

No persistent composition, pH, column solution, or reacted-inventory cache may survive a stage. Outlets and derivatives must reconstruct the same physical state. Repeating pure reconstruction is safe and cheap. Speciation or a column solve must use current stage inputs, including manifold outputs, not a previous report.

The fast reaction is not also registered as a slow source. The inlet can bring both reagents; their common progress is eliminated by the chart while products appear on reconstruction. This preserves sulfur/chlorine/sodium accounting without resolving the elementary fast timescale. Slow laws operate on reconstructed physical amounts/concentrations and must remain valid at a zero reactant boundary.

### Rates, extents, and reporting

For physical nonfast amount rate `F_nonfast`, the directional realized fast rate for the example is:

- `I > 0` (FAC excess): `xi_dot = F_nonfast_Sr`.
- `I < 0` (sulfide excess): `xi_dot = F_nonfast_F / d`.
- `I = 0`: for physically admissible supplies to the two absent reactants, `xi_dot = min(F_nonfast_Sr, F_nonfast_F/d)`; the sign of `I_dot` selects the outgoing branch.

The last expression is a complementarity **rate**, with units mol/h. It is not repeated removal of `min(current amounts)` from a derivative. Negative supply at an absent reactant indicates an invalid source/withdrawal law rather than a reversible interpretation of this irreversible fast reaction. Do not use automatic differentiation of `max` at zero as the physical directional rule: its numerical tie derivative can differ from the one-sided reaction law.

For accepted integration, the robust extent ledger is based on the same RK quadrature as the physical nonfast balance:

```text
Delta_n = n_end - n_start
Delta_xi = integral(F_nonfast_Sr dt) - Delta_n_Sr
check: Delta_n = integral(F_nonfast dt) + S_fast Delta_xi
```

This works even when the branch changes, provided the trial and quadratures were accepted consistently. Check every affected row and nonnegative cumulative irreversible progress within declared numerical tolerances. The ledger does not modify state again. Initialization extents are separate. For several fast directions, determining individual extents may need extra selection information; net invariant reduction alone need not identify them uniquely.

A stage directional rate is useful for diagnostics. Accumulating a reported rate at outer-step endpoints is not a conservative extent ledger, especially across `I=0`. Attach instantaneous diagnostics to an existing owning group; place accepted integrated extents in the integration result. A stateless manifold instead reports extent **flow** in mol/h.

## 4. Unavoidable integration work

### Keep the benchmark path intact

Do not change the arithmetic, projection order, default state structure, or default integration selection for BSM1. Preserve its current fixed RK4 path and establish zero float64 diff against the landed reference. The new guarded path is a capability for units with domain boundaries and hybrid laws, not an unrequested second implementation of holdup physics.

There is no valid D2 plant implementation that relies on `project` clipping `V` or chemical coordinates after a step. Identity projection is the default for the new holdup. Any future projection must state a conservative physical operation and its transferred amounts; a numerical clamp is not one.

### Small additive stage result

Factor the existing `Flowsheet.derivatives` resolution and derivative construction into a single canonical stage evaluator, conceptually `evaluate_rhs`, returning a typed result containing:

- The joint continuous derivative.
- Resolved stage streams and array diagnostics already produced by `_resolve`.

The existing `derivatives` obtains its derivative from that evaluator without changing numerical values. A guarded driver uses the result for instantaneous port-flow/amount-flow observables and validity flags. This avoids calling `report` and solving the graph a second time at each stage. The holdup can expose a typed pure physical-balance helper to the accounting policy; it may recompute its cheap reconstruction/hydraulics, not rerun the graph. Keep the result transient. Do not add arbitrary auxiliary objects to `GroupResult` or change every unit's derivative signature.

**WHY this extension is needed:** the conservative overflow/material ledger must use the streams evaluated at the actual RK stages. The current state-only step and independently evaluated report do not expose those stage rates. Passing a `HoldupEvaluation` into all unit derivatives is **not** required for the initial analytic closure. If sharing such an object later becomes necessary, it needs a separately reviewed typed contract and producer/consumer dependency model.

### Guarded joint RK4

The driver must advance the entire flowsheet together. Never run local `step_unit` for the sump while freezing the column/manifold during its four stages.

1. Check cheap state-domain conditions before invoking constitutive physics at each candidate stage: wet volume, finite coordinates, reconstructable products, and any declared physical constraints. Check numerical/chemistry/column validity after stage evaluation. An invalid stage must not feed an apparent valid zero reaction.
2. Generate one joint RK4 trial and matching quadratures with weights `(1,2,2,1)/6`. Check intermediate states and the candidate endpoint. Positivity is not guaranteed merely by a physically nonnegative boundary RHS.
3. Reject the **entire** trial on failure. Discard its state and all quadratures; reduce the interval globally and retry from the same accepted state. No unit or diagnostic accumulator may retain a rejected trial.
4. Locate the earliest active event, accept a segment ending at it, update its discrete mode/directional branch, then advance the remainder. Re-run joint stages during event location; do not interpolate concentrations or clip the endpoint and invent the missing discharge.
5. Return a typed result with final state, completed time, status, accepted quadratures, and explicit discrete modes. Bounded attempts/event iterations are necessary for JAX execution. On exhaustion or invalid physical continuation, return a failure/partial advance with completed time; do not claim the whole requested interval completed.

Halving alone is a rejection mechanism, not a full event method: it can approach a boundary indefinitely. Event surfaces include capacity entry/release, a declared wet-level draw interlock, and fast-chart branch changes. `I=0` is admissible and reconstruction is continuous but its derivative is kinked; segmenting there improves directional accounting and accuracy. Known product/depletion boundaries need an admissible continuation or a failure. Event root tolerances must be stated in physical units and checked against balance tolerances. Within a smooth branch, ordinary RK4 convergence remains the oracle; events limit global order if location is inaccurate.

Event isolation must not require evaluating chemistry at invalid negative volume/inventories. For a surface with a valid smooth continuation (for example inactive overflow through capacity), a trial can bracket the event without being accepted. Otherwise use admissible one-sided trials and bounded event location. If no admissible endpoint can be found within tolerance, fail with the remaining time. There is no permission to use fictitious epsilon-volume concentrations to cross depletion.

Start D2 with the wet domain. At `V_min`, turn off only those withdrawals that an explicit physical interlock declares disabled; then test whether the resulting dynamics point inward. If net volume still points out of the wet domain, stop with a depletion status. A genuinely empty vessel with retained dry solids and subsequent refill requires a separate dry-state/phase model and cannot be inferred by dividing zero inventory by zero volume.

Chemical inputs and actuator signals are held according to the current outer integration contract. Discrete mode changes are applied through an integration context and declared node inputs, frozen within a smooth RK segment. They are neither mutable unit attributes nor continuous state leaves. Differentiation through event timing/branch switches is nonsmooth; do not claim an ordinary smooth JAX derivative there. No new adaptive error estimator is necessary for the first guarded method, but its maximum smooth step and convergence study must establish accuracy separately from guard validity.

### Conservative quadratures and tolerances

For every accepted segment accumulate prescribed external inlet/draw/overflow volume and transported amount rates at the same RK stage states as the RHS. Internal ports cancel once when forming a plant budget; they are not external feed/discharge. Include explicit volume sources, slow reaction extents, transfer fluxes, and initial fast extents with their stated boundaries. Do not approximate overflow as final flow times the requested step.

For a holdup, compare the integrated physical nonfast rate and reconstructed inventory change using the fast-extent equation above. For an element/conserved coordinate `L` satisfying `L S_fast=0`, compare `L Delta_n` with its integrated nonfast supply. The invariant state's linear RK update and matching quadrature should satisfy this within float64 arithmetic tolerances. Accepting an event root with a silently snapped volume would break this check; either accept the computed endpoint within the declared boundary tolerance or account for a real conservative event transfer. No unreported reset.

Scale tolerances by inventories/flows and distinguish roundoff, integration error, and physical-domain failure. An arbitrary tolerance is not permission to clamp negative chemical products. Trials with materially negative reconstructed inventories are invalid. Do not silently remove a small mass and call the result conservative.

## 5. Serial algebraic injection manifold

Declare a stateless liquid unit with ordered recirculation and stock inlet ports and one outlet. All inputs use the same full carrier schema. D1 stock adapters expand measured stock composition into that carrier explicitly; there is no implicit positional alignment or automatic basis conversion at the manifold. Separate physical injection locations are separate serial manifolds when order/residence actually matters.

Its pure calculation is

```text
Q = sum(Q_i)
A = sum(Q_i C_i)                # transported amount flow, per hour
(A_reacted, xi_flow) = closure.transform_amounts(A)
C_out = A_reacted / Q           # Q > 0
```

With no fast chemistry, the closure is identity. The same fixed-yield amount transformation used for conservative initialization acts on **amount flows**, so `xi_flow` has mol/h units. No manifold volume, residence time, timestep, or persistent reacted inventory exists. This transformation is homogeneous for the fixed stoichiometric closure. A future nonlinear residence-dependent kinetic manifold is a different physical unit, not an optional large coefficient in this algebraic one.

Reuse the landed mixture's schema/phase checks and ordered flow arithmetic where practical. Do not refactor the BSM1 mixer just to implement this unit. At zero total flow, use a declared harmless placeholder composition consistent with existing mixture semantics, set extent flow to zero, and mark the nonflowing diagnostic. Do not react an unweighted average of stationary inlet compositions. An outlet with zero flow transports no amount; chemistry diagnostics on that placeholder must not be presented as a physical flowing-mixture pH.

| Group | Outputs | Inlet reads | Additional inputs |
|---|---|---|---|
| `flow` | Outlet FLOW | Every inlet FLOW | None |
| `composition` | Outlet COMPONENTS | Every inlet FLOW and COMPONENTS | Temperature only if the admitted closure/diagnostics needs it |
| context | Outlet TEMPERATURE_K, PRESSURE_PA | None | `PrescribedContext` inputs |

The flow and composition groups may repeat the cheap flow sum because their output ownership is separate. A group cannot read its own outlet FLOW as an inlet dependency. Attach extent-flow, conservation residual, nonflowing, and closure validity diagnostics to the composition group. Equilibrium/speciation may be evaluated there with declared inputs when required, but it must not create transported pH or alkalinity.

The initial manifold is isothermal with prescribed pressure: temperature/pressure are context values, not an enthalpy mix silently inferred from inlet temperatures. An energy-conserving mixer would declare those reads and energy balance explicitly. The fixed-yield fast endpoint itself does not need a pH solve to select `f`; `f` is the declared policy, not an instantaneous inferred yield.

Physical wiring is sump recirculation draw → measured stock injection manifold(s) → column liquid inlet → sump return. Stocks add both volume and all declared ions/products. Makeup enters the actual sump boundary. There is no stock injection directly into sump state simply because that is easier to code. The fast manifold policy is optional and explicitly identified: use equilibrium-only mixing if immediate oxidation in this manifold is not an admitted physical assumption. If both manifold and sump use the fast closure, unreacted incoming reagent is transformed only when newly coexisting; reconstruction/initialization is idempotent. The column's film reaction/transfer accounting remains the spec-03/spec-06 contract and must not also add a duplicate bulk sulfide demand.

## 6. Concrete commit sequence and validation

The sequence below is proposed for review, not authorization to edit code. Keep each production slice independently usable and typed. Additive validation can follow in its own passing commit; required contract-validation changes belong with their contract change. Split coherent helpers where possible rather than force a partially implemented public unit to meet a line budget. The event driver is the one substantial new numerical responsibility and should not be disguised as a small projection change.

| Order | Minimal coherent slice | WHY / validation oracle |
|---|---|---|
| 1 | Inventory chart protocol, identity chart, wet inventory state and pure amount/volume balance helper | Establish units and full-carrier transport before chemistry. No-reaction filling/draining, arbitrary opaque channel, exact linear volume balance; zero/negative volume rejected before division. |
| 2 | Fast declaration validation and analytic fixed-yield chart, initialization, reconstruction, directional rule | One canonical amount transformation. `US=0`, rank, pure-S0/pure-sulfate/mixed yields, both reagent-excess branches and equality, initialized/uninitialized reagent coexistence, idempotence, preserved S/Cl/Na, invalid products, untouched channels and explicit renamed bindings. |
| 3 | Geometry and finite-weir hydraulic law | Separate real storage head from capacity. Dimensioned formula, zero head, known head/width scaling, geometry monotonicity; unknown plant parameters remain unfilled. |
| 4 | Variable holdup unit with prescribed draws, weir groups, identity/fast charts and bound slow sources | Prove dependency contract: recycle comp/state root, draw independent of inlet flow, weir state-derived, inlet permutation with declared summation tolerance, draw exceeding inflow drains inventory, source converted concentration→amount once. Standalone derivative/group validation only; do not wire an unguarded dynamic plant. |
| 5 | Typed transient flowsheet RHS result; retain the existing derivative behavior | Expose actual stage streams for accounting without double graph solves. Graph outputs/diagnostics match existing report at the same state; BSM1 float64 trajectory diff remains zero. No `ContinuousUnit.derivatives` signature change. |
| 6 | Guarded joint RK4 trials with pre/post-stage checks, bounded retry and explicit result status | Prevent constitutive evaluation on inadmissible stages. Analytic draining case with a step too large for its intermediate states, invalid source at depleted reactant, nonfinite chemistry, rejected-trial rollback, whole-graph stage updates and partial-time failure. Keep fixed RK4 default unchanged. |
| 7 | Accepted stage quadratures and fast-extent balance ledger | Material accounting uses exactly accepted stages. No contribution from rejected trials; multi-inlet/outlet budgets, internal recycle cancellation, slow+fast source budget, initialization extent counted once. |
| 8 | Event location/restart primitive and wet-level/fast-branch policies | Halving alone is insufficient. Earliest-event selection, event at start, repeated events with bounded status, depletion interlock continuation/failure, fast I-crossing supply case; compare segmented trajectory with independently small-step integration. |
| 9 | Fixed-capacity law, declared mode input, and capacity event policy | Implements complementarity without clipping. The 9→10 m³ two-hour example above, overflow solute budget, net-flow reversal/release, no overflow below capacity, recycle graph stays acyclic. Mode excluded from RK leaves and preserved across calls. |
| 10 | Stateless amount-flow manifold using identity or the same fast closure | Injection is an actual connected material boundary. Two stocks plus recirc, dilution, reagent demand and products, Q=0, temperature context, no timestep dependence, extent-flow units and balance, group read/output coverage. |
| 11 | Small generic integration assembly examples and scrubber wiring specification | Validate holdup→manifold→return recycle with a conservative pass-through column surrogate, then the real column when available. Include every measured stock/makeup boundary. No guessed geometry, stock impurity, temperature or yield. |

The fixed-capacity option can land after the weir path; it is not required to run a physically calibrated finite-weir model. It is required before offering a fixed-level sump. A manifold can land earlier once the closure exists, but its dynamic sump assembly waits for guarded integration. Do not retain an unguarded alternate sump implementation as a compatibility path.

### Acceptance oracles that matter

1. **Transport:** equal-composition filling leaves concentration constant; draining with no reaction leaves concentration constant; a batch with no flows preserves volume; all inventories follow known analytic mixing curves. Concentration-only plots are insufficient to prove inventory conservation.
2. **Variable-volume reaction:** compare a simple integrable first-order source with its analytic inventory/concentration solution; verify the `V` multiplier and no duplicated dilution term.
3. **Fast chemistry:** one mol S(-II) consumes `1+3f` mol FAC and produces `(1-f)` mol S0, `f` mol sulfate, and `1+3f` mol chloride; preserve sodium and carbonate. Check `I<0`, `I>0`, `I=0`, continuous reconstruction, and accepted extent balance under alternate supplies and flow changes.
4. **Overflow/depletion:** volume and each conserved material budget across event timing, including a coarse requested step that crosses capacity or a wet draw interlock. Weir model compared at successively smaller smooth steps; fixed-capacity model compared with exact simple piecewise cases.
5. **Composition/architecture:** field graph compiles the complete recycle without cycles; deliberate bundling of recirc flow with inlet-dependent overflow is rejected. Permuted/renamed compatible carriers preserve locally declared chemistry and pass all untouched channels. No binding conversion/alias loophole is introduced.
6. **Numerical integration:** fourth-order convergence away from branches on smooth test systems; event tolerance controls boundary/budget error; invalid trials are transactional; bounded work returns an explicit non-success rather than an invented final state.
7. **Regression:** zero BSM1 float64 diff against milestone C. This is stronger than the earlier approximate reference tolerance and is achievable because D2 need not alter its physics or numerical path. Run repository-wide lint/type checks and focused behavioral tests during implementation; this design review has not run those checks.

Published bleach-decomposition/titration oracles from spec 06 remain chemistry-network checks, not substitutes for D2 transport/event budgets. D2 introduces no new measured plant facts and no new claim that the fast sulfate endpoint is experimentally established. Those assumptions remain explicit model policies for calibration/review.
