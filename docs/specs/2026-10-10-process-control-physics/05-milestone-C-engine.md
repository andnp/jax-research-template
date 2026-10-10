# Milestone C: phase-aware fields and atomic evaluation groups

Design only. Base implementation work on completed milestone B, especially its bound local kinetics and ReactiveCSTR; the source inspected for this review still included the pre-cutover reactor. Do not resurrect its superseded API. ADR 015's joint ODE remains authoritative: resolve the entire flowsheet separately at every RHS evaluation, then compute all state derivatives together. C replaces the outlet contract, not the integrator.

## 1. Replacement contract

Use four field identities: FLOW, COMPONENTS, TEMPERATURE, PRESSURE. Phase is static port/stream metadata, not a scheduled field. A field reference identifies exactly one named port and one field. Distinguish local references from global references rather than interpreting ambiguous strings.

A concrete static declaration sketch (illustrative Python, not implementation):

```python
class Phase(Enum):
    LIQUID = "liquid"
    GAS = "gas"

class StreamField(Enum):
    FLOW = "flow"
    COMPONENTS = "components"
    TEMPERATURE = "temperature_k"
    PRESSURE = "pressure_pa"

@dataclass(frozen=True)
class PortSpec:
    schema: ComponentSchema
    phase: Phase

@dataclass(frozen=True)
class LocalField:
    port: str
    field: StreamField

@dataclass(frozen=True)
class FieldRef:
    node: str
    port: str
    field: StreamField

@dataclass(frozen=True)
class EvaluationGroup:
    name: str
    outputs: tuple[LocalField, ...]
    reads: tuple[LocalField, ...]
    input_names: tuple[str, ...] = ()
    reads_state: bool = False
    diagnostic_names: tuple[str, ...] = ()

@dataclass(frozen=True)
class GroupResult:
    values: tuple[jax.Array, ...]
    diagnostics: tuple[jax.Array, ...] = ()
```

Register GroupResult as a pytree. Declaration objects and port specifications are static metadata. Values/diagnostics have fixed pytree structure and shapes for a built flowsheet. Diagnostics may have different shapes; their names and expected shapes belong to static declarations. Never use the existing all-data dataclass decorator for static schema/phase/dependency metadata.

The new unit interface has `inlet_ports: Mapping[str, PortSpec]`, `outlet_ports: Mapping[str, PortSpec]`, `evaluation_groups: tuple[EvaluationGroup, ...]`, `input_ports: tuple[str, ...]`, and **an explicit static `stateful: bool`**. Its evaluation method is:

```python
def evaluate_group(self, group_name: str, state,
                   reads: Mapping[LocalField, jax.Array],
                   inputs: Mapping[str, jax.Array], params):
    ...  # returns GroupResult in declaration order
```

The group name is a static Python dispatch value. `reads` contains only declared inlet fields; `inputs` contains only declared group signal names. There are no fabricated complete Streams with zero placeholders. A state-only group has `reads=()` and `reads_state=True`; no upstream edge follows from state access. Input-only and parameter-only groups likewise have no inlet edges. Each output field is owned exactly once. Each declared diagnostic is emitted exactly once by its owner group. Validate names, duplicate ownership, missing coverage, and tuple lengths at construction/tracing boundaries, not through data-dependent Python assertions under JIT.

Keep the current derivative signature with complete resolved inlet Streams, routed inputs, and params, plus projection. This is a transport/unit interface; the milestone-B chemistry binder still restricts local kernel reads. Do not pass every group's auxiliary results into every derivative method merely in anticipation of a future need. A later stateless column can expose its solve diagnostics through its GroupResult without needing derivatives. If a future dynamic unit needs an expensive algebraic closure shared by derivatives, add an explicit typed result-consumption contract then, rather than hiding a cache on the unit today.

Explicit statefulness is essential: the current `stateful_nodes` implementation excludes only Mixer and Splitter by concrete type. That would incorrectly demand a state entry for a stateless packed column. Use `unit.stateful`, and validate exact state keys against that declaration. Stateless groups receive None; stateful units can have groups that do not read state. This does not change ADR 015's state model.

### Today's declarations

Keep field groups separate unless a shared numerical computation actually requires joint evaluation. Grouping by identical dependency sets is not an identity: two groups with the same reads can perform different calculations.

* **Mixer:** flow group owns out.FLOW, reads each inlet.FLOW in the existing inlet tuple order. Composition group owns out.COMPONENTS, reads all inlet FLOW and COMPONENTS. It uses the current weighted matrix multiplication expression. A prescribed-context group owns out.TEMPERATURE and out.PRESSURE and reads no inlet fields. Do not group flow and composition: otherwise calculating a downstream flow would unnecessarily require recycle composition.
* **Splitter:** one draw-flow group owns all named draw FLOWs and reads only its draw signals; the remainder-flow group owns remainder.FLOW and reads in.FLOW plus draw signals. The latter owns the deficit diagnostic, including refused negative requests, matching the current formula. One composition group owns every outlet COMPONENTS and reads only in.COMPONENTS. Separate temperature-forward and pressure-forward groups own the respective fields across every outlet and read only in.TEMPERATURE or in.PRESSURE. Draw flows must not acquire a feed-flow dependency. Preserve current uncapped requested draws and reported deficits; the separate `streams.split` helper's capped allocation is a different policy.
* **LayeredSettler:** one flow group owns effluent.FLOW and underflow.FLOW, reads feed.FLOW and q_underflow, and preserves current clipping/subtraction order. One composition group owns both COMPONENTS fields. With FEED_FRACTIONS it reads feed.COMPONENTS and state; with LAYER_COMPOSITION it reads state only. These are distinct static declarations selected when the unit is built. Its prescribed-context group owns both outlets' T/P fields with no inlet reads. Replace the derivative's internal call to `outlet_flows` with a small canonical hydraulic calculation shared by derivative and group; retain its arithmetic. Cheap recomputation is acceptable; a second independently implemented formula is not.
* **ReactiveCSTR after B:** out.FLOW reads feed.FLOW; out.COMPONENTS reads state only. Prescribed-context T/P group reads the context temperature signal and prescribed pressure parameter, with no feed dependency. The derivative continues to use B's bound reaction/exchange sources and its existing biological temperature boundary. No oxygen-specific stream semantics enter the scheduler.

The group declaration is the union of dependencies of the jointly solved outputs. A column group can later own several fields on both gas/liquid outlets and declare the actual inlet fields required by its coupled solve. It is evaluated once, not once per field or outlet. Splitting an intrinsically coupled solve into fake independent groups to evade a cycle would be incorrect.

## 2. Stream representation and closures

Make Stream a pytree with dynamic `flow`, `components`, `temperature_k`, and `pressure_pa`; static `schema` and `phase`. Both port compatibility and external-stream validation compare schema **and** phase. T is absolute kelvin; P is absolute pascal. Liquid flow remains m³/h and its existing component bases remain unchanged. Gas flow is total mol/h; its component vector is a complete molecular mole-fraction vector with an explicit mol/mol species basis. Gas schema validation must reject COD/equivalent/proxy coordinates and aggregate element pools as molecular species. A dry-gas representation must be identified as such; it cannot silently count as complete wet gas when water transfer is modelled.

Do not perform automatic phase conversion or basis conversion when connecting ports. Milestone B's basis distinctions remain authoritative. Gas O2 mole fractions are not liquid s_o concentrations. A future interphase source binds explicit local roles to the appropriate carriers; C must permit different phase/schema specifications on different ports without requiring identical schemas for all ports of a unit.

T/P are resolved context, not proof that energy/hydraulic balances were solved. Make the closure an explicit unit construction choice. The minimal BSM1 policy is prescribed isothermal temperature and prescribed absolute pressure (101325 Pa is a chosen benchmark context, not a calibrated plant pressure). Add one context temperature signal in K, derived from the **effective** benchmark temperature already selected by its TemperatureModel. Retain the existing temperature signal/finished-B conversion path for biological calculations. In particular, seasonal temperature may differ from raw tabular-profile temperature: stamp the selected effective temperature at the benchmark boundary, not just the raw profile temperature returned by `tabular_influent.influent`.

Current BSM1 reactor and measurement paths consume temperature signals; they do not consume Stream temperature. Do not reroute these calculations through Stream.temperature_k in C. Thus adding T/P leaves need not change any material or ODE arithmetic. Supply context explicitly at production Stream construction sites. Avoid a universal implicit 298.15 K default: it would conceal a missing boundary condition. The public tabular-influent function can stamp its profile temperature, with the benchmark replacing that context after applying its configured temperature model; document that boundary clearly.

Mixing rules:

* Same-phase liquid with the same schema retains its present flow/concentration calculation, including the existing arithmetic mean of concentrations when total flow is zero. Preserve the exact `stack`, `sum`, `where`, and weighted matrix multiplication order. T requires an explicit closure: prescribed context is sufficient now. A later sensible-enthalpy closure must declare density/heat-capacity assumptions and its dependencies; flow-weighted T is justified only under the corresponding assumptions. Never average pressure as though it were an extensive quantity.
* Same-phase gas with the same complete species schema mixes species molar amounts: total F = sum F_i and y = sum(F_i y_i)/F. For F=0 choose a documented valid composition convention, such as the arithmetic average of normalized inlet fractions; no division by zero or implicit extra species. Do not renormalize every nonzero result as an unrelated numerical correction. Gas temperature likewise requires a prescribed or enthalpy closure; pressure is prescribed by a boundary or supplied by a declared hydraulic closure.
* Cross-phase mixing is rejected. A unit that performs absorption, condensation, flashing, or dissolution has separate phase ports and an explicit physical model.

Splitters forward context per field. Prescribed reactor/mixer/settler contexts provide upstream-independent anchors in BSM1, so context propagation does not introduce recycle equations. Do not default every generic unit to inheriting all inlet temperatures/pressures: that creates artificial context cycles in otherwise explicit flowsheets.

Static schema/phase requirements are construction checks. Numeric validity (positive T/P, nonnegative gas flow, normalized fractions) needs a defined boundary-validation/diagnostic policy compatible with JIT; it must not introduce clipping/renormalization into BSM1's arithmetic. No new general numeric validator is required to make the scheduler function.

## 3. Compilation, resolution, reporting

Build a plain-Python plan once in `build_flowsheet`:

1. Validate ports, one connection per inlet, existing connection/consumption rules, phase/schema equality, signal routing, group declarations, and full output-field coverage.
2. Map each global output FieldRef to its owner `(node, group_name)`. Translate each group's declared local inlet read through the connection map to an external field or a producer FieldRef.
3. Retain those labelled field edges for diagnostics; topologically order **atomic groups** using the owner map. A group becomes ready only when every declared inlet field is available. External fields and state/signals/params are roots. Collapse repeated producer-group edges for indegree accounting while retaining the field labels.
4. Reject cycles, including self-dependencies, with a witness naming the participating node/group and field edges. Report when grouping imposes a dependency that could be removed by a valid narrower declaration. Never automatically split a coupled group or attempt a global nonlinear solve.

Calling this a field DAG must not obscure atomicity: a field-only topological order is insufficient if it asks for one result of a solve before that solve's other dependencies exist. Cycle validation must cover the executable group graph. Field labels explain the cycle; atomic group ownership determines execution. Static prescribed context and state-only composition are explicit roots, not magical tear-stream rules.

Each `_resolve` invocation allocates local dictionaries/tuples for field values and group results. Iterate the compiled group order once. Gather only the declared reads and routed group inputs, call evaluate_group once, install every output value atomically, and save its diagnostics. Assemble complete Streams after resolution. This is the only resolution implementation: derivatives, report, and standalone `unit_outlets` reuse the same declaration/evaluation rules. The standalone adapter gathers declared fields from complete provided inlets; it must not retain old outlet callbacks.

The cache lifetime is exactly one resolution. No instance/global cache, no equality/hash of array values, and no reuse across RK stages, time steps, changed inputs, or report calls. Node identity matters even when several nodes share the same unit instance. Group identity matters even when read sets are equal. For a later packed column, one group result feeds both outlet Streams and report diagnostics within this invocation.

Preserve the existing report's outlet/stream/deficit access while adding a generic named diagnostic mapping, e.g. `(node, diagnostic_name) → array`. Derive splitter deficits from the saved remainder group's diagnostic; no separate outlet re-solve. Report resolves fresh at the supplied state/time. External streams are roots whose four data fields are available without a unit callback. Reporting fields does not add graph edges back into physical computation.

Use static Python tuples and loops; JAX traces numerical group computations once per scheduled node/group. Do not make a dense runtime field adjacency matrix, run graph algorithms in JAX, or issue one numerical callback per field. Small static dictionaries are trace-time routing, not runtime data structures. Keep declaration validation out of the repeatedly traced RHS. A few groups per unit should be comparable to the existing two-pass engine; a coupled column must remain a single solve, with a compact iteration primitive internally rather than a Python-unrolled iteration history. Measure trace/compile size if C actually regresses it; do not preemptively introduce a second execution engine.

## 4. Minimal commit sequence

Finish and validate B first. Commit counts below are approximate additions plus deletions, excluding unavoidable generated artifacts; respect independently passing states rather than manufacturing 20-line intermediate broken contracts. Pure additive declaration/compiler primitives may precede cutover. The old and new live unit/scheduler implementations must not coexist after cutover.

1. **Static field vocabulary** (~20–35 lines): Phase, StreamField, LocalField, FieldRef. WHY: establish unambiguous identities without changing runtime streams. Keep module ownership obvious; no generic graph framework.
2. **Port/group/result declarations** (~45–75 lines): PortSpec, EvaluationGroup, GroupResult registration and declaration validation. WHY: express coupled ownership and exact reads before integrating units. No unused alternate production resolver.
3. **Compiler primitive** (~70–120 lines): compile owner-labelled field dependencies into a deterministic atomic-group order with cycle witnesses. WHY: isolate the difficult graph invariant from physics and JAX. It consumes explicit declarations/connections, not the old Outlet API.
4. **Compiler contract tests** (~70–110 lines): acyclic recycle with state-only composition; a real same-field cycle; atomic grouping that changes readiness; duplicate/missing ownership; same reads but different group identities; external roots. WHY: validate execution semantics, not the chosen DFS implementation or arbitrary topological order.
5. **Single atomic production cutover** (~350–600 lines; verify actual diff before agreeing final split): Stream leaves/static phase; all public Stream constructors and helper results; new ports/groups/evaluators for Mixer, Splitter, settled-B reactor and LayeredSettler; one resolver/compiler integration; explicit statefulness; unit_outlets; report diagnostics; remove Outlet/dependency sets and old callbacks/cache/passes. Include activated-sludge assembly/context routing, tabular/benchmark effective-temperature boundary, all owned direct callers, test units/fakes, and required affected test fixes. WHY: replace the contract without broken intermediate users or parallel legacy behavior. Do not migrate standalone legacy ASM/physics functions that never implement the flowsheet contract except where an actual Stream constructor is affected.
6. **Phase/group behavior coverage** (~80–130 lines): gas molar mixing including zero flow; phase rejection; prescribed context; split field forwarding; exactly one group evaluation per resolution; new invocation evaluates again; stateless coupled two-outlet fake/real lightweight unit; report diagnostics correspond to its same result. WHY: protect externally meaningful contracts. A callback counter belongs in a focused non-JIT test; don't assert Python callback counts as runtime JIT execution counts.

If primitives require many imports/re-exports, simplify their placement instead of growing a framework. A pure static context-policy primitive can be folded into declaration commit 2; keep only the policies actually used in C. The atomic cutover is intentionally larger because required constructor and protocol changes cannot leave callers broken. Do not leave an adapter translating old Outlet objects into new groups or retain an old scheduler behind a flag.

### Numerical preservation gates

Before C's runtime cutover, capture completed-B float64 behavior on the same backend, JAX configuration, solver settings, seeds, input data, and controller/noise sequence. Compare post-cutover flow/components, RHS, one joint RK step, open/closed-loop trajectories, controller actions, and published reference metrics. First demand exact equality of existing material outputs in focused pre/post cases; report any differing values and maximum absolute/relative errors rather than claiming bitwise from matching schemas. Full reference reproduction must also meet the requested ~0.1% where the reference and protocol support it. Existing reference-test tolerances are looser (some open/closed metrics permit roughly 0.5%/1%, steady-state 0.2% plus absolute tolerance); a green suite alone cannot establish 0.1%. Record actual pre/post and reference deviations. An already-failing pre-C metric is a baseline limitation, not permission to silently relax C's gate.

Preserve these expressions and orders:

* Mixer inlet tuple order, stacked sum and weighted matrix multiplication, including zero-flow branch.
* Splitter Python `_total` reduction with its current scalar-zero initial value, draw order, remainder and refused-negative deficit calculation; no switch to a vector reduction or capped helper.
* Settler clipping, subtraction, feed fractions and TSS arithmetic, layer ordering, advection/settling combination.
* Completed-B derivative accumulation order, saturation calibration, projections, and named-source binding order; C must not add thermal corrections or change concentration basis.
* Joint RK4 stage construction, projection placement, benchmark substeps, controller timing, noise, and temperature selection.

Different scheduling order alone should not change independent pure calculations, but rewriting reductions or regrouping arithmetic can. New leaves can also alter compilation/fusion, so verify rather than promise universal bitwise behavior across compilers/platforms. Do not compare C against the older pre-B reactor and attribute B changes to C. No experiment launch is needed to develop the static compiler; reference verification should use the project's prescribed benchmark/test workflow.

## 5. Deliberate scope boundary and the column

C implements explicit field routing and phase/context representation. It does not implement energy storage/balances, heats of reaction, general gas equations of state, pressure-drop/hydraulic solving, valves, transfer correlations, packing geometry, gas holdup, or a gas unit. Gas mixing semantics are a representation contract; no generic scrubber performance claim follows from them. It does not solve algebraic recycle cycles or select tear streams.

The later packed column needs C to support: distinct gas/liquid schemas and phases on its ports; a stateless classification; a single evaluation group owning coupled outputs across both phases; exact field/signal dependencies; outlet T/P with explicit closures; and diagnostics from that same solve. It can retain prescribed isothermal T/P initially. Field ownership does not require every output to depend on every possible inlet field; declare the actual solver's reads. Its chemistry uses B's explicit local bindings and bases, not dense positional array guesses.

A packed-column group may own gas/liquid compositions and flows together when interphase transfer changes both phase totals. If dilute approximations hold flows prescribed, keep those flow groups independent only when the mathematical model actually permits it. A truly coupled recycle then requires a later explicit algebraic-solver design; C should reject it clearly, not make an expensive column solve appear state-only. Future dynamic holdup uses ADR 015's joint state and stage-local resolution, not caches between stages. These boundaries are sufficient for the column without prebuilding an energy or pressure network in C.
