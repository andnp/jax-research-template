# Finite stiff chemistry inside the canonical guarded advance

Design only, 10 October 2026. Extends [spec 08](08-guarded-integrator.md) after its canonical-integrator cutover; it does not reopen commits A–F or implement chemistry. The chlorine session owns mechanism selection and rate constants. This design owns the numerical and model-reduction boundary.

**Recommendation:** retain finite chloramine formation and breakpoint chemistry in chlorine v2 initially. Add one statically selected **RODAS3 embedded 3(2) trial kernel** to the same joint advance algorithm, using a dense exact JAX Jacobian first. All tanks, connected dynamic units and conservative ledgers advance together. Share guards, retry, rollback, events, modes and statuses with RK4. BSM1 continues to select its unchanged explicit kernel and fixed nominal substeps. Do not introduce a chloramine-unit solver, frozen-neighbor substepping, automatic solver fallback, or an instantaneous breakpoint shortcut.

## 1. Fast reduction is a physics decision

A rate being fast relative to a two-minute control interval does not establish an instantaneous endpoint. The external interval is an observation/control schedule; the canonical integrator may take smaller **joint** internal segments. Reduce only when the fast subsystem has a specified conservative endpoint/manifold, is asymptotically justified over the admitted operating domain, and has a negligible unresolved transient for the outputs of interest. Validate against finite integration across initial mixtures, depleted pools, pH, temperature, dosage changes and hydraulic residence times. Record the physical approximation and its domain, not a threshold based on the numerical timestep.

The isolated irreversible molecular reaction

```text
HOCl + NH3 -> NH2Cl + H2O
```

is balanced and has a fixed endpoint. Project it onto independently transported FAC, uncombined ammonia-family nitrogen and monochloramine amounts, with consistent stoichiometric mole bases. Its reduced direction is `(-1,-1,+1)`. For inventories `(F,A,M)`, an example chart is `I=F-A`, `J=M+A`; reconstruction is `A=max(-I,0)`, `F=max(I,0)`, `M=J-A`. Initialization extent is `min(F,A)`. Retain all unaffected strong ions/families and validate products. Acid-base equilibration changes molecular HOCl/NH3 fractions; it is not a reason to transport HOCl and NH3 alongside their analytical totals. Water/proton bookkeeping still belongs in balanced molecular records and the independent-basis reducer.

Thus this **isolated one-way pathway is mathematically reducible** by the existing fixed-stoichiometry closure. That does not make it the correct chlorine-v2 default. Hydrolysis/back reaction, further chlorination, acid-catalysed pathways, competing demands and breakpoint nitrogen products can change transient partitioning. If two fast reactions consume the same reagent, separate greedy closures are order-dependent. A breakpoint curve is not a unique instantaneous endpoint chosen by chlorine:ammonia ratio. A reversible fast subsystem, if reduced, needs an equilibrium or quasi-steady manifold with its own uniqueness/domain/conservative reconstruction; it is not this one-way limiting-reagent closure.

Use fast monochloramine formation only as a separately named reduced model after showing that competing pathways are slow on its relaxation scale, or after deriving a joint fast selection policy. Reagent-exhaustion branches and acid-base dependence must be checked throughout the domain. Keep dichloramine formation, decay and other admitted finite pathways in the same source network unless their own reductions have been justified. Never retain a reaction both in the fast closure and in finite kinetics.

The chemistry sources support this caution: Wahman's application is explicitly kinetic, based on Jafvert–Valentine and Vikesland mechanisms, and its example already produces both mono- and dichloramine during initial formation. It holds pH constant and uses batch/plug-flow-equivalent hydraulics; those assumptions cannot be silently imported into a dynamic charge-balanced tanks-in-series plant. [Wahman (2018), full paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6301018/).

## 2. Method selection and provenance

Select the **original Sandu et al. RODAS3**, not ROS3P and not the newly named RODAS3.1. It is an L-stable, stiffly accurate embedded order-3/order-2 pair with four linear solves and three RHS evaluations. The same matrix serves all solves. Sandu et al. provide its coefficients and compare chemistry solvers; their operator-split atmospheric experiments support a method choice, not permission to freeze neighbors in this engine. [Sandu, Verwer, Blom, Spee, Carmichael & Potra (1997), §4 and coefficients](https://ir.cwi.nl/pub/125/0125D.pdf).

| Candidate | Decision |
|---|---|
| ROS2 | Useful small verification method, but lower order makes it an unnecessary additional production kernel for this first extension. |
| ROS3P | Not selected: Lang–Verwer's method is A-stable, with a nonzero infinitely stiff stability limit (about 0.73 in magnitude). Its parabolic order-reduction design does not provide the strong fast-mode damping sought here. |
| RODAS3 | First production stiff kernel: L-stable 3(2) pair, moderate solve count and auditable coefficients. |
| RODAS4 | Potential later choice if measured accuracy/work demands it; six stages and a 4(3) estimator add cost. Do not implement a solver catalogue now. |

The ROS3P distinction is from [Lang & Verwer (2001), §§1 and 5](https://ir.cwi.nl/pub/1102/1102D.pdf). The Rosenbrock framework and RODAS family are described in Hairer & Wanner, *Solving Ordinary Differential Equations II*, second edition (1996), chapter IV; the authors also publish their [stiff solver descriptions and reference RODAS code](https://www.unige.ch/~hairer/software.html). Stable-method names alone do not certify positivity or accuracy for this network.

Use the primary paper's mathematical coefficients and an explicitly pinned reference implementation to verify any transformed table. Do not copy a rounded documentation table for a different ROS3/ROS4 method. The [KPP maintainers' RODAS3 documentation](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html#rodas-3) provides the transformed convention below. Its Jacobian typography must not be interpreted as a transpose of the engine's conventional `J_ij = d f_i/d x_j`.

## 3. Joint trial kernel

### State and RHS

Flatten only continuous dynamic plant leaves into a static-layout float64 vector `x` for the stiff kernel. Preserve the original pytree for guards, field resolution and returned state. Modes, inputs, static schemas, statuses, sensor/controller states advanced outside the plant interval, and accumulated ledgers are not ordinary physical RK leaves. Do not flatten BSM1's explicit kernel or alter its existing pytree arithmetic.

Define `f(x)` by unflattening, calling the canonical transient `evaluate_rhs`, and flattening its derivative. Every evaluation resolves the entire field graph from that trial's state. For constant-volume tanks in series,

```text
C_j_dot = (Q_j/V_j) * (C_upstream,j - C_j) + S r(C_j, context_j)
```

is part of the joint derivative. Upstream concentrations change within the same segment. For variable-volume/fast-chart units, use their declared inventory equations and reconstruction. The Jacobian includes upstream transport, recycles, algebraic manifold dependencies, and any state-dependent speciation/property effects. No tank-local Jacobian with all neighbors frozen is substituted.

Initial stiff scope is an autonomous segment: external inlets, ordinary signals and discrete modes are held as in spec 08. State-dependent pH is still inside `f`. If a future source varies explicitly with segment time, include the method's `f_t` terms and stage times, or extend the state with a correctly treated clock; do not use an autonomous table unchanged for a time-dependent forcing model.

### Transformed stage convention

For step `h`, form `J = d f/d x` at the accepted origin and `D = I - h gamma J`. With increments `k_i` in state units, solve

```text
X_i = x + sum(j<i, a_ij k_j)
D k_i = h gamma f(X_i) + gamma sum(j<i, c_ij k_j)
x_trial = x + sum(i, m_i k_i)
error = sum(i, e_i k_i)
```

This is equivalent to the `I/(h gamma)-J` formulation. Factor `D` once per trial and reuse for its stages. Factorization/increments are transient and disappear on rejection or a mode change. A failed/nonfinite solve or unacceptable residual invalidates the entire trial; never accept the explicit result as a fallback.

For original RODAS3, use `gamma=1/2`; rows of the strictly lower triangular transformed matrices are:

```text
a21=0;       a31=2, a32=0;       a41=2, a42=0, a43=1
c21=4;       c31=1, c32=-1;      c41=1, c42=-1, c43=-8/3
m=(2,0,1,1); e=(0,0,0,1)
```

The second stage reuses the first physical RHS because its evaluation state is identical. It still needs its own solve. Each distinct physical stage receives the same pre-domain and post-evaluation validity checks as an RK4 stage. The candidate endpoint requires its own check; neither the last evaluation nor the embedded approximation is automatically a substitute. Guard the physical stage locations `X_i`, not the increments `k_i`, which can legitimately be signed.

### Kernel boundary inside one advance algorithm

Generalize spec 08's trial responsibility to a static `TrialKernel` selection. A trial returns candidate state, optional embedded error plus its order metadata, integrated ledger increments, and validity/reason diagnostics. The driver alone owns acceptance, elapsed time, budgets, projection, context transitions and ledger commit. The kernel uses supplied guarded stage evaluation, so checks are not reimplemented by each method. Stage counts are kernel-static; no requirement that every kernel pretend it has four RK4 samples.

Select kernel **once for the joint plant**, via static integration configuration. The stiff plant uses RODAS3 for transport and all finite sources together. Do not dispatch different kernels by unit or change solver based on an instantaneous stiffness guess. Empty/error-disabled configuration keeps the RK4 expression sequence, nominal `dt/substeps`, outer static loop and projection placement specified in 08 exactly intact. No stiff flattening, Jacobian or tolerances are applied to BSM1 by this extension.

## 4. Error control in the shared driver

Use embedded control for the chlorine plant; fixed substeps alone can be stable yet inaccurate. Its two-minute control interval remains a requested endpoint. Existing nominal substeps may define maximum segment windows and projection boundaries; inside each window the same bounded driver chooses and retries stiff durations. Initial `h`, maximum/minimum `h`, attempt/segment budgets and tolerances are explicit. Do not confuse `substeps` with a separate local reaction solver.

For RODAS3's 3(2) pair, the difference estimates a local quantity of order `h^3`. Use component scales

```text
scale_i = atol_i + rtol_i * max(abs(x_i), abs(x_trial_i))
E = max_i(abs(error_i)/scale_i)
accept accuracy iff E <= 1
h_proposed = h * clip(0.9 * max(E, error_floor)^(-1/3), 0.2, 2.0)
```

This conservative max norm and bounded growth are proposed engine policy, not a quotation of a particular solver's defaults. Guard failure forces reduction independently of `E`; it cannot be overridden by a small error. After rejection prohibit immediate growth. A nonfinite estimator fails the trial. If tolerances/budgets cannot be met, return partial time/status; never silently proceed to the requested two-minute endpoint.

Absolute tolerances carry each state coordinate's units. Inventory, volume, concentration and signed invariant coordinates need different scales. For a fast chart, also map embedded endpoint differences through its reconstruction within the same branch to check important physical pools; small invariant error can otherwise hide cancellation in a small residual species. Branch crossing is an event, not a smooth error estimate. Trace intermediates and monochloramine/dichloramine outputs need tolerances appropriate to the observation target. Do not let a large irrelevant carrier coordinate drown their error in a global RMS norm.

Fixed stiff substeps can be offered as an explicitly error-disabled convergence-study configuration of the same driver. They are not the recommended production choice. RK4 has no embedded estimator in this design and retains its approved fixed schedule; do not add step doubling to its default path. Step-control decisions are transactional just like guards. Persistent next-step suggestions belong in numerical context, while physical modes remain distinguishable; reset suggestions at forcing jumps/events. Event-location trials use fixed pre-event mode and the selected kernel; they must meet validity/accuracy requirements before an event segment is accepted.

## 5. Conservation, positivity, and the ledger extension

For a closed system with a constant linear invariant `L f(x)=0` throughout a smooth domain, the exact Jacobian satisfies `L J=0`. Multiplying each stage equation by `L` shows recursively `L k_i=0`; hence `L x_trial=L x` in exact arithmetic. In floating arithmetic preservation is limited by RHS/Jacobian consistency and solve roundoff/residuals. This does not cover nonlinear concentration-times-variable-volume invariants, which is why generic wet holdup stores inventories. Nor does it certify open-system accounting with a separately approximated discharge.

L-stability and linear invariants **do not guarantee nonnegative stages/endpoints**. Keep positivity/domain guards and reject/reduce the entire joint trial. No independent species clamp or arbitrary matrix regularization repairs it. A zero reactant with a physical zero destruction rate must have a finite differentiable rate law; guard exhaustion at a boundary returns failure, not a claim that Rosenbrock makes that boundary admissible. Irreversible integrated reaction extents also need their stated sign/budget checks.

### Do not reuse RK4 ledger weights

Rosenbrock's state increment includes Jacobian corrections. Simply summing physical rates with `m_i` or using RK4 `(1,2,2,1)/6` weights generally does **not** yield a matching open-system material ledger. Treat each ledger `q` as a passive augmented ODE `q_dot=g(x)`. Its Jacobian block is

```text
J_aug = [ J  0 ]
        [ G  0 ]       G = d g/d x at the origin
```

No extra dense factorization is needed. Given physical increment `k_i`, the ledger increment `l_i` obeys

```text
l_i = h gamma g(X_i) + gamma sum(j<i, c_ij l_j) + h gamma G k_i
Delta_q = sum(i, m_i l_i)
```

The bottom-block multiplication `G k_i` can be obtained by a JVP of the ledger-rate extractor, avoiding a huge dense ledger Jacobian. Augmented ledger values do not feed back into chemistry. This passive augmentation supplies the same Rosenbrock discretization to state and budgets and preserves linear combined invariants, such as inventory minus cumulative net supply, up to solve precision. Rate extraction includes the same resolved stage flows as the RHS, and its derivatives include their state dependence.

This is the one material extension to spec 08: **the trial kernel returns method-consistent ledger increments; the driver commits them**. For RK4 those increments remain the exact existing accepted-stage quadratures. For RODAS3 use the augmented construction. Separate projection/event impulses and initialization extents remain as in 08. The fast-extent balance formula remains usable with the method-consistent integrated physical nonfast ledger. Do not reapply computed extents to the state.

If the system has prescribed sources/sinks or vents, full elemental invariants require their explicit augmented ledgers. Chlorine and nitrogen are not conserved merely within measured residual FAC/mono-/dichloramine pools: admitted chloride, nitrogen products and boundary losses must be accounted for. An empirical demand channel cannot claim molecular balance unless its products/boundary are represented.

## 6. Jacobians, solves and rate-law requirements

For a small tanks-in-series plant start with exact forward-mode JAX differentiation of the **joint flattened physical derivative** and dense LU plus four backsolves. Cost is O(N²) storage and O(N³) factorization, with forward Jacobian work comparable to N tangent directions; actual XLA batching changes constants. This is a reasonable first implementation for tens to a few hundred dynamic coordinates, subject to measured memory/latency. Avoid a blanket performance promise for large basin counts. [JAX `jacfwd`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html).

`linearize` supplies a reusable linear action at an origin; JVPs can construct Jacobian columns or support future matrix-free solves. It does not make a sparse factorization free. Prefer dense exact J initially; no stop-gradient approximate Jacobian masquerading as the specified Rosenbrock method. [JAX `linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html).

Serial tanks often yield block-lower-triangular transport coupling plus local chemistry blocks, but recycle/algebraic dependencies can fill other blocks. The field dependency graph is not by itself the derivative sparsity pattern. A future verified dependency pattern and block solve can reduce cost; omission of upstream Jacobian blocks changes the method. Matrix-free Krylov or approximate-J W-method variants need additional order/linear-residual review and are outside the first kernel. Closed chemical conservation makes J singular; `I-h gamma J` can still be nonsingular. Do not delete conserved coordinates merely to make J invertible.

The chemical network must supply:

- Balanced projected stoichiometry and finite extent rates on explicitly named totals/context, with consistent per-hour units. A seconds-based elementary constant is converted once by the network adapter.
- A locally smooth differentiable RHS within each admitted branch, including derivative paths through acid-base speciation. Differentiating discrete bisection choices in a pH root finder may produce a misleading Jacobian. Use a validated implicit derivative of charge balance or an equivalent differentiable closure, with domain/convergence checks. Verify directional derivatives against independent finite differences away from boundaries.
- Validity predicates for rate/property ranges, finite derivatives at depleted species, explicit handling of any intermediate denominators and empirical cutoffs. No clipping inside rates to conceal invalid stages, no unsupported zero-rate fallback, and no huge artificial rate replacing a fast closure.
- Explicit piecewise/event declarations for real discontinuities. Smooth policy-dependent rates and equilibrium closures remain inside the joint derivative; masks/statuses stay outside the differentiated physical RHS.

JIT/vmap use fixed layouts and bounded driver work as in 08. Invalid/inactive lanes need safe arguments before Jacobian and linear solves, not just before primal rate evaluation. Smooth-trajectory gradients through exact J and solves can involve second derivatives of the RHS; stopping J's gradient changes the numerical sensitivity. Control/event schedule gradients retain the limitations stated in 08. Do not claim a validated stiff sensitivity implementation from successful primal integration alone.

## 7. Chloramine mechanism provenance and benchmark scope

Use [Jafvert & Valentine (1992), *Reaction scheme for the chlorination of ammoniacal water*, ES&T 26, 577–586](https://doi.org/10.1021/es00027a022) for the reaction-scheme provenance; [Vikesland, Ozekin & Valentine (2001), *Monochloramine decay in model and distribution system waters*, Water Research 35, 1766–1776](https://doi.org/10.1016/S0043-1354(00)00406-1) for decay-model provenance; and [Wahman (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6301018/) for an accessible implementation/validation account. Do not conflate measured available chlorine, molecular chlorine-atom inventories and ammonia expressed as mg N/L. The chemistry session must specify conversions, temperature/pH assumptions and which pathways/products it actually includes.

The engine does not need a chloramine-specific stepping interface. Each tank binds the same network source to its local carrier, and its kinetics/transport enters one global `f`. Tank count and residence-time distribution are independent hydraulic model choices. Wahman's batch examples validate chemistry under their assumptions; they do not validate tanks-in-series mixing or dynamic pH. Validate constant-pH batch first, then nonreactive residence-time transport, then coupled reactive tanks. Disinfection/contact-time observables must be integrated on the accepted trajectory if they depend on transient residuals; final chlorine times two minutes can miss rapid formation/decay.

## 8. Additive commits G+ after A–F

**Do not change F's cutover scope or delay it for chloramine chemistry.** A–F establish the single guarded advance and RK4 migration described in 08. This extension is additive after F, with one shared driver; naming that driver `advance` rather than `advance_rk4` should be chosen when implementing F because the public responsibility is advancement. No second public `stiff_advance` API is introduced.

| Commit | Coherent responsibility | Estimated production size / validation |
|---|---|---|
| G | Static trial-kernel result/selection boundary; move existing trial unchanged behind it | 30–60 changed lines; zero BSM1 float64 diff, same nominal/projection order. No new solver used yet. |
| H | Stiff-only continuous-state layout, exact joint Jacobian and reusable dense solve helper | 50–90 lines; real small coupled systems, finite-difference J checks, solve residuals, guards before invalid Jacobians. |
| I | Original RODAS3 coefficient record and pure joint trial kernel | 70–110 lines; coefficient provenance, scalar stability/damping and third-order smooth convergence; stages guarded, no unit-local advancement. Pure trial helper until acceptance machinery is ready. |
| J | Method-consistent passive ledger augmentation | 40–80 lines; linear combined invariants for state plus boundary ledgers, recycle cancellation and failed-trial rollback; RK4 ledger unchanged. |
| K | Embedded error policy and selection in the same bounded driver | 50–90 lines; accuracy rejection, step growth cap, partial-time/budget failures, event trials use selected kernel. Activate RODAS3 only with J available. |
| L | Joint stiff event/vectorization/gradient acceptance tests | Primarily tests; capacity/depletion and stiff source together, I-branch event if reduced chart used, heterogeneous vmap lanes, no-event sensitivity checks. |
| M | Chlorine-v2 finite-network tank assembly and benchmark validation | Separate chemistry/benchmark session owns this; bind sources and explicitly select RODAS3/error policy. No rate constants are specified here. |

Additive tests may follow independently usable helpers; all tests needed for activating the kernel accompany that activation. Estimated sizes exceed 20 lines where one numerical formula/contract cannot safely be split; extract pure helpers before the activation commit rather than maintain two drivers. Existing F regression fixtures remain fixed; do not regenerate expected BSM1 values.

Acceptance oracles: zero BSM1 float64 diff before/after kernel abstraction; scalar `x_dot=-lambda x` with large negative `h lambda` demonstrates stiff damping and convergence; Robertson-type balanced finite chemistry or a small known stiff reversible network checks invariants and tolerance refinement; nonreactive serial-tank transport checks joint coupling; independent high-accuracy batch/tank solutions check transient FAC/mono-/dichloramine and breakpoint/decay outputs under stated chemistry; accepted state-plus-ledger budgets close; positivity failure rejects without clipping; events, retries and failed lanes do not contribute material. A slow-only control experiment must show that finite stiff integration converges to the same physical model as sufficiently fine joint explicit integration, while a proposed fast reduction is separately tested for approximation error.

No code, coefficients implementation, rate constants or runtime results were changed/generated in this design review. The primary references justify the method family and mechanism provenance; the proposed engine policy still requires implementation and validation.
