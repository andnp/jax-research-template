# Scrubber liquid chemistry: reactions, independent bases, stocks, and assembly

Design only, 10 October 2026. Read against the landed `components.py`, `binding.py`, `kinetics.py`, `units/reactive_cstr.py`, `units/exchange.py`, and `chemistry/{thermo,species,aqueous,speciation,solubility}.py`. This extends [the transfer/holdup design](03-transfer-column-holdup-orp.md), [milestone B](04-milestone-B-binding.md), and [the proposed milestone-C engine](05-milestone-C-engine.md). It does not implement chemistry or change BSM1.

The recommendation is to compile balanced molecular reactions into an explicit independent liquid basis, then use that same compiled chemistry for slow kinetics, film extents, and a separately declared instantaneous bulk closure. Transport analytical totals and spectator ions, derive pH, and measure stock composition. Do not integrate the elementary sulfide/chlorine coefficient in the existing RK4 reactor. Do not select sulfate solely because pH exceeds a threshold. Do not apply concentrated-bleach decomposition kinetics to sump liquor without demonstrating its domain.

## 1. What the landed code provides and what remains missing

`ComponentBasis` distinguishes species, elements, COD, equivalents, and proxies; `Binding.resolve` requires exact bases and explicit names where renamed. These are sufficient for local reaction-source binding. A basis reference such as `S` identifies the element being counted; it does not identify oxidation state or family. The chemistry assembly must explicitly bind the sulfide, sulfate, and elemental-sulfur pools rather than infer them from that reference.

`KineticsModel.rates(reads, temperature_k, params)` returns declared concentration sources per hour. `KineticSource` gathers/scatters and keeps model/schema/binding static, params dynamic. This is the right adapter for finite slow reactions. `ReactiveCSTR` is constant-volume, transports concentrations, adds all sources on the same state, and has an injected projection. Its default projection is identity. It is not the variable-volume sump, an invariant-state reactor, or a finite gas-reservoir column. `ExchangeSource` models a prescribed external reservoir and is likewise insufficient to represent two conservative phase outlets.

The registry already contains Na+, Cl-, H+/OH-, sulfide, hypochlorous, carbonate, and **bisulfate** families, plus CO2/O2 and molecular Henry records. Extend it with H2O, a declared elemental sulfur form, and ClO3-; add ClO2- only if its dynamics are explicitly modelled. `AnalyticalPool("CO2_star")` has no complete molecular formula. It cannot participate in a full H/O-balanced reaction validator as though it were CO2(aq).

`solve_ideal_charge_balance` already accepts independent family totals and net strong-ion charge. It reports residual, bracket, convergence, input validity, fitted-temperature validity, and truncated-family pH validity. Its ideal-activity applicability is explicitly **unknown**. A chemistry reducer, reaction compiler, stock boundary, fast manifold, and generic variable-volume reactive inventory are still needed. The code inspected still uses the old Outlet flow/composition contract; milestone C's phase/context groups are a prerequisite for the column assembly below, not a capability already landed.

## 2. Generic `chemistry/reactions.py`

### Static reaction truth

A frozen `Reaction` record should carry:

- Stable name, ordered `(registry species name, signed rational coefficient)` terms, and a stated extent normalization (for example one mol of S(-II) consumed).
- Phase roles when a reaction spans phases or involves a solid. Species chemistry and carrier transport roles are different metadata.
- Direction/reversibility, source links, and whether this is an elementary step or an explicitly reduced overall pathway.

Keep coefficients exact during build-time validation, for example integers/rational pairs. Compile once to float arrays for numerical evaluation. Reject duplicate terms, unknown species, zero/nonfinite coefficients, and empty reactions. Against **molecular Species records**, verify every element and net charge: `E ν = 0`, `zᵀ ν = 0`. Reject analytical pools in this validator. Do not invent a formula for CO2*. For carbonate reactions, write the molecular reaction with CO2/HCO3-/CO3-- and water; a separate equilibrium reduction maps that chemistry onto the carbonate total.

Registry membership alone does not validate a chemical mechanism. A charge-balanced sulfate endpoint is not evidence that sulfate forms instantaneously. Keep kinetic/product-regime assumptions on a separate policy record. Electrons may be useful in half-reaction derivations, but the complete reactions admitted to the simulator must cancel them; they are not stream channels with fabricated molecular mass.

For sulfur, either register thermodynamic elemental S(s) with explicitly per-mol-S normalization, or register actual S8(s). Prefer S8(s) when using the current molecular `Species` convention. The examples here use 1/8 mol S8 per mol of sulfur atoms. Molar mass and elemental counts must agree with that normalization; do not put a mass of 256 g/mol on a one-S formula. Its transported channel counts mol S, not mol S8.

### Independent-basis projection

An `AqueousBasis` declaration names every chemistry-active transported coordinate and its meaning:

- Acid-base family total, with a chosen conserved count (e.g. mol S in the sulfide family).
- Retained molecular ion, neutral species, or solid total.
- Explicit inert/opaque transport channel, excluded from this chemistry after a documented applicability decision.
- Eliminated equilibrium species/solvent, with its declared closure.

Build a static matrix `P` from molecular local amounts to independent basis amounts. Family-member columns carry the appropriate conserved-atom count; retained species/solid columns use their stated amount conversion. H+/OH- and solvent H2O have no independent transported row in this dilute aqueous model. They still occur in the validated molecular reaction, and their effects are recovered through charge equilibrium; their omission from the state is not permission to omit them from reaction validation.

For a molecular stoichiometric matrix `N`, compile `S = P N`. This is the liquid S matrix of design 4. For transfer compile a separate `B = P T`, where T identifies transferred molecules. An extent ξ produces `Δb = S ξ`; transfer produces `Δb = B t`. Acid-base redistribution has zero change in family totals. Equivalent molecular representations related by eliminated acid-base reactions must compile to the same S.

The compiler must reject a reaction product/reactant that has neither a retained row nor an explicitly declared eliminated/externally accounted role. Otherwise chlorine or sulfur can vanish behind an incomplete projection. H/O and solvent are special accounting boundaries here: validate the complete molecular equation, retain its solvent/proton bookkeeping, and state that dilute incompressible transport does not integrate total solvent mass or reaction-volume change. Claim exact S/Cl/Na/C inventory accounting within this reduction, not a full concentrated-mixture mass/energy model.

Do not add automatic basis conversion to `Binding`. The chemistry compiler converts its molecular extents into the declared local write bases; the existing binder then checks exact agreement. For g/m³ channels the compiler includes the appropriate molar/element mass. For this scrubber use mol/m³ throughout. Proxy/COD coordinates are not molecular stoichiometry unless a separate accounting model supplies that relation.

### Rate-law adapter and speciation ownership

Keep a rate law distinct from stoichiometry. A `ReactionRateLaw` declares the molecular concentrations it needs, any additional total/product/ionic-strength requirements, its time/rate units, and a typed parameter pack. It returns **extent rates**, not a full carrier derivative. A compiled `AqueousReactionNetwork` owns the reaction list, independent-basis layout, required local reads, speciation system, rate laws, and S. Its pure evaluation is:

```text
local totals + explicitly mapped ions + T + params
    → charge/property reducer
    → one SpeciationResult
    → required molecular concentrations and context
    → extent rates r (mol reaction / m³ / s)
    → local declared concentration sources 3600 S r
```

It implements `KineticsModel`; bind it with `KineticSource.bind` and explicit local→carrier maps. Speciation **may be inside** this KineticsModel adapter because its public inputs are totals. The inner elementary rate law sees only its declared molecular forms/context, not the carrier vector. Group the slow aqueous reaction network so it shares a speciation solve rather than solving pH independently for every reaction.

A law that uses only HS- and HOCl still indirectly needs all admitted buffers and ion contributions influencing pH. Compile that exact closure read set: sulfide, free chlorine, carbonate, sulfate/bisulfate, relevant extra families, and retained ions. Do not pretend carbonate is irrelevant because its name does not occur in the rate expression. Unknown chemistry-active channels require a contribution or rejection, as in ADR 016.

The landed kinetics protocol returns only arrays. Add a pure `evaluate` result for the network containing rates, extents, speciation, and domain flags; `rates` delegates to it. Diagnostic reporting can call the same function at the requested state without a persistent cache. A failed/unsupported speciation must not return apparently valid rates; return an explicitly unusable evaluation and, for the array-only adapter, nonfinite rates that stop the simulation according to the caller's numerical failure policy. Do not replace failure with zero reaction. Success requires the named policy's validity flags, not only `converged`. An explicitly admitted ideal or extrapolated research policy must remain identifiable in provenance.

This adapter is for integrable slow kinetics. Fast bulk reaction belongs to a `FastReactionClosure` over invariants, and film chemistry uses the same reaction records/S but its own boundary-value closure. Neither masquerades as a very large ordinary kinetic source.

## 3. Scrubber liquid basis, charge balance, and reaction matrix

### Recommended initial carrier

All concentrations below are mol/m³; inventories are the corresponding mol quantities. The following pool definitions are assembly metadata, separate from the basis reference.

| Local channel | Independent quantity | `Quantity` / reference | Transport/chemistry |
|---|---|---|---|
| `total_sulfide` | H2S + HS-, counted as S(-II) atoms | ELEMENT / S | Soluble sulfide-family total; current ladder only valid below its pH ceiling |
| `free_chlorine` | HOCl + OCl-, counted as Cl(+I) atoms | ELEMENT / Cl | Soluble hypochlorous-family total; one mol is one mol Cl2-equivalent oxidizing capacity |
| `total_carbonate` | CO2* + HCO3- + CO3--, counted as C | ELEMENT / C | Soluble carbonate-family total |
| `total_sulfate` | HSO4- + SO4--, counted as S(+VI) | ELEMENT / S | Soluble bisulfate-family total; do not count it again as fixed -2 charge |
| `elemental_sulfur` | Suspended S0, counted as S atoms | ELEMENT / S | Particulate; explicit suspension/retention policy |
| `sodium` | Na+ | SPECIES / Na+ | Soluble strong cation |
| `chloride` | Cl- | SPECIES / Cl- | Soluble strong anion, including stock salt and reaction product |
| `chlorate` | ClO3- | SPECIES / ClO3- | Soluble strong anion; separate from free chlorine |
| `dissolved_oxygen` (if needed) | Molecular O2(aq) | SPECIES / O2 | Needed if oxygen-producing bleach decay is included without an immediate vent boundary |

No transported alkalinity, pH, H+, OH-, HS-, HOCl, bicarbonate, or carbonate-ion coordinates are added alongside these totals. Each acid-base family has exactly one total. Sulfide, S0, and sulfate share an element but differ in oxidation state and participate differently in transport and reactions; replacing them by one total sulfur would lose chemistry. Adding chlorite/thiosulfate/sulfite later requires explicit rows/families, balances, and observations, not an anonymous extra chlorine/sulfur-demand sink.

Default suspended S0 is advected by liquid and leaves with well-mixed overflow; it is not automatically a deposit or instantly converted to sulfate. If using schema TSS, its factor is g S per mol S (about 32.06), not an ASM COD factor. Deposition/settling requires a separate retained-solid inventory and equal/opposite transfer. Omitting S0 is permitted only for an explicitly sulfate-only endpoint model; a mixed-yield model must retain or account for its sulfur product.

With the bisulfate family active, the strong-ion reducer is

```text
q_strong = [Na+] − [Cl-] − [ClO3-] + other explicitly included strong-ion contributions
0 = q_strong + [H+] − [OH-]
    − [HS-] − [OCl-] − [HCO3-] − 2[CO3--]
    − [HSO4-] − 2[SO4--] + charges of additional admitted families
```

All terms in this equation use mol charge/m³. The solver's pH uses mol/L; its existing standard concentration of 1000 mol/m³ handles this conversion. At alkaline pH a separately named reduced sulfate-as-fixed-ion policy can replace the bisulfate family, contributing `−2 total_sulfate` to q_strong. Never use both representations simultaneously.

Caustic increases the sodium coordinate; OH- is then reconstructed from water equilibrium/charge balance. That is the correct **reduced representation of adding NaOH**, not a claim that adding arbitrary sodium salts raises pH. NaCl adds both Na and Cl and has zero net ideal acid/base effect. NaOCl adds Na and the chlorine family, with further salt/caustic in actual stock. An HCl titrant adds Cl; no dynamic H+ inventory is needed. Stock dissociation supplies the eliminated counterion consistently.

The coordinates are non-redundant: family forms are reconstructed, q_strong is a derived sum, and alkalinity is a derived convention-dependent diagnostic. Separate Na/Cl remain necessary even though ideal pH sees only their difference: neutral salt accumulation changes ionic strength and affects bleach decay, activities and material accounting. In a closed vessel reactions imply **conserved combinations**, but that does not make these carrier coordinates redundant for an open plant with independent feeds/outflows. Fast-reaction invariants reduce the dynamic state further; they do not change what Streams carry.

### Example compiled S

Write balanced pathways per mol S:

```text
R0: HS- + OCl- → (1/8) S8(s) + Cl- + OH-
R6: HS- + 4 OCl- → SO4-- + 4 Cl- + H+
Rpost: (1/8) S8(s) + 3 OCl- + H2O → SO4-- + 3 Cl- + 2 H+
Rchlorate: 3 OCl- → ClO3- + 2 Cl-
Roxygen: 2 OCl- → O2 + 2 Cl-
```

These equations are balanced by construction; the proposed registry validator must independently establish that. R0/R6 are reduced endpoints, not elementary mechanisms. Using H2S instead gives `H2S + OCl- → (1/8)S8 + Cl- + H2O` and `H2S + 4OCl- → SO4-- + 4Cl- + 2H+`, with the same independent-basis columns.

| Basis row | R0 | R6 | Rpost | Rchlorate | Roxygen |
|---|---:|---:|---:|---:|---:|
| total_sulfide | -1 | -1 | 0 | 0 | 0 |
| free_chlorine | -1 | -4 | -3 | -3 | -2 |
| total_carbonate | 0 | 0 | 0 | 0 | 0 |
| total_sulfate | 0 | +1 | +1 | 0 | 0 |
| elemental_sulfur | +1 | 0 | -1 | 0 | 0 |
| sodium | 0 | 0 | 0 | 0 | 0 |
| chloride | +1 | +4 | +3 | +2 | +2 |
| chlorate | 0 | 0 | 0 | +1 | 0 |
| dissolved_oxygen | 0 | 0 | 0 | 0 | +1 |

A fixed effective yield f to sulfate gives `Sfast = (1−f) S_R0 + f S_R6`, demanding `d = 1+3f` mol free chlorine/mol reacted sulfide. Conserved sulfur is `n_sulfide+n_S0+n_sulfate`; conserved chlorine is `n_F+n_Cl+n_ClO3` (plus chlorite/other chlorine species if admitted). Carbon and sodium have their own boundary balances. Electron accounting distinguishes free oxidant from elemental chlorine: sulfate requires 8 electron equivalents/mol S, S0 requires 2, and each mol hypochlorite accepts 2. Chlorate formation redistributes chlorine oxidation states; it does not destroy elemental Cl or preserve free chlorine.

For actual sodium hypochlorite, complete sulfate oxidation of gaseous H2S has the reduced overall salt balance
`H2S + 4 NaOCl + 2 NaOH → Na2SO4 + 4 NaCl + 2 H2O`.
The S0 endpoint needs no net added caustic in that overall salt balance. Absorption into unoxidized HS- instead needs approximately one base equivalent/mol H2S at high pH. Replacing NaOCl with HOCl or Cl2 changes acid/base demand because their feed counterions differ. Therefore do not attach a universal caustic-per-sulfide coefficient to every oxidant feed. The charge solver and the actual stock basis recover that distinction.

## 4. Oxidation policy and fast bulk reduction

### What the primary literature supports

**Choppin & Faulkenberry (1937), JACS 59, 2203–2207:** the original study includes sulfur and sulfate measurements and explicitly describes more complicated alkaline products. Its preliminary mixing tests already contradict a universally one-for-one reaction. The scanned primary text is a reason to retain product inventories, not a validated Gold Bar yield curve. The publisher record and indexed scan were available; downloading the full scan failed in this review, so no numerical pH-switch fit is transcribed. [Publisher/DOI](https://doi.org/10.1021/ja01290a033), [primary scan](https://electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20the%20American%20Chemical%20Society%20US/1937%20%20%28vol%20059%29/11%20%20%282083-2480%29/2203-2207.pdf).

**Azizi et al. (2015), section 3.1:** with oxidant excess their experiments approached four chlorine equivalents and detected sulfate; with sufficient sulfide excess (initial sulfide/chlorine ratio > approximately 3), demand approached one, without detected sulfite/sulfate. The ratio here is **sulfide/chlorine**, not its reciprocal. Their kinetic result is about 1.2×10^9 L mol^-1 s^-1 for HS-/HOCl at 25°C, with acid-base kinetic limitations. This is neither a total-sulfide×total-chlorine coefficient nor a sulfate-formation rate for every subsequent step. [DOI](https://doi.org/10.1016/j.cherd.2014.07.023), [accepted manuscript, section 3.1 and Tables 5–7](https://scispace.com/pdf/competitive-kinetics-study-of-sulfide-oxidation-by-chlorine-3wtifkn2zq.pdf).

**Cadena & Peters (1988), JWPCF 60(7), 1259–1263:** their background equations use a 7.5 pH division between sulfur and sulfate. Their actual municipal-wastewater experiments were at pH 7.2–7.5 and about 25°C, with chlorine disappearance/removal assessed at the five-minute sampling limit. Their roughly 4.2 g chlorine/g sulfide demand is site/matrix-specific. This does not establish an elementary rate or a pH-only product switch for a packed scrubber operating near pH 9.4–10. Do not fit a sulfate fraction from that demand alone; competing wastewater chlorine demand also matters. [Primary paper](https://literature.wolkersdorfer.info/literature/Evaluation%20of%20Chemical%20Oxidizers%20for%20Hydrogen-Sulfide%20Control.pdf).

These sources support explicit endpoint constraints and the importance of exposure/reagent ratio. They do not support a hard generic `pH>7.5 → f=1` law. Local oxidant availability, mixing history, contact time, acid-base kinetics and subsequent heterogeneous/intermediate oxidation all matter. ORP is not a measurement of f. Stoichiometry cannot identify a kinetic product split.

### Initial policy recommendation

Implement **explicit fixed-yield endpoint policies** first: S0-dominated (f=0), sulfate-complete (f=1), and a declared fixed intermediate yield if measured sulfur products justify it. Use these as conservative model alternatives, not interchangeable claims about the plant. At Gold Bar's oxidant-residual operating condition sulfate completion is a plausible hypothesis; it still needs product measurements and residence-time validation. Do not silently choose it as established fact. A smooth empirical f(pH,ratio) can be a later named reduced model after data, with stated domain and uncertainty; it must not rewrite old product inventories when conditions change.

Existing S0 is a separate product. Enabling Rpost means choosing/sourcing a finite heterogeneous or otherwise reduced oxidation law and a solid-surface/availability model. Setting Rpost instantaneous merely because HS-/HOCl is fast would erase product memory and incorrectly impose sulfate completion. Intermediates such as sulfite/thiosulfate require added pools and kinetics if needed by the data. An unidentified chlorine demand should not be hidden in f.

### Bulk sump/manifold oxidation

Treat the **initial well-mixed dissolved-sulfide oxidation** as an instantaneous invariant closure only inside a validated regime where its relaxation time is negligible against mixing, transport and resolved dynamics. For a fixed f, let d=1+3f and evolve invariants, not two stiff competing reagent inventories. A concrete reduced chart is:

```text
I = n_F − d n_sulfide                  (signed)
J0 = n_S0 + (1−f) n_sulfide
J6 = n_sulfate + f n_sulfide
KCl = n_Cl + n_F
plus n_carbonate, n_Na, n_ClO3, optional n_O2, and volume V.

n_sulfide = max(−I/d, 0)
n_F       = max(I, 0)
n_S0      = J0 − (1−f) n_sulfide
n_sulfate = J6 − f n_sulfide
n_Cl      = KCl − n_F
```

The max expressions define the algebraic reagent-exhaustion chart; they are **not post-step inventory clipping**. Require all reconstructed physical amounts nonnegative. Negative reconstructed product/spectator amounts mean the invariant state is infeasible, not that another max should repair it. For more fast reactions use `U Sfast=0` with a declared constrained closure selecting physical amounts; conservation alone does not select products.

Conservative initialization is allowed to react initially coexisting reagents once, add the appropriate products, and preserve these invariants. It is not repeated finite subtraction inside an RHS. During dynamics, transport and genuinely slow reactions drive `dy/dt = U (transport + Sslow rslow + declared external sources)`. Recover physical concentrations and speciation from y at **each joint RK stage**. Fast extents follow from the reduced balances/closure; report them consistently with realized fluxes, including at reagent-switching events. Do not add a second fast oxidation source after this reduction.

Here f is fixed for a trajectory/model policy. A varying f changes the invariant chart; applying the old U after changing f is wrong. Time-varying pathway selection needs a more general progress/extent closure with irreversible product history and transition accounting. Calibration that changes a fixed f must recompute the initial invariant coordinates and every dependent reduction, not reuse cached static matrices from another f. Avoid making f dynamic during CSTR integration in the first implementation.

Fast initial chemistry does not prove every residual interaction is instantaneous. Very low HOCl/free-chlorine residuals, cold liquor, high pH, finite proton transfer, segregated feeds, and oxidation of S0/intermediates can invalidate that reduction. Demonstrate timescale separation across the intended domain, including approaching depletion. That test must include the entire selected endpoint: the HS-/HOCl coefficient alone cannot justify instantaneous sulfate completion. If subsequent oxidation is slow, use an initial S0-forming fast closure plus a separately validated slow product-oxidation model. If finite stiffness remains and cannot be reduced, ADR 018 requires a joint stiff/implicit integration extension; the present explicit RK4 is not the alternative. Do not step the sump separately with frozen column outlets.

The remaining invariant ODE still needs conservative stage/step admissibility and depletion/event treatment. Identity projection is appropriate; the landed `nonnegative` projection must not be used on signed I or as a way to repair negative material. A local algebraic closure can enforce chemistry but cannot make an infeasible hydraulic withdrawal safe.

## 5. Hypochlorite decomposition and chlorate

Keep stock aging and sump decay separate applications of the same balanced reaction infrastructure. In basic bleach the net chlorate branch is Rchlorate; an oxygen branch is Roxygen. If chlorite is resolved, use `2OCl- → ClO2- + Cl-` followed by `ClO2- + OCl- → ClO3- + Cl-`; do not also add the net chlorate reaction for the same conversion.

**Gordon, Adam & Bubnis (1995), not simply “Adam & Gordon 1995”:** report basic-bleach second-order decomposition with low-concentration chlorite intermediate and a smaller oxygen side reaction. [Journal AWWA 87(6), 97–106](https://doi.org/10.1002/j.1551-8833.1995.tb06382.x). Eliminating a fast chlorite intermediate is a declared reduction, not evidence that chlorate need not be tracked.

**Adam & Gordon (1999):** their concentrated, metal-catalyst-free regime covers 0.7–3.0 M hypochlorite, 0.001–1 M OH-, 15–55°C, ionic strength μ=1–6 M. They report chlorate formation 8.7 times the oxygen formation rate and extra chloride catalysis near pH 9–10. Their correlation is

```text
log10 k2 = 0.149 μ + log10[2.083×10^10 T
               × exp(−1.018×10^5/(R T)) × exp(−56.5/R)]
k2 in L mol^-1 s^-1; T in K; R in J mol^-1 K^-1.
```

[Primary paper/abstract and supporting-data description](https://doi.org/10.1021/ic980020q), [PubMed](https://pubmed.ncbi.nlm.nih.gov/11670917/). The full kinetic-equation normalization was not accessible here; verify whether its k2 multiplies a step extent or a disappearance rate before implementing it. Do not infer that from the abstract's units.

Define rate normalization explicitly in every decomposition policy. If an independently specified total chlorine disappearance is ℓ, and b is the **fraction of that disappearance** going through chlorate, then `r_chlorate=bℓ/3`, `r_oxygen=(1−b)ℓ/2`. The measured product-rate ratio `r_chlorate/r_oxygen=8.7` would give `b=26.1/28.1≈0.9288`, not 8.7/9.7. This is a stoichiometric derivation conditional on using that reported regime; do not export it as a universal product fraction. If k2 defines a reaction-step extent instead, convert once using its reaction coefficients.

For a declared disappearance law `ℓ=k_loss [OCl-]^2`, compute OCl- from local speciation. In the mol/m³, per-hour carrier its loss is `3600 k_loss [OCl-]_m3² /1000`, before partition into the balanced columns. This unit conversion does not settle the paper's normalization. A slow `AqueousReactionNetwork` can implement the policy once that normalization and parameter domain are verified.

Use explicit Na/Cl and other ions to estimate `I=0.5 Σ z_i² c_i` in mol/L after reconstructing forms, with completeness diagnostics. Net strong charge alone cannot supply ionic strength. An ideal pH solve plus an I estimate remains an ideal-pH approximation; it does not implement electrolyte activities at high salt. Concentrated stock is outside the landed ideal solver's demonstrated applicability, and dilute/cold sump liquor is outside the concentrated kinetic fit. Do not clamp either into a fitted regime or silently extrapolate its decay coefficient.

Initial assembly should use measured delivered stock assays (FAC, chloride, chlorate, excess base, age/T) rather than invent a dynamic storage tank. If storage aging is modelled, use a separate inventory/stock-domain closure. For sump self-decay, include a named finite-rate policy only when a dilute-liquor validation or an explicitly admitted extrapolation study justifies it; otherwise retain chlorate as a delivered/transported impurity and state that sump decay is unresolved. A fitted first-order lumped “chlorine loss” must identify its products/accounting boundary and must not be called this self-decomposition chemistry. Metals, organic demand, light, and chloramines are distinct mechanisms.

Oxygen formation needs a dissolved O2 row or a measured/declared immediate vent boundary with an equal material ledger. It is not silently lost and is not automatically routed through the future packed-column gas outlet if produced in the sump or stock tank.

## 6. Generic stock feeds and the H2S plant assembly

### Stock record and conversions

A `StockComposition` record supplies independent carrier concentrations or audited assay inputs compiled into them, plus density where mass-fraction conversion requires it, T/P boundary context, assay convention, source/date, and uncertainty. Its numerical assay values are parameter leaves; semantic layout/unit conventions are static. A separate pump adapter converts command/realized speed to delivered volume. The stock converter never receives a PI setpoint or guesses pump speed from ORP.

For a bleach produced by chlorine/caustic manufacture, `Cl2 + 2NaOH → NaOCl + NaCl + H2O` gives **equimolar initial NaCl and NaOCl** before excess base, feed impurities, processing and aging. That is a manufacturing prior, not a mandatory current assay relation. The manufacturer's handbook defines available chlorine as Cl2-equivalent oxidizing capacity, distinguishes wt%/g/L/trade percent, and documents excess caustic for stability. [OxyChem primary handbook, preparation/terminology/manufacturing](https://files.engineering.com/download.aspx?file=sodium_hypochlorite.pdf&folder=c789cf96-921c-48b4-8714-6f49ab9cee11).

Let A be available Cl2 in g/L, B excess NaOH in g/L, L chloride salt in mol/L, Z chlorate in mol/L, and C carbonate from stock contamination in mol C/L. For a declared Na2CO3 contamination convention:

```text
c_F       = A / M_Cl2                    mol/L, M_Cl2≈70.90 g/mol
c_Cl      = measured L                  (fresh manufacture prior: L≈c_F)
c_ClO3    = measured Z
c_C       = measured C
c_Na      = c_F + L + Z + B/M_NaOH + 2 C
```

Prefer measured sodium if available; the last equation is the analytical salt recipe's sodium balance, not a general replacement for an assay. Aged manufacture no longer guarantees `L=c_F`; chloride/chlorate persist as FAC declines. Carbonate formed by caustic carbonation retains sodium while changing carbonate/free-base composition. If other counterions occur, include them explicitly. Available Cl2 mass is not elemental Cl mass in the hypochlorite ion: **do not divide A by 35.45**. No physical Cl2 molecule is introduced into the liquid schema by this assay convention.

For caustic with NaOH mass fraction w and density ρ in kg/L, `c_NaOH=1000 ρ w /M_NaOH` mol/L, approximately `25 ρ w`. Its sodium carrier concentration is that value plus declared salt/carbonate impurities. Pure caustic adds no free chlorine. 25–50 wt% is an input scenario range until EPCOR's delivered grade is known, not a chosen plant fact. Density must correspond to w and stock temperature; water density is not adequate. The producer lists commercial 50% and diluted grades. [OxyChem product information and caustic handbook](https://www.oxychem.com/chlor-alkali/).

Given Q_L in L/h and c in mol/L, inlet amount rates are `Q_L c` mol/h; a liquid Stream uses `Q_m3=Q_L/1000`, `components_m3=1000 c`. Count chemical feed volume in the holdup balance. Do not add concentrated stock as a source of mol/h and separately add the same chemical Stream. The physical stock can be specified analytically without solving its pH using the dilute ideal solver; solve the diluted mixture within its admitted activity regime instead.

### EPCOR actuator/wiring facts

From the supplied EPCOR evidence, retained in [01](01-physics-review.md): bleach/caustic pumps are peristaltic and controller outputs are RPM. Duty conversions are bleach **1.632 L/h/RPM**, caustic **1.008 L/h/RPM**; standby caustic has **0.576 L/h/RPM**. Pump identity/duty selection belongs in the adapter. At 40 RPM the duty feeds are 65.28 and 40.32 L/h. A linear speed→volume policy assumes a calibrated head, tubing and backpressure regime; update it from flow tests rather than absorb every error into stock strength. Apply interlocks and speed limits before conversion, and retain realized speed/flow separately from requested command.

The hydraulic assembly is:

```text
sump outlet → recirc pump → stock injection/mixing manifold → packed column → sump
makeup and other documented returns → sump
sump continuous overflow → external discharge
```

Both chemical stocks enter the **recirculation pump discharge upstream of the tower**, not the sump. Makeup level close/reopen thresholds 695/600 mm belong to controller/actuator state; they are not tank volume or weir crest. At steady volume blowdown follows all net liquid additions/losses, including stocks, flush and evaporation, not an isolated arbitrary bleed coefficient. TI0880 measures gas intake temperature; it does not directly provide sump temperature. Geometry, stock strengths, recirculation/gas flow units and actual liquid temperature still need evidence.

### Equilibrium versus reaction in the manifold

Use a generic algebraic amount mixer followed by an explicitly declared fast chemistry transformation, rather than hiding oxidation in the mixer. Acid-base equilibrium is required to derive column-inlet speciation from the mixed independent totals. If returned sulfide and new bleach coexist and the manifold provides well-mixed contact long compared with validated relaxation time, also realize irreversible oxidation before the column, with the same product policy and conservative extents. This is a **serial physical location**, permitted by ADRs 017/018, not a sump time step.

A stream fast transformation acts on flowing **amounts** and residence assumptions once; it is distinct from mutating sump state inside an RHS. At positive flow, reactions operate on mol/h, preserve untouched amount rates, and return complete outlet composition/flow plus diagnostics. At zero flow, no sustained material or reaction rate is invented. A well-mixed instantaneous transformation is a reduction conditional on manifold residence/micromixing, not implied by “algebraic mixer.” If injection is segregated or residence too short, pass independent totals into an explicit finite-contact or column-entry closure; do not force immediate sulfate completion. Bulk equilibrium alone does not determine oxidation extents.

The column's film closure then operates on what actually enters it. Film-consumed sulfur is delivered as its products, while unreacted film-to-bulk sulfur can remain sulfide. Use either mapped gas transfer plus S×film extents or mapped film-to-bulk fluxes, not both. Sump fast reduction acts only on remaining bulk reagents. This prevents counting one absorbed sulfur mol as a film product, a fresh sump sulfide loading, and a second oxidation event.

## 7. CO2 co-absorption as an unknown disturbance

Include molecular CO2 in the complete gas mixture and carbonate total in liquid. CO2 transfer uses molecular solubility/speciation, not a total-carbonate Henry constant. The landed registry distinguishes molecular CO2 from CO2*, so their dilute low-temperature approximation must remain explicit. Carbonate uptake consumes base through charge redistribution and competes with H2S absorption; it does not require an empirical alkalinity sink.

The WRF fermenter design compendium explicitly identifies CO2 production during fermentation and headspace corrosion from carbonic acid, but supplies no representative Gold Bar gas fraction. [WRF 2019, printed p.19](https://www.waterrf.org/sites/default/files/file/2021-07/Fermenters-for-BPR.pdf). Do not use anaerobic-digester biogas percentages as a ventilated sludge-fermenter foul-air prior; ventilation dilution and fermentation regime differ.

A primary municipal-wastewater putrefaction study measured approximately **0.5–2.0 vol% CO2** in raw/treated experimental gas. Its setup was 5 L wastewater in 6 L anaerobic lab reactors, 20°C, intermittently stirred; it was not a continuously ventilated odor duct. [Dębowski, Kazimierowicz & Zieliński (2022), sections 2.3/3.4, Table S4](https://doi.org/10.3390/atmos13101724), [primary PDF](https://mdpi-res.com/d_attachment/atmosphere/atmosphere-13-01724/article_deploy/atmosphere-13-01724.pdf). A WWTP NDIR chamber study starts near **450 ppmv** ambient and demonstrates process-dependent CO2 release, but chamber buildup is likewise not a ventilated duct concentration. [Pilot field study (2018)](https://doi.org/10.1016/j.scitotenv.2018.03.174). The putrefaction study's dry/wet reporting convention was not established here; convert measured plant boundaries explicitly rather than copying those numbers as exact wet-gas fractions.

Recommend a provisional **engineering scenario envelope** from measured/local ambient to 0.02 mole fraction, with checks near 450, 1000, 5000 and 20000 ppmv on a stated dry/wet basis. A log-uniform excess-above-ambient prior is a defensible broad uncertainty model only when explicitly labelled a design choice; use a small positive scale near zero excess. These references bracket plausible scenarios; they do **not** establish a universal upper bound or a validated statistical prior for Gold Bar. Permit expansion when measured duct CO2 or ventilation/source balances warrant it. A more physical disturbance is `F_CO2,in = F_air x_CO2,ambient + G_CO2,fermenter`, with production G uncertain and ventilation measured. That makes dilution dependence visible.

Maintain complete gas fractions: adding CO2 changes carrier fractions/total flow consistently, rather than adding another ppm channel to fractions already summing to one. Use one low-dimensional disturbance model initially (constant per operating period or slowly varying source rate), not a free CO2 value for every pH sample. CO2, stock excess base, initial carbonate, makeup composition and blowdown are confounded in pH/caustic-only data. Need gas CO2, TIC/carbonate titration, makeup chemistry, and salt/overflow measurements to distinguish them. Do not fit all of these independently from one pH trace.

## 8. Sump and column contract requirements

The sump is a generic variable-volume holdup over independent physical amounts, with a fast-state reconstruction policy when enabled. Evolve V and invariant amounts; transport every declared channel according to its soluble/particulate classification. Multiple inlets, recirculation, makeup and continuous overflow contribute to amount/volume balances. Reconstructed outlet composition is state-derived; prescribed recirc pump flow breaks the flow recycle. Never use a concentration-only constant-volume ReactiveCSTR as though it already implements this sump.

Under milestone C, the column is stateless and owns an atomic evaluation group spanning its coupled outlet fields. Reads include actual gas/liquid flows, compositions and thermodynamic context needed by its solve. Static independent-basis/phase-role metadata bind the liquid S/B matrices and complete gas molecular basis. The same solve emits both phase outlets, reaction extents, product partitions, convergence/validity, and conservation residuals. A stock mixer can keep flow separate from chemistry-dependent composition when its incompressible policy actually permits that separation.

Fast sump reconstruction creates a new shared evaluation need: its physical outlet state and slow derivative terms must use the same admitted closure semantics. Prefer a small typed `HoldupEvaluation` containing reconstructed amounts/concentrations, speciation and closure diagnostics, obtained once per joint RHS and passed explicitly where consumed. This is the concrete need anticipated but deferred in C's design; do not create a generic persistent cache or reevaluate with inconsistent policies. If the engine does not yet pass such results into derivatives, straightforward repeated **pure** closure evaluation is correct initially, although more costly. Reporting uses a fresh resolution at the requested state. No cache survives a stage or parameter change.

T is actual local liquid T in K for chemistry; stock/gas/liquid contexts may differ. A prescribed isothermal scrubber policy is allowed if stated; TI0880 does not validate it. At 3°C the landed Henry guard (5–35°C) reports out-of-domain. The sulfide ladder is truncated above pH 12, the sulfate temperature range is provisional, and concentrated caustic stock lies outside dilute-ideal applicability. Preserve these flags rather than expanding domains just to accept plant data. Energy, pressure-drop and nonideal activity extensions remain separate capabilities.

Calibration requires derivative checks through the charge/fast/column closures. The fast manifold has a real reagent-switch kink; diagnose/note nonsmoothness instead of assuming one smooth derivative everywhere. A converged algebraic solve alone does not establish usable gradients. BSM1 uses neither these families nor fast states; preserve its landed bindings, rates, temperature boundary and projection policy.

## 9. Minimal validation oracles and implementation order

These are proposed checks, not test results from this design review. No code/tests/experiments were run. Prioritize independently known balances and chemical measurements over tests duplicating the implementation.

1. **Reaction/compiler contracts:** balanced molecular R0/R6/Rpost/chlorate/oxygen records; deliberately wrong charge/element records fail; unknown/unmapped products fail; acid-base-equivalent molecular equations compile to the same S. S8-to-mol-S conversion must exercise its factor of eight. Reordered/extended opaque carriers give identical bound local sources and correctly transported untouched channels.
2. **Stoichiometric demands:** one mol S(-II) to S0 consumes one mol FAC; to sulfate consumes four. For mass comparisons use 70.90 g available Cl2/mol FAC and explicitly distinguish mol S, g S, and g H2S. Derived demand is approximately 2.21/8.85 g Cl2 per g S, or 2.08/8.32 per g H2S. Check chloride/product formation and total chlorine/sulfur, not only disappearance of sulfide. Mixed fixed yield uses exactly 1+3f; competing paths cannot each spend the same FAC.
3. **Dilute titration oracles:** pure water at temperature-dependent pKw/2; NaOH/HCl amount titration including dilution; NaHCO3 and Na2CO3 carbonate curves against the already verified chemistry/independent reference; dilute NaOCl hydrolysis plus controlled excess NaOH where molecular Cl2 formation is negligible; adding NaCl leaves ideal pH unchanged while changing the diagnosed I. Acidifying chlorine into a Cl2-forming regime needs a chlorine-family/gas-closure extension. Sulfate must not be counted as both a family and a fixed ion. Match an independently evaluated reference solver or measured titration curve across the stated dilute domain, not merely another call to the same charge solver.
4. **Fast manifold:** conservative initialization; both reagent-excess branches, exact stoichiometric endpoint, zero reagent, product memory, signed I, and inadmissible reconstructed products. Feed/withdrawal/recycle examples verify full integrated inventories and discharge. Near depletion compare the admitted reduction to an independently solved finite-kinetics reference at decreasing relaxation time; include endpoint/yield uncertainty. Column film products must not react a second time as fresh sulfide in the sump.
5. **Decomposition:** coefficient transcription checks in the actual published domain. Direct evaluation of the displayed Adam–Gordon expression at 298.15 K gives k2≈1.4333×10^-8 at μ=1 and 7.9678×10^-8 at μ=6 L mol^-1 s^-1 (derived scalar checks, not measured disappearance rates). Then validate normalization/product rates against the paper/supporting concentration-time data before enabling that model. For an explicitly normalized single second-order disappearance law at fixed context, `F(t)=F0/(1+k_loss F0 t)` is an analytic oracle; use matched chloride/chlorate/O2 balances. Out-of-domain cold/dilute lanes must be identified, not silently accepted.
6. **Stocks/assembly:** assay-unit conversion catches factor 2 in Cl versus Cl2-equivalent and factors 1000/3600; RPM conversion distinguishes duty/standby; stock volume/density/mass-fraction tests use stated synthetic values. Mixing/fast transformation at pump discharge produces the physically appropriate tower inlet, with zero flow emitting zero material. Makeup hysteresis remains distinct from overflow geometry. CO2-only loading changes TIC/pH/base demand without inventing sulfur or chlorine; absorption and stripping preserve signed interphase balances.

Minimal dependency order: (a) registry additions and exact balanced reaction records; (b) explicit aqueous-basis reducer and stoichiometric projection; (c) slow aqueous reaction-network/diagnostic adapter, initially demonstrated with a genuinely slow law; (d) generic stock recipe and calibrated pump boundary; (e) generic fast invariant reconstruction plus conservative integration contract and variable-volume holdup; (f) serial stock manifold and column/liquid assembly after C. Add a validated decomposition policy only after its normalization/data gate, and choose the scrubber product policy explicitly before enabling fast oxidation. Each primitive should stand on generic contracts; the H2S factory supplies pool names, stock assays, interlocks, geometry/context and product assumptions.

The critical unresolved measurements are delivered FAC/NaOH/NaCl/chlorate and density, makeup/TIC/ionic composition, sump/recirc/overflow volumes and flows, local liquid T, gas CO2/dry-wet convention, and sulfur product assays/solid retention. These affect actual balances and regime selection; calibration cannot make them disappear behind an efficiency coefficient.
