# Buddy design 4 — transfer, packing, column, holdup, electrode

Design only. Read against the current ADRs 015–018; no repository implementation changes. Equations below use SI seconds internally. The carrier converts once to the flowsheet's hours: mol/s × 3600, m³/s × 3600. Temperature is K, pressure absolute Pa, concentration mol/m³, diffusivity m²/s, kL m/s, pressure-based kG mol/(m² s Pa). This avoids embedding hour conversions inside kinetic constants. Suggested signatures describe contracts, not a requested implementation plan.

## Decisions to carry forward

Use one conservative segmented column kernel, with a pluggable local film closure and packing strategy. Exact physical and first-order film solutions are worthwhile cheap closures and test oracles. A scalar E(Ha,E∞) is a restricted approximation for one irreversible reaction, not a sufficient general chemistry interface. Use a steady reaction–diffusion film solve when parallel reactions, finite proton transfer, finite bulk concentrations, or product partitioning require it. Keep column axial cells and film cells separate numerical dimensions. Both solves are local algebraic closures under ADR 018, not separate time integrators.

The first deliverable need not implement every closure. Deliver exact physical/first-order limits, conservative column residuals, diagnostics, and property injection first; admit richer film chemistry only when its transport and kinetics are specified. Do not label an equilibrium-HS approximation a validated Scrubber 4 mechanism.

## 1. Films: solve the interface and account for the reaction

Suggested contracts:

```text
film(p_g_bulk, c_liquid_bulk, hcp, k_l, k_g_pressure,
     diffusivities, reaction_parameters, temperature_k) -> FilmResult
film_area_rates(film_result, area_m2) -> PhaseRates
```

`p_g_bulk` is the partial pressure at the outer edge of the gas film. It is not interfacial pressure. Interfacial p_i and molecular C_Ai = Hcp p_i are solved. If a caller supplies p_i, expose a distinct liquid-film boundary problem; applying a gas resistance again would double-count it. Hcp is molecular solubility, never a total-family Henry coefficient.

`FilmResult` contains gas→film fluxes, film→liquid-bulk fluxes, reaction extents per area, interface values, Ha and resistance diagnostics where defined, scaled balance/root residuals, and validity/convergence flags. Scalar A fluxes have mol/(m² s); each extent has mol reaction/(m² s) with a declared reaction normalization. Area multiplication belongs to the carrier/geometry layer, preserving ADR 016's separation. Allow signed transfer and signed bulk-boundary fluxes.

### Exact physical and pseudo-first-order closures

These formulas are derived directly from the steady film boundary problem D_A C'' − k_app C = 0, C(0)=C_Ai, C(δ)=C_Ab, δ=D_A/kL. With Ha=√(D_A k_app)/kL, define φ=Ha coth(Ha), ψ=Ha csch(Ha). Then:

```text
C_Ai = (kG p_g_bulk + kL ψ C_Ab)/(kG/Hcp + kL φ)
J_interface = kG(p_g_bulk − C_Ai/Hcp)
            = kL(φ C_Ai − ψ C_Ab)
J_bulk      = kL(ψ C_Ai − φ C_Ab)
ξ_A         = J_interface − J_bulk
            = kL(φ−ψ)(C_Ai+C_Ab)
```

J_bulk is positive into the bulk. It can be negative: bulk A can diffuse back into the film and react. Therefore film consumption can exceed net gas absorption without violating conservation. This is precisely why returning only an enhanced transfer flux is insufficient.

At Ha→0, φ=ψ=1 and both boundary fluxes reduce to
J=(p_g_bulk−C_Ab/Hcp)/(1/kG+1/(Hcp kL)); reaction vanishes. For C_Ab=0, E=φ, while unreacted A reaching the bulk is kL ψ C_Ai. At large Ha, ψ→0 and φ≈Ha. Implement the small-Ha series φ=1+Ha²/3−Ha⁴/45, ψ=1−Ha²/6+7Ha⁴/360, and large-Ha exponential forms; evaluating sinh at enormous Ha is unnecessary. Physical stripping must remain possible.

For truly parallel irreversible first-order losses of the same A, k_app=Σ k_j, and ξ_j=(k_j/k_app)ξ_A. The zero-rate case has zero extents. For k_j=k_2,j C_Bj this partition is valid only when each B_j is approximately uniform, available, and undepleted in the film. Reagent supply and balanced product routing remain required; mathematical first-order reservoirs are not finite chemical inventories.

### Which E(Ha,E∞)?

For A+νB→products, C_Ab≈0, a common stagnant film thickness, and irreversible absorption, choose the **van Krevelen–Hoftijzer film approximation** as the scalar helper:

```text
E∞ = 1 + D_B C_Bb/(ν D_A C_Ai)
q = Ha √[(E∞−E)/(E∞−1)]
E = q coth(q),  1 ≤ E ≤ E∞
Ha = √(D_A k2 C_Bb)/kL
```

Solve E jointly with the gas-interface condition; E∞ depends on C_Ai, not Hcp p_g_bulk. Handle B=0 and Ha=0 by the physical limit without dividing by E∞−1. It tends to Ha coth Ha for unlimited B and to E∞ for instantaneous reaction. The primary source is [van Krevelen & Hoftijzer (1948)](https://doi.org/10.1002/recl.19480670708). This is an approximation, not a generic multi-reaction law. Do not infer J_bulk or separate pathway extents solely from E. When the fast-film assumption justifies J_bulk≈0, label it and test its domain; otherwise use the numerical film closure.

Do **not** silently substitute DeCoursey's formula into this film model. Its primary [1974 paper](https://doi.org/10.1016/0009-2509(74)85003-7) develops a Danckwerts surface-renewal approximation. In its conventional notation, with b=Ha²/[2(E∞−1)], E=√[b²+E∞Ha²/(E∞−1)+1]−b. Its unlimited-reactant limit is √(1+Ha²), not Ha coth Ha. These theories also differ in diffusivity scaling. A separate declared surface-renewal policy is possible, but it is not the default here.

### A general film closure

For finite reagent transport use a fixed-grid steady boundary problem:

```text
0 = −∂J_s/∂x + Σ_r ν_sr r_r(c,T)
J_s = −D_s ∂c_s/∂x                 [neutral dilute Fick closure]
c_s(δ) = c_s,bulk
J_A(0) = kG[p_g_bulk − c_A(0)/Hcp]
J_nonvolatile(0) = 0
ξ_r = ∫₀^δ r_r dx
```

For volatile products, declare their gas boundary/Henry relation too; zero gas flux is not automatic. If using δ=D_A/kL, all species share that film geometry and their transport scales with D_s. Do not assign independently chosen thicknesses and retain the above E∞ formula.

For ionic species, independent Fick fluxes with arbitrary diffusivities are not an automatically charge-consistent electrolyte model. Declare a dilute supporting-electrolyte approximation and its neglected charge effects, or solve electroneutral multicomponent/Nernst–Planck transport with spectator ions and zero-current closure. Milestone A bulk speciation does not supply an electrolyte transport model. The film's species representation is local algebraic state, so it does not violate ADR 016's one independent transported basis per bulk family.

At every film node, return actual reaction extents; integrate them with the same quadrature used in the flux residual. Verify J_s(δ)−J_s(0)=Σν_sr ξ_r. The numerical reference handles finite bulk A, depletion and reversibility; it cannot rescue missing kinetics or unidentified products. A locally stiff algebraic solve is permitted by ADR 018.

### H2S, HS−, chlorine and proton transfer

Under *local ideal acid-base equilibrium at a prescribed film pH*, [HS−]/[H2S]=10^(pH−pKa). If oxidation is r_ox=[HS−](k_HOCl[HOCl]+k_OCl[OCl−]), its equivalent loss coefficient relative to molecular H2S is:

```text
k_app,ox = 10^(pH−pKa) × (k_HOCl C_HOCl + k_OCl C_OCl)
```

This is a derivation conditional on those assumptions. SI conversion is k[m³/(mol s)]=10⁻³ k[L/(mol s)]. It is not valid to multiply the HS− coefficient directly by total sulfide and total chlorine, or to hold bulk pH constant across a film depleted of base.

H2S+OH−⇌HS−+H2O is a reversible conversion within the sulfide family, not irreversible sulfur destruction. It cannot be added to the oxidation k_app as an independent terminal sink. As an illustrative local reduction, if molecular A→HS has forward frequency f, reverse frequency b, and HS oxidation frequency o, an immobile quasi-steady HS intermediate gives k_eff=f o/(b+o). This explicitly saturates at the proton-transfer supply rate when oxidation becomes arbitrarily fast. HS diffusion and buffering invalidate that simple reduction and require the coupled film problem.

[Azizi et al. (2015)](https://doi.org/10.1016/j.cherd.2014.07.023) report ~1.2×10⁹ L/(mol s) for HS−/HOCl at 25°C and identify acid-base kinetic limitations. [Vilmain et al. (2014)](https://doi.org/10.1016/j.cherd.2013.07.015), [author manuscript](https://hal.science/hal-00875684), infer apparent H2S/ClO− and H2S/HOCl coefficients of 6.75×10⁶ and 1.62×10⁵ L/(mol s) at 293 K. They are effective absorption-model coefficients, not interchangeable elementary HS− constants. Do not combine these two descriptions as independent reactions. The manuscript is readable in this [mirror](https://paperzz.com/doc/7870168/kinetic-study-of-hydrogen-sulfide-absorption-in-aqueous); full publisher access was unavailable.

Suggested two chemical policies: (a) explicitly named empirical apparent H2S/chlorine absorption kinetics with its fitted regime; (b) mechanistic finite proton transfer plus HS oxidation, only after reverse rates, buffers, ion transport, and product pathways are sourced. Reactions with OH−, HOCl proton donation, water or other buffers may participate in proton transfer. Assuming only OH− supplies protons/base is itself a reduced model.

**Oracles:** exact physical resistance sum; first-order two-boundary solution above; nonzero bulk back-diffusion; single-reaction instantaneous plane; numerical second-order film versus VKH across Ha/E∞; shared reagent exhaustion; both absorption and stripping. At Ha=1, φ≈1.3130353 and ψ≈0.8509181 (derived checks). A gas-resistance ceiling applies to zero-bulk absorption with a liquid sink, not universally to stripping. Flag an ill-defined E when its physical-reference driving force vanishes; boundary fluxes remain the useful outputs.

## 2. Packing strategy and physical properties

```text
packing_transfer(geometry, packing_data, u_g, u_l,
                 gas_properties, liquid_properties, diffusivities) -> TransferData
```

Return kL, pressure-based kG, effective interfacial area per bed volume, liquid holdup if supported, and regime/domain diagnostics. Compute u_g=F_g RT/(P A_cross) for ideal gas and u_l=Q_l/A_cross. The packing model receives physical superficial velocities, not gas molar flow interpreted as liquid volume. Its species diffusivities affect each coefficient; the geometric area is shared.

### Onda (1968), random packing

With a_t geometric area/bed volume, d_p nominal size, L=ρ_L u_L, G=ρ_G u_G:

```text
ReL = L/(a_t μL); FrL = a_t L²/(g ρL²); WeL = L²/(ρL σ a_t)
a_w/a_t = 1 − exp[−1.45(σ_c/σ)^0.75 ReL^0.1 FrL^−0.05 WeL^0.2]
kL = 0.0051(μL g/ρL)^(1/3) [L/(a_w μL)]^(2/3)
     × [μL/(ρL DL)]^−1/2 (a_t d_p)^0.4
kG = c (a_t DG)/(RT) [G/(a_t μG)]^0.7
     × [μG/(ρG DG)]^(1/3) (a_t d_p)^−2
```

Use SI throughout, including σ and σ_c in N/m. **Primary text uses c=2.00 for small rings/saddles below 15 mm, otherwise 5.23; do not silently use the 12 mm threshold from later summaries.** Applicability includes the tested random packing shapes/materials; liquid correlation requires surface tension roughly above 0.050 N/m. Reported liquid and wet-area scatter is about ±20%, gas vaporization about ±30%. These are not guarantees for fouled industrial packing. [Onda, Takeuchi & Okumoto, primary paper, equations 1–3](https://www.jstage.jst.go.jp/article/jcej1968/1/1/1_1_56/_pdf).

Keep σ_c explicit and material-specific. No invented universal plastic value. At u_L=0 branch to the declared dry-bed model before evaluating singular groups. No residual 98% removal factor.

### Billet & Schultes (1999), below loading

Let a be geometric area/bed volume, ε void fraction, ν=μ/ρ, d_h=4ε/a, h_L liquid volume/bed volume:

```text
hL = [12 μL uL a²/(g ρL)]^(1/3)
kL = CL 12^(1/6) √(uL/hL) √(DL/dh)
kG = CV DG/[RT √(ε−hL)] × √(a/dh)
     × [uG/(a νG)]^(3/4) × (νG/DG)^(1/3)
a_e/a = 1.5(a dh)^−0.5 (uL dh/νL)^−0.2
        × (uL² ρL dh/σ)^0.75 × (uL²/(g dh))^−0.45
```

CL/CV are packing-specific tabulated constants. These equations apply at/below loading; higher loads need their holdup/area extensions and loading/flooding calculation. Mass-transfer data span gas load factor uG√ρG≈0.003–2.77 √Pa and liquid loads 0.256–118 m³/(m² h); broad database ranges do not validate every packing at every combination. Source: [Billet & Schultes (1999), equations 6–11 and Table 1](https://doi.org/10.1205/026387699526520), [primary-paper scan](https://www.researchgate.net/profile/M-Schultes-2/publication/257096495_Prediction_of_Mass_Transfer_Columns_with_Dumped_and_Arranged_Packings_Updated_Summary_of_the_Calculation_Method_of_Billet_and_Schultes/links/5acb038faca272abdc625a68/Prediction-of-Mass-Transfer-Columns-with-Dumped-and-Arranged-Packings-Updated-Summary-of-the-Calculation-Method-of-Billet-and-Schultes.pdf).

Do not call this a complete hydraulic correlation above loading, or assume its countercurrent validation establishes co-current hydraulics. The effective area a_e is not necessarily constrained by a geometric wetted-fraction interpretation: jets/droplets can contribute interface. Do not clip a_e/a to one and still claim the original correlation. Keep a_w (wetted solid) and a_e (gas–liquid interface) semantically distinct even where Onda equates them.

### Property providers and source confidence

Inject properties independently of packing. Water approximations are explicitly dilute-water policies; chloride-rich liquor can have different μ, σ, density, diffusivity and solubility. Milestone A evaluates Hcp(T) and aqueous equilibrium; it is not automatically a liquid-mixture property package.

| Property | Recommended source/policy | Check and limitation |
|---|---|---|
| Water μ(T,ρ) | [IAPWS R12-08](https://iapws.org/technical-guidance/release/viscosity); near-critical correction unnecessary here | About 0.890 mPa s at 298.15 K near atmospheric pressure. Use tabulated release checks before claiming an implementation validated. |
| Water ρ(T,P), simpler μ(T) at 0.1 MPa | [IAPWS SR6-08(2011)](https://iapws.org/technical-guidance/release/LiquidWater) | Temperature-only low-pressure policy, 253.15–383.15 K including metastable liquid; not concentrated-stock data. |
| Water σ(T) | [IAPWS surface-tension release](https://iapws.org/technical-guidance/release/Surf-H2O) | σ=0.2358 τ^1.256(1−0.625τ) N/m, τ=1−T/647.096; triple to critical point on saturation curve. About 0.07197 N/m at 25°C. |
| Molecular H2S diffusivity | [Tamimi, Rinker & Sandall (1994)](https://doi.org/10.1021/je00014a031), water measurements 293–368 K; [readable primary-paper reproduction](https://www.scribd.com/document/502656944/H2S-en-agua) | Table 2: T=(293,298,303,313,333,353,368) K, D=(1.75,1.93,2.07,2.55,3.87,4.62,5.49)×10⁻⁹ m²/s. Interpolation of these measured values is immediately defensible; the viscosity-fit typography must be checked visually before transcribing its coefficient. |
| HOCl and OCl− diffusivity | [Chao (1968)](https://doi.org/10.1149/1.2410933) | Vilmain reproduces D_HOCl=10^(−945/T−5.72), D_OCl=10^(−1060/T−5.40) m²/s; at 293 K ~1.13 and 0.96×10⁻⁹. Original Chao temperature range/media not verified here: these are provisional reproductions, not a source-validated universal default. |

Where a primary D(T) fit is unavailable, an explicitly estimated interpolation D(T)=D_ref (T/T_ref) μ(T_ref)/μ(T) is a Stokes–Einstein-inspired engineering approximation, not measured H2S/HOCl data. Carry that uncertainty and do not fit it simultaneously with unrestricted film coefficients. Gas diffusivity and gas viscosity/density are also required; use measured mixture data or a separately sourced gas-property strategy. Water-only data cannot provide DG.

**Unknown packing:** inject a reduced conductance strategy now; defer named packing constants until geometry/material arrive. Under dilute zero-bulk absorption, measurements constrain K_G a_e V_bed/F_g through −ln(y_out/y_in), where 1/K_G=1/kG+1/(Hcp kL E). They identify a_w kG only when gas control is independently justified. Unknown volume adds another confounder. Fit one effective NTU/conductance scale with positive parameters and engineering bounds tied to geometry/loads; no defensible universal numeric prior bounds exist without those data. Separating area, gas film, liquid film and kinetics needs independent measurements or deliberately varied regimes. Validate with held-out changes in recirculation, gas load, reagent and temperature, not just one operating plateau.

**Oracles:** dimensions; the Onda primary equations and Billet packing constants reproduced from a selected verified packing row; monotone/finite conductance in the admitted domain; zero irrigation branch; domain/flooding rejection; reference property values. Source-fit scatter is model uncertainty, not a numerical tolerance.

## 3. Canonical quasi-steady packed column

```text
solve_column(gas_in, liquid_in, context, geometry, parameters,
             film_policy, packing_policy, static_grid) -> ColumnResult
```

One implementation owns co-current and countercurrent boundary placement. Static N axial finite volumes is the canonical kernel. Countercurrent is a boundary-value problem with both inlet boundary conditions; marching the liquid and gas independently or using yesterday's liquid profile is not the same problem. Co-current permits a forward solve, but retains the same balances and local closures. First-order upwind cell closures are a sensible initial choice; increase N until both outlets and reagent/pH profiles are converged. A small fixed N is a discretization decision, not a physical constant.

Each cell has complete gas component molar face flows and declared liquid basis amount face flows. Define t_i as the molecular transfer vector from gas into the liquid film, and ξ_i as integrated film/bulk reaction extents. In coordinates following each phase's own flow:

```text
gas_out_i − gas_in_i    = −t_i                [no gas reaction in this policy]
liquid_out_i − liquid_in_i = B t_i + S ξ_i
t_i = A_cross Δz_i a_e,i J_interface,i
```

B maps transferred molecular species to independent liquid totals/strong-charge basis. S maps balanced reactions into that same basis. Film molecular boundary fluxes provide an equivalent accounting route, **not a second additive source**: do either B t_i+S ξ_film,i or the mapped film→bulk boundary flux vector. Add bulk slow reactions only for an explicitly modeled bulk liquid volume, excluding the separately modeled film volume. Reaction source rates × hL ΔV are appropriate only when that holdup and residence model are declared. A pH equilibrium solve alone supplies no irreversible oxidation extent.

Resolve each cell's bulk speciation, film/interface state and packing properties consistently. Use stoichiometrically balanced full molecular reaction records (including water/protons where needed), then reduce to independent totals. Chlorine electrons/product assumptions are explicit. Do not impose sulfate as the only fate of all absorbed sulfur without evidence; elemental sulfur or intermediates need their own channels/retention model. Finite reagent supply is enforced by the coupled column residual and admissible solution, not min-clipping separately requested reactions.

`ColumnResult`: complete gas and liquid outlets, species transfer rates, named film and bulk extents, cell profiles, scaled material/element/charge/oxidant residuals, root status, property-domain flags and derivative status. Untouched gas molar flows pass through before total flow and fractions are recomputed. Untouched liquid amounts undergo the declared flow/volume change. If incompressible constant liquid Q is prescribed, state that absorbed-solute partial-volume effects are neglected; evaporation or materially significant volume effects require explicit solvent transfer/volume closure. Prescribed isothermal T and P are legitimate initial policies; TI0880 is gas intake T, not a substitute for liquid T. A nonisothermal/pressure-drop extension needs energy/hydraulic residuals and dependencies, not output temperature averaging.

### Analytic oracle, not a second plant kernel

For constant total gas F, constant P, constant K_G a_e, and constant equilibrium y*=C_Ab/(Hcp P), the derived plug-flow limit is:

```text
NTU = K_G a_e V_bed P/F
y_out = y* + (y_in−y*) exp(−NTU)
```

It is valid for dilute gas and an undepleted effectively fixed liquid sink. At y*=0, removal=1−exp(−NTU). Use this as a solver oracle and potentially a narrowly declared local limit, not a parallel scrubber model chosen by an arbitrary efficiency switch. With finite first-order film chemistry replace the local sink relation consistently using φ/ψ; don't assume the ordinary physical equilibrium y* remains its no-flux point. For variable liquid chemistry, integrate the full residual; recirculation rate affects reagent supply and wetting as well as liquid contact, so removal is not fixed by L/G alone.

### Absent phases and quasi-steady limits

Zero gas inlet in the **throughflow-only, no gas-generation policy** returns zero gas material and no interphase transfer; liquid bulk chemistry can still transform a flowing liquid. Volatile generation/evaporation into an initially absent gas requires an explicit pressure/vent model and is outside that policy. Zero liquid flow is not physically synonymous with a dry tower: retained liquid can absorb transiently. The canonical drained-bed policy declares zero liquid holdup/area at zero irrigation, gas bypass, and no film reaction. Modeling wet shutdown requires dynamic liquid holdup/drainage, not borrowing an unlimited stagnant sink. Zero bed/area bypasses interphase transfer; separately declared liquid residence chemistry remains distinct.

Quasi-steady neglect of column storage needs τ_column much shorter than forcing and sump dynamics. Check hL V_bed/Q_l and gas void residence time once geometry arrives. If appreciable changing packing inventory exists, quasi-steady outlets alone cannot conserve transient system inventory. Adopt a dynamic column extension instead of hiding storage as sump inventory.

### JAX and calibration

N, film-grid size, species/reaction ordering, geometry topology and boundary direction are static. Use a bounded fixed-iteration Newton/trust-region or continuation schedule with admissible step lengths; compute a scaled final residual, never equate exhausting the iteration budget with success. Countercurrent recycle iteration can fail near pinch; do not promise simple repeated marching converges. Pure deterministic initial guesses avoid hidden mutable warm-start caches. Report invalid states without manufacturing a valid outlet; the caller decides evaluation failure handling.

For converged regular F(x,θ)=0, use implicit derivatives dx/dθ=−F_x⁻¹F_θ. Check linear-solve residuals and conditioning; verify against finite differences and increased solver accuracy. Differentiating a finite number of unfinished iterations gives the algorithm's derivative, not necessarily the column's. Depletion, dry-bed transitions and controller events have genuine nonsmooth boundaries; gradients there need event-aware treatment or a declared restricted calibration domain. Check mesh convergence of derivatives too.

**Oracles:** analytic NTU; co-current physical equilibration without overshoot; countercurrent finite-capacity/pinch cases from independent high-resolution solves; transfer reversal; oxidant exhaustion; cell and whole-column balances; zero-phase cases; inert extension/reordering; mesh convergence; JIT/vmap equivalence; directional gradients away from regime boundaries. No numerical mesh size or plant removal target is itself an oracle.

## 4. Generic reactive variable-volume holdup

```text
holdup_rhs(inventories, volume, inlet_amount_rates, outlet_flows,
           temperature_context, bulk_chemistry, parameters) -> HoldupRates
overflow_flow(level, downstream_level, geometry, parameters) -> FlowResult
```

Store independent inventories n and actual V, including signed strong-charge/equivalent inventories where that is the chosen basis. Recover C=n/V only in an admitted positive-volume domain. Molecular n is mol; COD/proxy inventories retain their own declared bases. Do not pretend every channel is molecular.

For a well-mixed, incompressible, constant-density policy:

```text
dV/dt = ΣQ_in − ΣQ_out + Q_volume_sources
dn/dt = Σ(amount_rate_in) − ΣQ_out C + S r_slow(C,speciation,T) V
        + declared external material sources
```

Volume sources may include evaporation/condensation, with matched solvent mass; dissolved transfer can neglect volume only under a stated dilute partial-volume approximation. A variable-density formulation conserves solvent/material mass and derives V from a density/partial-volume closure. Never combine constant-density volume bookkeeping with unexplained reaction shrinkage.

Each outlet carries its own hydraulic draw with the common well-mixed concentration, unless settling/selective removal is explicitly added. Thus overflow and recirculation export opaque channels too. Fast acid-base speciation reads the current totals and charge (milestone A). Slow kinetics consumes those molecular concentrations. Instantaneous irreversible bulk chemistry instead uses conserved invariants plus a feasible algebraic manifold and conservative initialization (ADR 018); no large elementary rate constant or repeated min subtraction in RHS. Film reaction occurs in the column and returns products; the sump cannot count it again.

For overflow choose one physical policy: (a) finite weir/outlet hydraulics, e.g. rectangular free overflow Q=(2/3) Cd b √(2g) [h−h_crest]_+^(3/2), with specified geometry/free-discharge/approach assumptions; (b) ideal fixed-level complementarity at capacity, with matched integrated discharge and event handling. Submerged weirs, pipes and siphons need their own laws. [USBR Water Measurement Manual, rectangular-weir rating](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_06.html) includes effective-head/width corrections and geometry-dependent coefficients; the simple formula here is an engineering closure, not its complete rating procedure or a known Scrubber 4 overflow curve. Obtain the correct outlet geometry before selecting it. Soft arbitrary drain plus hard volume clip is not a conservation policy.

For this plant, compose sump → recirc pump → stock mixer/fast closure → column → sump. Makeup enters the sump; stock feeds enter the pump discharge per the supplied EPCOR extraction. Pump RPM→stock volume is a boundary conversion, with measured stock concentration as separate data. Generic holdup can accept any number of inlets without hardcoding this placement. Makeup hysteresis at 695/600 mm is actuator/controller state, not weir crest or Vmax. Recirc low-level interlocks and overflow dynamics are separate. At steady level, overflow follows the total volume balance, not makeup alone when stock/flush/evaporation matter.

**Oracles:** no-reaction dilution; pure withdrawal leaves C constant while n and V decrease; stationary recirculation conserves external totals; every stock/makeup/overflow balance; zero reagent and signed charge; weir onset; integrated overflow at capacity; empty-tank draw feasibility. Numerical integration must stop/reduce an infeasible draw conservatively; clipping n or V afterwards is forbidden. No assumption that nonnegative instantaneous rates make RK intermediate stages feasible.

## 5. Electrode observation: thermodynamic features, empirical response

```text
orp_target(speciation, chloride_activity, temperature_k,
           electrode_parameters, optional_reductant_features) -> ObservationResult
electrode_rhs(sensor_state, target_potential, parameters) -> SensorRates
```

For HOCl+H⁺+2e⁻⇌Cl⁻+H₂O, thermodynamic derivation gives:

```text
E_pair = E°(T) + [RT ln(10)/(2F)]
         × [log10(a_HOCl) − pH − log10(a_Cl−) − log10(a_water)]
E_measured,ideal = E_pair − E_reference(T)
```

Activities are dimensionless and use one declared standard state. In the ideal dilute initial model a_water≈1 and a_s≈C_s/(1000 mol/m³). The 25°C feature coefficient is 29.58 mV/decade: ×10 HOCl raises E 29.58 mV; ×10 chloride or +1 pH lowers it 29.58 mV holding other activities fixed. At constant total chlorine, increasing pH additionally reduces HOCl through speciation, so that is not a constant −29.58 mV/pH experiment. The HOCl/OCl− ratio is an acid-base ratio, not the oxidized/reduced redox couple.

For an actual probe start with a declared **Nernst-feature observation surrogate**:

```text
E_target = b0 + bT(T−Tref) + s × [RT ln(10)/(2F)]
           × [log a_HOCl − pH − log a_Cl−]
```

Here `log` means log10. Center features for fitting. b0 absorbs reference and formal-potential offsets; s is empirical. This preserves a defensible structural relationship without claiming equilibrium ORP of reacting scrubber liquor. With enough independent data, permit separate coefficients on the three features, explicitly giving up the coupled Nernst slopes. Fit bT only if temperature variation identifies it; scaling RT does not specify E°(T) or reference T dependence. Chloride activity must be represented or independently observed, not fitted out of existence while chloride accumulates.

Primary electrode evidence supports caution: [Victorin, Hellström & Rylander (1972)](https://doi.org/10.1017/S0022172400022361) show chlorine-demand and compound effects on measured potentials. [Zimmermann et al. (2018)](https://doi.org/10.3390/s18082404) show platinum surface conditioning/oxide and mixed-potential behavior in an oxygen system. That is evidence for electrode-state dependence, not a numerical chlorine-probe calibration. A mechanistic mixed potential solves Σ_j i_j(E,c,T,surface)=0 using measured polarization/current laws; it does not average redox potentials or add an arbitrary sulfide Nernst term.

If reduced sulfur improves held-out residuals, add a labeled empirical feature −bS log10(1+C_reduced/C_scale), with C_scale fixed or independently constrained. This is a candidate surrogate, **not a sourced universal poisoning law**. Sulfur adsorption on Pt is experimentally documented in [Sethuraman & Weidner's fuel-cell study](https://arxiv.org/abs/1108.0609), but its electrode/electrolyte/temperature are not this ORP probe. Actual poisoning can have memory and hysteresis; only add an adsorption/recovery state when clean/reagent-switch/recovery observations identify it. A static sulfide correction cannot distinguish mixed current from surface poisoning.

Separate lag with dE_sensor/dt=(E_target−E_sensor)/τ, plus explicitly modeled drift/noise if needed. Location, electrode/reference type, maintenance/cleaning history, pH/chlorine/Cl− covariation and redundant-probe selection are calibration inputs. The supplied plant extraction does not establish probe location; don't silently attach it to sump rather than tower return or dosing line. ORP outputs never enter the chemistry as a thermodynamic state or force free chlorine from a fixed lookup table. A controller may act on the observation, while chemistry determines true inventories independently.

At zero HOCl or Cl−, the ideal single-couple logarithm is outside its observation domain. Return a validity/detection-limit indicator or use a specifically calibrated low-signal observation policy; arbitrary positive floors are numerical surrogates with arbitrary voltage and gradient consequences. Fit on actual sensor data and check calibration identifiability; ORP alone cannot identify oxidant inventory, chloride, sulfur pathway and electrode offset simultaneously.

**Oracles:** ideal feature slopes and reference-sign convention; temperature units; paired calibration-solution checks; finite-difference sensitivity; lag response E(t)=E_target+(E0−E_target)e^(−t/τ); low-oxidant invalidity; held-out switching/cleaning episodes. Sensor correction must leave chemical mass and charge balances unchanged.

## Composition with milestone A and ADRs

Milestone A supplies sourced temperature-dependent equilibrium constants, molecular identities, aqueous speciation and molecular Henry functions. Add diffusivity/fluid-property providers beside that layer, not inside a scrubber class. Reaction records add sourced rate expressions and balanced stoichiometry; the transport binder maps molecular fluxes to family totals and charge without separately storing equilibrium species.

ADR 016: kernels gather only declared local fields, return local rates/extents, and never own or truncate a whole carrier vector. The carrier maps both gas and liquid changes and preserves untouched amounts. ADR 017: the column's two outlets and diagnostics belong to one atomic coupled evaluation group, with explicit flow/composition/T/P dependencies; pressure or thermal policies add only the dependencies they use. Results are reused only within one resolution call, not between RK stages. Holdup outlet concentration remains state-derived while pump flow can be prescribed. ADR 018: local algebraic film/column/speciation solves are valid within a declared regime; finite stiff dynamic bulk chemistry still needs a joint implicit plant integrator. ADR 015's joint continuous-time plant remains intact.

The genuinely missing plant information remains geometry/packing and hydraulic direction, validated gas/recirc flow units, liquid temperature, stock concentrations/active strength, probe locations and chloride/buffer inventory. These affect identifiable model choices. None is filled by adopting a famous packing correlation or by matching >95% removal in one steady condition.
