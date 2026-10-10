# Process-control physics reviews and EPCOR plant evidence

Restored on 10 October 2026 from the original session log after the scratchpad was erased. The two reviews and equipment extraction below retain their original text and citations, including later corrections to Review 2. They describe the source at review time; later milestones supersede migration instructions. Original scratchpad references are historical and no longer resolve.

## Original review 1

# Independent review: H₂S scrubber physics and reusable building blocks

Reviewed 2026-10-10. Read-only review; no repository files changed, no experiments launched. Core HEAD at inspection: `9513d97fc2cc4c058fe38f810bca84504019994a`; unrelated working-tree changes were present and left alone. Line numbers below refer to the live files inspected, not indexed documentation. Ragdocs found the original reduced-order design and calibration records; Repowise returned no matching live targets, so source was read directly. Memory search returned no clearly applicable physics record.

## Decision

Replace the empirical removal multiplier and sigmoid chemistry with one conserved, rate-based composition of thermodynamics, film transport/reaction, liquid inventory, and instrument models. A stateless tower plus dynamic sump remains a viable reduction when tower residence times are short relative to control steps. The defect is not statelessness: it is that the interface communicates only absorbed H₂S mass, excluding the oxidant consumption, products, and acid/base consequences of reaction inside the tower.

The strongest new findings are the split temperature semantics, inconsistent pH in calibrated stripping, absent carbonate/CO₂ balance, and conservation-breaking integration/clamping. The reported ~0.73 removal ceiling is reproducible from the *initial default conditions*, but is not a universal bound. The current benchmark uses base efficiency **0.99**, overriding the unit's **0.90**. At 25 °C and nominal flows its ideal-chemistry upper bound is ~0.807, still below the 0.95 target. Cold measured replay or fitted parameters can exceed that bound.

## Scope and shorthand

All of `gas_liquid_contactor.py`, `scrubber_sump.py`, `dosing_system.py`, `chemistry/ph_model.py`, and benchmark wiring were read end to end. Also inspected all chemistry modules, `components.py`, `streams.py`, `kinetics.py`, `transport.py`, unit/flowsheet contracts, CSTR, tank, aerated bioreactor, and relevant digester helpers. Read 009 diary headings and recent entries, 015 plan/diary, and calibration/replay parameter mappings.

In findings below:

- **GC** = `core/domains/process-control/src/process_control/units/gas_liquid_contactor.py`
- **SS** = `core/domains/process-control/src/process_control/units/scrubber_sump.py`
- **HB** = `core/domains/process-control/src/process_control/benchmarks/h2s_scrubber.py`
- **DS** = `core/domains/process-control/src/process_control/actuators/dosing_system.py`
- **PH** = `core/domains/process-control/src/process_control/chemistry/ph_model.py`
- **CAL** = `projects/process-control-baselines/process_control_baselines/h2s_calibration.py`
- **DATA** = `projects/process-control-baselines/process_control_baselines/h2s_real_data.py`

“Confirmed” describes source behavior or algebra. Plant significance and unmeasured mechanisms are identified separately; no multi-season fit or numerical plant trajectory was performed here.

## Literature and first-principles structure

### Solubility, speciation, and pH

Choose an explicit Henry convention: `Hcp = molecular dissolved concentration / gas partial pressure`, in mol m⁻³ Pa⁻¹. Then `C*H2S = Hcp(TL) pH2S`. Sander's current compilation gives representative H₂S values near `1.0e-3 mol m⁻³ Pa⁻¹` at 298.15 K and reciprocal-temperature coefficients near 2100 K. Use `Hcp(T) = Hcp_ref exp(B(1/T − 1/Tref))`; increasing temperature lowers this solubility constant. The 2023 compilation supersedes 2015. Constants apply to **molecular H₂S**, not total dissolved sulfide. [Sander H₂S data](https://henrys-law.org/henry/casrn/7783-06-4), [compilation](https://doi.org/10.5194/acp-23-10901-2023), [NIST convention and temperature equation](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7783064&Mask=1F).

For an ideal diprotic acid with `h=[H+]`, `D=h²+Ka1 h+Ka1 Ka2`, the fractions are `α0=h²/D`, `α1=Ka1 h/D`, `α2=Ka1 Ka2/D`. Apply to H₂S/HS⁻/S²⁻ and the apparent CO₂*/HCO₃⁻/CO₃²⁻ family. HOCl/OCl⁻ uses the monoprotic version. These are equilibrium fractions, not efficiency multipliers. At plant pH around 9.4–10, HS⁻ predominates and molecular H₂S is small; omission of S²⁻ is a defensible approximation there. Do not casually choose the historically disputed pKa2≈13 for general high-pH use: primary spectroscopy revisited it. [Giggenbach, 1971](https://doi.org/10.1021/ic50101a002). Temperature-dependent HOCl dissociation is independently measured. [Morris, 1966](https://doi.org/10.1021/j100884a007).

Solve electroneutrality from analytical totals plus strong-ion charge, or from a precisely defined reaction invariant. Water, carbonate, sulfide, chlorine, and relevant product ions contribute. The existing Henson–Seborg solver is the right numerical starting point, but currently covers only a fixed carbonate family. [Henson & Seborg, 1994](https://sites.chemengr.ucsb.edu/~ceweb/faculty/seborg/cv/pub_pdfs/134.pdf).

Important bookkeeping distinction: neutral H₂S absorption consumes **free hydroxide** as HS⁻ forms, but does not simply destroy generalized titration alkalinity when HS⁻ itself is included in that alkalinity definition. Likewise neutral CO₂ absorption changes inorganic-carbon total and pH, while caustic is needed to hold pH; it is not automatically an equal loss of total alkalinity. Choose one invariant definition and derive its changes from stoichiometry. Do not transplant the present “alkalinity consumed per mg” into a charge-balance model unchanged.

CO₂ is a required potential co-absorbate, not an optional cosmetic correction. Its uptake competes for caustic and changes carbonate buffering; its magnitude in Gold Bar gas is unmeasured here. Simultaneous H₂S/CO₂ absorption has been modeled and measured with coupled diffusion/reaction. [Bontozoglou & Karabelas, 1993](https://doi.org/10.1021/ie00013a022). Trimeric also documents operating/design consequences and selective-contact strategies; its refinery examples are not direct calibration data for this WWTP. [Trimeric, AFPM 2014](https://trimeric.com/assets/caustic-scrubber-designs-for-h2s-removal-from-refinery-gas-streams---afpm-2014.pdf).

### Packed tower and reaction enhancement

Obtain physical `kL`, pressure-based `kgp`, wetted interfacial area `a`, and liquid holdup from packing geometry/properties and phase superficial velocities. Onda is a random-packing correlation family; Billet–Schultes covers random/structured packings with packing-specific parameters and hydraulics. Neither is a universal “kLa from recirc flow” formula. Need tower cross section/height, packing identity, voidage, area, fluid density/viscosity/surface tension, diffusivities, pressure, and gas/liquid throughput. [Onda et al., 1968](https://doi.org/10.1252/jcej.1.56), [Billet & Schultes, 1999](https://doi.org/10.1205/026387699526520).

Physical two-film flux obeys `J=kgp(pbulk−pi)=kL(Ci−Cb)` with `Ci=Hcp pi`; thus `1/Kp=1/kgp+1/(Hcp kL)` for a pressure driving force. Chemical enhancement modifies the liquid film, not an arbitrary maximum efficiency. In the appropriate pseudo-first-order, negligible-bulk-solute regime, `Ha=sqrt(D k_app)/kL` and film `E=Ha/tanh(Ha)`. For fast reaction E grows with reaction rate, but liquid-reactant depletion limits it and gas-film resistance eventually dominates. For finite bulk solute the first-order film solution is instead `J=kL Ha[Ci coth(Ha)−Cb csch(Ha)]`; this reduces to physical transfer as Ha→0. A single E multiplied by an arbitrary bulk gradient is not general. These standard film reductions are documented in primary absorption work; they are **conditional reductions**, not the proposed universal sulfide law. [Vilmain et al., 2014](https://doi.org/10.1016/j.cherd.2013.07.015), [Rochelle group absorption report, eqs. 5.17–5.18](https://sites.utexas.edu/rochelle/wp-content/uploads/sites/3310/2015/06/Texas-Carbon-Management-Program-Fourth-Quarterly-Report-2015.pdf).

In a packed column, integrate gas and liquid balances axially with the actual co-/countercurrent arrangement. A useful limiting oracle for dilute gas with an essentially constant liquid sink and constant coefficients is `yout=yin exp(−NTU)`, `NTU=Kp a P A Z / Fgas`, where Fgas is molar carrier throughput. Then 95% requires NTU≈2.996 and 99%≈4.605. Those numbers are algebraic consequences, not equipment calibration. For finite loading use the equilibrium driving force and operating line; solve the countercurrent boundary-value problem when warranted. Products and reaction extents must leave the tower in its liquid outlet, with equal/opposite gas transfer.

Primary scrubber studies demonstrate high removal but cannot be copied as a plant guarantee. Chen et al. report packed-column NaOCl/NaOH performance including 99.2% in their apparatus and an HTU range; their pH-11 operating conclusion is specific to that study, not a universal minimum for Gold Bar. [Chen, Huang & Yang, 2001](https://doi.org/10.1002/ep.670200313). Biard/Couvert et al. achieve >95% in a compact high-velocity co-current mesh contactor; that is a different hydraulic device from a conventional packed tower. [Biard et al., 2010](https://doi.org/10.1002/cjce.20310).

### Oxidation: chemical species and products matter

Balanced limiting net reactions on the current H₂S/NaOCl mass basis are:

- `H₂S + OCl⁻ → S⁰ + Cl⁻ + H₂O`: one oxidant mole per sulfur mole.
- `H₂S + 4OCl⁻ → SO₄²⁻ + 4Cl⁻ + 2H⁺`: four oxidant moles per sulfur mole.
- On an HS⁻ basis: `HS⁻ + OCl⁻ → S⁰ + Cl⁻ + OH⁻`, versus `HS⁻ + 4OCl⁻ → SO₄²⁻ + 4Cl⁻ + H⁺`.

The sulfate limit needs `4×74.44/34.08≈8.737 mg NaOCl/mg H₂S`; the elemental-S limit needs ~2.184. Product distribution also depends on oxidant availability and reaction progression, not just pH. Track S⁰ and sulfate at minimum; additional intermediate sulfur pools require evidence-supported closure. [Biard et al., competitive kinetics, 2015](https://doi.org/10.1016/j.cherd.2014.07.023).

That study identifies rapid HS⁻+HOCl reaction over pH 6–12, with `k≈1.2e9 L mol⁻¹ s⁻¹` at 25 °C, and acid/base reaction limitations in its kinetic simulations. Thus the small HOCl fraction at high pH can still matter greatly. **Do not apply this number to total sulfide×total chlorine without speciation.** Vilmain's earlier absorption study identifies H₂S-specific HOCl/OCl⁻ channels; these are different reactant pairs, not interchangeable rate constants. It also explicitly questions assumptions underlying Bonnin's old apparent rate estimate. Do not canonize the often repeated `1.8e8` apparent Bonnin constant. [2015 study](https://doi.org/10.1016/j.cherd.2014.07.023), [2014 study](https://doi.org/10.1016/j.cherd.2013.07.015).

My order-of-magnitude calculation, assuming ideal equilibrium, pKa(HOCl)=7.537, 200 mg NaOCl/L and pH 9.8, gives HOCl≈1.46e−5 M and a HS⁻ pseudo-first-order coefficient ~1.75e4 s⁻¹ at 25 °C. This is not a complete multistep sulfate oxidation rate or a prediction for this plant; it shows why resolving chlorine/sulfide chemistry directly with 30-second explicit RK4 substeps is physically inappropriate. Film reaction/extents or an appropriate stiff/equilibrium treatment are needed.

### ORP is an instrument closure, not a sulfur/chlorine equilibrium constant

For an equilibrated HOCl/Cl⁻ couple, `E=E0−RT/(2F) ln(aCl/(aHOCl aH))`; subtract the actual reference-electrode potential to compare with probe mV. At 25 °C the log coefficient is ~29.58 mV/decade for two electrons. Using total free chlorine at pH well above HOCl pKa adds speciation dependence, producing a limiting ~−59.16 mV/pH slope. This explains a limiting slope; it does not justify 100 mV per decade of a NaOCl-mass/H₂S-mass ratio. [Emerson's primary instrument application note](https://www.emerson.co.jp/is/content/emerson/en/measurement-instrumentation/technical/products/unaudited/documents/dl-rmt-00800-0100-4044.pdf).

Actual chlorine/reductant mixtures yield electrode-dependent mixed potentials. A calibrated observation model can be legitimate, provided chemistry is independently conserved and electrode/reference, ionic conditions, lag and fit domain are explicit. A mechanistic mixed-potential model would solve zero net electrode current across electroactive couples; it requires kinetic evidence, not just standard potentials. ORP limitations in chlorinated mixtures are documented in experimental work. [ORP/free-chlorine possibilities and limitations](https://www.sciencedirect.com/science/article/pii/S092552141930290X), [platinum sensor experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539692/).

## Ranked defect ledger

Ranking reflects expected fidelity impact, then numerical/conservation severity. An omitted mechanism is a structural gap, not proof of its measured importance in Scrubber 4.

| Rank | Impact | Location | Finding and consequence |
|---|---|---|---|
| 1 | Critical | GC:137–175 | No Henry equilibrium, gas/liquid films, geometry, axial residence, or NTU. Weighted Monod chemistry and power laws determine removal. Cannot distinguish gas-side limitation, liquid-film enhancement, reagent exhaustion, or fouling; fitted coefficients lack transferable equipment meaning. |
| 2 | Critical | SS:295–305; HB:424 | pH is an alkalinity-only tanh. Same reported alkalinity gives the same pH regardless of sulfide, chlorine, carbonate, products, temperature or ionic strength. Default nonnegative alkalinity restricts pH to approximately 5.25–10, not the docstring's theoretical 4–10. At 20 meq/L the output is ~10 regardless of buffer composition; this is not a charge-balance result. |
| 3 | Critical | SS:59–73, 204–215; GC:99–105 | No CO₂ gas input, inorganic carbon state, or carbonate buffering. Caustic demand comes only from H₂S and dilution. The absence is certain; CO₂ concentration and its plant-scale demand are unknown. |
| 4 | Critical | GC:180–193; SS:179–183, 195, 223 | All removed gas enters the sump as reduced sulfide, with oxidation only afterward. Reactive absorption can consume chlorine in the film. The interface cannot return film oxidation, sulfate/S⁰/Cl⁻ or acid/base sources, so tower reagent depletion and liquid return chemistry cannot be represented. |
| 5 | Critical | SS:181–183, 245–288; integration.py:29–38 | Explicit RK4 plus post-substep clipping is not a reagent-conserving stiff chemistry solver. Intermediate RK stages can be negative even when initial/projected states are positive; the sulfide/(sulfide+1) term has a pole at −1. Clipping oxidant and sulfide separately after consuming them can break the fixed stoichiometric ratio and remove/create inventory. Ten substeps is not a stability guarantee. |
| 6 | High | HB:362–366, 418–428; SS:84–100, 221–231, 252; DATA:180–183 | Measured TI temperature is assigned to `GasSourceTrajectory.temperature`, affects contactor efficiency and gas mass conversion, but does not force liquid temperature or sump kinetics. Sump starts at 25 °C, relaxes toward default ambient 25 °C, and clamps at ≥5 °C. Plant liquid temperature reaches ~3 °C. Henry equilibrium should use liquid/interfacial temperature; actual-volume gas conversion should use gas temperature. Current replay conflates these and cannot represent seasonal reaction kinetics. |
| 7 | High | GC:143–154 | Loading factor depends on total sulfide with fixed 10 mg/L half-saturation, independently of inlet H₂S partial pressure or molecular fraction. A liquid at equilibrium still absorbs; supersaturated liquid cannot desorb. The same total sulfide imposes the same loading suppression at different pH apart from a separate multiplier, instead of entering `Cbulk,H2S=α0 ST`. |
| 8 | High | SS:76, 181–183 | Oxidation is empirical oxidant×Monod(total sulfide), with hidden “1.0 mg/L” saturation scale and no HOCl/OCl⁻ speciation, channel-specific rates or activation energy. Dimensional comment on oxidation_rate is inconsistent with its use as a concentration-rate coefficient. Incorrect pH response is especially consequential across pH-control changes. |
| 9 | High | SS:50–57, 70–73, 223 | Sulfate yield and acid consumption fixed for every oxidized sulfide molecule, even oxidant-starved operation. No S⁰/sulfate/intermediates/chloride inventories. `8.7` is a reasonable rounded sulfate-limit mass ratio, **not itself an arithmetic error**; the defect is unconditional product closure. |
| 10 | High | SS:69, 73, 207–215 | “Alkalinity” has no defined invariant/species basis. Free-caustic consumption, neutral-acid absorption and oxidation proton release are manually subtracted. It can approximate free-base reserve under assumptions, but cannot serve as conserved generalized alkalinity. Bleach's own OCl⁻ acid/base contribution, stock NaOH and makeup buffering are absent. |
| 11 | High | SS:188–190, 250, 283–289; HB:156–157, 532 | Drain starts above 10 m³ and reaches steady state at `10+Qmakeup(m³/h)/10`, e.g. **10.12 m³** at 20 L/min. Diagnostic accumulates only volume above **15 m³**, missing the modeled drain. It reports zero while physical overflow persists; nominal==max is permissible, the overflow closure/report is wrong. |
| 12 | High | SS:217–226; GC:178; HB:492, 519 | Sump stripping removes sulfur from liquid without adding it to any gas outlet/report. No receiving headspace partial pressure or transfer driving force. Sulfur disappears and computed removal omits re-emission, particularly during pH collapse. |
| 13 | High | SS:308–326 | ORP uses fixed 450+100 log10((oxidant+.01)/(sulfide+.01))−pH correction. Chloride/product activity, electrode reference, temperature and competing couples omitted. Ratio compares mass concentrations of species that are not a conjugate redox pair; arbitrary .01 floors set oxidant-free behavior. Suitable only as an explicitly empirical probe surrogate. |
| 14 | High | CAL:217–227, 284; SS:221; HB:424 | Calibration populates dynamic.ph_amplitude but omits sump.ph_amplitude, leaving default 3.0. Reported pH/contactor/ORP can use a fitted amplitude while stripping uses another pH. Same split can occur for contactor pKa changes: the factory sets contactor pKa but does not propagate it to sump pKa. One shared speciation closure removes both discrepancies. |
| 15 | High | DATA:71, 284, 317; HB:181, 243; CAL:330 | Raw recirc transmitter is declared m³/h and its median is used without verified conversion; its variability is discarded. At 2375 L/min it would be 142.5 m³/h; at 2375 US gpm ~539.4 m³/h, versus raw 2375 m³/h. No choice is justified yet. Gas flow actual/standard, pressure, pump percent-to-volume calibration and solution concentrations also need plant metadata. |
| 16 | High | CAL:898–925 | Stoichiometry/supply physics are free fit dimensions: H₂S alkalinity consumption starts at 1.0 with range .01–5 vs sourced .0293, and caustic yield starts .5 with range .01–5 vs .019 for stated 50% NaOH. These large bounds permit missing CO₂, pump scaling and pH errors to be absorbed into supposedly physical yields. Separate uncertain plant supply metadata from fixed molar stoichiometry. |
| 17 | High, default regime | HB:172–180, 191; GC:170–175 | Warm default ceiling conflicts with target: at 25 °C, Qg=500, Ql=50, chemistry fractions≤1, η≤`.99×.98^10×(.98+.02×50/55)`≈**.80743**. Initial oxidant 200, alkalinity 20, sulfide .5 gives **.73493**. These are analytic evaluations, not a demonstrated max-setpoint rollout. Cold replay/fitted coefficients can exceed .95; a universal “can't tune around .73” claim is false. |
| 18 | Medium–high | GC:160–172 | At Ql=0, hydraulic recirc multiplier retains `(1−weight)`, 0.98 in benchmark; wet packed contact persists without irrigation. At Qg→0 the gas power term is arbitrarily regularized. No dry bed, wetting threshold, loading/flooding regime, pressure drop or entrainment. Infinite chemistry must not overcome a dry column. |
| 19 | Medium–high | HB:423–479, 481–524 | Contactor evaluated once from old sump, then load held over five minutes while chemistry evolves; newly calculated pump flow applied throughout that step. This lag/splitting can bias breakthrough, reagent demand and action timing. Observation mixes new temperature/volume with old-chemistry outlet/true PVs. Quasi-steady contactor should be reevaluated within the coupled RHS; instrument timestamps should be explicit. |
| 20 | Medium | SS:185–190, 194–215 | Volume includes makeup only, excluding liquid chemical feed, bleed, evaporative loss and entrainment. Chemical feed volume is small at defaults but grows with dose. Makeup is pure diluent with no minerals/temperature. Salt/product flushing therefore cannot explain the full makeup tradeoff. |
| 21 | Medium | SS:228–231, 252 | Thermal balance omits incoming-water/gas sensible heat, evaporation and stock dilution heat. reaction_heat uses a whole gas-to-sulfate enthalpy proxy in the sump although heat can be released in the tower. Hard temperature clipping creates/deletes energy. Thermodynamic heat capacity and process boundaries should replace unrelated degree-per-mass factors when temperature is predictive. |
| 22 | Medium | GC:163–166 | Constant θ-per-degree modifies efficiency directly, rather than Hcp(T), diffusivity, viscosity, dissociation and rates. In gas-film-limited enhanced absorption, reduced liquid solubility need not produce the same efficiency slope. A fixed .98 coefficient can confound seasonal hydraulics/chemistry. |
| 23 | Medium | GC:188–193 | Ideal-gas mass conversion is correct **if** Qg is actual m³/h at Tgas and 101.325 kPa. No pressure or normalized-flow basis is represented. Using this temperature correction with Nm³/h would be wrong. Humid/wet ppm basis is also unstated. This is a metadata/model-boundary defect, not a missing factor of 1000. |
| 24 | Medium | SS:48–49, 196; HB:158; SS:159–162 | Unit bleach_yield default 1 mg/mL differs by 150× from benchmark 150. Concentration is not identified as active NaOCl vs available chlorine as Cl₂; natural decay has no products and one fixed rate. Unit reset ignores RNG and always uses the create() 25 °C/chemistry defaults. A factory should specify stock composition, initial totals and temperature explicitly. |
| 25 | Medium | DS:124–130; HB:95 | Sensor lag is per-step and random-walk drift has fixed per-step variance, so changing dt changes physical instrument dynamics. Use α=exp(−dt/τ) and diffusion increment σ√dt with stated time basis. At fixed five-minute replay a fitted lag is usable, but not transferable across step sizes. |
| 26 | Medium, objective | HB:194–196, 342–345, 499–504 | Unit costs labeled $/mL or $/L multiply flow/min to form a cost rate, then called per-step OPEX without `60dt` integration. Compliance deficit also summed per step; relative scaling changes if only OPEX is fixed. Monetary units/prices are unverified, and fee/penalty weights are choices, not physics. Integrate both desired running objectives consistently; distinguish costs from arbitrary reward weights. |

Two claims worth resisting: (1) concentration derivatives need not have an explicit overflow sink when `dV/dt=Qin−Qout` is consistently included—well-mixed outflow cancels from dC/dt; SS's omission there is not independently a double-counting bug. Track inventories for clarity and discharge reporting. (2) makeup control is a **flow** loop in actual wiring (HB:443, 461–467), consistent with the brief, despite stale “level control”/m³ comments at HB:141–143. Do not redesign it as level control without P&ID evidence.

## Existing primitives and what to extend

| Existing location | What already exists | Reuse / limitation |
|---|---|---|
| `components.py:9–39` | Static ComponentSchema, names, particulate flags, COD→TSS mapping | Keep static ordering and construction validation. It has no molar mass, charge, elemental composition or concentration basis; add chemical metadata separately, rather than treating COD/mg/L and mol/m³ vectors as interchangeable. |
| `streams.py:12–44` | JAX Stream and conservative flow-weighted mix; split also exists | Reuse routing/mixing for consistent liquid species bases. Stream currently lacks pressure, temperature and phase/basis metadata. Add explicit phase thermodynamic context, especially before gas/liquid exchange. |
| `units/contract.py:27–47` | ContinuousUnit derivatives/outlet flows/outlet compositions/project | Adopt this as the unit boundary rather than inventing a scrubber-specific plant engine. Support different schemas on gas and liquid ports; reactions/transfer can remain pure strategies. |
| `flowsheet.py:136–195` | Coupled RHS, whole-plant RK integration, outlets/report | Already solves interaction within derivative evaluations, unlike HB's old-sump/frozen-load step. Reuse for recirculation; take stiffness and accumulated discharge diagnostics seriously. |
| `units/tank.py:69–116` | Realized flow, capacity overflow, unmet withdrawal | Use its inventory-limit semantics; extend to carry composition/amount and discharged species. Do not duplicate the soft 10×volume drain. Event-aware/conservative chemistry integration still needed. |
| `units/cstr.py:34–58` | Exact constant-volume linear mixing | Keep for its valid constant-volume regime; not a complete variable-volume reactive sump. |
| `chemistry/ph_model.py:45–95` | Fixed-iteration damped Newton charge-balance solver | Generalize to arbitrary weak-acid families/temperature/totals and residual diagnostics. Its carbonate-only helper is reusable evidence; copying a second bespoke scrubber solver is unnecessary. |
| `kinetics.py:8–24` | Parameterized reaction-rate/temperature protocol | `oxygen` is mandatory, making it an aeration-facing interface rather than a truly general reaction interface. Separate reaction capability from aeration species selection. Do not fabricate an oxygen component for chlorine chemistry. |
| `units/bioreactor.py:30–74` | Vector CSTR, injected kinetics and oxygen saturation | Reuse reaction/mixing shape. General film transfer should eventually also serve aeration without converting oxygen units incorrectly. |
| `units/anaerobic_digester.py:235–303` | Private temperature-corrected carbonate split, Henry law, exact gas relaxation | Harvest general thermodynamic/transfer math into canonical helpers, preserving explicit unit conversions. Digester's reduced carbonate approximation is not the general charge solver. |
| `units/contact_basin.py`; chemistry demand modules | Chlorine speciation-related behavior and demand consumption | Reuse species constants and bounded demand/extent concepts; do not copy empirical wastewater demand coefficients into scrubber kinetics. |
| `transport.py:10–89` | Water-treatment Hydraulics/Composition/BulkProperties object | Fixed chlorine/demand/ammonia/turbidity/organics fields are not a general species transport representation. Prefer existing Stream for new chemistry, with explicit thermodynamic metadata. |
| `integration.py` / `_jax_dataclass.py` | Pytree RK4, JAX registration | Retain pytree/JIT conventions. Existing RK4 projection is not a valid stiff reaction solver merely because it is convenient. |

## Target decomposition and interfaces

Names are proposed responsibilities, not an instruction to implement every file immediately. Keep general physics in process_control under `core/domains/process-control`; plant-specific values/calibration stay in the benchmark/project. Do not introduce a parallel legacy physics path.

| General building block | Proposed boundary | Other concrete reuse |
|---|---|---|
| `chemistry/species.py` | Static species metadata: IDs, charge, molar mass, elements, phase, analytical-total grouping, parameter/source provenance. Construction-time atom/charge validation. Dynamic concentration arrays in a stated molar basis. | Chlorination, ferric precipitation, digestion, ammonia stripping; interoperates with ComponentSchema ordering. |
| `chemistry/equilibria.py` + generalized `ph_model.py` | Weak-acid family speciation from pH/T and Ka parameters; charge solve from totals and strong-ion invariant. Return pH, species concentrations, charge residual and convergence/domain status. | Existing neutralization benchmark, carbonate in digester, HOCl in contact basin, sulfide scrubber. |
| `chemistry/solubility.py` | Hcp(T, parameters), equilibrium concentration/partial pressure and documented units; salt correction only when supported by ionic measurements/model. Harvest digester helper. | O₂ aeration, CH₄/CO₂ headspace exchange, NH₃/H₂S stripping, gas scrubbers. |
| `chemistry/reactions.py` + `kinetics.py` | Static stoichiometric matrix plus injected species-dependent rate laws; net rates `ν r`, common reaction extents, product/heat sources. Fixed channel data with sources, not a runtime string registry in JIT. Chemistry model excludes mandatory oxygen property. | Chemical chlorine demand, sulfate/S⁰ pathways, precipitation, biological reaction systems with their existing COD conversion boundaries. |
| `mass_transfer/films.py` | Two-film resistance and reaction–diffusion closure. Inputs include phase state, Hcp, diffusivities, physical kL/kgp and reaction model. Return interfacial species fluxes **and integrated film reaction extents**. Physical and pseudo-first-order analytic limits are validated special regimes of the same physical contract. | Absorbers/strippers, gas exchange in digester, reactive CO₂ uptake; no assumption that all transferred pollutant survives to bulk. |
| `mass_transfer/packing.py` | Injected packing correlation with validated parameterization; geometry+phase rates/properties → kL, kgp, area, holdup, pressure drop, operating-domain status. | H₂S, NH₃, SO₂ and CO₂ packed towers; Onda/Billet selection depends on actual packing. |
| `units/packed_column.py` | Axial balance kernel using injected transport/reaction chemistry. Ports: gas in/out, liquid in/out. Return species-conservative outlet streams, transfer/reaction accounting and hydraulic diagnostics. Quasi-steady spatial solve when timescale evidence supports it. | Countercurrent chemical scrubbers, physical absorbers and strippers using the same balance. |
| Extend `units/tank.py` / reusable liquid holdup | State V, species amounts n, optionally energy. RHS `dV=ΣQin−ΣQout`, `dn=ΣQin Cin−ΣQout C+Vνr+transfer`; actual overflow/bleed carries C and energy. Integrate cumulative overflow/discharge with actual realized fluxes. | Scrubber sump, blend/storage/neutralization tanks, variable-volume treatment basins. |
| `sensors/redox.py` | Explicit electrode observation strategy, consuming speciated solution/T/reference metadata. Nernst couple helper is general; fitted mixed-potential probe closure is labeled/calibrated independently. | Chlorine residual monitoring, dechlorination, sulfide/oxidation endpoints. |
| Existing dosing system, with instrument time constants | Sensor receives predicted true pH/ORP/flow; PI receives signals; pump/valve receives commands. Supply adapter converts actual volume rate and stock composition into molar/volumetric stream. | All chemical dosing and makeup loops; general actuator does not own NaOH chemistry. |

Conservation is the shared truth: no separate manually coded alkalinity penalty, sulfur depletion, chlorine depletion, product generation and heat coefficients that can disagree. Derive those from one reaction extent and one selected species/invariant basis.

### H₂S scrubber composition

1. Gas boundary supplies H₂S, CO₂ and carrier throughput with absolute pressure, Tgas, humidity/normalization basis. Makeup/chemical stock boundaries supply composition, temperature and measured/calibrated volume rates.
2. Sump inventories determine analytical totals; one charge/speciation solver supplies all molecular species, pH and chlorine fractions.
3. Recirculation supplies the tower's liquid inlet. Packed-column kernel computes gas outlet and liquid return including film reaction consumption/products; gas/liquid elemental transfers agree exactly.
4. Sump balances liquid return minus recirc draw, supplies, bleed/overflow and bulk reactions. Film and bulk reaction extents are distinct spatial contributions; do not oxidize absorbed gas twice or omit products made in the film.
5. Solve coupled tower/sump RHS at each integration stage. If finite packing holdup is material, give it dynamic state; otherwise use a justified quasi-steady column solve. Controllers update at their sampled interval while process integration respects the faster physics.
6. pH/ORP/flow instrument closures drive the existing dosing loops. Report outlet sulfur, discharge, reagent consumption and integrated operating costs with consistent timing.

A dynamic well-mixed sump breaks the composition recycle algebraic loop naturally: its recirc outlet composition depends on state, not instantaneous return composition. Recirc outlet flow is an imposed pump/hydraulic signal. The liquid returning from a quasi-steady tower depends on this outlet and on the gas inlet, so flowsheet composition ordering can remain acyclic.

## Architecture blockers and required decisions

- **Current contactor contract is too narrow.** `ContactorResult` (GC:109–114) must be replaced by phase outlets plus transfer/reaction accounting. Merely replacing its efficiency formula cannot conserve enhanced absorption chemistry.
- **Two pH ownership paths already disagree.** Remove sump sigmoid/amplitude and separate contactor/sump pKa parameters as independent truths; all consumers use the same chemical state/speciation result. CAL's amplitude omission is evidence of the architectural failure.
- **General stream basis is implicit.** Stream can route arrays but cannot distinguish mol/m³, mg as H₂S, mg as S, mg as Cl₂, COD, or actual versus normalized gas flow. Require construction-time basis/phase metadata and explicit adapters; do not overhaul unrelated ASM schemas speculatively.
- **Flowsheet rejects algebraic cycles** (`flowsheet.py:334–353`). A well-mixed holdup avoids the recirc loop, but a fully quasi-steady countercurrent column has an *internal* boundary-value problem. Solve that inside the column kernel; if a P&ID truly requires plant-wide algebraic loops, extend the engine deliberately rather than silently lagging streams.
- **Split flow/composition passes matter.** ContinuousUnit.outlet_flows only receives inlet flows (contract:41–43). Exact gas volumetric shrinkage depending on uptake cannot fit that boundary without representation changes. Dilute carrier throughput may be held constant with an explicit accuracy assumption; significant multicomponent absorption needs molar throughput/phase context or a unified phase-outlet solve.
- **Inventory constraints and stiff chemistry need integration policy.** RK4+project does not account for lost material at capacity/depletion. Use conservative realized boundary fluxes and common bounded reaction extents; extremely fast species kinetics need equilibrium/film reduction or suitable stiff integration, not bigger arbitrary substep counts.
- **Calibration factory duplicates physics assembly.** HB and CAL independently construct parameter trees; CAL omits values that HB propagates. Keep one construction mapping from named physical quantities to the canonical model. Retire empirical fit dimensions alongside the superseded model and update owned tests/callers/config together, with no shadow/fallback physics.
- **Generality is justified now, but scope should follow shared truth.** Extract Henry/speciation/holdup primitives with the existing digester/pH/tank consumers in mind. A data-driven stoichiometric matrix is useful; a giant generic reaction-registry framework is not required. Preserve named strategy interfaces and fixed JAX shapes.

## Plant evidence needed before claiming predictive fidelity

Confirm P&ID/packing geometry and contact direction; Qg and Ql engineering units and standard/actual basis; TI0880 sensor location; system pressure; NaOH/bleach active strengths, density, stock NaOH/NaCl and pump calibration; makeup composition/temperature; bleed/overflow/weir arrangement; ORP probe and reference. Measure or bound CO₂ and competing chlorine demand. Do not infer these solely by optimizing efficiency.

015's screened multi-month train/validation/locked-test structure is useful and should remain frozen. Its diary reports that 009 is superseded and that the warm-window fit generalizes unevenly. Fix structural mass/speciation defects before interpreting bound-pinned yield/ORP coefficients as chemistry. Efficiency and outlet H₂S are derived from the same analyzers; they are not independent evidence. Do not spend the locked months to select between structural physics candidates.

## Verification proposed for the next implementation phase

No tests or experiments run in this phase. The two numerical examples above were evaluated with scalar algebra, not a simulator run.

- Species fractions sum to one; charge solver satisfies a scaled electroneutrality residual across buffered/unbuffered dilution and temperature cases, with an independent equilibrium oracle.
- Balanced reaction channels conserve atoms/charge and produce the correct one/four oxidant-equivalent limits. Reagent depletion limits a common reaction extent; products and consumed reagent cannot diverge after a step.
- Physical absorber reproduces Henry equilibrium, zero driving-force flux and desorption; dry packing/zero area removes no gas. Zero flow has a defined boundary behavior, not invented removal efficiency.
- Film closure reproduces physical and pseudo-first-order analytical limits, reagent-depletion and gas-resistance limits. Finite-bulk flux is checked independently, rather than only against the implementation's E formula.
- Column reproduces the dilute perfect-sink exponential NTU solution, and conserves each element between two phase outlets including products.
- Liquid inventory at capacity reports the actual overflow and its discharged amounts; closed vessel/dilution/bleed cases conserve inventory. Step refinement must converge without chemistry clipping or hidden sulfur loss.
- Coupled composition conserves sulfur/chlorine through tower, sump and discharge; measured **liquid** temperature changes equilibrium and kinetics consistently. Factory and calibration must describe one physical model.
- Instrument physical time constants are invariant to timestep. Nernst-helper reference/temperature semantics are independently checked; real mixed-potential fit gets held-out instrument data.
- Then use 015 per-component residual checks and seasonal validation, distinguishing instrument error, missing inputs and transport/chemistry errors. Real plant removal is validation data, not a hard-coded maximum.

When code changes are authorized, use the existing Ruff/Pyrefly repository checks (project explicitly excludes Pyright) and behavior-focused small/medium JIT integration tests. Agree the commit plan and required core/project ownership before implementation. This report does not authorize or initiate edits.

## Source-access limits

Publisher abstracts/extracts were available for several papers; full-text access was not universal. Onda's official J-STAGE landing page confirms the primary paper; no packing constants were copied without reading/calibrating their applicable correlation. Bontozoglou/Giggenbach DOI metadata and author-paper extracts were found, but direct ACS open calls failed. The ORP mixed-potential paper was discoverable with relevant extracts but a direct full-text open failed. The existing SS comment's specific PMC6111276 citation could not be verified in this session; I do not rely on its claim that a fitted slope well above 59.2 is automatically expected. No Kohl & Nielsen numerical formula is cited because the book text was not inspected.

---

## Original review 2 (corrected)

# Review 2: named physics modules, opaque transport, corrected plant wiring

Read-only design review, 10 October 2026. No production code was edited. This reviews `lead_review_1.md`, `buddy_note_2.md`, the current process-control toolkit, and the supplied `scrubber4_equipment.md` extraction. Equipment claims below come from that extraction; I have not independently opened its Drive originals. Interface snippets describe contracts, not implementation proposals ready to merge.

## 1. Disagreements and corrections

1. **Neither “instantaneous” nor “gas-film controlled” is a universal column model.** Fast HS−/HOCl chemistry supports an enhanced absorption closure, but the controlling resistance depends on speciation, reagent delivery, wetting, loading, and packing. Low chlorine or pH can reintroduce liquid-side resistance and gas backpressure. An exponential NTU expression is useful after defining its driving force and transfer coefficient; it does not establish a universal exponent or reaction regime. The old efficiency formula is inadequate, but replacing it with a different empirical expression does not itself supply physical closure.
2. **Film reaction does not eliminate bulk chemistry.** Fresh chemicals meet recycled liquor before entering the tower. Residual sulfide and oxidant can react there and in the sump. Slow sinks can coexist with effectively instantaneous reactions. The error is resolving a fast reaction with explicit slow-step RK4, not locating all reaction outside the film.
3. **Our ceiling language needs qualification.** The approximately 0.73 result was for warm default initial conditions. The ideal chemistry bound for the cited default flow factors is approximately 0.807, and changing temperature or fitted parameters changes either result. Neither is a global ceiling. A required removal efficiency must be a validation criterion, not a clipping parameter that ensures success.
4. **I retract the description of TI0880 as liquid temperature.** The equipment extraction identifies it as *Fermenter Scrubber Intake Temperature*. Passing it to gas properties is appropriate; requiring the sump to track it is not. Gas and liquid temperatures must be distinct. Liquid temperature is presently unmeasured, requiring a heat balance or an explicitly uncertain parameter. An arbitrary liquid lower bound still needs justification, but this sensor alone cannot refute it.
5. **The sump CSTR accusation was too strong.** With inventory `n = V C`, outflow terms can cancel from the concentration derivative when the same outflow appears in `dV/dt`. Absence of an explicit `−Qout C/V` term is not proof of failed solute conservation. The concrete defects are the physical outlet law and incomplete drain reporting/composition routing. Verify the amount balance before condemning the concentration equation.
6. **CO2 importance is unresolved.** A result from another industrial scrubber does not establish that CO2 dominates this plant's alkali demand. Include carbonate chemistry and uncertainty, but distinguish a credible mechanism from measured dominance. CO2 uptake changes TIC, pH, and hydroxide availability; it does not simply destroy generalized total alkalinity. Likewise, 0.0293 meq/mg H2S describes a free-base equivalent consumption convention, not universal removal of alkalinity that includes HS−.
7. **8.7 mg NaOCl/mg H2S is not intrinsically an erroneous stoichiometry.** It corresponds approximately to oxidation to sulfate; elemental sulfur requires approximately 2.18. The unsupported choice is an unconditional product pathway. Bleach strength and the “as NaOCl” versus “as Cl2” reporting basis must be explicit rather than accepted as “okish.”
8. **The proposed ORP ratio is chemically invalid.** Free chlorine and HOCl are not an oxidized/reduced redox pair. A HOCl/Cl− equilibrium expression needs chloride, hydrogen activity, temperature, and an electrode reference. Actual ORP may be a mixed potential with sensor dynamics; a calibrated observation model is permissible, but must not masquerade as an equilibrium free-chlorine measurement.
9. **The phase and hydraulic facts materially change our earlier assumptions.** Dosing is into recirculation-pump discharge upstream of the tower, not into the sump. Chemical outputs are RPM, not percent or mL/min. Duty-pump calibrations are bleach 1.632 L/h/RPM and caustic 1.008 L/h/RPM; standby caustic is 0.576. At 40 RPM the duty flows are 65.28 and 40.32 L/h, so feed liquid volume must be counted. Stock concentrations remain unknown. Makeup has flow PV/SP in L/min and valve output in percent: those are different signals, requiring valve hydraulics or measured flow rather than a percentage-to-flow fiction.
10. **The revised hydraulic topology is continuous overflow.** Makeup, chemical feeds, mist-eliminator flush, and evaporation contribute to the volume balance. Makeup closes at level 695 mm and reopens at 600 mm; these are controller thresholds, not demonstrated weir elevation or tank volume. Recirculation trips below 300 mm. Neither level-to-volume geometry nor overflow capacity is known. FI0871's identity as a gas flow measurement is itself questionable, not merely its engineering-unit conversion. FI0872 units and tower geometry remain unknown. Do not identify fitted NTU with a measured packing coefficient under these conditions.
11. **The original schema proposal conflicts with Andy's requirement.** Charge, molar mass, equilibrium constants, and sources belong in a species registry. A transport schema carries channel names and reporting/transport basis; it must not become a second chemistry database. Repeated Henry formulas are candidates for consolidation, but reduced ADM1 carbonate closure and a full charge-balance solver are not equivalent merely because both mention carbonate.

The existing CAL pH-amplitude split, nonconservative clipping risk, unaccounted stripping outlet, and removal at zero recirculation remain concrete issues from review 1. The plant reward described in the extraction is removal above 90% **or** outlet below 1000 ppb; the benchmark's 95% target is a separate design decision. A five-minute replay grid also does not establish the reported thirty-minute agent action period.

## 2. Separate the carrier, physics requirements, and chemistry registry

`ComponentSchema` already supports an arbitrary vector; `Stream.mix` already transports every coordinate. The restriction arises when a physics model makes its local species list the entire stream schema. Keep the full carrier schema in the flowsheet; give each physics kernel only gathered inputs and allow it to return only declared source coordinates.

Illustrative interfaces:

```python
@dataclass(frozen=True)
class ComponentBasis:
    unit: str                 # e.g. mol/m3, mg/L
    quantity: str             # species, elemental total, COD, equivalents, proxy
    reference: str            # e.g. H2S, S, N, Cl2, COD

@dataclass(frozen=True)
class ComponentSchema:
    names: tuple[str, ...]
    bases: tuple[ComponentBasis, ...]
    particulate: tuple[bool, ...]
    tss_factor: tuple[float, ...]

@dataclass(frozen=True)
class NamedAccess:
    reads: tuple[str, ...]
    writes: tuple[str, ...]

@dataclass(frozen=True)
class SpeciesRecord:
    charge: int
    molar_mass_kg_per_mol: float
    elements: tuple[tuple[str, int], ...]
    acid_base_family: str | None
    henry: HenryParameters | None
    sources: tuple[str, ...]

class SpeciesRegistry(Protocol):
    def species(self, name: str): ...
    def family(self, name: str): ...
```

Registry records describe molecular species. Separate family records describe conserved analytical totals and equilibria. Biomass/COD channels, turbidity, “chlorine demand,” and equivalent alkalinity are not molecules: do not invent a molecular mass or charge for them. Their channel descriptors specify their actual basis and admissible transport operations. Parameter provenance belongs with registry records; calibratable numerical parameters remain explicit JAX leaves rather than mutations of a global registry.

ASM1 demonstrates why basis is per coordinate: organic channels are mg COD/L, oxygen is mg O2/L, nitrogen channels are mg N/L, and `s_alk` uses mol HCO3−/m3. ASM2d adds phosphorus and storage pools with further reporting conventions. Multiplying the whole vector by a single molar conversion is impossible. Exact schema equality at connections should continue to include bases. A deliberate unit converter is a visible operation; silently accepting unequal bases is forbidden.

At construction, a binder resolves declared names to fixed index tuples, checks basis compatibility, and builds constant gather/scatter operations. Missing reads/writes fail before tracing. The kernel can know the local order of its own declarations, as existing ASM kinetics do; it never receives an extra tracer's name, value, or Python type. No runtime dictionary lookup, per-species Python loop inside JIT, or union of every module's fields is required.

For a holdup, the framework computes full-vector transport and adds scattered physical sources. For a column, it first constructs full-vector outlet amounts from transport, then applies the kernel's declared transfer/reaction extents and closes each phase. Reads alone never grant permission to overwrite an outlet coordinate. Multiple source modules add their declared rates; sequential transformations require explicit composition and evaluation order rather than accidental last-writer behavior.

**Pass-through means unchanged amount except generic transport, not unchanged concentration.** An unknown soluble tracer dilutes when water enters, leaves with overflow, and accumulates in a holdup. An unknown particulate component follows settling because its transport classification specifies that behavior. A gas inert's mole fraction can increase when H2S is absorbed even though its molar flow is unchanged. Generic transport knows the vector and its physical basis; the reaction module does not know the untouched values.

There is a necessary limit to “unknown”: an arbitrary new salt cannot be declared invisible to charge balance while claiming correct pH. Compile property reducers from registry metadata, such as strong-ion charge and participating weak-family totals, and supply only those reduced quantities to the speciation kernel. An unregistered channel can be transported opaquely, but cannot silently receive zero charge or molecular mass when chemistry needs it. Assembly must either establish its chemical contribution, designate an intentionally inert/proxy channel, or reject that chemistry configuration.

## 3. One stream carrier, explicit phase closure

Use **one `Stream` carrier**, with static phase/basis metadata and numerical temperature/pressure fields. Liquid flow is m3/h with channel concentrations in their declared bases; gas flow is total mol/h with molecular mole fractions. T is kelvin and P is absolute pascals. A build-time phase check prevents liquid and gas ports connecting accidentally. Existing process-control consumers can retain their familiar concentration representation through explicit basis descriptors rather than wholesale conversion of COD models into molecular chemistry.

If a gas boundary is specified by inert-carrier molar flow, retain that boundary input and derive total flow using its carrier fraction. True mole fractions require `F_total = F_carrier / y_carrier`; multiplying true mole fractions by carrier flow is wrong. An alternative solute/carrier ratio representation is valid, but those coordinates are not mole fractions. Prefer total molar flow internally, preserving each inert's molar amount across the column and renormalizing all output fractions afterward.

Mixing must operate on phase-appropriate amounts. Gas temperature mixing needs an energy/enthalpy closure; liquid temperature averaging requires stated heat-capacity assumptions. Do not reuse concentration weighting blindly for temperature, pressure, or nonadditive proxies. A turbidity channel can remain a volume-weighted empirical proxy when preserving the existing contact-basin model, but cannot enter elemental or charge conservation tests.

### The current two-pass engine needs a real extension

`ContinuousUnit.outlet_flows` currently receives inlet flows only, and `flowsheet._resolve` schedules flows before compositions. Exact gas absorption makes gas outlet total flow depend on inlet composition and reaction. The present contract cannot represent that exactly. Freezing gas flow or inserting an artificial time lag would hide the limitation.

Generalize outlet dependency declarations to field references and compile a single dependency DAG:

```python
@dataclass(frozen=True)
class FieldRef:
    port: str
    field: str                # flow, components, temperature, pressure

@dataclass(frozen=True)
class Outlet:
    schema: ComponentSchema
    dependencies: tuple[FieldRef, ...]
    evaluation_group: str
```

Resolve a node's grouped outputs once when their declared inputs are available; cache the result during that RHS evaluation. A column solves its coupled phase outlets once, rather than independently repeating chemistry for flow and concentration callbacks. Existing mixers/splitters declare the same dependencies they have today. State-dependent holdup composition still breaks recycle composition cycles, and a prescribed recirculation pump breaks the corresponding flow dependency. Genuine algebraic cycles remain build errors unless a specifically supported coupled solver owns them. A packed-column countercurrent boundary-value solve belongs inside its unit, not as an accidental global flowsheet cycle. Co-/countercurrent plant configuration is still unverified.

This is a coherent engine-contract change, not a second gas-only flowsheet implementation. Convert the existing declarations and scheduler together; remove the superseded two-pass implementation in that same change. Prefer one moderately sized, independently valid commit over artificial twenty-line slices that cannot build.

## 4. General reactions, transfer, and fast chemistry

Remove the mandatory `oxygen` property and full-carrier `schema` from the physics-facing kinetics protocol:

```python
class KineticsModel(Protocol[P]):
    access: NamedAccess
    def rates(self, reads: Array, temperature_k: Array, params: P): ...
    # Returns rates ordered by access.writes, with declared bases per hour.

class TransferModel(Protocol[P]):
    access: NamedAccess
    def rates(self, reads: Array, temperature_k: Array, params: P): ...

@dataclass(frozen=True)
class FilmResult:
    transfer_mol_per_hour: Array
    reaction_extents_mol_per_hour: Array
    heat_watts: Array
```

The binder, not the kinetics model, owns the carrier schema. A generic reactive holdup owns full inventories and transport; a separately composed oxygen-transfer module declares O2 and its physical saturation law. ASM1's `s_o` slot is connected explicitly to this module at assembly. That is a static channel binding, not a legacy alias or an oxygen requirement imposed on anaerobic kinetics. Keep temperature dependence in model-specific rate evaluation; sharing a temperature argument does not mandate a common Arrhenius law.

The holdup evolves inventory and volume. Inventory units follow each channel's basis; conserved molecular families preferably use molar inventories. Signed equivalent quantities cannot receive unconditional nonnegative clipping. Physical molecular amounts must remain admissible through bounded fluxes, appropriate integration, and timestep control rather than post hoc clipping that creates matter.

At every coupled RHS evaluation:

1. Obtain concentrations from admissible inventories and volume, then solve fast acid-base speciation and charge balance algebraically using local liquid T.
2. Solve the column's transfer/film reaction closure using inlet states and available reagent. Return matched transfer and reaction extents, with stoichiometric updates on both phases. Absorbed sulfur already converted into product in the film must not also become an unreacted sulfide source to the sump.
3. Evaluate genuinely slow bulk reaction and hydraulic/thermal rates on the resulting reduced state.

**Instantaneous irreversible bulk reaction is not acid-base equilibrium.** A `min(sulfide, oxidant)` subtraction inside every RHS evaluation is not a rate and will repeatedly consume inventory. If bulk oxidation is treated as instantaneous, evolve compatible reaction invariants and solve a constrained fast-reaction/product closure; initialize onto that manifold conservatively. Reagent-feed fluxes then induce extents through the reduced balances. Sulfur product selection must be an explicit closure with supporting assumptions. If finite chemistry cannot be reduced reliably, use a suitable stiff solver for that block. Do not promise that ordinary RK4 becomes safe merely because one large coefficient was removed.

No explicit RK4 stage should integrate the approximately 1e9 L/mol/s HS−/HOCl reaction. Equilibrium and film solves need bounded solutions and conservation diagnostics; differentiability and convergence must be assessed if calibration differentiates through them.

## 5. Correct plant assembly

The topology should be:

`sump → recirculation pump → bleach/caustic stock-feed mixer → fast mixing chemistry → packed column → sump`

Gas enters the column on its separate port. Makeup and intermittent mist-eliminator drainage enter the appropriate liquid holdup; continuous overflow leaves with sump composition. The chemical manifold's transit/holdup may initially be negligible if justified, but its inlet dosing location cannot be moved to the sump: fresh reagent can alter tower performance before changing the well-mixed sump concentration.

Actuator adapters convert RPM to stock liquid flow using the duty/standby calibration, and stock composition converts liquid flow to chemical equivalents. Include stock-feed water and sensible heat. Makeup valve position needs a measured flow input or an identified valve/supply-pressure relation. The level hysteresis, fan/recirculation dosing permissives, and low-level trip are generic sampled controller state, separate from molecular chemistry. They still affect actual realized physical flows. Level-to-volume mapping, overflow law, mist-flush schedule, gas throughput, stock strength, and probe sampling location remain explicit unknowns.

Do not assume the pH/ORP probes measure sump or manifold until their locations are established. Make the observation tap configurable at assembly with documented uncertainty. Separate sensor selection/lag, PI/ramp logic, thirty-minute agent updates, and five-minute data resampling. The supplied report's bleach cost coefficient is approximately 1.35456 dollars/hour/RPM, matching the old numerical 1.355 only if interpreted in RPM; integrating it as a cost per mL is wrong.

## 6. Existing consumers and migration boundaries

| Consumer | Required change | Boundary and risk |
|---|---|---|
| ASM1 / aerated reactors | Declare local named reads/writes; bind its existing local rate vector into a larger carrier. Move oxygen transfer out of the generic kinetics requirement. Add per-channel bases. | Preserve canonical local ordering and validated rate equations. Do not optimize sparse declarations and change equations simultaneously. |
| BSM1 / activated-sludge assembly | Construct the carrier schema and bind reactor, mixer, splitter, and settler modules. Settler transports all coordinates using particulate metadata. | Verify float64 reference behavior. There is ongoing untracked `plants/activated_sludge.py` work; coordinate its ownership before implementation. Existing standalone BSM1 paths need caller inspection, not presumed retirement. |
| ASM2d / combined NP | Give the reduced 17-channel kinetics the same named-access adapter and basis declarations. Replace the affected direct reactor-step path when that consumer is migrated. | It does not currently expose the same KineticsModel interface. Sharing transport does not mean identical ASM1/ASM2d chemistry. Remove superseded local transport/integration only when its live callers have moved. |
| ADM1 | Harvest shared Henry temperature/basis conversion primitives; migrate all helper callers, including benchmark imports. Share chemical constants without pretending reduced AM2 speciation equals full equilibrium. | Henry currently uses mol/(L atm); explicit conversion is essential. Full named-carrier ADM1 conversion can be its own milestone, rather than forcing it into H2S cutover. |
| pH neutralization | Generalize the charge/speciation solver, then migrate its buffer/base-excess interpretation with explicit mol/L↔mol/m3 conversion. | Replace the old owned solver in the consumer migration. Preserve a clearly specified reduced buffer model if that is the benchmark contract; no silent new carbonate assumption. |
| chlorine / contact basin | Replace fixed five-column assembly with generic segment transport and a named chlorine/demand/ammonia kernel. Carry extra coordinates through all segments. | Preserve existing bounded demand kinetics first. Turbidity is a proxy, not conserved molecular mass. Its current model does not implement HOCl speciation; do not claim that upgrading transport does. Migrate its owned producer/consumer chain atomically without deleting unrelated uses of the broader Transport type. |

Generic transport, Henry conversion, and charge-balance machinery remove shared truth. Product yields, AM2 reduction assumptions, ASM substrate definitions, contact-basin demand policy, and ORP observations remain model-specific. This distinction avoids both duplicated infrastructure and a universal “chemistry engine” whose flags encode unrelated models.

## 7. Milestones and validation

These are design milestones, not authorization to implement. Each may contain several commits, with explicit commit-plan agreement before code. A public-contract replacement must include all required caller/test updates in its own valid commit; the approximately twenty-line preference cannot justify broken intermediate states.

1. **Document the physical contracts and uncertainties.** Establish channel bases, temperature/pressure units, registry identities, pump calibration/stock-strength distinction, and source provenance. Record unknown geometry and tag meanings. This is independently reviewable without changing the benchmark.
2. **Introduce explicit channel bases and registry primitives.** Update existing schema constructors together; do not install guessed default bases. Add molecular/family records with sources and unit conversions. Opaque tracer transport must work without a chemistry registry entry.
3. **Introduce named binding and migrate aerobic holdup composition.** Preserve the existing ASM rate kernel, gather/scatter around it, remove mandatory oxygen, and compose oxygen transfer separately. Update affected ASM1/BSM1 consumers in valid contract-changing slices. There must be one canonical reactor integration path for each migrated consumer.
4. **Migrate other generic-transport consumers.** ASM2d and the contact basin can be independently migrated and verified. Remove each superseded owned path with its replacement. This proves the abstraction with consumers that do not share H2S chemistry; it need not force full ADM1 state redesign.
5. **Consolidate physical-property and fast-speciation primitives.** Move Henry handling with all donor call sites; extend charge balance with a neutralization consumer migration. Use explicit reduced closures for consumers that require them. Validate temperature/basis conversions before introducing scrubber chemistry.
6. **Provide inventory/volume holdup and constrained chemistry contracts.** Validate inlet/outlet inventory conservation, variable volume, overflow, depletion, charge/speciation, and fast-manifold initialization. Keep irrelevant empirical sensor or controller behavior outside this layer.
7. **Extend the canonical stream/field dependency engine.** Add phase/T/P closure and replace the two-pass scheduler with grouped field dependencies. Migrate current unit declarations in the same coherent change; validate current recycle behavior and new composition-dependent gas flow. No separate gas runner or hidden frozen-flow approximation.
8. **Build isolated column and plant actuator modules.** Test film extents, two-phase elemental balances, reagent limits, gas renormalization, zero-wetting behavior, stock feed conversion, overflow, and controller permissives. These are standalone capabilities, not a second live H2S benchmark factory. Geometry-dependent coefficients stay explicit uncertain parameters until evidence identifies them.
9. **Cut over H2S atomically.** Rewire dosing before the tower, integrate liquid inventories/overflow and observation taps, update replay units and calibration parameterization, then delete the old empirical contactor/sump implementations and their owned callers, tests, configuration, and fit parameters in this same milestone. No shadow run, fallback, deprecated wrapper, or legacy factory. Include required behavior tests in the cutover commit if the contract changes inseparably.

Meaningful validation includes a schema extension/reordering test where a new tracer changes neither known reaction rates nor access declarations; liquid tracer amount balance under dilution and overflow; preserved inert molar flow with changed gas fractions; summed gas/liquid sulfur and chlorine balances; electrode observation units; interlock hysteresis; and reference ASM/BSM behavior. A full-vector equality test would be wrong for dilution or fraction renormalization. Chemical tests must distinguish atoms/equivalents from COD and proxy transport, and test depleted-reagent and zero-flow boundaries without asserting a chosen internal algorithm.

For implementation, run the project's required whole-repository Ruff and Pyrefly checks and relevant behavior tests. No simulator, experiment, or software test was run for this read-only review. The proposed migration is tractable in independently valid slices, but the stream scheduler and final H2S replacement are inherently coupled changes and should not be forced into tiny broken commits.

## Evidence pointers

- Current toolkit under `core/domains/process-control/src/process_control/`: `components.py`, `streams.py`, `kinetics.py`, `flowsheet.py`, `units/contract.py`, `units/bioreactor.py`, `units/asm1.py`, `units/asm2d.py`, `units/anaerobic_digester.py`, `chemistry/ph_model.py`, `benchmarks/ph_neutralization.py`, and `units/contact_basin.py`; ADR 015 explains the current two-pass flow/composition design. Paths should be verified again immediately before implementation because another agent is working in this repository.
- Local plant extraction: [scrubber4_equipment.md](./scrubber4_equipment.md). Its draft narrative and later telemetry/report evidence have different dates and authority; discrepancies are not resolved by treating the draft as current operations.
- Extraction's primary references: [control narrative](https://drive.google.com/file/d/1KkdNWzyqDohazWkZvpT0MSjVhHbjA9ml/view), [December 2025 report](https://docs.google.com/document/d/1pUZ7l6DtW9uNrPcqVy2rEJ6ETvCr2pGK6P7DMZuVHBc/edit), [dashboard export](https://drive.google.com/file/d/1M8HtY0wkyyQWJ8pD-naWDmoNH6o5qH4b/view), and [RL configuration](https://drive.google.com/file/d/1hBdTfF0S4h_4NtT0amlVUFlcbgI2w2ky/view). These links identify the extraction's sources, not independent verification in this review.
- Literature and numerical detail supporting the disputed physics are documented in [buddy_review_1.md](./buddy_review_1.md). The temperature interpretation and pump-unit assumptions in that first review are superseded explicitly above.

---

## Original EPCOR equipment extraction

# EPCOR Gold Bar Scrubber #4 (Fermenter Scrubber): equipment/process metadata from Drive

Read-only extraction. Page numbers are NOT available from the Drive text export; locations are given as section titles (the narrative .doc has 55 pages, but the export has no page breaks).

Abbreviations for sources:
- NARR = GB-PCA-NAR-ODR-001.doc "Foul Air Scrubbers Control Narrative" (id 1KkdNWzyqDohazWkZvpT0MSjVhHbjA9ml; rev 1b, Dec 6 2022 draft; modified 2023-01-10)
- QA = Google Doc "H2S Scrubber #4" (id 1su1UlIlplmaod9nvXoGdBnUoHrw4Jos4qc8FVo3rcBs; Feb 2024)
- YAML = scrubber4.yaml, RLCore agent config (id 1hBdTfF0S4h_4NtT0amlVUFlcbgI2w2ky; 2026-09)
- GRAF = "Scrubber 4-1740678292819.json", Grafana dashboard export (id 1M8HtY0wkyyQWJ8pD-naWDmoNH6o5qH4b; Feb 2025)
- TELE = telegraf-postgres.conf, OPC-UA tag map (id 1GMXtIdbZQvCBbywOSpPYvwZYtmNE2JVh)
- RPT25 = "EPCOR Scrubber Project Report_Dec2025" (id 1pUZ7l6DtW9uNrPcqVy2rEJ6ETvCr2pGK6P7DMZuVHBc)
- RPT24 = "Epcor Scrubber Report" (id 1U9kPLYNQUjNNhSCXtbb_jezGp-WjDnUbvgVOK9PEZNw)
- DC = Google Doc "Data Cleaning" (id 1iwW_rgvlAVeM0Vr9fgbY3LoETFAEjAwjA_f_xOnBcBY; Q&A with operators, 2024)
- AEC = AEC_2025-Scrubber-EPCOR.pptx (id 1EQm0oP8oTeywMhrdqzSUo31sjQqgPFQa); AWWOA = AWWOA_2025-Scrubber-EPCOR.pptx (id 137xUsLq0Atq_LrRv6M4mJFC5Ao-uTnU_)

## Source-coverage notes (read first)
- "Scrubber 4.zip" (both copies, 20,809,148 bytes) could NOT be downloaded: the Drive MCP download tool has a 10 MB limit. The zip's contents are unread. It is the most likely home of tower dimensions/packing/vendor data and P&ID-ODR-400.
- "P&ID.zip" (id 1o0jSIc8Xqjl_HIUD3xTBrtN_7JyLzWtQ) is NOT about Gold Bar or Copper Basin. It holds 12 PDFs, extracted in scratchpad/pid_zip/ and inspected with pdftotext only, titled "Area 40 Aeration Blowers / Area 41 Aeration Area ... Bioreactor 1/2/3 ... Foul Air Blowers, Dwg P40 520", with the title block "Alberta Capital Region ... ACRWC project no." (Alberta Capital Region Wastewater Commission, a different utility). No Scrubber-4 content found.
- The narrative has no packed-tower geometry, packing data, design air flow, blower sizing, sump volume or chemical strengths. Nothing in the other readable docs has them either (see "Not found").
- Narrative text covers the fermenter scrubber's control only; the H2S analyzers AI-0879A/B/C are not described in the narrative (it only lists East/West AI-0888/AI-0980).

---------------------------------------------------------------------

## 1. Tower

| Item | Value | Quote | Source + location |
|---|---|---|---|
| Packed bed present | yes (media type unspecified) | "These chemicals work with the scrubber “packing” media and softened water to purify the foul air. The recirculation pump is used to circulate the scrubbant solution over the packing media inside the scrubber tower." | NARR sec. 2 "Process Description and Control Philosophy", subsection "Scrubbing" |
| Spray/packed tower, flow arrangement | "packed bed with a spray tower"; liquid goes from tower bottom to top | "The H2S gas stream passes through a packed bed with a spray tower where it comes into contact with a scrubbing liquid." / "The recirculation pump circulates water injected with bleach and caustic from the bottom of the scrubber tower to the top of the scrubber tower." | AEC slide "Scrubber Process Overview"; NARR sec. "Fermenter Scrubber Recirculation Pump", Objective |
| Co-current vs counter-current | NOT STATED. INFERRED only: liquid is fed at the top and drains to the bottom sump; the foul-air fan "pushes" air into the tower ("The foul air is pushed into the scrubber by a dedicated VFD exhaust fan"), consistent with a typical bottom-inlet counter-current packed tower but gas inlet location is not documented | see quotes | NARR sec. "Foul Air"; inference, unverified |
| Mist eliminator | Not documented for #4 in the narrative. A mist-eliminator valve tag FV3735_PV ("scrubber mist eliminator valve status") exists in the data feed, so scrubber 4 has an (intermittently wetted) mist eliminator. The operator Q&A says the daily sump-level bump is from "Mist eliminator, valves open". East scrubber has "mist eliminator valve FV-0925" and West has PDI-0992 mist-eliminator dP; no #4 equivalent | "scrubber mist eliminator valve status" ; "Mist eliminator, valves open" | TELE (node FV3735_PV comment); DC "Questions" item 7.2 |
| Packing dP / conductivity / inlet temp sensors on #4? | Open question in DC: "The control narrative mentions that scrubbers 5 & 6 have sensors for (1) Differential pressure across scrubber packing media, (2) Conductivity ..., (3) Temp of air entering the scrubbers. Does the fermenter scrubber also have these sensors?" Answer not recorded. (The data feed does have TI0880 inlet temp and AI0897B softened-water conductivity, but no packing dP tag.) | quoted | DC final question; YAML input_tags |
| Diameter, bed height, number of beds, packing type/size/manufacturer, specific area, void fraction | NOT FOUND (searched 'Tri-Pack', 'Jaeger', 'packing', 'fermenter scrubber') | | |

## 2. Gas side

| Item | Value | Quote | Source + location |
|---|---|---|---|
| Foul air sources for #4 | grit tanks 4 and 5, building ventilation, sludge fermenters | "The Fermenter Scrubber #4 receives odorous air from grit tanks 4 and 5, building ventilation and the sludge fermenters." | NARR sec. "Foul Air" |
| Fan | dedicated VFD exhaust fan "GEF-47150 & MTR-47150" (heading); VFD "GBB-28521" with bypass M-3739A | "The foul air is pushed into the scrubber by a dedicated VFD exhaust fan. The rest of the scrubber system components will not operate unless the exhaust fan is in operation." | NARR sec. "Foul Air"; sec. "Fermenter Scrubber Blower Fan and Motor (GEF-47150 & MTR-47150)", DCS Modules table |
| Fan control | In CAS, VFD driven by PID PDIC-3738 from the differential of PIT-0168 (inlet pressure) and PIT-0169 (discharge pressure) | "When in CAS, the blower fan VFD is controlled by PID controller PDIC-3738 based on the pressure differential measured by PIT-0168 and PIT-0169. The PID’s setpoint is set at 0.5 kPa." | NARR same section, "Functional Description" |
| PDIC-3738 setpoint conflict | Narrative: 0.5 kPa. Operator Q&A: "Differential pressure setpoint is always 2". 2024 operator note: "For a high flowrate (60L) and DP of 2". YAML PDIC3738_PV operating_range [-0.1, 3] (no unit given). Unit is almost certainly kPa (INFERRED from narrative, same loop) | "Differential pressure setpoint is always 2, PID output seems to just be fixed (changes but only in discrete chunks), is this just in manual mode?" ; "Why is the PDIC-3738 measured value very frequently quite different from the specified setpoint? ..." | QA sec. "Blower"; DC Q2; YAML input_tags |
| PDIC-3738 history | Originally constant fan speed; converted to PID during the 2024 pilot | "Conversion of the differential air pressure controller to PID control (previously operating with a constant fan speed)." | RPT24 "Additional Benefits" |
| PI-0169 unit | Not stated explicitly. YAML range [-0.1, 3] identical to PDIC3738_PV, so likely kPa (INFERRED) | "- name: PI0169 # blower discharge pressure / operating_range: [-0.1, 3]" | YAML input_tags; TELE ("blower discharge pressure") |
| PI-0168 | "Fermenter Scrubber Inlet Pressure"; PI-0169 "Fermenter Scrubber Discharge Pressure" | | NARR DCS Modules table, fan section |
| TI-0880 | "Fermenter Scrubber Intake Temperature" (the narrative's wording); RLCore tag comment "scrubber inlet temperature"; range 0 to 75 (unit not stated; INFERRED deg C because NARR uses deg C for fan interlocks) | "TI-0880 | Fermenter Scrubber Intake Temperature" ; "- name: TI0880 # scrubber inlet temp / operating_range: [0, 75]" | NARR fan DCS table; YAML |
| AI-0881 | "Fermenter Scrubber Intake RH" | | NARR fan DCS table |
| FI-0871 | Tag exists (OPC node "FI_0871") with NO comment, description or unit anywhere. YAML has no FI0871 entry. The only flow indicators documented are FI0872 (recirc, see sec. 3) and FI-0887 (softened water). So "design air flow / FI-0871 unit" is NOT FOUND. RPT25 even states there is no gas flow measurement: "the absence of a flow meter for accurate loading calculation" ; AWWOA: "no available flow meter to calculate loading" | | TELE; RPT25 "Key Observations"; AWWOA "Operator Feedback" |
| Design air flow, blower curve/size, VFD hp | NOT FOUND | | |
| H2S inlet/outlet | AI-0879A inlet, range [0,100] (ppm per Grafana: "H2S In (ppm)"); AI-0879B outlet range [0,30000] (ppb per Grafana: "H2S Out (ppb)"); AI-0879C efficiency [0,100] % | "displayName: H2S Out (ppb)", "H2S In (ppm)", "Efficiency (%)" | GRAF "Efficiency" panel; YAML input_tags |
| Observed inlet H2S (not design) | median 28.61 ppm (Oct 22 - Nov 22 2024); median 9.57 ppm (same days 2023); 29.16 ppm (Sep 1 - Oct 2 2023). Outlet medians 1464.56 ppb (2024 agent), 1934.85 ppb, 6613.25 ppb | table in report | RPT25 Appendix; RPT24 Reference Periods table |
| Design/expected inlet H2S, CO2 or other gas composition | NOT FOUND | | |
| Efficiency target | agent reward: "(efficiency > 90%) OR (outlet_h2s < 1000 ppb)" | | YAML reward |

## 3. Liquid side

| Item | Value | Quote | Source + location |
|---|---|---|---|
| Recirc pump | "PCP-28529 & MTR-28530"; started by DeltaV 11 s after blower start; stops with blower; low-water flow switch on recirculation line | "The recirculation pump motor is started by DeltaV 11 seconds after the blower fan has started." / "There is also a low water flow switch on the recirculation line." | NARR sec. "Fermenter Scrubber Recirculation Pump" |
| Recirc pump interlock | Tower sump level LO at 300 mm (LI-3734); fail-to-start 4 s | "Tower sump level LO at 300 mm (LI-3734)" | NARR same, "Shutdowns/Interlocks" |
| Recirc design flow | NOT FOUND. Only an instrument range: FI0872 "scrubber recirc flow" operating_range [0, 4000], unit not stated (INFERRED L/min, since sibling make-up flow is L/min, but at 4000 L/min that would be a very large range; could be L/h or gpm. Do not rely on it). A 2024 operator note mentions "high flowrate (60L)" which refers to make-up water FIC-3734, not recirc | "- name: FI0872 # scrubber recirc flow / operating_range: [0, 4000]" | YAML input_tags; TELE |
| Sump level LI-3734 | Unit mm. Range [0, 700]. Make-up valve closes at 695 mm and reopens at 600 mm; recirc pump LO trip at 300 mm. Daily bump ~570 to ~580 (mist eliminator valves open) | "DeltaV also automatically closes the makeup water valve if the level in the scrubber tower increases to 695 mm ... If the scrubber tower level decreases to 600 mm, the makeup water valve is once again opened" | NARR sec. "Fermenter Scrubber Make-up Water Control Valve", Functional Description; YAML (LI3734 [0,700]); DC Q7.2 |
| Sump volume, diameter/footprint, level-to-volume relation, overflow weir elevation | NOT FOUND. INFERRED only: the 695 mm close point is "above" nothing stated; for scrubber 5 the narrative says its 1016 mm close level "is a level that is above the sump overflow", suggesting the overflow elevation on #4 is somewhere just below 695 mm. Unverified | "which is a level that is above the sump overflow" (scrubber 5 text) | NARR sec. EPT Scrubber 5 Make-up Water Control Valve |
| Blowdown arrangement | Continuous overflow-driven blowdown; no level-controlled drain or separate bleed valve. Blowdown rate set by make-up flow | "This valve also governs the blowdown which is continuously removed from the scrubber sump via a sump overflow. The blowdown rate increases when make-up water flow rates increase and vice versa. The make-up water rate must always be high enough so that blowdown is constantly removed from the scrubber sump." ; "A make-up water control valve is used to initially fill the scrubber sump and to continuously waste excess scrubbant solution out through the sump overflow." | NARR sec. "Fermenter Scrubber Make-up Water Control Valve", Objective; sec. 2 "Scrubbing" |
| Overflow destination | "Fermenter Scrubber Effluent Sump Pumps (PCP-47250/47252 & MTR-47251/47253)", lead/lag, in the East + Fermenter scrubber chemical storage building, controlled by multiranger LIT-4620 | | NARR sec. "Fermenter Scrubber Effluent Sump Pumps" |
| Make-up water | softened water entering "the bottom of the scrubber tower"/sump through FV-3734; flow meter FI-0887 "Fermenter Scrubber Softened Water Flow" (NOTE: the user's tags "FIC-3734" = flow controller; narrative calls the module FV-3734 and the meter FI-0887) | | NARR make-up water section, DCS Modules |
| FIC-3734 units | PV in L/min (Grafana axis label); SP in L/min (YAML range [10, 30]); OUT = valve position in % (Grafana "Valve Position", axis "%"; YAML [0,100]). Narrative: operator enters "percentage open" setpoint. FIC3734 is the flow PID added during the 2024 AI-readiness work ("Addition of PID control to the flow rate controller") | "displayName: Process Value ... axisLabel: L/min" ; "displayName: Valve Position ... axisLabel: %" | GRAF "Makeup Water" panel; YAML; RPT24 |
| Make-up flow values seen | SOP 19 L/min; EPCOR fixed-setpoint test 16 L/min; agent range 10-30 L/min; operator Q&A mention 30, 45, 60 L tests and flows "above 80" seen as non-normal. For comparison, scrubber 5's narrative gives make-up "0.005 Ml/d to 0.008 Ml/d (3.8 l/min to 5.7 l/min)" (different scrubber) | "Softened Water Flow: 19 L/min" (SOP); "Softened Water Flow: 16 L/min" (fixed) | RPT25 "Results Achieved"; DC Q2, Q7; NARR EPT 5 make-up section |
| AI-0897B softened-water conductivity | Tag exists ("scrubber softened water conductivity"), range [0, 30]; unit not stated (INFERRED mS/cm or similar; do not rely). Narrative only lists "AI-0897 East Scrubber #1 Softened Water Hardness" | "- name: AI0897B # scrubber softened water conductivity / operating_range: [0, 30]" | YAML input_tags; TELE |
| Temperature / sump conductivity | No sump conductivity probe documented for #4 (narrative says conductivity probes AIT-5619/5650 exist only on scrubbers 5 & 6) | | NARR Misc Items 5 & 6 |

## 4. Chemicals

| Item | Value | Quote | Source + location |
|---|---|---|---|
| Injection point | Recirc-pump discharge, upstream of the scrubber inlet (not the sump) | "feeds it to the recirculation pump discharge upstream of the scrubber inlet" (both NaOCl and NaOH) | NARR sec. "Fermenter NaOCl and NaOH Chemical Dosing Pumps" |
| Arrangement | two bleach + two caustic pumps, duty/standby; independent of recirc pump | "There are two bleach pumps and two caustic pumps installed for each scrubber. They are configured in a duty/standby configuration." | NARR same, Objective |
| Pump tags | Caustic (pH): M-3730A PDP-65322, M-3730B PDP-65324. Bleach (ORP): M-3731A PDP-65313, M-3731B PDP-65314 | | NARR same, DCS Modules |
| Pump type/capacity | Peristaltic ("Flex Pro Peristaltic Pumps"). Bleach PDP-65313/14 model M-424-MNHH: 1.632 L/h per RPM. Caustic duty PDP-65322 model M-324-MNKL: 1.008 L/h per RPM; standby PDP-65324 "lower capacity flow head (0.576 LPH/RPM)". (The OPEX tag in YAML/Grafana uses 1.355 $/h per RPM bleach and 0.5455 or 0.4536 $/h per RPM caustic, which are cost factors not flows: 1.632 x $0.83/L = 1.355; 1.008 x $0.5412/L = 0.5455) | "Flow Coefficient: $1.632 LPH per RPM" ; "Flow Coefficient: $1.008 LPH per RPM" | RPT25 Appendix "Methodology: Chemical Spend Estimation" |
| AIC-3730_OUT / AIC-3731_OUT mapping | OUT is pump SPEED in RPM, not %, despite the "OUT" name: Grafana labels "Pump Speed", axis "RPM"; YAML range [0, 125]; red (alarm-guard) bounds upper 60 (caustic) and 50 (bleach). RPT25: ORP sensor failure drove bleach pump "to unsafe speeds (greater than 80 rpm)". Flow (L/h) = RPM x coefficient above (INFERRED mapping: the coefficient is flow per RPM; max flow at 125 RPM would be 204 L/h bleach and 126 L/h caustic, but 125 is a tag range, not a documented pump maximum) | "displayName: Pump Speed ... axisLabel: RPM" | GRAF "pH"/"ORP" panels; YAML; RPT25 "Emergent Safety" |
| FIC-3734_OUT | valve position %, 0-100 | see sec. 3 | GRAF; YAML |
| Cost basis | bleach $0.83/L; caustic $0.5412/L | | RPT25 Appendix |
| Hypochlorite strength (% or g/L available Cl) | NOT FOUND (narrative says only "sodium hypochlorite storage tank") | | |
| Caustic strength (%) | NOT FOUND | | |
| Storage | Caustic and bleach tanks shared with the East scrubber ("Both the East and Fermenter Scrubbers use the same chemical tanks"); tag names in narrative are East (LI-0911 caustic, LI-0913 bleach) | | NARR sec. "Fermenter Chemical Storage" |
| Chemistry stated | Step 1 H2S + NaOH to sulphide salts; Step 2 NaOCl oxidises to sulphates or elemental sulphur. "Alkaline solution - pH between 10-12" (AEC deck; the typo "NaCl (Caustic)" in the AEC deck is corrected to NaOH in AWWOA) | | AEC "The Chemistry of the Scrubbing Process"; AWWOA same slide |
| Dosing interlocks | trip on: recirc pump PCP-28529 off; low level on corresponding chemical tank (LAL-0911 NaOH, LAL-0913 NaOCl); blower GBB-28521 off | | NARR same, Interlocks/Trips |
| Dosing control | NaOCl feed by metering pump speed; NaOH by "stroke speed" (narrative wording; the real pumps are peristaltic so this is probably loose wording, INFERRED) | | NARR same |

## 5. Control

| Item | Value | Quote | Source + location |
|---|---|---|---|
| pH | PID AIC-3730 (output = NaOH pump). Narrative recommended SP 9.5. Alarms: pH low 8.5, high 10.5. Redundant probes AI-3730A / AI-3731A, operator picks avg/max/min | "The recommended setpoint for pH will be 9.5." ; "pH Alarm recommended to be set at Low at 8.5, High at 10.5." | NARR dosing section |
| ORP | PID AIC-3731 (output = NaOCl pump). Narrative recommended SP +700 mV (East/West use +670 mV). Alarms: ORP low 500 mV, high 850 mV. Probes AI-3730B / AI-3731B (tag naming: ORP probes carry the 3730B/3731B names) | "The recommended setpoint for ORP will be +700 mV" ; "ORP Alarms recommended to be set at Low 500 mV, High 850mV." | NARR dosing section |
| Setpoints actually in logs | QA: "the setpoints from the log files are 670 and 9.8". RPT25: SOP ORP 670 mV, pH 9.5, make-up 19 L/min; EPCOR fixed test 693 mV / pH 9.6 / 16 L/min. Answer to why 700/9.5 vs 670/9.8 was not recorded | | QA Q1; RPT25 |
| Agent limits (YAML) | pH SP operating_range [9.4, 10.0], action bounds +/-0.5 around SP; ORP SP range [650, 750], bounds from computed tags (typ. min(max(SP-20, 650),680) to max(min(SP+20, 700),670), AIC3731_upper range [650,850]); make-up flow SP range [10, 30] L/min, +/-10 | | YAML output_tags / computed_tags |
| Controller mode codes | 4 = Off, 8 = Man, 16 = Auto, 32 = Cas (Grafana value mappings). Agent cascade: operator mode value 16, agent mode value 32 ("op_mode_val: 16, ai_mode_val: 32"); agent writes `_SPRL` setpoints only while in Cas. NARR glossary: "CAS = Cascade Control Mode". DC: "find_manual_mode.py will output all timestamps that mode doesn't equal to 16" | | GRAF mappings; YAML cascade; DC; NARR Glossary |
| Quality status strings | "GoodNonCascade ... NotLimited": data quality good / reading hits limit | | DC Q6 |
| Fan/pump interlocks | Fan: VFD fault; fail-to-start 3 s; permissive HOA = Auto. Recirc: sump LO 300 mm; fail-to-start 4 s; stops with blower. Make-up valve: auto-close at 695 mm, reopen at 600 mm. Chem pumps: see sec. 4 | | NARR sections |
| Timing | Agent obs/action period 30 min (YAML "00:30:00"); ORP PV has a long lag causing oscillation; 15-min data sampling in cost calc | | YAML timing; RPT24 Challenges; RPT25 Appendix |
| ORP delay | "There was a significant delay in response of the ORP process value to changes in the setpoint" | | RPT24 "Challenges Identified" |

## 6. How Scrubber 4 differs from the others
- Scrubbers #1-#3 (and East/West): "dedicated single-speed exhaust fan"; #4 has "a dedicated VFD exhaust fan"; #5/#6 have "a dedicated variable speed exhaust fan located at the top of each scrubber", fan speed set manually (SC-5671) from foul-air duct pressures (NARR "Foul Air").
- #1-#4 "are connected to dedicated sources"; #5 & #6 can pull from multiple sources via manual dampers.
- #4 is controlled on fan blower differential (PDIC-3738); #5/#6 have packing dP (PDI-5612/5644), fan dP and sump conductivity transmitters; East/West have inlet/outlet H2S, packing and mist-eliminator dP (West), humidity and make-up flow; #4 narrative lists only intake RH/temp, inlet/discharge pressure.
- #4's make-up valve has level closes at 695/600 mm; East 663/620, West 678/600, #5 1016/600 (TBD), #6 695/600 (TBD) (from NARR make-up sections; the East/West assignment of 663/678 is inferred from section order, not verified).
- ORP recommended SP: #4 +700 mV, East/West +670, #5 +600 (600-650 design), #6 +700. ORP alarms: #4 500/850, East 500/800.
- Chemical tanks are shared between the East and Fermenter scrubbers; effluent sump pumps are shared in the same building.
- Fermenter air has the highest/most variable H2S (sludge fermenters); scrubber was chosen as the RL pilot. Observed median inlet 28.6 ppm (2024).

---------------------------------------------------------------------

## NOT FOUND (searched Drive full text for 'fermenter scrubber', 'packing', 'Tri-Pack', 'Jaeger', 'scrubber 4', 'recirculation pump', 'sump', 'FI-0871/FI0872', 'LI3734', 'hypochlorite')
- Tower diameter, packed height, number of beds/stages, packing type/size/vendor, specific area, void fraction, flow direction (co/counter-current), mist-eliminator specifics.
- Design air flow and its engineering unit; FI-0871 unit (tag exists, undocumented); fan/blower hp and curve; any gas-composition (CO2 etc.) data.
- Recirc pump design flow and FI-0872 unit (only the [0,4000] tag range); sump volume/dimensions/normal level/level-volume relation; overflow weir elevation.
- NaOCl and NaOH strengths; dosing pump max flow (only L/h per RPM and 0-125 tag range, no documented max rpm).
- Units of PI-0169, PDIC-3738, TI-0880, AI-0897B (all INFERRED above).
- Rationale for 670/9.8 vs 700/9.5 (question asked, no answer in Drive).
- Contents of "Scrubber 4.zip" (too big to download) and P&ID-ODR-400 (the P&ID the narrative cites for the fermenter scrubber). Recommended next step: download the zip via a non-MCP route (Drive web/API) or ask the user to upload it somewhere under 10 MB.
- Not Gold Bar: the P&ID.zip (ACRWC), "Copper Basin ..." docs, and "40 61 96.70 Odor Control Chemical Feed System" spec were not used (the latter two appeared in searches for Copper Basin; not opened).
- Housekeeping: telegraf-postgres.conf contains plaintext credentials; nothing from them was used or copied.

