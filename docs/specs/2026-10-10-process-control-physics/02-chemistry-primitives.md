# Milestone A: additive chemistry primitives

Read-only co-design, 10 October 2026. No existing or new production/test code was written. The only deliverable is this document. A should supply reusable, independently testable aqueous thermodynamic functions, not change component schemas, kinetics, flowsheet resolution, or any current benchmark while the BSM1 session is landing. Reviewed the committed ADR text, current `chemistry/ph_model.py`, and `units/anaerobic_digester.py`; the ADR status paragraphs still say “Proposed,” so do not infer an implementation status from their presence in history.

## A. Interface decisions

### 1. Keep AnalyticLogK as an expression, not as the universal thermodynamic model

The proposed six-coefficient form matches PHREEQC's convention:

`L(T) = log10 K = a1 + a2 T + a3/T + a4 log10(T) + a5/T² + a6 T²`, T in K.

The [official PHREEQC species documentation](https://water.usgs.gov/water-resources/software/PHREEQC/documentation/phreeqc3-html/phreeqc3-50.htm) defines this expression and distinguishes association reactions from dissociation reactions. Reversing the reaction negates every coefficient. A pKa expression must likewise be negated before storing it as log10 Ka. Never infer that direction from a species name.

For constant reaction enthalpy, with the reaction direction matching K:

`log10 K(T) = Lref − ΔH/(R ln 10) × (1/T − 1/Tref)`.

Thus `a3 = −ΔH/(R ln 10)` and `a1 = Lref − a3/Tref`; the other coefficients vanish. Positive dissociation ΔH makes Ka increase when temperature rises. Require finite coefficients, positive Tref, an ordered positive temperature range, and explicit J/mol. Do not equate the numeric Sander inverse-temperature slope in K with an enthalpy in J/mol.

Separate the expression from a sourced constant record. The record should carry reaction/quantity identity, standard-state convention, sources with equation/table identifiers, and domain. “One representation for every temperature-dependent constant” overcommits: modern Kw depends on density as well as T, and salting/pressure corrections require more context. The expression is sufficient for A's chosen low-pressure correlations; it is not a prohibition on later physical property strategies. PHREEQC syntax does not validate the empirical range of a particular fit.

Return the evaluated value and its domain diagnostic together, or require a single evaluation object that downstream functions propagate. A naked `log10_k(T)` result followed by an optional range check is too easy to misuse. Distinguish invalid numerical input, extrapolation, and unsupported physical regime. Extrapolation may return a finite diagnostic value, but must not become `in_domain=True`; do not silently clamp T into the fitted range.

### 2. Registry: explicit namespaces and analytical pools

Use an immutable, validated collection with separate species, families/pools, and solubility namespaces. A single `name → record` mapping cannot hold both molecular CO2 and CO2's Henry record under that same key. Separate typed lookups avoid a union return type and runtime kind checks. A frozen dataclass containing a mutable dict is not immutable; `MappingProxyType` alone is not hashable. Tuple-backed records with constructor-time indexes are sufficient. Closing an object over a JIT function does not require the entire registry to be a static argument to every operation.

**CO2* = CO2(aq) + H2CO3 is not a molecular species.** Its effective first dissociation constant is around 6.35 at 25 °C, rather than the dissociation constant of isolated hydrated H2CO3. Do not attach CO2* a fabricated exact molecular formula/molar mass, then demand that its conversion to bicarbonate is a one-H atom deletion. CO2(aq) + water → HCO3− + H+ includes solvent water; the combined pool also contains hydration states with different H/O counts.

Keep real molecular `Species` records. Add an explicit analytical-form/pool descriptor for the carbonate neutral form, with charge zero and one conserved carbon per form. Families can reference molecular species or such declared forms. Validate common conserved family moieties and a one-unit charge decrease across each acid dissociation. Validate complete elemental stoichiometry separately against balanced reactions including solvent water where required. Molecular-only families such as HOCl/OCl− can additionally validate the exact one-H difference. Do not weaken the whole registry's molecular atom validation merely to accommodate CO2*.

Returned carbonate concentrations must be named `CO2_star`, HCO3−, and CO3²− unless hydration is modeled explicitly. A Henry record for molecular CO2 cannot be blindly applied to the combined pool. A low-temperature approximation `CO2* ≈ CO2(aq)` may be stated and bounded; an effective CO2* Henry constant must be paired with the matching pool convention. This is a model choice, not a unit conversion. Full hydration kinetics is outside A.

Other construction checks: unique IDs within each namespace; known references; nonempty ordered family forms; exactly N−1 acid constants for N forms; finite positive molecular masses; integer charges and positive integer element counts with no duplicates; no duplicate species in a family; no overlapping independently counted analytical totals in a selected speciation system. Do not require monotonically increasing successive pKas as a universal validation rule. Family membership is derived from references, avoiding a second manually synchronized membership field in each Species.

Include H+ and OH− as real species but handle water autoionization separately. Neither is an independent family total. Na+ and Cl− are spectator species, not acid-base families. Molecular records need only modest precision using one declared atomic-mass convention; validating a formula against a mass must allow source rounding and the negligible ion-electron mass convention.

### 3. Compile a speciation layout, then evaluate numerical constants

`solve_charge_balance(families, totals, strong_charge, T, ...)` is missing **Kw**. Make the chosen water constant explicit. Also specify result ordering, because different families have different numbers of forms. Prefer a constructor-validated static `SpeciationSystem` holding ordered families, flattened output-form names, offsets, and charges. Dynamic arrays hold totals and evaluated constants; no dynamic dictionary/species lookup occurs in JIT.

Suggested signatures, not implementations:

```python
def evaluate_constants(system: SpeciationSystem, temperature_k: Array): ...
# EvaluatedConstants(log_ka, log_kw, numerical_valid, temperature_in_domain)

def fractions(log_ka: Array, ph: Array): ...
# A single family's ordered fractions; charge/proton steps are validated statically.

def solve_ideal_charge_balance(
    system: SpeciationSystem,
    totals_mol_per_m3: Array,
    strong_ion_charge_eq_per_m3: Array,
    constants: EvaluatedConstants,
    controls: SolveControls,
): ...

def henry_hcp(record: HenrySolubility, temperature_k: Array): ...
# PropertyEvaluation(value, numerical_valid, in_domain)

def equilibrium_concentration(hcp: Array, partial_pressure_pa: Array): ...
def equilibrium_partial_pressure(hcp: Array, molecular_concentration_mol_per_m3: Array): ...
```

The final two functions are simple inverse physical relations; evaluating Henry once avoids three places independently performing temperature/domain handling. Accepting evaluated arrays creates a natural future calibration/activity boundary without adding fitted-pKa options in A. Use one canonical solver kernel; an ergonomic system evaluator delegates to that kernel rather than implementing a second solver.

Make concentrations' trailing axes explicit: totals `(..., Nfamilies)`, logKa in a fixed compiled ordering, flattened forms `(..., Nforms)`, and scalar/batched pH and diagnostics. Either document direct batch broadcasting or require `vmap`; do not leave implicit broadcasting able to align the family axis with a temperature batch accidentally. Empty-family water/strong-acid cases are valid. Zero totals are valid; negative totals and nonfinite inputs are invalid, not silently clipped.

At ideal molar standard state `C° = 1000 mol/m3`, define:

`H = C° 10^(−pH)`, `OH = C° 10^(logKw + pH)`;

`R(pH) = S + H − OH + Σ Ct,f Σi zi αf,i = 0`,

where `S = Σ spectator zi Ci` is **positive for net cation excess**. This is not alkalinity and not its negative unless a consumer's invariant definition establishes that mapping. Residual is eq/m3. Using `10^−pH` directly with mol/m3 totals introduces a factor-of-1000 error; using a mol/m3 hydrogen concentration product with an unconverted molar Kw introduces 10^6.

For independent ideal acid families, stable fractions follow log weights `wi = i pH + Σj≤i logKa,j`, normalized in log space. The balance is monotone decreasing: `R′ = −ln(10)[H + OH + Σ Ct Var(z)]`. That supports safeguarded Newton with a maintained sign bracket and a bisection fallback. These identities are design/oracle derivations, not a reason to test private implementation details. Track bracket failure, finite numerical values, final unscaled and scaled residual, pH-interval width, and convergence separately. A bounded iteration count alone is not convergence. Solver pH search bounds are numerical bounds; model pH validity is a separate diagnostic.

Scale residuals with a declared absolute charge tolerance plus a concentration-dependent relative tolerance, avoiding division by zero for pure water. Require a meaningful pH/root error bound as well; a large total concentration can make a small relative residual conceal a bad pH in a weakly buffered case. Fixed iterations and JIT-safe status are compatible with this contract. Failed results may be returned for diagnosis, but must never claim convergence or physical validity.

Constants from molal thermodynamic standard states require an explicit convention. A's ideal concentration model may use the dilute-water approximation that 1 mol/kg solvent ≈ 1 mol/L, but must label it and its error budget. At 50 °C the water-density difference alone is around one percent, relevant to tight precision tests. A later molality/activity implementation needs solvent mass per liquid volume, not just the word “Davies.” Preserve source standard-state metadata now; do not present approximate converted constants as exact molarity constants.

### 4. Ideal-only in A; no assertion that a salty sump is ideal

I recommend **ideal-only A**, with an explicit model label and documented applicability. The proposed I = 0.05–0.2 M is a hypothesis, not a plant measurement. If true, ideal chemistry is not a quantitative scrubber-pH model. Using the 25 °C Davies estimate `log10 γz = −0.509 z²[√I/(1+√I) − 0.3I]`, monovalent γ is approximately 0.822 and 0.747 at the two endpoints; divalent γ approximately 0.455 and 0.311. These are illustrative scalar calculations from the equation, not evidence of plant salinity or validation of Davies at those conditions. The [PHREEQC manual](https://water.usgs.gov/water-resources/software/PHREEQC/documentation/phreeqc3-html/phreeqc3-50.htm) documents its Davies treatment.

Net strong-ion charge loses ionic-strength information: equal 100 mM Na+ and Cl− have net charge zero but large I, just like a charge-balanced solution containing almost no salt. Family totals and `S` alone therefore cannot implement Davies. Future input must preserve individual spectator concentrations or, for an explicitly limited charge-only activity closure, at least their validated `Σz²C/2` contribution. Coupled I depends on weak-family speciation too. pH then means H+ activity, not concentration; weights require the appropriate species activity-coefficient ratios and the water relation changes accordingly.

Davies also does not supply neutral-gas salting-out, specific ion pairing, concentrated-stock behavior, or a reliable mixed-electrolyte thermodynamic model merely by adding γ. Do not ship it as a false guarantee of sump fidelity. Provide an input-side external ionic-strength estimate/status for applicability diagnostics if desired; never infer “ideal valid” from net charge. Without that information, report ionic-strength applicability as unknown rather than silently true. A's `in_domain` must distinguish known temperature/numerical domain from the unverified activity approximation. Activity extension must precede claims of predictive saline-sump chemistry, even if it follows A.

### 5. Keep calibration outside A, but keep numerical parameters explicit

No fitted-pKa shift API or calibration registry is needed in A. Sourced records are immutable defaults, not tunable truth. Separating constant evaluation from the numerical solver allows an explicit future parameter pytree to supply evaluated logKa/logKw/Henry values without mutating static metadata or changing the carrier schema. Avoid baking every numeric constant into a giant static solver object; that forces recompilation and prevents sensitivity evaluation unnecessarily.

Temperature is already a numerical leaf. Differentiating a fixed bracketed algorithm can give algorithmic derivatives, especially with a bisection fallback, rather than equilibrium derivatives. Do not advertise gradient-safe calibration solely because values are JIT-safe. A should test temperature sensitivity in smooth valid regimes; a later differentiable solver can use the implicit derivative `dpH/dθ = −Rθ/RpH` with convergence/domain checks. This is a scoped future capability, not justification to implement a calibration stack now.

## B. Constants I would adopt

All tuples below are **log10 dissociation K**, in the six-coefficient order above. Numbers are specified for reproducibility; more digits in a correlation do not imply matching physical accuracy. Source-reported range, our chosen use range, and numerical extrapolation range are different metadata. Use around-atmospheric pressure in A. Thermodynamic molal constants are used under the declared dilute concentration approximation unless an exact conversion is provided.

### Water Kw

Best current reference is [IAPWS R11-24 (2024)](https://iapws.org/public/documents/5pjeC/Kw-2024.pdf), based on Bandura–Lvov with revised Arcis et al. parameters. It is a T-and-water-density function, **not** exactly the PHREEQC six-term T-only form. Its documented range is 273.15–1273.15 K and pressures to 1000 MPa for stable fluid states. It provides a direct code oracle at T = 300 K and density = 1 g/cm3: pKw = 13.906672. That is not a test of a one-atmosphere T-only fit.

For narrowly scoped A I would use the directly readable low-pressure correlation in [Bates & Pinching (1949), NBS RP1982](https://nvlpubs.nist.gov/nistpubs/jres/42/jresv42n5p419_A1b.pdf), equation 21, attributed there to Harned & Owen's evaluation of four determinations:

`pKw = 4470.99/T − 6.0875 + 0.017060 T`.

Tuple: `(6.0875, −0.017060, −4470.99, 0, 0, 0)`.

Source fit: 0–60 °C, mean deviation 0.0005 against the historical evaluation. Chosen A envelope: 0–50 °C, dilute low-pressure liquid water. Identify it as that historical fit, not “IAPWS 2024” or an exact reproduction of Harned & Hamer (1933). Its predictions are sufficiently close to the modern low-pressure table for a reduced A model; do not force agreement beyond the chosen fit/standard-state accuracy. If modern water thermodynamics is required in A, implement the actual density-dependent IAPWS property and provide density, rather than inventing six coefficients attributed to it.

For verification of that alternative, the official formula is:

`pKw = −2n[log10(1+Z) − Z/(1+Z) ρ(β0+β1/T+β2ρ)] + pKw_g + 2log10(Mw/1000)`;

`Z = ρ exp(α0 + α1/T + α2 ρ^(2/3)/T²)`;

`pKw_g = 0.61415 + 48251.33/T − 67707.93/T² + 10102100/T³`.

Here ρ is g/cm3, n = 6, Mw = 18.015268 g/mol; `(α0,α1,α2) = (−0.702132,8681.05,−24145.1)` and `(β0,β1,β2) = (0.813876,−51.4471,−0.469920)`, with units as in IAPWS Table 1. These are supplied for a future explicit alternative, not a request for two live default Kw paths.

### Carbonate effective pK1 and pK2

Use [Plummer & Busenberg (1982)](https://doi.org/10.1016/0016-7037(82)90056-4), *The solubilities of calcite, aragonite and vaterite…*, with the effective dissolved CO2 convention. The [authors' USGS record](https://pubs.usgs.gov/publication/70011789) identifies the evaluation and pressure conditions. Exact coefficient signs are independently checkable in the [official PHREEQC database](https://github.com/phreeqc-dev/phreeqc3/blob/master/database/phreeqc.dat): its carbonate entries are association constants, so reverse them or subtract reactions to obtain the successive acid constants.

`logKa1 = −356.3094 − 0.06091964 T + 21834.37/T + 126.8339 log10 T − 1684915/T²`;

`logKa2 = −107.8871 − 0.03252849 T + 5151.79/T + 38.92561 log10 T − 563713.9/T²`.

Tuples: `(-356.3094, -0.06091964, 21834.37, 126.8339, -1684915, 0)` and `(-107.8871, -0.03252849, 5151.79, 38.92561, -563713.9, 0)`.

Choose 0–90 °C for A's documented carbonate range; do not advertise the paper's broader high-temperature evaluation without its pressure/water-state treatment. At 25 °C expect approximately 6.352 and 10.329. The effective first constant must not be paired with isolated H2CO3 as though it were that molecule's intrinsic Ka. This section uses official model/database verification; publisher full text was not independently readable in this session.

### H2S / HS−

Use the infinite-dilution equation reproduced as equation 4 of [Millero, Plese & Fernandez (1988)](https://doi.org/10.4319/lo.1988.33.2.0269), sourced there to [Hershey, Plese & Millero (1988)](https://doi.org/10.1016/0016-7037(88)90183-4):

`pKa1 = −98.080 + 5765.4/T + 15.0455 ln T`.

Tuple: `(98.080, 0, −5765.4, −15.0455 ln(10), 0, 0)`; `a4 ≈ −34.64354401664192`.

The [primary-paper scan](https://www.researchgate.net/profile/Y-Xu-6/post/Do-you-have-any-references-for-rate-of-dissociation-of-Hydrogen-Sulfide-in-Seawater/attachment/5ae443a0b53d2f63c3c899ac/AS:620327293968384@1524908960030/download/Millero_et_al-1988-Limnology_and_Oceanography.pdf), pp. 269–271, confirms **15.0455**, not the 15.04555 often reproduced elsewhere. It states a 0–300 °C correlation with σ ≈ 0.05 pK, with low-temperature 0–45 °C data based on the newer measurements and higher-temperature thermodynamic input. Choose **0–45 °C** for A. The seawater salinity adjustment and total-proton pH scale are separate conditional models; do not apply those coefficients to a NaCl/bleach sump or combine them with Davies.

Excluding S²− is reasonable within a bounded ordinary-temperature, moderate-pH model. [Ellis & Giggenbach (1971)](https://doi.org/10.1016/0016-7037(71)90036-6) give pK2 >17 at 25 °C; [Schoonen & Barnes (1988)](https://doi.org/10.1016/0016-7037(88)90326-2) estimate 18.51 ±0.56 at 20 °C. These are not interchangeable exact constants. State a conservative pH use ceiling, for example 12 in A, and report out of model domain beyond it. Do not assert this truncation is universally valid at high temperature or for metal-sulfide solubility models. No pK2 coefficient is proposed in A.

### HOCl / OCl−

Use [J. Carrell Morris (1966)](https://doi.org/10.1021/j100884a007), *The Acid Ionization Constant of HOCl from 5 to 35°*, equation 13:

`pKa = 3000/T − 10.0686 + 0.0253 T`.

Tuple: `(10.0686, −0.0253, −3000, 0, 0, 0)`.

The original paper's indexed scan gives computed values from 0–35 °C and accuracy about 0.01 pK. Choose **0–35 °C**, noting the measured temperatures approximately 4–34 °C and the endpoint extension. Above 35 °C return an extrapolation diagnostic. Do not adopt the often repeated “0–50 °C” range as though Morris validated it. Published rounded values include pKa = 7.754 at 5 °C, 7.537 at 25 °C, and 7.463 at 35 °C. Source access: [primary journal scan](https://lib3.dss.go.th/fulltext/scan_ebook/j.of_physical_1966_v70_n12.pdf), pp. 3798–3805, was available in indexed excerpts; complete download timed out. Lead should verify equation 13/Table VII directly before adopting the record.

### NH4+ / NH3

Use the directly readable **1949 ammonium-acid** paper [Bates & Pinching, NBS RP1982](https://nvlpubs.nist.gov/nistpubs/jres/42/jresv42n5p419_A1b.pdf), equations 20 and Table 4:

`pKa = 2835.76/T − 0.6322 + 0.001225 T`.

Tuple: `(0.6322, −0.001225, −2835.76, 0, 0, 0)`.

Range **0–50 °C**. The reported mean fit deviation is 0.0011 pK. This is not the common `0.09018 + 2729.92/T` approximation and must not be cited as that approximation. It is also not Bates & Pinching's separate 1950 paper about the basic dissociation of aqueous ammonia. The 1949 Table 3 directly lists acid pKas of 10.0813 at 0 °C, 9.2449 at 25 °C, and 8.5387 at 50 °C. Preserve its standard-state convention and fit tolerance rather than requiring exact equality to a modern database's rounded 9.25.

### Sulfate / bisulfate

Do **not** permanently classify sulfate as pH-independent strong anion. At pH 4 with pKa ≈2, HSO4− is approximately one percent of sulfate-family total in the ideal limit; at pH 2 it is comparable to SO4²−. Even a small fraction can matter to a tight charge residual at large sulfate loading.

Include a named HSO4−/SO4²− family if A claims sulfate chemistry across acidic solutions. A defensible official coefficient record is the reversal of PHREEQC/WATEQ's association expression:

`logKa = 56.889 − 0.006473 T − 2307.9/T − 19.8858 log10 T`;

tuple `(56.889, −0.006473, −2307.9, −19.8858, 0, 0)`.

The [USGS thermodynamic appendix](https://pubs.usgs.gov/pp/1728/pdf/p1728_508.pdf), Table 1–1, explicitly documents the association expression and logK = 1.988 at 25 °C. The [official database](https://github.com/phreeqc-dev/phreeqc3/blob/master/database/phreeqc.dat) agrees. This is a **database/evaluation source**, not yet a verified experimental-primary coefficient provenance/range in this review. [Covington, Dobson & Wynne-Jones (1965)](https://doi.org/10.1039/TF9656102057) is a relevant 25 °C primary determination, but its publisher page did not expose coefficients or a temperature correlation; do not falsely attribute the above fit to it.

My concrete A choice: include molecular HSO4− and SO4²− records now; adopt the family only after the lead verifies the coefficient provenance and low-temperature range. Alternatively ship an explicitly sulfate-as-SO4²− reduced assembly valid only above its stated pH threshold, with the omitted protonation error documented, while leaving the generic solver capable of the family. Do not call an unverified broad-temperature bisulfate correlation “sourced primary data.” A primary-temperature range for this optional fit is an unresolved adoption gate, not a number to guess.

## C. Henry solubility

The proposed convention and SI units are correct. Use:

`Hcp(T) = Hcp_ref exp[B(1/T − 1/Tref)]`, Tref = 298.15 K;

`C_molecular,eq = Hcp p`, `p_eq = C_molecular/Hcp`.

B is `d ln Hcp / d(1/T)` in K, positive for these usual exothermically dissolving gases. This is a constant-slope local approximation, not a universal wide-temperature fit. If using AnalyticLogK for Henry, explicitly mean `log10(Hcp / Hcp_unit)` with Hcp_unit = 1 mol/(m3 Pa); Hcp is dimensional, unlike thermodynamic K. Prefer a separate Henry record using that same mathematical evaluator internally.

For A's default simple records I would select the following specific assessed rows from [Sander (2023), compilation 5.0.0](https://doi.org/10.5194/acp-23-10901-2023), not average incompatible rows:

| Molecular gas | Hcp_ref, mol/(m3 Pa) | B, K | Selected assessed row/source |
|---|---:|---:|---|
| H2S | 1.0e−3 | 2100 | Burkholder et al. (2019) review row in [H2S table](https://www.henrys-law.org/henry/casrn/7783-06-4) |
| CO2 | 3.4e−4 | 2300 | Burkholder et al. (2019) review row in [CO2 table](https://henrys-law.org/henry/casrn/124-38-9) |
| O2 | 1.3e−5 | 1500 | Burkholder et al. (2019) review row in [O2 table](https://www.henrys-law.org/henry/casrn/7782-44-7) |
| NH3 | 5.9e−1 | 4200 | Burkholder et al. (2019) review row in [NH3 table](https://www.henrys-law.org/henry/casrn/7664-41-7) |
| CH4 | 1.4e−5 | 1600 | Burkholder et al. (2019) pure-water row in [CH4 table](https://www.henrys-law.org/henry/casrn/74-82-8), excluding the distinct seawater row |

These are rounded assessed constants, not values independently read from each experimental primary paper. Preserve Sander row attribution and precision. Its notes explicitly warn that the two-parameter temperature slope can be only a local summary of a more detailed original fit. Chosen A operating guard **5–35 °C** is a conservative design envelope for this approximation, **not** a published validation range shared by all five rows. Store original-range status as unknown until verified, and label approximation adequacy separately from temperature guard. If the requested definition of `in_domain` requires experimentally verified fit range, these records cannot honestly claim it yet.

Useful primary/authoritative follow-up sources, rather than invented ranges: H2S [Clarke & Glew (1971)](https://doi.org/10.1139/v71-116) and [Carroll & Mather (1989)](https://doi.org/10.1016/0016-7037(89)90053-7); CO2 [Weiss (1974)](https://doi.org/10.1016/0304-4203(74)90015-2); O2 [Rettich et al. (2000)](https://www.henrys-law.org/henry/casrn/7782-44-7) as recommended by the IUPAC-NIST evaluation linked there; NH3 [Clegg & Brimblecombe (1989)](https://doi.org/10.1021/j100357a041); CH4 [Rettich et al. (1981)](https://www.henrys-law.org/henry/casrn/74-82-8). The compilation gives multiple competing rows; switching to an original full correlation is a record/model change with independent validation, not silently replacing its slope while keeping the old source.

Sander's H2S note 168 supplies a useful richer **compilation-fitted** form from Clarke & Glew data: `Hcp = exp(−133.37136 + 7422.07661/T + 17.82904 ln T)` in mol/(m3 Pa). It fits the proposed AnalyticLogK form after dividing the natural-log coefficients by ln 10 and converting the lnT term appropriately. Do not misattribute these fitted coefficients as a verbatim original-paper equation. No reason to implement this second H2S default in A unless its precision is needed.

Only molecular NH3 and H2S concentrations enter these relations, never ammonia or sulfide totals. Effective solubility after protonation belongs to the speciation/transfer composition. CO2* requires the explicit hydration convention described above. Henry's law here assumes dilute solute, pure-water reference, and partial pressure approximately equal to fugacity; high pressure, saline solvent, or concentrated ammonia requires a different declared closure. Davies does not supply those corrections. Validate positive finite Hcp and nonnegative finite p/C; zero p/C must return exactly zero counterpart.

## D. Test oracles and acceptance cases

No tests were written or run. The following is a test design, with independently published checks separated from scalar calculations of the proposed formulas. Do not create a test that repeats the implementation equation and calls that independent physical validation.

**Published-value checks:**

- Morris's table: pKa(HOCl) 7.754 at 5 °C, 7.537 at 25 °C, 7.463 at 35 °C; tolerance around 0.01 pK for physical agreement, stricter only for the rounded correlation values.
- Bates & Pinching Table 3: pKa(NH4+) 10.0813 at 0 °C, 9.2449 at 25 °C, 8.5387 at 50 °C. A ≈0.003 pK fit allowance covers these anchors without pretending equation 20 exactly interpolates every measured point.
- IAPWS modern low-pressure table gives rounded pKw approximately 14.95, 13.99, and 13.26 at 0, 25, and 50 °C. Use roughly 0.02 pK as a comparison envelope for the chosen historical T-only fit, not a 1e−6 assertion against a different density/standard-state model. If implementing actual IAPWS instead, its T = 300 K, ρ = 1 g/cm3 test value is 13.906672, with a code-verification tolerance appropriate to the printed digits.
- Official carbonate association values at 25 °C give successive dissociation pKas approximately 6.352 and 10.329. Check reaction reversal/subtraction explicitly with an independently tabulated reference, not merely constructor output.
- The H2S primary scan's infinite-dilution equation is the adopted correlation; published approximate room-temperature pKa ≈7.0 from Ellis & Giggenbach is a coarse independent check. For precision, use Hershey's primary tables after confirming their pH/molality scale; do not test ideal freshwater output against a seawater conditional table.
- Each selected Sander Henry reference row at 298.15 K is a tabulated numeric oracle. Use the row's printed precision. For a temperature oracle independent of a constant-slope model, acquire an original measured/full-correlation value first and give the approximation an accuracy envelope; the rounded B is not an exact measured curve.

**Derived numeric regression anchors**, calculated here with scalar JavaScript, not a simulator or experiment:

| T, °C | pKw (NBS eq.21) | carbonate pKa1 | carbonate pKa2 | H2S pKa1 | HOCl pKa | NH4+ pKa |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.940698 | 6.578562 | 10.628572 | 7.432656 | 7.825071 | 10.084104 |
| 5 | 14.731764 | 6.516302 | 10.554413 | 7.326154 | 7.754142 | 9.903608 |
| 25 | 13.994713 | 6.351864 | 10.328854 | 6.980438 | 7.536644 | 9.244219 |
| 45 | 13.393227 | 6.289410 | 10.195832 | 6.741680 | **7.410109 extrapolated** | 8.670814 |
| 50 | 13.261088 | 6.285801 | 10.174401 | **6.695904 beyond chosen A guard** | **7.390709 extrapolated** | 8.539026 |

These help catch sign/log-base/transcription mistakes and support a documented formula regression; they are not independent measurements.

**Behavior/contract cases:**

1. Empty-family water with S = 0 has pH = pKw/2 and H = OH. For prescribed S, compare to the independent quadratic `h² + (S/C°)h − Kw = 0` in molar-normalized variables, using a cancellation-safe oracle. Include dilute strong acid/base and salt with net zero charge. This exposes sign and factor-of-1000 mistakes.
2. Monoprotic family at pH = pKa has fractions 1/2,1/2; a one-pH-unit displacement gives 1/11,10/11. For diprotic carbonate use an independent closed-form fraction oracle at pH near each transition, including extreme but valid pH. Fractions remain finite/nonnegative and sum to one; no unstable raw exponential normalization.
3. Flattened species/forms sum to each family total. Charge residual, reconstructed charge, and root bracket agree. Test zero totals, no buffer, multiple families, strong-charge sign changes, and a trace family beside a large buffer. Do not count water H/O as independently conserved analytical totals.
4. Require failed-bracket, iteration-limit, invalid-temperature, negative-total, nonfinite-input, and out-of-fit-range statuses. `converged` and `in_domain` are distinct. Reordering families permutes outputs without changing pH; adding a zero-total family cannot change valid ideal charge accounting.
5. Compare eager, JIT, and vmap values/status; assert numerical tolerances, not exact algorithmic iteration counts. Check temperature sensitivity against an independent perturbation oracle in smooth converged cases. If gradients are not validated, document that limit.
6. Henry forward/inverse relations agree with independent SI conversion anchors: `1 mol/(L atm) = 1000/101325 mol/(m3 Pa)`. Test zero partial pressure, zero molecular concentration, reference T, sign of thermal trend, and a molecular-versus-total example. These primitives must permit desorption through the driving force later; they do not themselves clamp transfer to absorption.
7. Registry validation rejects malformed references/family sizes/charges and molecular atom imbalance, while accepting explicitly modeled carbonate hydration/pools. Tests should distinguish molecular stoichiometry from analytical-pool bookkeeping rather than encode a fake CO2* molecule.

## E. Future replacement constraints

- **`chemistry/ph_model.py` (MIGRATED 2026-10-10; file deleted, `benchmarks/ph_neutralization.py` now uses the charge-balance solver with `S`/`Ct` invariants, so this bullet is historical):** its `base_excess` is mol/L, positive base excess; `buffer_capacity` is total carbonate mol/L. With the current ideal carbonate convention, future charge-solver mapping is `S = 1000 base_excess`, `Ct = 1000 buffer_capacity`, and `R_new = −1000 F_old`. The current nominal pKa2 is **10.25**, not the recommended 10.328854; Kw is fixed 14 and pKa1 6.35. Preserve that nominal benchmark parameterization for an explicitly behavior-preserving migration, or declare/test a physical-model change. Do not call the new defaults numerically equivalent. A does not edit or wrap this existing solver; canonical replacement waits for its consumer migration.
- **Digester Henry helper:** accepts °C and returns mol/(L atm). Convert T using +273.15 and Hcp using ×1000/101325. Its `dh_r_*` numbers are used as J/mol in code despite ambiguous coefficient comments; the slope B matching its formula is `dh_r/R`. Its actual R is 8.3145 J/(mol K). Existing CH4 values 0.0014 mol/(L atm), 14240 J/mol yield Hcp_ref ≈1.38169e−5 and B ≈1712.670636 K; CO2 0.035, 19410 yield ≈3.45423e−4 and B ≈2334.475916 K. Those are explicit donor parameters and differ from rounded Sander defaults. Migrating the helper must not quietly change them. Replacing its rounded R by 8.314462618 changes B slightly too; specify a tolerance or an intentional numerical change.
- **Digester carbonate:** `_pka1_co2` uses pKa_ref 6.35 and ΔH = 7646 J/mol at 298.15 K. The analytic expression can represent its existing formula exactly: Lref = −6.35, a3 ≈−399.37646384, a1 ≈−5.01048478 using its R = 8.3145. A constructor using modern R = 8.314462618 instead gives a3 ≈−399.37825944 and a1 ≈−5.01047875; close, but not bitwise equivalent. Its reduced bicarbonate/alkalinity split and pH clipping are separate model policy, not the full charge solver. Keep property harvesting distinct from replacing that chemistry model.
- **Basis conversion:** digester TIC is mg/L as CO2, ammonia mg/L as N, alkalinity mg CaCO3/L, methane mg COD/L. They require separate explicit factors; there is no common “convert chemistry to molar” multiplication. Current ADM1-style time base is days; new equilibrium properties have no time base. Kinetic migration must retain explicit per-day/per-hour conversion.
- **ASM/BSM:** do not change `s_alk`, aeration saturation law, temperature factors, constructors, or precision defaults in A. A's constants are not permission to replace BSM1 calibration values. No mandatory package-wide re-export or `__init__.py` edit is needed for direct new-module imports.
- **Scrubber:** shared properties will later replace duplicated pKas and empirical pH ownership. A contains no ORP fit, product-yield choice, film enhancement law, pump/valve behavior, or sulfate-oxidation rate. Plant stock strength/salinity and measured liquid temperature remain unknown. The neutralization/digester compatibility cases are evidence that primitives compose, not a reason to carry two live implementations after those consumers eventually migrate.

Implementation scope remains new chemistry files only. Source records with unresolved provenance/ranges should be withheld or clearly identified as provisional evaluation records; unknown range must never be transformed into a guessed published range. Ideal speciation can be useful and testable before it is a credible saline-sump model. The key additions that prevent a corner are explicit Kw/standard-state input, separate analytical pools, numerical evaluated constants, deterministic output layout, and honest validity/convergence diagnostics.
