"""Buswell-stoichiometry / carbonate-speciation behavior of the digester.

Complements `TestDigesterUnit`/`TestDigesterCODConservation` in
test_process_control_new_modules.py (COD balance, basic ranges, pH-crash
failure mode), which are unaffected by the CH4:CO2 mechanism added here.
"""

import jax
import jax.numpy as jnp
from process_control.units.anaerobic_digester import DigesterParams, DigesterState
from process_control.units.anaerobic_digester import reset as digester_reset
from process_control.units.anaerobic_digester import step as digester_step


def _run(params: DigesterParams, n_steps: int, q_feed: float = 125.0, feed_cod: float = 30000.0, seed: int = 0):
    state = digester_reset(feed_cod, params, jax.random.PRNGKey(seed))
    dt = jnp.array(0.04167)
    feed_cod_a = jnp.array(feed_cod)
    q_feed_a = jnp.array(q_feed)
    temperature = jnp.array(35.0)

    def body(state: DigesterState, _step: None):
        state, q_biogas, ch4_frac, ph = digester_step(state, feed_cod_a, q_feed_a, temperature, params, dt)
        return state, (q_biogas, ch4_frac, ph)

    state, (q_biogas, ch4_frac, ph) = jax.lax.scan(body, state, None, length=n_steps)
    return state, q_biogas[-1], ch4_frac[-1], ph[-1]


class TestDigesterMethaneFraction:
    def test_steady_state_ch4_fraction_above_raw_buswell_ratio(self) -> None:
        """The default 60-day run (125 m3/d, 30 g COD/L, 35C) settles above the raw
        Buswell gross-production ratio (1/(1+0.6) = 62.5%) and below the acetate/
        carbohydrate 50% baseline: NH4+ from ammonification pairs with some of the
        produced CO2 as HCO3-, which leaves dissolved with the effluent rather than
        venting, so net vented CO2 is a smaller share of gross CO2 production than
        of gross CH4 production (whose dissolved standing pool is much smaller, CH4
        being far less soluble). This is a substrate-specific outcome for the
        C10H19O3N formula (`DigesterParams.co2_ch4_molar_ratio`/`f_in_hyd`), not a
        target tuned to a literature range — the ~55-70% quoted for "mixed
        primary+WAS sludge" (Buswell & Mueller 1952) applies to a more
        carbohydrate-rich composite than this primary-sludge-only formula.
        """
        params = DigesterParams()
        _, _, ch4_frac, _ = _run(params, n_steps=60 * 24)
        assert 0.625 < float(ch4_frac) < 0.80

    def test_higher_co2_ch4_ratio_lowers_ch4_fraction(self) -> None:
        """A more carbohydrate-like substrate (CO2:CH4 -> 1.0, the pure-acetate
        case) must produce less CH4-rich biogas than the default sludge-like
        composite substrate (CO2:CH4 = 0.6).
        """
        params_default = DigesterParams()
        params_carb = DigesterParams(co2_ch4_molar_ratio=1.0)
        _, _, ch4_default, _ = _run(params_default, n_steps=30 * 24)
        _, _, ch4_carb, _ = _run(params_carb, n_steps=30 * 24)
        assert float(ch4_carb) < float(ch4_default)


class TestDigesterCarbonBalance:
    def test_inorganic_carbon_balance_over_multiple_steps(self) -> None:
        """CO2 production (Buswell stoichiometry, tied to CH4 production) plus
        dilution outflow equals vented CO2 plus the change in dissolved
        inorganic-carbon storage, within numerical tolerance — the carbon
        analogue of `test_cod_balance_over_multiple_steps`. Neither CH4 nor
        CO2 production is directly observable from `step()`'s outputs, so
        both are recovered from the same feed-less-washout identity that test
        already uses for COD (mass balance of a pool that receives no feed
        input is exactly `dilution * (0 - pool_before)`). The whole S_IC
        pool (CO2(aq) and HCO3- together) washes out through the effluent,
        so the washout term here uses total S_IC, matching `step()`.
        """
        params = DigesterParams()
        state = digester_reset(30000.0, params, jax.random.PRNGKey(1))
        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        q_feed = jnp.array(125.0)
        temperature = jnp.array(35.0)
        dilution = q_feed / params.volume

        ch4_before = state.s_ch4
        ic_before = state.s_ic

        def body(state: DigesterState, _step: None):
            ch4_start, ic_start = state.s_ch4, state.s_ic
            state, q_biogas, ch4_frac, _ = digester_step(state, feed_cod, q_feed, temperature, params, dt)
            ch4_vol = q_biogas * ch4_frac * dt
            co2_vol = q_biogas * (1.0 - ch4_frac) * dt
            terms = (
                dilution * (0.0 - ch4_start) * dt,
                dilution * (0.0 - ic_start) * dt,
                ch4_vol * 1000.0 / 22.4 * 64.0 / params.volume,
                co2_vol * 1000.0 / 22.4 * 44.0 / params.volume,
            )
            return state, terms

        state, terms = jax.lax.scan(body, state, None, length=200)
        ch4_dilution_loss_total, ic_dilution_loss_total, ch4_vented_total, co2_vented_total = (t.sum() for t in terms)

        ch4_production_total = ch4_vented_total + (state.s_ch4 - ch4_before) - ch4_dilution_loss_total
        co2_production_total = ch4_production_total * params.co2_ch4_molar_ratio * (44.0 / 64.0)

        delta_ic_storage = state.s_ic - ic_before
        residual = co2_production_total + ic_dilution_loss_total - co2_vented_total - delta_ic_storage
        assert abs(float(residual)) < 0.02 * abs(float(co2_production_total))

    def test_alkalinity_tracks_ammonium_charge_balance_at_steady_state(self) -> None:
        """At steady state (VFA net change ~0, so the acid/base swing term in the
        alkalinity balance vanishes), alkalinity must equal the ammonium charge
        it is paired with: `alkalinity = s_in * _ALK_PER_N` (1 eq HCO3- generated
        per eq NH4+ released) — alkalinity and S_IC are separately conserved
        quantities (net charge vs. carbon mass), each balanced on its own, but
        alkalinity's OWN source/sink pair must close against S_IN's.
        """
        params = DigesterParams()
        state, _, _, _ = _run(params, n_steps=150 * 24, seed=4)

        alk_per_n = 50.0 / 14.0
        predicted_alk = float(state.s_in) * alk_per_n
        assert abs(float(state.alkalinity) - predicted_alk) < 0.05 * predicted_alk

    def test_organic_overload_still_crashes_ph_with_new_carbon_pathway(self) -> None:
        """The acid-crash failure mode (test_organic_overload_drops_ph in the
        shared test module) must still trigger with the Buswell/bicarbonate
        carbon pathway in place, not just under the old fixed acetate ratio.
        """
        params = DigesterParams()
        state, _, _, ph_baseline = _run(params, n_steps=24 * 10, seed=2)

        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        temperature = jnp.array(35.0)
        overload_q_feed = jnp.array(1500.0)

        def body(state: DigesterState, _step: None):
            state, _, _, ph = digester_step(state, feed_cod, overload_q_feed, temperature, params, dt)
            return state, ph

        _, ph_trace = jax.lax.scan(body, state, None, length=120)
        ph_overload = ph_trace[-1]

        assert float(ph_overload) < float(ph_baseline) - 0.5

    def test_jit_compatible(self) -> None:
        """The new S_IC/pH-speciation path must remain JIT-traceable."""
        params = DigesterParams()
        state = digester_reset(30000.0, params, jax.random.PRNGKey(3))

        def step_fn(s: DigesterState):
            return digester_step(s, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167))

        jit_step = jax.jit(step_fn)
        s2, qb, ch4, ph = jit_step(state)
        assert jnp.isfinite(qb)
        assert jnp.isfinite(ch4)
        assert jnp.isfinite(ph)
