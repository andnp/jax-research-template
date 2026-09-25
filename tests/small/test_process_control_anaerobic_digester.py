"""Buswell-stoichiometry / carbonate-speciation behavior of the ADM1 digester.

Complements `TestADM1Unit`/`TestADM1CODConservation` in
test_process_control_new_modules.py (COD balance, basic ranges, pH-crash
failure mode), which are unaffected by the CH4:CO2 mechanism added here.
"""

import jax
import jax.numpy as jnp
from process_control.units.anaerobic_digester import ADM1Params, ADM1State, _speciate_inorganic_carbon
from process_control.units.anaerobic_digester import reset as adm1_reset
from process_control.units.anaerobic_digester import step as adm1_step


def _run(params: ADM1Params, n_steps: int, q_feed: float = 125.0, feed_cod: float = 30000.0):
    state = adm1_reset(feed_cod, params, jax.random.PRNGKey(0))
    dt = jnp.array(0.04167)
    feed_cod_a = jnp.array(feed_cod)
    q_feed_a = jnp.array(q_feed)
    temperature = jnp.array(35.0)
    q_biogas = ch4_frac = ph = jnp.array(0.0)
    for _ in range(n_steps):
        state, q_biogas, ch4_frac, ph = adm1_step(state, feed_cod_a, q_feed_a, temperature, params, dt)
    return state, q_biogas, ch4_frac, ph


class TestADM1MethaneFraction:
    def test_steady_state_ch4_fraction_in_literature_range(self) -> None:
        """The default 60-day run (125 m3/d, 30 g COD/L, 35C) lands in the 55-70% CH4
        range Buswell & Mueller (1952) report as typical for sludge digestion, once
        both Buswell substrate stoichiometry and carbonate/bicarbonate CO2 retention
        are modeled — not the 50% a pure-acetate/carbohydrate stoichiometry gives.
        """
        params = ADM1Params()
        _, _, ch4_frac, _ = _run(params, n_steps=60 * 24)
        assert 0.55 < float(ch4_frac) < 0.70

    def test_higher_co2_ch4_ratio_lowers_ch4_fraction(self) -> None:
        """A more carbohydrate-like substrate (CO2:CH4 -> 1.0, the pure-acetate
        case) must produce less CH4-rich biogas than the default sludge-like
        composite substrate (CO2:CH4 = 0.6).
        """
        params_default = ADM1Params()
        params_carb = ADM1Params(co2_ch4_molar_ratio=1.0)
        _, _, ch4_default, _ = _run(params_default, n_steps=30 * 24)
        _, _, ch4_carb, _ = _run(params_carb, n_steps=30 * 24)
        assert float(ch4_carb) < float(ch4_default)


class TestADM1CarbonBalance:
    def test_inorganic_carbon_balance_over_multiple_steps(self) -> None:
        """CO2 production (Buswell stoichiometry, tied to CH4 production) plus
        dilution outflow equals vented CO2 plus the change in dissolved
        inorganic-carbon storage, within numerical tolerance — the carbon
        analogue of `test_cod_balance_over_multiple_steps`. Neither CH4 nor
        CO2 production is directly observable from `step()`'s outputs, so
        both are recovered from the same feed-less-washout identity that test
        already uses for COD (mass balance of a pool that receives no feed
        input is exactly `dilution * (0 - pool_before)`). Only the free
        CO2(aq) share of S_IC is diluted directly through the effluent — the
        HCO3- share's export is already carried by alkalinity's own
        dilution term (see the `d_s_ic_dil` comment in `step()`) — so the
        washout term here uses CO2(aq), not the whole S_IC pool.
        """
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(1))
        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        q_feed = jnp.array(125.0)
        temperature = jnp.array(35.0)
        dilution = q_feed / params.volume

        ch4_dilution_loss_total = jnp.array(0.0)
        ch4_vented_total = jnp.array(0.0)
        ic_dilution_loss_total = jnp.array(0.0)
        co2_vented_total = jnp.array(0.0)
        ch4_before = state.s_ch4
        ic_before = state.s_ic

        for _ in range(200):
            ch4_start = state.s_ch4
            co2aq_start, _ = _speciate_inorganic_carbon(state.s_ic, state.alkalinity, temperature)
            state, q_biogas, ch4_frac, _ = adm1_step(state, feed_cod, q_feed, temperature, params, dt)
            ch4_dilution_loss_total += dilution * (0.0 - ch4_start) * dt
            ic_dilution_loss_total += dilution * (0.0 - co2aq_start) * dt
            ch4_vol = q_biogas * ch4_frac * dt
            co2_vol = q_biogas * (1.0 - ch4_frac) * dt
            ch4_vented_total += ch4_vol * 1000.0 / 22.4 * 64.0 / params.volume
            co2_vented_total += co2_vol * 1000.0 / 22.4 * 44.0 / params.volume

        ch4_production_total = ch4_vented_total + (state.s_ch4 - ch4_before) - ch4_dilution_loss_total
        co2_production_total = ch4_production_total * params.co2_ch4_molar_ratio * (44.0 / 64.0)

        delta_ic_storage = state.s_ic - ic_before
        residual = co2_production_total + ic_dilution_loss_total - co2_vented_total - delta_ic_storage
        assert abs(float(residual)) < 0.02 * abs(float(co2_production_total))

    def test_organic_overload_still_crashes_ph_with_new_carbon_pathway(self) -> None:
        """The acid-crash failure mode (test_organic_overload_drops_ph in the
        shared test module) must still trigger with the Buswell/bicarbonate
        carbon pathway in place, not just under the old fixed acetate ratio.
        """
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(2))
        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        temperature = jnp.array(35.0)

        ph_baseline = jnp.array(7.0)
        for _ in range(24 * 10):
            state, _, _, ph_baseline = adm1_step(state, feed_cod, jnp.array(125.0), temperature, params, dt)

        overload_q_feed = jnp.array(1500.0)
        for _ in range(120):
            state, _, _, ph_overload = adm1_step(state, feed_cod, overload_q_feed, temperature, params, dt)

        assert float(ph_overload) < float(ph_baseline) - 0.5

    def test_jit_compatible(self) -> None:
        """The new S_IC/pH-speciation path must remain JIT-traceable."""
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(3))

        def step_fn(s: ADM1State):
            return adm1_step(s, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167))

        jit_step = jax.jit(step_fn)
        s2, qb, ch4, ph = jit_step(state)
        assert jnp.isfinite(qb)
        assert jnp.isfinite(ch4)
        assert jnp.isfinite(ph)
