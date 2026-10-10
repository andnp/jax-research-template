"""Combined nitrogen and phosphorus control benchmark (ASM2d + FeCl₃).

The hardest variant: biological N removal (nitrification/denitrification) and
biological + chemical P removal in a single plant. Conflicting requirements:
  - High aeration helps nitrification but hurts anaerobic PAO zone
  - Denitrification needs anoxic conditions but NO₃ intrusion kills bio-P
  - Chemical P backup (FeCl₃) can compensate but costs money

FeCl₃ is dosed into the aerobic zone (simultaneous precipitation). Captured P
is removed from the reactor state immediately, as if wasted with the sludge
(reduced-order: no metal-phosphate solids state), and the reduced PO₄
recirculates to the anaerobic zone with the RAS. Rates are per hour (ASM time
unit): dt = 0.02 h and steps_per_day = 1200, so a 10-day horizon is 12000 steps;
nitrifier washout is hysteretic over days, so evaluate over 8-10 simulated days.

Reward is euros per step (energy, chemicals, exceedance fines). Influent NH4
and PO4 load follows the diurnal flow, which the flow observation exposes.

Action (4D): [kla / kla_max, Q_recycle / Q_max, FeCl₃ dose / dose_max, carbon_dose / carbon_max]
Observation (9D):
  effluent PO₄ / 10
  effluent NH₄ / 35
  effluent NO₃ / 20
  anaerobic PO₄ / 30
  aerobic DO / 8
  PHA/PAO ratio
  flow / mean_flow
  FeCl₃ dose / dose_max
  carbon dose / carbon_max
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.precipitation import PrecipitationParams, precipitate
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as analyzer_reset
from process_control.sensors.lagged import step as analyzer_step
from process_control.units.asm2d import ASM2dParams, make_default_influent_asm2d, mix, perfect_clarifier, reactor_step


@jax_dataclass
class CombinedNPState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    anaerobic_state: jax.Array
    anoxic_state: jax.Array
    aerobic_state: jax.Array
    po4_eff_sensor: LaggedSensorState
    nh4_eff_sensor: LaggedSensorState
    no3_eff_sensor: LaggedSensorState
    po4_ana_sensor: LaggedSensorState
    last_q_in: jax.Array


@dataclass(frozen=True)
class CombinedNPConfig:
    dt: float = 0.02
    steps_per_day: int = 1200

    # Reactor volumes
    v_anaerobic: float = 500.0
    v_anoxic: float = 1000.0
    v_aerobic: float = 2000.0

    # Aeration range; calibrated for the benchmark so full aeration over-supplies
    # oxygen (DO ~3 mg/L) rather than only just nitrifying
    kla_max: float = 15.0
    s_o_sat: float = 8.0

    # Flow
    mean_flow: float = 500.0
    diurnal_amplitude: float = 100.0
    min_flow: float = 300.0
    max_flow: float = 700.0
    drift_scale: float = 0.1

    # Influent NH4 and PO4 concentrations rise and fall with the diurnal flow (load
    # peaks with flow): relative swing per unit relative flow excursion. Assumed
    # for the benchmark so the flow observation carries load information.
    load_swing: float = 0.5

    # Recycle: internal nitrate recycle (aerobic → anoxic, action-controlled)
    # plus a fixed external RAS + wastage loop through an ideal clarifier
    # (aerobic → anaerobic). Without wastage, SRT = HRT (~7 h) and
    # PAOs/nitrifiers wash out well before they can establish; the waste
    # ratio sets SRT ≈ 12 d given the reactor volumes and mean_flow.
    # The clarifier is asm2d.perfect_clarifier (solids-free overflow, RAS and waste underflows).
    q_recycle_max: float = 1500.0
    return_sludge_ratio: float = 1.0
    waste_sludge_ratio: float = 0.0123

    # Chemical P (FeCl₃)
    # Sized to meet the 1 mg P/L effluent limit, not to capture all influent P:
    # 6 mg P/L x 1.8 mol Fe/mol P needs ~20 mg Fe/L stoichiometrically, but the
    # precipitation rate also carries the Monod factor P/(P + k_half), so 20 mg
    # Fe/L still leaves ~0.3 mg P/L at the default k_half.
    fe_dose_max: float = 20.0
    precip: PrecipitationParams = PrecipitationParams()

    # External carbon
    carbon_dose_max: float = 50.0

    # ASM2d
    asm2d: ASM2dParams = ASM2dParams()

    # Sensors
    po4_noise_std: float = 0.1
    nh4_noise_std: float = 0.3
    no3_noise_std: float = 0.3
    sensor_lag: float = 0.7
    sensor_sample_period: int = 5

    # Reward in euros per step. Prices and fines are benchmark utility
    # assumptions, not statutory charges; compliance limits are calibrated
    # component limits (NH4 and NO3 separately, not a total-nitrogen permit).
    price_kwh: float = 0.15  # EUR/kWh, assumed EU industrial tariff
    kwh_per_kg_o2: float = 1.0 / 1.8  # BSM1 aeration energy: 1.8 kg O2/kWh (Copp 2002); zero-DO proxy
    kwh_per_m3_recycle: float = 0.004  # BSM1 internal-recycle pumping energy (Copp 2002)
    price_fe_kg: float = 2.0  # EUR/kg Fe, assumed: 40% FeCl3 solution at ~0.30 EUR/kg
    price_cod_kg: float = 1.0  # EUR/kg COD, assumed external carbon (acetate/methanol class)
    fine_po4_kg: float = 50.0  # EUR/kg P exceeding the limit, calibrated so compliance beats paying
    fine_nh4_kg: float = 10.0  # EUR/kg N exceeding the limit, calibrated
    fine_no3_kg: float = 10.0  # EUR/kg N exceeding the limit, calibrated
    po4_limit: float = 1.0
    nh4_limit: float = 4.0
    no3_limit: float = 8.0


def make_combined_np_benchmark(config: CombinedNPConfig):
    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=0.0,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )
    po4_analyzer = LaggedSensorParams(
        noise_std=config.po4_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )
    nh4_analyzer = LaggedSensorParams(
        noise_std=config.nh4_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )
    no3_analyzer = LaggedSensorParams(
        noise_std=config.no3_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )

    q_max = jnp.array(config.q_recycle_max)
    kla_max = jnp.array(config.kla_max)
    fe_max = jnp.array(config.fe_dose_max)
    carbon_max = jnp.array(config.carbon_dose_max)
    mean_flow = jnp.array(config.mean_flow)
    influent = make_default_influent_asm2d()
    zero_kla = jnp.array(0.0)
    initial_po4 = influent[13]

    def _observe(
        sensed_po4: jax.Array,
        sensed_nh4: jax.Array,
        sensed_no3: jax.Array,
        sensed_ana_po4: jax.Array,
        aerobic_do: jax.Array,
        pha_pao_ratio: jax.Array,
        q_in: jax.Array,
        fe_dose: jax.Array,
        carbon_dose: jax.Array,
    ):
        return jnp.array(
            [
                sensed_po4 / 10.0,
                sensed_nh4 / 35.0,
                sensed_no3 / 20.0,
                sensed_ana_po4 / 30.0,
                aerobic_do / 8.0,
                pha_pao_ratio,
                q_in / mean_flow,
                fe_dose / fe_max,
                carbon_dose / carbon_max,
            ]
        )

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)
        src = source_reset(k1)

        init_state = influent.at[4].set(2500.0).at[5].set(150.0)
        init_state = init_state.at[14].set(500.0).at[15].set(100.0).at[16].set(100.0)

        state = CombinedNPState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            anaerobic_state=init_state.at[7].set(0.0),
            anoxic_state=init_state.at[7].set(0.0),
            aerobic_state=init_state.at[7].set(2.0),
            po4_eff_sensor=analyzer_reset(initial_po4, k2),
            nh4_eff_sensor=analyzer_reset(init_state[9], k3),
            no3_eff_sensor=analyzer_reset(init_state[8], k4),
            po4_ana_sensor=analyzer_reset(init_state[13], k5),
            last_q_in=jnp.array(config.mean_flow),
        )
        obs = _observe(
            state.po4_eff_sensor.held_value,
            state.nh4_eff_sensor.held_value,
            state.no3_eff_sensor.held_value,
            state.po4_ana_sensor.held_value,
            state.aerobic_state[7],
            state.aerobic_state[15] / (state.aerobic_state[14] + 1e-10),
            mean_flow,
            jnp.array(0.0),
            jnp.array(0.0),
        )
        return state, obs

    def step(state: CombinedNPState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        # 1. Flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Actions
        kla = jnp.clip(action[0], 0.0, 1.0) * kla_max
        q_recycle = jnp.clip(action[1], 0.0, 1.0) * q_max
        fe_dose = jnp.clip(action[2], 0.0, 1.0) * fe_max
        carbon_dose = jnp.clip(action[3], 0.0, 1.0) * carbon_max

        # 3. Clarifier (ideal): settles the *previous* step's aerobic-zone
        # liquor into a solids-free overflow and a concentrated RAS underflow
        # (one-step lag avoids an algebraic loop)
        q_ras = q_in * config.return_sludge_ratio
        q_waste = q_in * config.waste_sludge_ratio
        _, return_sludge = perfect_clarifier(state.aerobic_state, q_in + q_ras, q_ras, q_waste)

        # 4. Anaerobic zone: influent + carbon + RAS (A2O: no nitrate/DO here)
        load = 1.0 + config.load_swing * (q_in - config.mean_flow) / config.diurnal_amplitude
        loaded_influent = influent.at[9].set(influent[9] * load).at[13].set(influent[13] * load)
        inf_with_carbon = loaded_influent.at[1].set(loaded_influent[1] + carbon_dose)
        anaerobic_inlet = mix(inf_with_carbon, q_in, return_sludge, q_ras)
        new_anaerobic = reactor_step(
            state.anaerobic_state,
            anaerobic_inlet,
            q_in + q_ras,
            config.v_anaerobic,
            zero_kla,
            config.s_o_sat,
            config.asm2d,
            jnp.array(config.dt),
        )

        # 5. Anoxic zone: anaerobic effluent + internal nitrate recycle from
        # aerobic (A2O: the recycle carrying NO3/DO goes here, not anaerobic)
        anoxic_inlet = mix(new_anaerobic, q_in + q_ras, state.aerobic_state, q_recycle)
        new_anoxic = reactor_step(
            state.anoxic_state,
            anoxic_inlet,
            q_in + q_ras + q_recycle,
            config.v_anoxic,
            zero_kla,
            config.s_o_sat,
            config.asm2d,
            jnp.array(config.dt),
        )

        # 6. Aerobic zone (with variable aeration); carries the internal
        # recycle flow, which splits off again at this outlet
        new_aerobic = reactor_step(
            state.aerobic_state,
            new_anoxic,
            q_in + q_ras + q_recycle,
            config.v_aerobic,
            kla,
            config.s_o_sat,
            config.asm2d,
            jnp.array(config.dt),
        )

        # 7. Simultaneous precipitation in the aerobic zone. fe_dose is mg Fe per
        # L of influent; alkalinity is consumed by all dosed Fe (hydrolysis),
        # not only the Fe that captures P.
        fe_conc = fe_dose * q_in * config.dt / config.v_aerobic
        final_po4, fe_consumed = precipitate(new_aerobic[13], fe_conc, config.precip)
        alk_consumed = config.precip.alk_per_fe * fe_conc / config.precip.mw_fe
        p_captured_kg = (new_aerobic[13] - final_po4) * config.v_aerobic / 1000.0
        new_aerobic = new_aerobic.at[13].set(final_po4)
        # Non-negative concentration invariant, as in reactor_step (asm2d.py)
        new_aerobic = new_aerobic.at[12].set(jnp.maximum(new_aerobic[12] - alk_consumed, 0.0))

        # 8. Effluent quality
        eff_nh4 = new_aerobic[9]
        eff_no3 = new_aerobic[8]
        ana_po4 = new_anaerobic[13]
        aerobic_do = new_aerobic[7]
        pha_pao_ratio = new_aerobic[15] / (new_aerobic[14] + 1e-10)

        # 9. Sensors
        new_po4_s, sensed_po4 = analyzer_step(state.po4_eff_sensor, final_po4, po4_analyzer, k2)
        new_nh4_s, sensed_nh4 = analyzer_step(state.nh4_eff_sensor, eff_nh4, nh4_analyzer, k3)
        new_no3_s, sensed_no3 = analyzer_step(state.no3_eff_sensor, eff_no3, no3_analyzer, k4)
        new_po4a_s, sensed_ana_po4 = analyzer_step(state.po4_ana_sensor, ana_po4, po4_analyzer, k5)

        # 10. Observation
        obs = _observe(sensed_po4, sensed_nh4, sensed_no3, sensed_ana_po4, aerobic_do, pha_pao_ratio, q_in, fe_dose, carbon_dose)

        # 11. Reward: euros per step, fines on true effluent exceedance load
        vol_kg = q_in * config.dt / 1000.0  # m3 x mg/L -> kg
        aeration_kwh = config.s_o_sat * config.kwh_per_kg_o2 * config.v_aerobic * kla * config.dt / 1000.0
        pumping_kwh = config.kwh_per_m3_recycle * q_recycle * config.dt  # m3/h x h = m3
        cost = (
            config.price_kwh * (aeration_kwh + pumping_kwh)
            + config.price_fe_kg * fe_dose * vol_kg
            + config.price_cod_kg * carbon_dose * vol_kg
        )
        fines = vol_kg * (
            config.fine_po4_kg * jnp.maximum(final_po4 - config.po4_limit, 0.0)
            + config.fine_nh4_kg * jnp.maximum(eff_nh4 - config.nh4_limit, 0.0)
            + config.fine_no3_kg * jnp.maximum(eff_no3 - config.no3_limit, 0.0)
        )
        reward = -(cost + fines)

        new_state = CombinedNPState(
            step_count=state.step_count + 1,
            source_state=new_source,
            anaerobic_state=new_anaerobic,
            anoxic_state=new_anoxic,
            aerobic_state=new_aerobic,
            po4_eff_sensor=new_po4_s,
            nh4_eff_sensor=new_nh4_s,
            no3_eff_sensor=new_no3_s,
            po4_ana_sensor=new_po4a_s,
            last_q_in=q_in,
        )
        info: dict[str, jax.Array] = {
            "po4_eff": final_po4,
            "nh4_eff": eff_nh4,
            "no3_eff": eff_no3,
            "po4_anaerobic": ana_po4,
            "do_aerobic": aerobic_do,
            "x_pao": new_aerobic[14],
            "x_pha": new_aerobic[15],
            "x_pp": new_aerobic[16],
            "fe_consumed": fe_consumed,
            "p_captured_kg": p_captured_kg,
            "q_in": q_in,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
