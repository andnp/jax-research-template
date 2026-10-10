"""Drinking water treatment train benchmark.

Multi-stage cascade: coagulation → membrane filtration → chlorine disinfection.
Upstream decisions affect downstream performance:
  - Underdosing coagulant → higher particle load on membrane → faster fouling
  - Underdosing coagulant → less DOC/NOM removed → UF passes DOC through
    unchanged (it only rejects particulates) → more chlorine demand
  - Overdosing coagulant → wasted chemical, possible re-stabilisation
  - Backwash timing affects throughput and downstream chlorine demand

Action (3D): [coag_dose / dose_max, backwash_trigger (>0.5 = wash), Cl2_dose / Cl2_max]
Observation (8D):
  raw_turbidity / 50
  post_coag_tss / 100
  TMP / max_TMP
  permeate_flow / max_flow
  Cl2_residual / 5
  hours_since_backwash
  coag_dose / dose_max
  flow / mean_flow
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.coagulation import CoagulationParams, coagulate
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as analyzer_reset
from process_control.sensors.lagged import step as analyzer_step
from process_control.units.membrane import MembraneParams, MembraneState, compute_tmp
from process_control.units.membrane import reset as membrane_reset
from process_control.units.membrane import step as membrane_step


@jax_dataclass
class DrinkingWaterTrainState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    turbidity_drift: jax.Array
    membrane_state: MembraneState
    cl2_residual: jax.Array  # chlorine residual at basin outlet (mg/L)
    tmp_sensor: LaggedSensorState
    cl2_sensor: LaggedSensorState
    last_q_in: jax.Array


@dataclass(frozen=True)
class DrinkingWaterTrainConfig:
    dt: float = 0.01
    steps_per_day: int = 2400

    # Raw water source
    mean_flow: float = 200.0  # m³/h
    diurnal_amplitude: float = 40.0
    min_flow: float = 120.0
    max_flow: float = 300.0
    drift_scale: float = 0.2

    # Raw water turbidity
    turb_mean: float = 10.0  # NTU
    turb_amplitude: float = 5.0
    turb_min: float = 2.0
    turb_max: float = 50.0  # storm events
    turb_drift_scale: float = 0.4

    # Coagulation
    coag: CoagulationParams = CoagulationParams()
    coag_dose_max: float = 80.0  # mg/L

    # Membrane
    membrane: MembraneParams = MembraneParams(
        # Assumed benchmark sizing (not from a source): rated flux flux_max =
        # 0.06 m/h over this area passes max_flow, and mean_flow gives 40 LMH,
        # a plausible UF design flux; allowable flux is site/pilot specific
        # (EPA Membrane Filtration Guidance Manual, 2005). Diurnal flow
        # spans 24-60 LMH, crossing flux_critical (30 LMH).
        area=5000.0,
        # Coagulated floc particles are much larger than raw colloidal TSS,
        # which lowers specific cake resistance (Carman-Kozeny: alpha ~ 1/r²);
        # reduced-order tuning, not calibrated to a specific measured cake.
        # 5e15 sits inside the 1e13-1e16 m/kg range cited in MembraneParams and
        # makes backwash necessary within about a day (assumed for the benchmark).
        alpha_cake=5e15,
    )
    flux_max: float = 0.06  # m/h

    # Chlorine disinfection (simplified contact basin)
    cl2_dose_max: float = 5.0  # mg/L
    cl2_decay_rate: float = 0.3  # 1/h (first-order in-basin decay)
    # HRT = volume / post-membrane flow (not a fixed constant). Assumed sizing:
    # 200 m³ gives 0.67-1.67 h over the 120-300 m³/h feed range, inside the
    # 0.5-2 h range typical of chlorine contact design.
    contact_basin_volume: float = 200.0  # m³
    cl2_demand_per_tss: float = 0.02  # mg Cl2 / mg/L TSS demand

    # DOC/NOM: raw DOC in the 2-10 mg/L range is typical for surface water
    # (USGS/AWWA source-water surveys); coagulation removes a fraction of it
    # per Edwards, JAWWA 89:5:78, 1997, saturating with dose. UF passes DOC
    # unchanged (it only rejects particulates), so it reaches the chlorine
    # stage regardless of membrane performance.
    raw_doc: float = 4.0  # mg/L at mean turbidity
    # Storm runoff raises DOC together with turbidity. ASSUMED linear,
    # reduced-order coupling for the benchmark; plausible range ~0.1-0.5
    # mg/L per NTU. Result is clipped to the 2-10 mg/L source-water range.
    doc_per_ntu: float = 0.25  # mg/L DOC per NTU above/below turb_mean
    doc_removal_max: float = 0.4  # fractional DOC removal at saturating dose
    doc_removal_half_dose: float = 15.0  # mg/L coag dose at half-max removal
    # ASSUMED reduced-order value (the former 0.8 was equally unsourced);
    # plausible range ~0.2-0.8. 0.4 keeps the needed dose inside 0-cl2_dose_max.
    cl2_demand_per_doc: float = 0.4  # mg Cl2 demand / mg DOC

    # Sensors
    tmp_noise_std: float = 500.0
    tmp_lag: float = 0.9
    cl2_noise_std: float = 0.05
    cl2_lag: float = 0.8
    sensor_sample_period: int = 5

    # Reward
    reward_w_cl2_low: float = 2.0  # penalty for residual < target
    reward_w_cl2_high: float = 0.5  # penalty for residual > ceiling
    # Loss per step once TMP reaches max_tmp (the damage limit); ramps up
    # quadratically from tmp_alarm and is capped there. Assumed benchmark value.
    reward_w_tmp: float = 5.0
    reward_w_coag_cost: float = 0.05
    reward_w_cl2_cost: float = 0.1
    reward_w_throughput: float = 0.3
    cl2_target: float = 0.5  # mg/L minimum residual
    cl2_ceiling: float = 2.0  # mg/L max residual (taste/DBP)
    tmp_alarm: float = 1.5e5  # Pa


def make_drinking_water_train_benchmark(config: DrinkingWaterTrainConfig):
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
    turb_channel = ChannelParams(
        mean=config.turb_mean,
        amplitude=config.turb_amplitude,
        min_value=config.turb_min,
        max_value=config.turb_max,
        drift_scale=config.turb_drift_scale,
        drift_clip=config.turb_amplitude * 2.0,
    )
    tmp_analyzer = LaggedSensorParams(
        noise_std=config.tmp_noise_std,
        lag_coefficient=config.tmp_lag,
        sample_period=config.sensor_sample_period,
    )
    cl2_analyzer = LaggedSensorParams(
        noise_std=config.cl2_noise_std,
        lag_coefficient=config.cl2_lag,
        sample_period=config.sensor_sample_period,
    )

    dt = jnp.array(config.dt)
    dose_max = jnp.array(config.coag_dose_max)
    cl2_max = jnp.array(config.cl2_dose_max)
    flux_max = jnp.array(config.flux_max)
    # NTU->TSS is water-specific; typical reported ratios run ~1-2.5 mg/L per
    # NTU. 2.5 sits at the high end (unverified for this benchmark's water).
    raw_tss_per_turbidity = jnp.array(2.5)
    basin_volume = jnp.array(config.contact_basin_volume)
    raw_doc = jnp.array(config.raw_doc)
    max_tmp = jnp.array(config.membrane.max_tmp)
    mean_flow = jnp.array(config.mean_flow)
    area = jnp.array(config.membrane.area)
    operating_flux = mean_flow / area
    # Wash water is drawn from product at bw_flux_ratio times the operating
    # flux in force when the wash starts, for the wash duration.
    wash_volume_per_flux = config.membrane.bw_flux_ratio * area * config.membrane.bw_duration

    def _observe(
        raw_turbidity: jax.Array,
        post_coag_tss: jax.Array,
        sensed_tmp: jax.Array,
        q_permeate: jax.Array,
        sensed_cl2: jax.Array,
        hours_since_bw: jax.Array,
        coag_dose: jax.Array,
        q_in: jax.Array,
    ):
        return jnp.array(
            [
                raw_turbidity / 50.0,
                post_coag_tss / 100.0,
                sensed_tmp / max_tmp,
                q_permeate / (flux_max * area),
                sensed_cl2 / 5.0,
                hours_since_bw,
                coag_dose / dose_max,
                q_in / mean_flow,
            ]
        )

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)
        src = source_reset(k1)
        mem = membrane_reset(config.membrane, k2)
        initial_cl2_residual = jnp.array(1.0)
        state = DrinkingWaterTrainState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            turbidity_drift=jnp.array(0.0),
            membrane_state=mem,
            cl2_residual=initial_cl2_residual,
            tmp_sensor=analyzer_reset(compute_tmp(operating_flux / 3600.0, mem, config.membrane), k3),
            cl2_sensor=analyzer_reset(initial_cl2_residual, k4),
            last_q_in=jnp.array(config.mean_flow),
        )
        raw_turbidity = jnp.array(source_params.base_turbidity)
        post_coag_tss, _ = coagulate(raw_turbidity * raw_tss_per_turbidity, raw_turbidity, jnp.array(0.0), config.coag)
        obs = _observe(
            raw_turbidity,
            post_coag_tss,
            state.tmp_sensor.held_value,
            operating_flux * area,
            state.cl2_sensor.held_value,
            mem.hours_since_bw,
            jnp.array(0.0),
            mean_flow,
        )
        return state, obs

    def step(state: DrinkingWaterTrainState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Raw water flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Raw water turbidity
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        turb_signal = jnp.sin(2.0 * jnp.pi * t)
        new_turb_drift, raw_turbidity = channel_step(
            state.turbidity_drift,
            turb_signal,
            turb_channel,
            k2,
        )

        # 3. Coagulation stage
        coag_dose = jnp.clip(action[0], 0.0, 1.0) * dose_max
        # Approximate TSS from turbidity (rough NTU → mg/L)
        raw_tss = raw_turbidity * raw_tss_per_turbidity
        post_coag_tss, coag_eta = coagulate(raw_tss, raw_turbidity, coag_dose, config.coag)

        # 4. Membrane filtration
        # Production follows raw feed up to the rated flux; air scour and flux
        # are not actions in this benchmark.
        flux = jnp.minimum(q_in / area, flux_max)
        do_backwash = action[1]
        # Air scour at constant 50% (could be action)
        air_scour = jnp.array(0.5)

        new_mem, tmp, permeate_tss, q_permeate = membrane_step(
            state.membrane_state,
            post_coag_tss,
            flux,
            air_scour,
            do_backwash,
            config.membrane,
            dt,
        )

        # 5. Chlorine disinfection
        cl2_dose = jnp.clip(action[2], 0.0, 1.0) * cl2_max
        # Coagulation removes a saturating fraction of DOC with dose (Edwards
        # 1997); UF passes whatever DOC remains straight through, unlike TSS.
        doc_removal = config.doc_removal_max * coag_dose / (coag_dose + config.doc_removal_half_dose)
        raw_doc_now = jnp.clip(raw_doc + config.doc_per_ntu * (raw_turbidity - config.turb_mean), 2.0, 10.0)
        permeate_doc = raw_doc_now * (1.0 - doc_removal)
        # Chlorine demand depends on permeate quality: residual particulate
        # (TSS, now tiny post-membrane) plus dissolved organics (DOC, which
        # the membrane does not reject).
        cl2_demand = permeate_tss * config.cl2_demand_per_tss + permeate_doc * config.cl2_demand_per_doc
        # Contact time is basin HRT = volume / post-membrane flow, which dips
        # while a backwash interrupts production.
        contact_time = basin_volume / jnp.maximum(q_permeate, 1e-6)
        applied_cl2 = cl2_dose - cl2_demand
        new_cl2_residual = jnp.maximum(
            0.0,
            applied_cl2 * jnp.exp(-config.cl2_decay_rate * contact_time),
        )

        # 6. Sensors
        new_tmp_sensor, sensed_tmp = analyzer_step(state.tmp_sensor, tmp, tmp_analyzer, k3)
        new_cl2_sensor, sensed_cl2 = analyzer_step(state.cl2_sensor, new_cl2_residual, cl2_analyzer, k4)

        # 7. Observation
        obs = _observe(raw_turbidity, post_coag_tss, sensed_tmp, q_permeate, sensed_cl2, new_mem.hours_since_bw, coag_dose, q_in)

        # 8. Reward
        # Throughput is net product: gross permeate minus the wash water drawn
        # from it, charged once at the wash start.
        starts_bw = (
            (do_backwash > 0.5)
            & ~state.membrane_state.is_backwashing
            & ~state.membrane_state.backwash_trigger_latched
        )
        backwash_water = wash_volume_per_flux * flux * starts_bw
        q_net = q_permeate - backwash_water / dt
        # Scored on true plant state: analyzers only feed the observation, so a
        # lagged/held sensor cannot hide a bad action from the objective.
        cl2_low = jnp.maximum(config.cl2_target - new_cl2_residual, 0.0)
        cl2_high = jnp.maximum(new_cl2_residual - config.cl2_ceiling, 0.0)
        tmp_violation = jnp.clip((tmp - config.tmp_alarm) / (max_tmp - config.tmp_alarm), 0.0, 1.0)

        reward = (
            config.reward_w_throughput * (q_net / (flux_max * area))
            - config.reward_w_cl2_low * cl2_low**2
            - config.reward_w_cl2_high * cl2_high**2
            - config.reward_w_tmp * tmp_violation**2
            - config.reward_w_coag_cost * (coag_dose / dose_max)
            - config.reward_w_cl2_cost * (cl2_dose / cl2_max)
        )

        new_state = DrinkingWaterTrainState(
            step_count=state.step_count + 1,
            source_state=new_source,
            turbidity_drift=new_turb_drift,
            membrane_state=new_mem,
            cl2_residual=new_cl2_residual,
            tmp_sensor=new_tmp_sensor,
            cl2_sensor=new_cl2_sensor,
            last_q_in=q_in,
        )
        info: dict[str, jax.Array] = {
            "raw_turbidity": raw_turbidity,
            "post_coag_tss": post_coag_tss,
            "coag_eta": coag_eta,
            "tmp": tmp,
            "cl2_residual": new_cl2_residual,
            "q_in": q_in,
            "q_permeate": q_permeate,
            "q_net": q_net,
            "coag_dose": coag_dose,
            "cl2_dose": cl2_dose,
            "permeate_doc": permeate_doc,
            "contact_time": contact_time,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
