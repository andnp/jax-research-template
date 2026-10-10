"""Membrane fouling control benchmark.

The agent controls membrane flux, air scour, and backwash timing to
maintain throughput while managing fouling.

Action (3D): [flux position in flux_min..flux_max, air_scour (0-1), backwash_trigger (>0.5 = wash)]
Observation (7D):
  sensed TMP / max_TMP
  permeate_flow / max_flow
  air_scour
  hours_since_backwash / bw_age_ref (capped at 1)
  feed_tss / feed_tss_max
  water demand / max_flow
  forced-CIP time left / cip_duration
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as analyzer_reset
from process_control.sensors.lagged import step as analyzer_step
from process_control.units.membrane import MembraneParams, MembraneState, compute_tmp
from process_control.units.membrane import reset as membrane_reset
from process_control.units.membrane import step as membrane_step


@jax_dataclass
class MembraneFoulingBenchState:
    step_count: jax.Array
    membrane_state: MembraneState
    tss_drift: jax.Array
    tmp_sensor: LaggedSensorState
    last_feed_tss: jax.Array
    cip_remaining: jax.Array  # h of forced cleaning left


@dataclass(frozen=True)
class MembraneFoulingConfig:
    dt: float = 0.01  # hours (~36 sec, needs fine resolution for backwash)
    steps_per_day: int = 2400

    # Membrane
    # Calibrated for the benchmark, not measured: alpha_cake sits inside the cited
    # 1e13-1e16 m/kg range and makes TMP rise on a 1-3 h backwash timescale;
    # irr_fraction is raised so irreversible fouling matters within days; the
    # wash is a 1-minute UF-style backwash. The unit's timed CIP is disabled; the
    # TMP trip below is the only CIP path.
    membrane: MembraneParams = MembraneParams(alpha_cake=5e14, irr_fraction=0.1, bw_duration=1.0 / 60.0, cip_interval=1e9)

    # Operating range
    flux_max: float = 0.05  # m/h (50 LMH — typical UF)
    flux_min: float = 0.01  # m/h

    # Feed TSS variation
    feed_tss_mean: float = 50.0  # g/m³ (typical MF/UF feed)
    feed_tss_amplitude: float = 20.0
    feed_tss_min: float = 10.0
    feed_tss_max: float = 150.0
    # Wider wander than the diurnal swing so no single backwash timer or scour
    # setting is best at every feed quality (assumed level).
    feed_tss_drift_scale: float = 1.0
    feed_tss_drift_clip: float = 40.0

    bw_age_ref: float = 6.0  # h; backwash-age observation saturates here

    # TMP sensor
    tmp_noise_std: float = 500.0  # Pa
    tmp_lag: float = 0.9
    tmp_sample_period: int = 5

    # Diurnal water demand as a fraction of full capacity (flux_max * area); the
    # trough is at step 0. Assumed levels: peak demand exceeds the critical flux.
    demand_mean: float = 0.65
    demand_amp: float = 0.2

    # Reward
    reward_w_tmp: float = 1.0  # penalty for high TMP

    # Operating economics. Nominal assumptions for the benchmark, not sourced
    # from a specific plant; scour energy is the main trade-off knob.
    water_value: float = 0.5  # $/m3 permeate; also the reward normaliser
    elec_price: float = 0.10  # $/kWh
    pump_eff: float = 0.7  # permeate pump wire-to-water efficiency
    scour_kw_per_m2: float = 0.02  # air scour power at full intensity (kW/m2)
    bw_water_price: float = 0.5  # $/m3 of permeate used as backwash water

    # Hard limit: when operating TMP reaches membrane.max_tmp the module trips to a
    # forced CIP with no production. Duration and chemical cost are assumed values.
    cip_duration: float = 4.0  # h offline
    cip_chem_cost: float = 5.0  # $ per forced CIP
    tmp_target: float = 0.8e5  # Pa — target TMP (0.8 bar)


def make_membrane_fouling_benchmark(config: MembraneFoulingConfig):
    tss_channel_params = ChannelParams(
        mean=config.feed_tss_mean,
        amplitude=config.feed_tss_amplitude,
        min_value=config.feed_tss_min,
        max_value=config.feed_tss_max,
        drift_scale=config.feed_tss_drift_scale,
        drift_clip=config.feed_tss_drift_clip,
    )
    tmp_analyzer_params = LaggedSensorParams(
        noise_std=config.tmp_noise_std,
        lag_coefficient=config.tmp_lag,
        sample_period=config.tmp_sample_period,
    )

    dt = jnp.array(config.dt)
    flux_max = jnp.array(config.flux_max)
    flux_min = jnp.array(config.flux_min)
    max_tmp = jnp.array(config.membrane.max_tmp)
    tmp_target = jnp.array(config.tmp_target)
    q_cap = config.flux_max * config.membrane.area
    # Reward is expressed per step of full-capacity production value.
    ref_value = config.water_value * q_cap * config.dt
    # Nominal wash flux is a multiple of the maximum operating flux.
    backwash_water_m3 = config.membrane.bw_flux_ratio * q_cap * config.membrane.bw_duration

    def demand_at(step_count: jax.Array):
        phase = (step_count % config.steps_per_day) / config.steps_per_day
        return q_cap * (config.demand_mean + config.demand_amp * jnp.sin(2.0 * jnp.pi * phase - jnp.pi / 2.0))

    def _observe(
        sensed_tmp: jax.Array,
        q_permeate: jax.Array,
        air_scour: jax.Array,
        hours_since_bw: jax.Array,
        feed_tss: jax.Array,
        demand: jax.Array,
        cip_remaining: jax.Array,
    ):
        return jnp.array(
            [
                sensed_tmp / max_tmp,
                q_permeate / q_cap,
                air_scour,
                jnp.minimum(hours_since_bw / config.bw_age_ref, 1.0),
                feed_tss / config.feed_tss_max,
                demand / q_cap,
                jnp.minimum(cip_remaining / config.cip_duration, 1.0),
            ]
        )

    def reset(rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)
        mem = membrane_reset(config.membrane, k1)
        tmp_sensor = analyzer_reset(compute_tmp(jnp.array(0.0), mem, config.membrane), k2)
        state = MembraneFoulingBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            membrane_state=mem,
            tss_drift=jnp.array(0.0),
            tmp_sensor=tmp_sensor,
            last_feed_tss=jnp.array(config.feed_tss_mean),
            cip_remaining=jnp.array(0.0),
        )
        obs = _observe(
            tmp_sensor.held_value,
            jnp.array(0.0),
            jnp.array(0.0),
            mem.hours_since_bw,
            jnp.array(config.feed_tss_mean),
            demand_at(jnp.array(0)),
            jnp.array(0.0),
        )
        return state, obs

    def step(state: MembraneFoulingBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)

        # 1. Feed TSS
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        tss_signal = jnp.sin(2.0 * jnp.pi * t)
        new_tss_drift, feed_tss = channel_step(state.tss_drift, tss_signal, tss_channel_params, k1)

        # 2. Actions
        offline = state.cip_remaining > 0.0
        flux = jnp.where(offline, 0.0, flux_min + jnp.clip(action[0], 0.0, 1.0) * (flux_max - flux_min))
        air_scour = jnp.where(offline, 0.0, jnp.clip(action[1], 0.0, 1.0))
        do_backwash = jnp.where(offline, 0.0, action[2])

        # 3. Membrane step
        new_mem, tmp, permeate_tss, q_permeate = membrane_step(
            state.membrane_state,
            feed_tss,
            flux,
            air_scour,
            do_backwash,
            config.membrane,
            dt,
        )

        # Forced CIP: true operating TMP at the limit, whatever the analyzer reads.
        tripped = tmp >= max_tmp
        new_mem = replace(
            new_mem,
            r_reversible=jnp.where(tripped, 0.0, new_mem.r_reversible),
            r_irreversible=jnp.where(
                tripped, new_mem.r_irreversible * (1.0 - config.membrane.cip_recovery), new_mem.r_irreversible
            ),
        )
        cip_remaining = jnp.where(tripped, config.cip_duration, jnp.maximum(state.cip_remaining - dt, 0.0))

        # 4. TMP sensor
        new_tmp_sensor, sensed_tmp = analyzer_step(state.tmp_sensor, tmp, tmp_analyzer_params, k2)

        # 5. Permeability ratio (clean vs current)
        r_total = config.membrane.r_membrane + new_mem.r_reversible + new_mem.r_irreversible
        perm_ratio = config.membrane.r_membrane / (r_total + 1e-10)

        # 6. Observation
        obs = _observe(sensed_tmp, q_permeate, air_scour, new_mem.hours_since_bw, feed_tss, demand_at(state.step_count + 1), cip_remaining)

        # 7. Reward: throughput benefit minus costs
        tmp_violation = jnp.maximum(tmp - tmp_target, 0.0)
        starts_bw = (
            (do_backwash > 0.5)
            & ~state.membrane_state.is_backwashing
            & ~state.membrane_state.backwash_trigger_latched
        )
        pump_energy_kwh = tmp * q_permeate * config.dt / config.pump_eff / 3.6e6  # Pa * m3 = J
        scour_energy_kwh = config.scour_kw_per_m2 * config.membrane.area * air_scour * config.dt
        backwash_water = backwash_water_m3 * starts_bw
        cost = config.elec_price * (pump_energy_kwh + scour_energy_kwh) + config.bw_water_price * backwash_water
        demand = demand_at(state.step_count)
        shortfall_cost = config.water_value * jnp.maximum(demand - q_permeate, 0.0) * config.dt
        cip_cost = config.cip_chem_cost * tripped
        tmp_penalty = config.reward_w_tmp * ref_value * (tmp_violation / (max_tmp - tmp_target)) ** 2
        reward = -(shortfall_cost + cost + cip_cost + tmp_penalty) / ref_value

        new_state = MembraneFoulingBenchState(
            step_count=state.step_count + 1,
            membrane_state=new_mem,
            tss_drift=new_tss_drift,
            tmp_sensor=new_tmp_sensor,
            last_feed_tss=feed_tss,
            cip_remaining=cip_remaining,
        )
        info: dict[str, jax.Array] = {
            "tmp": tmp,
            "perm_ratio": perm_ratio,
            "q_permeate": q_permeate,
            "demand": demand,
            "tripped": tripped,
            "cip_remaining": cip_remaining,
            "feed_tss": feed_tss,
            "permeate_tss": permeate_tss,
            "pump_energy_kwh": pump_energy_kwh,
            "scour_energy_kwh": scour_energy_kwh,
            "backwash_water_m3": backwash_water,
            "r_reversible": new_mem.r_reversible,
            "r_irreversible": new_mem.r_irreversible,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
