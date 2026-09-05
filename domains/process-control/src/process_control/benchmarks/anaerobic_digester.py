"""Anaerobic digester control benchmark.

The agent controls feed rate and temperature setpoint to optimise
biogas production while preventing digester upset (VFA accumulation,
pH drop, biomass washout).

Action (2D): [Q_feed / Q_max, T_setpoint / 45]
Observation (7D):
  biogas_flow / 2000
  ch4_fraction
  VFA / 1000
  pH / 9
  temperature / 45
  feed_cod / 50000
  biomass / 5000
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.anaerobic_digester import ADM1Params, ADM1State
from process_control.units.anaerobic_digester import reset as digester_reset
from process_control.units.anaerobic_digester import step as digester_step


@jax_dataclass
class AnaerobicDigesterBenchState:
    step_count: jax.Array
    digester_state: ADM1State
    cod_drift: jax.Array
    vfa_sensor: ResidualAnalyzerState
    ph_sensor: ResidualAnalyzerState
    biogas_sensor: ResidualAnalyzerState
    temperature: jax.Array
    last_q_feed: jax.Array


@dataclass(frozen=True)
class AnaerobicDigesterConfig:
    dt: float = 0.04167  # days (~1 hour per step)
    steps_per_day: int = 24

    # Digester
    digester: ADM1Params = ADM1Params()

    # Feed COD variation
    feed_cod_mean: float = 30000.0  # mg/L (3% TS sludge)
    feed_cod_amplitude: float = 8000.0
    feed_cod_min: float = 15000.0
    feed_cod_max: float = 50000.0
    feed_cod_drift_scale: float = 0.2

    # Feed flow range
    q_feed_max: float = 200.0  # m³/d
    q_feed_min: float = 50.0

    # Temperature range
    t_min: float = 30.0
    t_max: float = 40.0
    t_ramp_rate: float = 2.0  # °C/d max temperature change rate

    # Sensors
    vfa_noise_std: float = 20.0
    vfa_lag: float = 0.8
    ph_noise_std: float = 0.05
    ph_lag: float = 0.9
    biogas_noise_std: float = 50.0
    biogas_lag: float = 0.5
    sensor_sample_period: int = 1

    # Reward
    reward_w_biogas: float = 1.0  # reward for biogas production
    reward_w_ch4: float = 0.5  # reward for high CH₄ content
    reward_w_vfa: float = 2.0  # penalty for VFA > threshold
    reward_w_ph: float = 3.0  # penalty for pH outside safe range
    reward_w_heat: float = 0.05  # penalty for heating energy
    vfa_limit: float = 500.0  # mg-COD/L — alarm threshold
    ph_safe_low: float = 6.5
    ph_safe_high: float = 7.8


def make_anaerobic_digester_benchmark(config: AnaerobicDigesterConfig):
    cod_channel_params = ChannelParams(
        mean=config.feed_cod_mean,
        amplitude=config.feed_cod_amplitude,
        min_value=config.feed_cod_min,
        max_value=config.feed_cod_max,
        drift_scale=config.feed_cod_drift_scale,
        drift_clip=config.feed_cod_amplitude,
    )
    vfa_analyzer = ResidualAnalyzerParams(
        noise_std=config.vfa_noise_std,
        lag_coefficient=config.vfa_lag,
        sample_period=config.sensor_sample_period,
    )
    ph_analyzer = ResidualAnalyzerParams(
        noise_std=config.ph_noise_std,
        lag_coefficient=config.ph_lag,
        sample_period=config.sensor_sample_period,
    )
    biogas_analyzer = ResidualAnalyzerParams(
        noise_std=config.biogas_noise_std,
        lag_coefficient=config.biogas_lag,
        sample_period=config.sensor_sample_period,
    )

    dt = jnp.array(config.dt)
    q_max = jnp.array(config.q_feed_max)
    q_min = jnp.array(config.q_feed_min)
    t_max = jnp.array(config.t_max)
    t_min = jnp.array(config.t_min)
    vfa_limit = jnp.array(config.vfa_limit)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)
        dig = digester_reset(config.feed_cod_mean, config.digester, k1)
        state = AnaerobicDigesterBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            digester_state=dig,
            cod_drift=jnp.array(0.0),
            vfa_sensor=analyzer_reset(k2),
            ph_sensor=analyzer_reset(k3),
            biogas_sensor=analyzer_reset(k4),
            temperature=jnp.array(35.0),
            last_q_feed=jnp.array(config.q_feed_max * 0.5),
        )
        obs = jnp.array(
            [
                1000.0 / 2000.0,  # initial biogas
                0.65,  # CH₄ fraction
                200.0 / 1000.0,  # VFA
                7.2 / 9.0,  # pH
                35.0 / 45.0,  # temperature
                config.feed_cod_mean / 50000.0,
                1500.0 / 5000.0,  # biomass
            ]
        )
        return state, obs

    def step(state: AnaerobicDigesterBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Feed COD (slow variation, ~daily timescale)
        t_frac = (state.step_count % config.steps_per_day) / config.steps_per_day
        cod_signal = jnp.sin(2.0 * jnp.pi * t_frac)
        new_cod_drift, feed_cod = channel_step(state.cod_drift, cod_signal, cod_channel_params, k1)

        # 2. Actions
        q_feed = q_min + jnp.clip(action[0], 0.0, 1.0) * (q_max - q_min)
        t_setpoint = t_min + jnp.clip(action[1], 0.0, 1.0) * (t_max - t_min)

        # Temperature ramp limiting
        max_dt_change = config.t_ramp_rate * config.dt
        new_temp = state.temperature + jnp.clip(
            t_setpoint - state.temperature,
            -max_dt_change,
            max_dt_change,
        )

        # 3. Digester step
        new_dig, q_biogas, ch4_frac, ph = digester_step(
            state.digester_state,
            feed_cod,
            q_feed,
            new_temp,
            config.digester,
            dt,
        )

        # 4. Sensors
        new_vfa_s, sensed_vfa = analyzer_step(state.vfa_sensor, new_dig.s_vfa, vfa_analyzer, k2)
        new_ph_s, sensed_ph = analyzer_step(state.ph_sensor, ph, ph_analyzer, k3)
        new_bg_s, sensed_biogas = analyzer_step(state.biogas_sensor, q_biogas, biogas_analyzer, k4)

        # 5. Observation
        obs = jnp.array(
            [
                sensed_biogas / 2000.0,
                ch4_frac,
                sensed_vfa / 1000.0,
                sensed_ph / 9.0,
                new_temp / 45.0,
                feed_cod / 50000.0,
                new_dig.x_biomass / 5000.0,
            ]
        )

        # 6. Reward
        vfa_violation = jnp.maximum(sensed_vfa - vfa_limit, 0.0)
        ph_low_viol = jnp.maximum(config.ph_safe_low - sensed_ph, 0.0)
        ph_high_viol = jnp.maximum(sensed_ph - config.ph_safe_high, 0.0)
        heating_cost = jnp.abs(new_temp - 20.0) / 25.0  # normalised heating above ambient

        reward = (
            config.reward_w_biogas * (sensed_biogas / 2000.0)
            + config.reward_w_ch4 * ch4_frac
            - config.reward_w_vfa * (vfa_violation / vfa_limit) ** 2
            - config.reward_w_ph * (ph_low_viol + ph_high_viol) ** 2
            - config.reward_w_heat * heating_cost
        )

        new_state = AnaerobicDigesterBenchState(
            step_count=state.step_count + 1,
            digester_state=new_dig,
            cod_drift=new_cod_drift,
            vfa_sensor=new_vfa_s,
            ph_sensor=new_ph_s,
            biogas_sensor=new_bg_s,
            temperature=new_temp,
            last_q_feed=q_feed,
        )
        info: dict[str, jax.Array] = {
            "q_biogas": q_biogas,
            "ch4_frac": ch4_frac,
            "ph": ph,
            "vfa": new_dig.s_vfa,
            "biomass": new_dig.x_biomass,
            "feed_cod": feed_cod,
            "q_feed": q_feed,
            "temperature": new_temp,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
