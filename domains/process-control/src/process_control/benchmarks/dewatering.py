"""Sludge dewatering control benchmark.

Thickened sludge arrives at a stochastic rate and concentration into a well-mixed
holding tank; the agent sets polymer dose and belt speed to draw the tank down
through the press. A slow belt lets the tank fill and overflow (hauled away as
liquid sludge); a fast belt or the wrong polymer dose sends solids back in the
filtrate.

Action (2D): [polymer_dose / dose_max, belt_speed (0-1)]
Observation (6D):
  cake_dryness (fraction, 0 when the belt is idle)
  filtrate_tss / 500 (analyzer, held while the belt is idle)
  throughput / max_throughput
  polymer_dose / dose_max
  tank_tss / 15000
  tank_level (fraction of tank_volume_max)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as analyzer_reset
from process_control.sensors.lagged import step as analyzer_step
from process_control.units.dewatering import DewateringParams, DewateringState
from process_control.units.dewatering import reset as dewatering_reset
from process_control.units.dewatering import step as dewatering_step


@jax_dataclass
class DewateringBenchState:
    step_count: jax.Array
    dewatering_state: DewateringState
    tss_drift: jax.Array
    q_drift: jax.Array
    filtrate_sensor: LaggedSensorState
    tank_volume: jax.Array  # m³
    tank_solids: jax.Array  # kg


@dataclass(frozen=True)
class DewateringConfig:
    dt: float = 0.05  # hours per step (~3 min)
    steps_per_day: int = 480

    # Dewatering unit
    dewatering: DewateringParams = DewateringParams()

    # Feed sludge TSS variation
    feed_tss_mean: float = 10000.0  # g/m³ (1% solids)
    feed_tss_amplitude: float = 3000.0
    feed_tss_min: float = 5000.0
    feed_tss_max: float = 18000.0
    feed_tss_drift_scale: float = 100.0  # g/m³ per step random walk (assumed)

    # Sludge inflow to the holding tank; all values assumed for the benchmark.
    q_in_mean: float = 22.0  # m³/h
    q_in_amplitude: float = 5.0
    q_in_min: float = 3.0
    q_in_max: float = 45.0
    q_in_drift_scale: float = 0.5
    q_in_drift_clip: float = 8.0

    # Well-mixed holding tank (~2.7 h of mean inflow, assumed); overflow is hauled as liquid sludge.
    tank_volume_max: float = 60.0  # m³
    tank_init_fraction: float = 0.5

    # Polymer: mg/L dose that, at feed_tss_mean, gives ~15 kg/tonne DS — the
    # top of the centrifuge practice range (see units/dewatering.py).
    dose_max: float = 150.0  # mg/L max polymer dose

    # Sensor
    filtrate_noise_std: float = 10.0
    filtrate_lag: float = 0.6
    filtrate_sample_period: int = 3

    # Reward: operating cost in $/h plus a quadratic penalty on true filtrate above the limit.
    # Prices describe a nominal benchmark scenario (assumed, not site accounting data).
    polymer_cost: float = 5.0  # $/kg active polymer
    disposal_cost: float = 80.0  # $/wet tonne of cake
    haul_cost: float = 15.0  # $/m³ of overflowed liquid sludge
    recycle_cost: float = 1.0  # $/kg TSS returned to the plant in filtrate
    cost_scale: float = 50.0  # $/h normalising the cost term
    reward_w_filtrate: float = 1.0
    # ~95% capture at 1% feed solids, reachable at peak polymer dose (assumed)
    filtrate_limit: float = 500.0  # g/m³


def make_dewatering_benchmark(config: DewateringConfig):
    tss_channel_params = ChannelParams(
        mean=config.feed_tss_mean,
        amplitude=config.feed_tss_amplitude,
        min_value=config.feed_tss_min,
        max_value=config.feed_tss_max,
        drift_scale=config.feed_tss_drift_scale,
        drift_clip=config.feed_tss_amplitude,
    )
    inflow_channel_params = ChannelParams(
        mean=config.q_in_mean,
        amplitude=config.q_in_amplitude,
        min_value=config.q_in_min,
        max_value=config.q_in_max,
        drift_scale=config.q_in_drift_scale,
        drift_clip=config.q_in_drift_clip,
    )
    filtrate_analyzer_params = LaggedSensorParams(
        noise_std=config.filtrate_noise_std,
        lag_coefficient=config.filtrate_lag,
        sample_period=config.filtrate_sample_period,
    )

    dt = jnp.array(config.dt)
    dose_max = jnp.array(config.dose_max)
    tank_volume_max = jnp.array(config.tank_volume_max)
    filtrate_limit = jnp.array(config.filtrate_limit)

    def _select_if(condition: jax.Array):
        def select(new: jax.Array, old: jax.Array):
            return jnp.where(condition, new, old)

        return select

    def _observe(
        dryness: jax.Array,
        sensed_filtrate: jax.Array,
        throughput: jax.Array,
        polymer_dose: jax.Array,
        tank_tss: jax.Array,
        tank_volume: jax.Array,
    ):
        return jnp.array(
            [
                dryness,
                sensed_filtrate / 500.0,
                throughput / config.dewatering.max_throughput,
                polymer_dose / dose_max,
                tank_tss / 15000.0,
                tank_volume / tank_volume_max,
            ]
        )

    def reset(rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)
        initial_volume = config.tank_init_fraction * config.tank_volume_max
        state = DewateringBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            dewatering_state=dewatering_reset(config.dewatering, k1),
            tss_drift=jnp.array(0.0),
            q_drift=jnp.array(0.0),
            filtrate_sensor=analyzer_reset(0.0, k2),
            tank_volume=jnp.array(initial_volume),
            tank_solids=jnp.array(initial_volume * config.feed_tss_mean / 1e3),
        )
        obs = _observe(
            jnp.array(0.0),
            state.filtrate_sensor.held_value,
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(config.feed_tss_mean),
            state.tank_volume,
        )
        return state, obs

    def step(state: DewateringBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3 = jax.random.split(rng_key, 3)

        # 1. Inflow to the holding tank
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        new_tss_drift, inflow_tss = channel_step(state.tss_drift, jnp.sin(2.0 * jnp.pi * t), tss_channel_params, k1)
        new_q_drift, q_in = channel_step(state.q_drift, jnp.cos(2.0 * jnp.pi * t), inflow_channel_params, k3)
        volume = state.tank_volume + q_in * dt
        solids = state.tank_solids + inflow_tss * q_in * dt / 1e3  # kg
        tank_tss = 1e3 * solids / volume  # g/m³

        # 2. Actions
        polymer_dose = jnp.clip(action[0], 0.0, 1.0) * dose_max
        belt_speed = jnp.clip(action[1], 0.0, 1.0)

        # 3. The press can draw the whole tank down; the belt limits what it takes
        new_dw, dryness, filtrate_tss, q_filtrate, bypass_flow = dewatering_step(
            state.dewatering_state,
            tank_tss,
            volume / dt,
            polymer_dose,
            belt_speed,
            config.dewatering,
            dt,
        )
        throughput = volume / dt - bypass_flow
        running = throughput > 1e-6
        volume = volume - throughput * dt
        solids = solids - tank_tss * throughput * dt / 1e3

        # 4. Volume above the tank limit overflows at tank concentration
        overflow_volume = jnp.maximum(volume - tank_volume_max, 0.0)
        overflow_solids = tank_tss * overflow_volume / 1e3
        volume = volume - overflow_volume
        solids = solids - overflow_solids

        # 5. Filtrate analyzer samples only while filtrate is flowing
        new_filt_sensor, sensed_filtrate = analyzer_step(
            state.filtrate_sensor,
            filtrate_tss,
            filtrate_analyzer_params,
            k2,
        )
        new_filt_sensor = jax.tree.map(_select_if(running), new_filt_sensor, state.filtrate_sensor)
        sensed_filtrate = jnp.where(running, sensed_filtrate, state.filtrate_sensor.held_value)

        # 6. Observation
        obs = _observe(dryness, sensed_filtrate, throughput, polymer_dose, tank_tss, volume)

        # 7. Reward: the quality penalty uses the true filtrate, not the analyzer reading
        polymer_kg_h = polymer_dose * throughput / 1e3
        cake_solids = new_dw.cake_produced - state.dewatering_state.cake_produced
        wet_cake_tonnes_h = jnp.where(running, cake_solids / dt / (dryness + 1e-10) / 1e3, 0.0)
        cost = (
            config.polymer_cost * polymer_kg_h
            + config.disposal_cost * wet_cake_tonnes_h
            + config.haul_cost * overflow_volume / dt
            + config.recycle_cost * filtrate_tss * q_filtrate / 1e3
        )
        filtrate_violation = jnp.maximum(filtrate_tss - filtrate_limit, 0.0)
        reward = -(cost / config.cost_scale + config.reward_w_filtrate * (filtrate_violation / filtrate_limit) ** 2)

        new_state = DewateringBenchState(
            step_count=state.step_count + 1,
            dewatering_state=new_dw,
            tss_drift=new_tss_drift,
            q_drift=new_q_drift,
            filtrate_sensor=new_filt_sensor,
            tank_volume=volume,
            tank_solids=solids,
        )
        info: dict[str, jax.Array] = {
            "dryness": dryness,
            "filtrate_tss": filtrate_tss,
            "polymer_dose": polymer_dose,
            "tank_tss": tank_tss,
            "tank_level": volume / tank_volume_max,
            "throughput": throughput,
            "q_filtrate": q_filtrate,
            "inflow": q_in,
            "inflow_tss": inflow_tss,
            "overflow_volume": overflow_volume,
            "overflow_solids": overflow_solids,
            "cake_solids": cake_solids,
            "cost": cost,
            "filtrate_solids": filtrate_tss * q_filtrate * dt / 1e3,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
