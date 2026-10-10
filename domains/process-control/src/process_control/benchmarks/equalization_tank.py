from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import reset as actuator_reset
from process_control.actuators.ramp_limited import step as actuator_step
from process_control.controllers.pi_controller import PIControllerParams, PIControllerState
from process_control.controllers.pi_controller import reset as pi_reset
from process_control.controllers.pi_controller import step as pi_step
from process_control.disturbances.schedule import (
    DisturbanceSchedule,
    add_event,
    apply_active,
    create_empty,
    validate_disturbance_events,
)
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as level_sensor_reset
from process_control.sensors.lagged import step as level_sensor_step
from process_control.units.tank import TankParams, TankState
from process_control.units.tank import reset as tank_reset
from process_control.units.tank import step_with_result as tank_step


@dataclass(frozen=True)
class EqualizationTankBenchmarkConfig:
    target_level_fraction: float = 0.5  # fraction of max_level; centre of the level band
    dt: float = 0.25  # time step (same units as flow / area)

    # Tank geometry
    tank_max_level: float = 5.0
    tank_min_level: float = 0.0
    # Assumed benchmark size, not a plant design: 60 m2 x 5 m = 300 m3, 16.7% of the
    # 1800 m3/day mean inflow (mean_flow 75 m3/h x 24 h). EPA's flow-equalization report
    # gives 10-20% of daily flow as a screening range for diurnal equalization basins
    # (https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=300007H3.TXT); a real design would
    # integrate an inflow mass curve.
    tank_area: float = 60.0  # level changes as flow / area * dt

    # Outlet pump (action = desired outlet flow set point)
    pump_max_flow: float = 150.0
    pump_min_flow: float = 0.0
    pump_max_ramp_rate: float = 100.0

    # PI controller (proportional-integral on level error → outlet flow).
    # Reverse-acting: raising the outlet flow *drains* the tank, so a
    # too-high level (positive measurement, negative error) must *increase*
    # output, and a too-low level must decrease it — hence negative gains.
    # Gains are per raw level unit and tuned (assumed, benchmark-specific) for averaging
    # level control: loose enough to buffer the diurnal wave, tight enough to hold the
    # sustained-rain stress case inside the level band.
    pi_kp: float = -20.0
    pi_ki: float = -0.1
    pi_ff: float = 75.0  # steady-state outlet ≈ mean inlet
    pi_output_min: float = 0.0
    pi_output_max: float = 150.0

    # Reward weights (all terms are dimensionless; assumed/calibrated for the benchmark).
    flow_rate_weight: float = 1.0  # outlet change per step / (max ramp * dt)
    flow_deviation_weight: float = 1.0  # outlet deviation from mean_flow / mean_flow
    level_band_weight: float = 20.0  # squared excursion outside the band, fraction of range
    level_band_low: float = 0.15  # band edges as fractions of the level range
    level_band_high: float = 0.85
    overflow_penalty: float = 10.0
    unmet_outlet_penalty: float = 10.0

    # Level sensor
    level_noise_std: float = 0.02
    level_lag_coefficient: float = 0.1

    # Inlet source (DiurnalSourceState)
    mean_flow: float = 75.0
    diurnal_amplitude: float = 20.0
    min_flow: float = 50.0
    max_flow: float = 100.0
    demand_offset: float = 0.0
    flow_demand_coefficient: float = 0.005
    demand_noise_std: float = 0.01
    drift_scale: float = 0.2
    steps_per_day: int = 96

    max_disturbance_events: int = 16
    disturbance_events: tuple[tuple[int, int, float, int], ...] = ()


@jax_dataclass
class EqualizationPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    tank_state: TankState
    outlet_pump_state: RampLimitedActuatorState
    level_sensor_state: LaggedSensorState
    pi_state: PIControllerState
    last_outlet_flow: jax.Array
    disturbance_schedule: DisturbanceSchedule


def _validate_reward_config(config: EqualizationTankBenchmarkConfig) -> None:
    weights = (
        config.flow_rate_weight,
        config.flow_deviation_weight,
        config.level_band_weight,
        config.overflow_penalty,
        config.unmet_outlet_penalty,
    )
    if min(weights) < 0.0:
        raise ValueError("reward weights must be non-negative")
    if not 0.0 <= config.level_band_low < config.level_band_high <= 1.0:
        raise ValueError("level band must satisfy 0 <= low < high <= 1")
    if config.pump_max_ramp_rate * config.dt <= 0.0 or config.mean_flow <= 0.0:
        raise ValueError("pump ramp rate, dt and mean_flow must be positive")


def make_equalization_tank_benchmark(
    config: EqualizationTankBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[EqualizationPlantState, jax.Array]],
    Callable[[EqualizationPlantState, jax.Array, jax.Array], tuple[EqualizationPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    _validate_reward_config(config)
    max_disturbance_events = validate_disturbance_events(
        config.max_disturbance_events,
        config.disturbance_events,
    )
    tank_params = TankParams(
        max_level=config.tank_max_level,
        min_level=config.tank_min_level,
        cross_section_area=config.tank_area,
    )
    pump_params = RampLimitedActuatorParams(
        max_output=config.pump_max_flow,
        min_output=config.pump_min_flow,
        max_ramp_rate=config.pump_max_ramp_rate,
    )
    pi_params = PIControllerParams(
        kp=config.pi_kp,
        ki=config.pi_ki,
        ff=config.pi_ff,
        output_min=config.pi_output_min,
        output_max=config.pi_output_max,
    )
    level_sensor_params = LaggedSensorParams(
        noise_std=config.level_noise_std,
        lag_coefficient=config.level_lag_coefficient,
    )
    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=config.demand_offset,
        flow_demand_coefficient=config.flow_demand_coefficient,
        demand_noise_std=config.demand_noise_std,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )
    dt = jnp.array(config.dt)
    target_level = jnp.array(config.target_level_fraction * config.tank_max_level)
    pump_max_flow = jnp.array(config.pump_max_flow)
    tank_max_level = jnp.array(config.tank_max_level)
    min_level = jnp.array(config.tank_min_level)
    rate_scale = jnp.array(config.pump_max_ramp_rate * config.dt)

    def reset(rng_key: jax.Array) -> tuple[EqualizationPlantState, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        tank_state = tank_reset(tank_params)
        pump_state = actuator_reset(k2)
        level_initial = tank_state.level
        ls_state = level_sensor_reset(level_initial, k3)
        pi_state = pi_reset(k4)
        disturbance_schedule = create_empty(max_disturbance_events)
        for start_step, end_step, magnitude, type_id in config.disturbance_events:
            disturbance_schedule = add_event(
                disturbance_schedule,
                start_step=start_step,
                end_step=end_step,
                magnitude=magnitude,
                type_id=type_id,
            )

        plant_state = EqualizationPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            tank_state=tank_state,
            outlet_pump_state=pump_state,
            level_sensor_state=ls_state,
            pi_state=pi_state,
            last_outlet_flow=jnp.array(0.0),
            disturbance_schedule=disturbance_schedule,
        )

        measured_level = level_initial
        obs = jnp.array(
            [
                measured_level / tank_max_level,
                jnp.array(0.0),
                jnp.array(config.mean_flow) / pump_max_flow,
                (target_level - measured_level) / tank_max_level,
            ]
        )
        return plant_state, obs

    def step(
        state: EqualizationPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[EqualizationPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k2, k3 = jax.random.split(rng_key, 3)

        # 1. Source generates inlet flow
        new_source_state, transport, inlet_flow, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 1.5 Apply active disturbances to transport (modifies inlet flow via hydraulics)
        transport = apply_active(state.disturbance_schedule, transport, state.step_count)
        inlet_flow = transport.hydraulics.flow

        # 2. Outlet pump realizes the desired outlet flow set point
        new_pump_state, requested_outlet = actuator_step(
            state.outlet_pump_state,
            action,
            pump_params,
            dt,
        )

        # 3. Tank level update
        tank_result = tank_step(
            state.tank_state,
            inlet_flow,
            requested_outlet,
            tank_params,
            dt,
        )
        new_tank_state = tank_result.state
        realized_outlet = tank_result.realized_outlet_flow
        new_pump_state = RampLimitedActuatorState(current_output=realized_outlet)
        true_level = new_tank_state.level

        # 4. Level sensor
        new_ls_state, measured_level = level_sensor_step(
            state.level_sensor_state,
            true_level,
            level_sensor_params,
            k2,
        )

        # 5. PI controller (for info/comparison — acts on level error → outlet flow)
        new_pi_state, pi_outlet = pi_step(
            state.pi_state,
            measured_level,
            target_level,
            pi_params,
            dt,
        )

        # 6. Observation: [normalized_level, normalized_outlet, normalized_inlet, level_error]
        obs = jnp.array(
            [
                measured_level / tank_max_level,
                realized_outlet / pump_max_flow,
                inlet_flow / pump_max_flow,
                (target_level - measured_level) / tank_max_level,
            ]
        )

        # 7. Reward: equalize the outlet flow, keep the true level inside its band
        # and avoid physical constraint violations
        flow_rate_penalty = config.flow_rate_weight * (
            (realized_outlet - state.last_outlet_flow) / rate_scale
        ) ** 2
        flow_deviation_penalty = config.flow_deviation_weight * (
            (realized_outlet - config.mean_flow) / config.mean_flow
        ) ** 2
        level_fraction = (true_level - min_level) / (tank_max_level - min_level)
        band_excursion = jnp.maximum(config.level_band_low - level_fraction, 0.0) + jnp.maximum(
            level_fraction - config.level_band_high, 0.0
        )
        level_band_penalty = config.level_band_weight * band_excursion**2
        overflow_penalty = config.overflow_penalty * (tank_result.overflow_flow / pump_max_flow) ** 2
        unmet_outlet_penalty = config.unmet_outlet_penalty * (
            tank_result.unmet_outlet_flow / pump_max_flow
        ) ** 2
        constraint_penalty = overflow_penalty + unmet_outlet_penalty
        reward = -(
            flow_rate_penalty + flow_deviation_penalty + level_band_penalty + constraint_penalty
        )

        new_state = EqualizationPlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            tank_state=new_tank_state,
            outlet_pump_state=new_pump_state,
            level_sensor_state=new_ls_state,
            pi_state=new_pi_state,
            last_outlet_flow=realized_outlet,
            disturbance_schedule=state.disturbance_schedule,
        )

        info: dict[str, jax.Array] = {
            "true_level": true_level,
            "measured_level": measured_level,
            "requested_outlet": requested_outlet,
            "realized_outlet": realized_outlet,
            "overflow_flow": tank_result.overflow_flow,
            "unmet_outlet_flow": tank_result.unmet_outlet_flow,
            "tank_constraint_status": tank_result.constraint_status,
            "flow_rate_penalty": flow_rate_penalty,
            "flow_deviation_penalty": flow_deviation_penalty,
            "level_band_penalty": level_band_penalty,
            "overflow_penalty": overflow_penalty,
            "unmet_outlet_penalty": unmet_outlet_penalty,
            "constraint_penalty": constraint_penalty,
            "pi_outlet": pi_outlet,
            "inlet_flow": inlet_flow,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
