from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams
from process_control.actuators.dose_pump import DosePumpState
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.controllers.pi_controller import PIControllerParams
from process_control.controllers.pi_controller import PIControllerState
from process_control.controllers.pi_controller import reset as pi_reset
from process_control.controllers.pi_controller import step as pi_step
from process_control.disturbances.schedule import DisturbanceSchedule
from process_control.disturbances.schedule import apply_active, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams
from process_control.scenarios.diurnal_source import DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.level_sensor import LevelSensorParams
from process_control.sensors.level_sensor import LevelSensorState
from process_control.sensors.level_sensor import reset as level_sensor_reset
from process_control.sensors.level_sensor import step as level_sensor_step
from process_control.units.tank import TankParams
from process_control.units.tank import TankState
from process_control.units.tank import reset as tank_reset
from process_control.units.tank import step as tank_step


@dataclass(frozen=True)
class EqualizationTankBenchmarkConfig:
    target_level_fraction: float = 0.7  # fraction of max_level
    dt: float = 0.25  # time step (same units as flow / area)

    # Tank geometry
    tank_max_level: float = 5.0
    tank_min_level: float = 0.0
    tank_area: float = 300.0  # level changes as flow / area * dt

    # Outlet pump (action = desired outlet flow set point)
    pump_max_flow: float = 150.0
    pump_min_flow: float = 0.0
    pump_max_ramp_rate: float = 100.0

    # PI controller (proportional-integral on level error → outlet flow)
    pi_kp: float = 5.0
    pi_ki: float = 0.5
    pi_ff: float = 75.0  # steady-state outlet ≈ mean inlet
    pi_output_min: float = 0.0
    pi_output_max: float = 150.0
    pi_max_integral: float = 50.0

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


@jax_dataclass
class EqualizationPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    tank_state: TankState
    outlet_pump_state: DosePumpState
    level_sensor_state: LevelSensorState
    pi_state: PIControllerState
    last_outlet_flow: jax.Array
    disturbance_schedule: DisturbanceSchedule


def make_equalization_tank_benchmark(
    config: EqualizationTankBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[EqualizationPlantState, jax.Array]],
    Callable[[EqualizationPlantState, jax.Array, jax.Array], tuple[EqualizationPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    tank_params = TankParams(
        max_level=config.tank_max_level,
        min_level=config.tank_min_level,
        cross_section_area=config.tank_area,
    )
    pump_params = DosePumpParams(
        max_dose=config.pump_max_flow,
        min_dose=config.pump_min_flow,
        max_ramp_rate=config.pump_max_ramp_rate,
    )
    pi_params = PIControllerParams(
        kp=config.pi_kp,
        ki=config.pi_ki,
        ff=config.pi_ff,
        output_min=config.pi_output_min,
        output_max=config.pi_output_max,
        max_integral=config.pi_max_integral,
    )
    level_sensor_params = LevelSensorParams(
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

    def reset(rng_key: jax.Array) -> tuple[EqualizationPlantState, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        tank_state = tank_reset(tank_params)
        pump_state = dose_pump_reset(k2)
        level_initial = tank_state.level
        ls_state = level_sensor_reset(config.tank_max_level * 0.5, k3)
        pi_state = pi_reset(k4)

        plant_state = EqualizationPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            tank_state=tank_state,
            outlet_pump_state=pump_state,
            level_sensor_state=ls_state,
            pi_state=pi_state,
            last_outlet_flow=jnp.array(0.0),
            disturbance_schedule=create_empty(config.max_disturbance_events),
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
        new_pump_state, realized_outlet = dose_pump_step(
            state.outlet_pump_state,
            action,
            pump_params,
            dt,
        )

        # 3. Tank level update
        new_tank_state = tank_step(
            state.tank_state,
            inlet_flow,
            realized_outlet,
            tank_params,
            dt,
        )
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

        # 7. Reward: negative MSE on level setpoint
        reward = -((true_level - target_level) ** 2)

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
            "realized_outlet": realized_outlet,
            "pi_outlet": pi_outlet,
            "inlet_flow": inlet_flow,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
