from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.controllers.pi_controller import PIControllerParams
from process_control.controllers.pi_controller import reset as pi_reset
from process_control.controllers.pi_controller import step as pi_step
from process_control.disturbances.schedule import apply_active, create_empty
from process_control.plant_state import PlantState
from process_control.scenarios.diurnal_source import DiurnalSourceParams
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.flow_sensor import FlowSensorParams
from process_control.sensors.flow_sensor import reset as flow_sensor_reset
from process_control.sensors.flow_sensor import step as flow_sensor_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams
from process_control.sensors.residual_analyzer import reset as residual_reset
from process_control.sensors.residual_analyzer import step as residual_step
from process_control.signal_bus import SignalBus
from process_control.units.contact_basin import ContactBasinParams
from process_control.units.contact_basin import reset as basin_reset
from process_control.units.contact_basin import step as basin_step
from process_control.units.mixer import MixerState
from process_control.units.mixer import step as mixer_step


@dataclass(frozen=True)
class ChlorineBenchmarkConfig:
    target_residual: float = 1.5
    dt: float = 0.25

    basin_volume: float = 400.0
    basin_segments: int = 10
    basin_tau: float = 1.0

    pump_max_dose: float = 5.0
    pump_min_dose: float = 0.0
    pump_max_ramp_rate: float = 10.0

    pi_kp: float = 0.1
    pi_ki: float = 0.1
    pi_ff: float = 3.0
    pi_output_min: float = 1.5
    pi_output_max: float = 3.5
    pi_max_integral: float = 10.0

    flow_noise_std: float = 0.5
    flow_bias: float = 0.0
    flow_dropout_probability: float = 0.0

    residual_noise_std: float = 0.02
    residual_lag_coefficient: float = 0.3
    residual_sample_period: int = 1

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


def _make_params(config: ChlorineBenchmarkConfig) -> tuple[ContactBasinParams, DosePumpParams, PIControllerParams, FlowSensorParams, ResidualAnalyzerParams, DiurnalSourceParams]:
    basin_params = ContactBasinParams(
        total_volume=config.basin_volume,
        n_segments=config.basin_segments,
        tau=config.basin_tau,
    )
    pump_params = DosePumpParams(
        max_dose=config.pump_max_dose,
        min_dose=config.pump_min_dose,
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
    flow_sensor_params = FlowSensorParams(
        noise_std=config.flow_noise_std,
        bias=config.flow_bias,
        dropout_probability=config.flow_dropout_probability,
    )
    residual_params = ResidualAnalyzerParams(
        noise_std=config.residual_noise_std,
        lag_coefficient=config.residual_lag_coefficient,
        sample_period=config.residual_sample_period,
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
    return basin_params, pump_params, pi_params, flow_sensor_params, residual_params, source_params


def make_chlorine_benchmark(
    config: ChlorineBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[PlantState, jax.Array]],
    Callable[[PlantState, jax.Array, jax.Array], tuple[PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    basin_params, pump_params, pi_params, flow_sensor_params, residual_params, source_params = _make_params(config)
    dt = jnp.array(config.dt)
    target_residual = jnp.array(config.target_residual)

    def _build_observation(
        signal_bus: SignalBus,
        last_dose: jax.Array,
        target_residual: jax.Array,
    ) -> jax.Array:
        return jnp.array([last_dose, signal_bus.outlet_residual, target_residual, signal_bus.flow])

    def reset(rng_key: jax.Array) -> tuple[PlantState, jax.Array]:
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

        src_state = source_reset(k1)
        bas_state = basin_reset(basin_params, k2)
        fs_state = flow_sensor_reset(k3)
        rs_state = residual_reset(k4)
        dp_state = dose_pump_reset(k5)
        pi_state = pi_reset(k6)
        last_dose = jnp.array(0.0)

        plant_state = PlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            basin_state=bas_state,
            flow_sensor_state=fs_state,
            residual_sensor_state=rs_state,
            dose_pump_state=dp_state,
            pi_state=pi_state,
            last_dose=last_dose,
            disturbance_schedule=create_empty(config.max_disturbance_events),
        )

        signal_bus = SignalBus(flow=jnp.array(0.0), outlet_residual=jnp.array(0.0))
        obs = _build_observation(signal_bus, last_dose, target_residual)
        return plant_state, obs

    def step(state: PlantState, action: jax.Array, rng_key: jax.Array) -> tuple[PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, _k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

        # 1. Source generates flow + demand
        new_source_state, transport, flow, demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 1.5 Apply active disturbances
        transport = apply_active(state.disturbance_schedule, transport, state.step_count)

        # 2. Dose pump realizes the agent's requested dose
        new_pump_state, realized_dose = dose_pump_step(
            state.dose_pump_state, action, pump_params, dt,
        )

        # 3. Mixer injects dose into stream
        _, mixed_transport = mixer_step(MixerState(), transport, realized_dose, dt, k3)

        # 4. Contact basin advances
        new_basin_state, outlet_residual = basin_step(
            state.basin_state,
            mixed_transport,
            basin_params,
            dt,
            k4,
        )

        # 5. Sensors sample
        new_flow_state, sensed_flow = flow_sensor_step(
            state.flow_sensor_state,
            flow,
            flow_sensor_params,
            k5,
        )
        new_residual_state, sensed_residual = residual_step(
            state.residual_sensor_state, outlet_residual, residual_params, k6,
        )

        # 6. PI controller (for info/comparison — not driving action)
        new_pi_state, pi_dose = pi_step(
            state.pi_state, sensed_residual, target_residual, pi_params, dt,
        )

        # 7. Build observation
        signal_bus = SignalBus(flow=sensed_flow, outlet_residual=sensed_residual)
        obs = _build_observation(signal_bus, realized_dose, target_residual)

        # 8. Reward (from sensor reading — matches what an online agent would see)
        reward = -((sensed_residual - target_residual) ** 2)

        new_state = PlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            basin_state=new_basin_state,
            flow_sensor_state=new_flow_state,
            residual_sensor_state=new_residual_state,
            dose_pump_state=new_pump_state,
            pi_state=new_pi_state,
            last_dose=realized_dose,
            disturbance_schedule=state.disturbance_schedule,
        )

        info: dict[str, jax.Array] = {
            "pi_dose": pi_dose,
            "outlet_residual": outlet_residual,
            "flow": flow,
            "demand": demand,
            "realized_dose": realized_dose,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
