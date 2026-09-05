from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.dosing_system import (
    DIRECT,
    DosingSystemParams,
    DosingSystemState,
)
from process_control.actuators.dosing_system import reset as dosing_reset
from process_control.actuators.dosing_system import step as dosing_step
from process_control.disturbances.schedule import DisturbanceSchedule, apply_active, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.flow_sensor import FlowSensorParams, FlowSensorState
from process_control.sensors.flow_sensor import reset as flow_sensor_reset
from process_control.sensors.flow_sensor import step as flow_sensor_step
from process_control.signal_bus import SignalBus
from process_control.transport import Composition, Hydraulics, Transport
from process_control.units.contact_basin import ContactBasinParams, ContactBasinState
from process_control.units.contact_basin import reset as basin_reset
from process_control.units.contact_basin import step as basin_step
from process_control.units.mixer import MixerState
from process_control.units.mixer import step as mixer_step


@dataclass(frozen=True)
class ChlorineTwoStageBenchmarkConfig:
    """Two contact basins in series with a single upstream dose point.

    Doubles the effective contact time compared to the single-stage chlorine
    benchmark, producing slower outlet dynamics and a different RL challenge shape.
    """

    target_residual: float = 1.5
    dt: float = 0.25

    # ── Control mode ──────────────────────────────────────────────
    control_mode: int = DIRECT

    # Contact basin 1 (primary disinfection)
    basin1_volume: float = 200.0
    basin1_segments: int = 10
    basin1_tau: float = 1.0

    # Contact basin 2 (secondary disinfection / distribution proxy)
    basin2_volume: float = 200.0
    basin2_segments: int = 10
    basin2_tau: float = 1.0

    # Dose pump
    pump_max_dose: float = 5.0
    pump_min_dose: float = 0.0
    pump_max_ramp_rate: float = 10.0  # symmetric (chemical dosing pump)

    # PI controller
    pi_kp: float = 0.1
    pi_ki: float = 0.1
    pi_ff: float = 3.0
    pi_output_min: float = 1.5
    pi_output_max: float = 3.5
    pi_max_integral: float = 10.0

    # Flow sensor
    flow_noise_std: float = 0.5
    flow_bias: float = 0.0
    flow_dropout_probability: float = 0.0

    # Residual sensor (on outlet of basin 2)
    residual_noise_std: float = 0.02
    residual_lag_coefficient: float = 0.3
    residual_drift_rate: float = 0.001

    # Diurnal source
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
class TwoStageChlorinePlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    basin1_state: ContactBasinState
    basin2_state: ContactBasinState
    flow_sensor_state: FlowSensorState
    dosing_loop: DosingSystemState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


def make_chlorine_two_stage_benchmark(
    config: ChlorineTwoStageBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[TwoStageChlorinePlantState, jax.Array]],
    Callable[[TwoStageChlorinePlantState, jax.Array, jax.Array], tuple[TwoStageChlorinePlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    basin1_params = ContactBasinParams(
        total_volume=config.basin1_volume,
        n_segments=config.basin1_segments,
        tau=config.basin1_tau,
    )
    basin2_params = ContactBasinParams(
        total_volume=config.basin2_volume,
        n_segments=config.basin2_segments,
        tau=config.basin2_tau,
    )
    flow_sensor_params = FlowSensorParams(
        noise_std=config.flow_noise_std,
        bias=config.flow_bias,
        dropout_probability=config.flow_dropout_probability,
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

    dosing_params = DosingSystemParams(
        control_mode=config.control_mode,
        base_setpoint=config.target_residual,
        sensor_noise_std=config.residual_noise_std,
        sensor_lag=config.residual_lag_coefficient,
        sensor_drift_rate=config.residual_drift_rate,
        kp=config.pi_kp,
        ki=config.pi_ki,
        ff=config.pi_ff,
        output_min=config.pump_min_dose,
        output_max=config.pump_max_dose,
        max_integral=config.pi_max_integral,
        max_ramp_up=config.pump_max_ramp_rate,
        max_ramp_down=config.pump_max_ramp_rate,
    )

    dt = jnp.array(config.dt)
    target_residual = jnp.array(config.target_residual)

    def _build_observation(
        signal_bus: SignalBus,
        last_dose: jax.Array,
        target_residual: jax.Array,
    ) -> jax.Array:
        return jnp.array([last_dose, signal_bus.outlet_residual, target_residual, signal_bus.flow])

    def reset(rng_key: jax.Array) -> tuple[TwoStageChlorinePlantState, jax.Array]:
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        src_state = source_reset(k1)
        b1_state = basin_reset(basin1_params, k2)
        b2_state = basin_reset(basin2_params, k3)
        fs_state = flow_sensor_reset(k4)
        ds_state = dosing_reset(0.0, config.pi_ff, k5)
        last_dose = jnp.array(0.0)

        plant_state = TwoStageChlorinePlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            basin1_state=b1_state,
            basin2_state=b2_state,
            flow_sensor_state=fs_state,
            dosing_loop=ds_state,
            last_dose=last_dose,
            disturbance_schedule=create_empty(config.max_disturbance_events),
        )

        signal_bus = SignalBus(flow=jnp.array(0.0), outlet_residual=jnp.array(0.0))
        obs = _build_observation(signal_bus, last_dose, target_residual)
        return plant_state, obs

    def step(
        state: TwoStageChlorinePlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TwoStageChlorinePlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

        # 1. Source
        new_source_state, transport, flow, demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 1.5 Apply active disturbances
        transport = apply_active(state.disturbance_schedule, transport, state.step_count)

        # 2. Dosing loop: reads basin 2 outlet from previous step
        current_residual = state.basin2_state.segments[-1, 0]
        new_dosing, sensed_residual, realized_dose, pi_dose = dosing_step(
            state.dosing_loop,
            action,
            current_residual,
            dosing_params,
            dt,
            k2,
        )

        # 3. Mixer injects dose into stream
        _, mixed_transport = mixer_step(MixerState(), transport, realized_dose, dt, k3)

        # 4. Basin 1 advances
        new_basin1_state, _outlet_1 = basin_step(
            state.basin1_state,
            mixed_transport,
            basin1_params,
            dt,
            k4,
        )

        # 5. Build transport for basin 2 from basin 1's outlet
        outlet_1_segments = new_basin1_state.segments[-1]
        transport_2 = Transport(
            hydraulics=Hydraulics(flow=transport.hydraulics.flow),
            composition=Composition(
                chlorine_residual=outlet_1_segments[0],
                demand=outlet_1_segments[1],
                ammonia=outlet_1_segments[2],
                turbidity=outlet_1_segments[3],
                organics=outlet_1_segments[4],
            ),
            bulk_properties=transport.bulk_properties,
        )

        # 6. Basin 2 advances
        new_basin2_state, outlet_residual = basin_step(
            state.basin2_state,
            transport_2,
            basin2_params,
            dt,
            k5,
        )

        # 7. Flow sensor
        new_flow_state, sensed_flow = flow_sensor_step(
            state.flow_sensor_state,
            flow,
            flow_sensor_params,
            k6,
        )

        # 8. Build observation
        signal_bus = SignalBus(flow=sensed_flow, outlet_residual=sensed_residual)
        obs = _build_observation(signal_bus, realized_dose, target_residual)

        # 9. Reward
        reward = -((sensed_residual - target_residual) ** 2)

        new_state = TwoStageChlorinePlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            basin1_state=new_basin1_state,
            basin2_state=new_basin2_state,
            flow_sensor_state=new_flow_state,
            dosing_loop=new_dosing,
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
