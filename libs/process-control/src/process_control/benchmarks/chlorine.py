import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

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
from process_control.disturbances.schedule import (
    DisturbanceSchedule,
    add_event,
    apply_active,
    create_empty,
)
from process_control.disturbances.types import (
    DISTURBANCE_DEMAND_SLUG,
    DISTURBANCE_NONE,
    DISTURBANCE_RAIN_STORM,
)
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.flow_sensor import FlowSensorParams, FlowSensorState
from process_control.sensors.flow_sensor import reset as flow_sensor_reset
from process_control.sensors.flow_sensor import step as flow_sensor_step
from process_control.signal_bus import SignalBus
from process_control.units.contact_basin import ContactBasinParams, ContactBasinState
from process_control.units.contact_basin import reset as basin_reset
from process_control.units.contact_basin import step as basin_step
from process_control.units.mixer import MixerState
from process_control.units.mixer import step as mixer_step

DisturbanceEvent = tuple[int, int, float, int]


def _validate_disturbance_events(config: "ChlorineBenchmarkConfig") -> int:
    """Validate static event data before it enters a JAX episode state."""
    max_events = config.max_disturbance_events
    if isinstance(max_events, bool) or not isinstance(max_events, Integral):
        raise ValueError("max_disturbance_events must be an integer")
    max_events = int(max_events)
    if max_events <= 0:
        raise ValueError("max_disturbance_events must be positive")

    events = config.disturbance_events
    if not isinstance(events, tuple):
        raise ValueError("disturbance_events must be a tuple of event tuples")
    if len(events) > max_events:
        raise ValueError("disturbance_events exceeds max_disturbance_events")

    supported_types = {DISTURBANCE_NONE, DISTURBANCE_DEMAND_SLUG, DISTURBANCE_RAIN_STORM}
    for index, event in enumerate(events):
        if not isinstance(event, tuple) or len(event) != 4:
            raise ValueError(f"disturbance event {index} must be (start_step, end_step, magnitude, type_id)")
        start_step, end_step, magnitude, type_id = event
        if isinstance(start_step, bool) or not isinstance(start_step, Integral):
            raise ValueError(f"disturbance event {index} start_step must be an integer")
        if isinstance(end_step, bool) or not isinstance(end_step, Integral):
            raise ValueError(f"disturbance event {index} end_step must be an integer")
        if start_step < 0:
            raise ValueError(f"disturbance event {index} start_step must be non-negative")
        if end_step <= start_step:
            raise ValueError(f"disturbance event {index} end_step must be after start_step")
        if isinstance(magnitude, bool) or not isinstance(magnitude, Real) or not math.isfinite(float(magnitude)):
            raise ValueError(f"disturbance event {index} magnitude must be a finite real number")
        if isinstance(type_id, bool) or not isinstance(type_id, Integral) or int(type_id) not in supported_types:
            raise ValueError(f"disturbance event {index} type_id is unsupported")
    return max_events


@jax_dataclass
class PlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    basin_state: ContactBasinState
    flow_sensor_state: FlowSensorState
    dosing_loop: DosingSystemState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


@dataclass(frozen=True)
class ChlorineBenchmarkConfig:
    target_residual: float = 1.5
    dt: float = 0.25
    reward_profile: str = "tracking"
    quality_floor: float = 1.0
    dose_cost_weight: float = 0.0
    dose_movement_weight: float = 0.0

    # ── Control mode ──────────────────────────────────────────────
    # DIRECT (0): RL action is raw dose command (original behavior)
    # SUPERVISORY (1): RL action is residual setpoint → PI → dose
    # FEEDFORWARD (2): PI runs at target, RL action is delta correction
    control_mode: int = DIRECT

    basin_volume: float = 400.0
    basin_segments: int = 10
    basin_tau: float = 1.0

    pump_max_dose: float = 5.0
    pump_min_dose: float = 0.0
    pump_max_ramp_rate: float = 10.0  # symmetric (chemical dosing pump)

    pi_kp: float = 0.1
    pi_ki: float = 0.1
    pi_kd: float = 0.0
    pi_ff: float = 3.0
    pi_output_min: float = 1.5
    pi_output_max: float = 3.5
    pi_max_integral: float = 10.0

    flow_noise_std: float = 0.5
    flow_bias: float = 0.0
    flow_dropout_probability: float = 0.0

    residual_noise_std: float = 0.02
    residual_lag_coefficient: float = 0.3
    residual_drift_rate: float = 0.001

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
    # (start_step, end_step, magnitude, type_id); windows are half-open and
    # magnitudes use the native units of the selected disturbance type.
    disturbance_events: tuple[DisturbanceEvent, ...] = ()


def make_chlorine_benchmark(
    config: ChlorineBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[PlantState, jax.Array]],
    Callable[[PlantState, jax.Array, jax.Array], tuple[PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    max_disturbance_events = _validate_disturbance_events(config)
    if config.reward_profile not in {"tracking", "supervisory-floor"}:
        raise ValueError("reward_profile must be tracking or supervisory-floor")
    if config.quality_floor < 0.0:
        raise ValueError("quality_floor must be non-negative")
    if config.dose_cost_weight < 0.0:
        raise ValueError("dose_cost_weight must be non-negative")
    if config.dose_movement_weight < 0.0:
        raise ValueError("dose_movement_weight must be non-negative")
    dose_range = config.pump_max_dose - config.pump_min_dose
    if dose_range <= 0.0:
        raise ValueError("pump_max_dose must exceed pump_min_dose")

    basin_params = ContactBasinParams(
        total_volume=config.basin_volume,
        n_segments=config.basin_segments,
        tau=config.basin_tau,
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

    # Dosing loop bundles residual sensor + PI controller + dose pump
    dosing_params = DosingSystemParams(
        control_mode=config.control_mode,
        base_setpoint=config.target_residual,
        sensor_noise_std=config.residual_noise_std,
        sensor_lag=config.residual_lag_coefficient,
        sensor_drift_rate=config.residual_drift_rate,
        kp=config.pi_kp,
        ki=config.pi_ki,
        kd=config.pi_kd,
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

    def reset(rng_key: jax.Array) -> tuple[PlantState, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        bas_state = basin_reset(basin_params, k2)
        fs_state = flow_sensor_reset(k3)
        ds_state = dosing_reset(0.0, config.pi_ff, k4)
        last_dose = jnp.array(0.0)

        disturbance_schedule = create_empty(max_disturbance_events)
        for start_step, end_step, magnitude, type_id in config.disturbance_events:
            disturbance_schedule = add_event(
                disturbance_schedule,
                start_step=start_step,
                end_step=end_step,
                magnitude=magnitude,
                type_id=type_id,
            )

        plant_state = PlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            basin_state=bas_state,
            flow_sensor_state=fs_state,
            dosing_loop=ds_state,
            last_dose=last_dose,
            disturbance_schedule=disturbance_schedule,
        )

        signal_bus = SignalBus(flow=jnp.array(0.0), outlet_residual=jnp.array(0.0))
        obs = _build_observation(signal_bus, last_dose, target_residual)
        return plant_state, obs

    def step(state: PlantState, action: jax.Array, rng_key: jax.Array) -> tuple[PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        # 1. Source generates flow + demand
        new_source_state, transport, flow, demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 1.5 Apply active disturbances at the influent boundary. All
        # downstream units and measured influent signals must use this result.
        transport = apply_active(state.disturbance_schedule, transport, state.step_count)
        flow = transport.hydraulics.flow
        demand = transport.composition.demand

        # 2. Dosing loop: reads previous outlet residual from sensor state,
        #    computes dose based on control mode
        current_residual = state.basin_state.segments[-1, 0]
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

        # 4. Contact basin advances
        new_basin_state, outlet_residual = basin_step(
            state.basin_state,
            mixed_transport,
            basin_params,
            dt,
            k4,
        )

        # 5. Flow sensor
        new_flow_state, sensed_flow = flow_sensor_step(
            state.flow_sensor_state,
            flow,
            flow_sensor_params,
            k5,
        )

        # 6. Build observation
        signal_bus = SignalBus(flow=sensed_flow, outlet_residual=sensed_residual)
        obs = _build_observation(signal_bus, realized_dose, target_residual)

        # 7. Reward (from sensor reading — matches what an online agent would see)
        tracking_error_cost = (sensed_residual - target_residual) ** 2
        quality_cost = jnp.maximum(
            jnp.array(config.quality_floor) - sensed_residual,
            0.0,
        ) ** 2
        normalized_dose = (
            realized_dose - config.pump_min_dose
        ) / dose_range
        dose_cost = config.dose_cost_weight * normalized_dose**2
        normalized_dose_delta = (realized_dose - state.last_dose) / dose_range
        dose_movement_cost = (
            config.dose_movement_weight * normalized_dose_delta**2
        )
        if config.reward_profile == "tracking":
            reward = -tracking_error_cost
        else:
            reward = -(quality_cost + dose_cost + dose_movement_cost)

        new_state = PlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            basin_state=new_basin_state,
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
            "tracking_error_cost": tracking_error_cost,
            "quality_cost": quality_cost,
            "dose_cost": dose_cost,
            "dose_movement_cost": dose_movement_cost,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
