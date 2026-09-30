from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import reset as actuator_reset
from process_control.actuators.ramp_limited import step as actuator_step
from process_control.chemistry.ph_model import PhModelParams, compute_ph
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
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as ph_sensor_reset
from process_control.sensors.lagged import step as ph_sensor_step
from process_control.units.cstr import CSTRParams, CSTRState
from process_control.units.cstr import reset as cstr_reset
from process_control.units.cstr import step as cstr_step


@dataclass(frozen=True)
class PhNeutralizationBenchmarkConfig:
    target_ph: float = 7.0
    dt: float = 1.0  # minutes

    # CSTR
    cstr_volume: float = 500.0  # L

    # Dose pump (base dosing, mol/min). Sized with headroom over the
    # steady-state dose implied by the acid load and pH-model parameters
    # below (~0.15-1.1 mol/min across the acid concentration/flow range).
    pump_max_dose: float = 3.0
    pump_min_dose: float = 0.0
    pump_max_ramp_rate: float = 1.0

    # PI controller (operates in pH space → outputs dose in mol/min). Gains
    # are small because the titration curve is steep near neutrality (see
    # ph_model.py): the process gain from dose to pH is far higher than for
    # a linear process, so a much larger kp/ki would overshoot and oscillate.
    pi_kp: float = 0.02
    pi_ki: float = 0.005
    pi_ff: float = 0.4544  # steady-state dose at mean acid load and flow
    pi_output_min: float = 0.0
    pi_output_max: float = 3.0

    # pH sensor
    ph_noise_std: float = 0.05
    ph_lag_coefficient: float = 0.3

    # pH model: total carbonate buffer capacity (mol/L); see ph_model.py.
    ph_buffer_capacity: float = 0.005

    # Inlet flow (L/min), via DiurnalSourceState/Params (demand output unused).
    mean_flow: float = 50.0
    diurnal_amplitude: float = 15.0
    min_flow: float = 30.0
    max_flow: float = 80.0
    drift_scale: float = 0.1
    steps_per_day: int = 96

    # Acid inlet concentration (mol/L strong acid), via its own diurnal
    # channel (see chemistry/ph_model.py and benchmarks/chem_p_dosing.py for
    # the same pattern). 1e-3-1e-2 mol/L spans the free-mineral-acidity scale
    # used to titrate industrial acidic wastewater
    # (https://law.resource.org/pub/us/cfr/ibr/002/apha.method.2320.1992.html).
    acid_mean: float = 0.005
    acid_amplitude: float = 0.002
    acid_min: float = 0.001
    acid_max: float = 0.01
    acid_drift_scale: float = 0.0002

    max_disturbance_events: int = 16
    disturbance_events: tuple[tuple[int, int, float, int], ...] = ()


@jax_dataclass
class PhPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    acid_drift: jax.Array
    cstr_state: CSTRState
    actuator_state: RampLimitedActuatorState
    ph_sensor_state: LaggedSensorState
    pi_state: PIControllerState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


def make_ph_neutralization_benchmark(
    config: PhNeutralizationBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[PhPlantState, jax.Array]],
    Callable[[PhPlantState, jax.Array, jax.Array], tuple[PhPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    max_disturbance_events = validate_disturbance_events(
        config.max_disturbance_events,
        config.disturbance_events,
    )
    cstr_params = CSTRParams(volume=config.cstr_volume)
    pump_params = RampLimitedActuatorParams(
        max_output=config.pump_max_dose,
        min_output=config.pump_min_dose,
        max_ramp_rate=config.pump_max_ramp_rate,
    )
    pi_params = PIControllerParams(
        kp=config.pi_kp,
        ki=config.pi_ki,
        ff=config.pi_ff,
        output_min=config.pi_output_min,
        output_max=config.pi_output_max,
    )
    ph_sensor_params = LaggedSensorParams(
        noise_std=config.ph_noise_std,
        lag_coefficient=config.ph_lag_coefficient,
    )
    ph_model_params = PhModelParams(buffer_capacity=config.ph_buffer_capacity)
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
    acid_channel_params = ChannelParams(
        mean=config.acid_mean,
        amplitude=config.acid_amplitude,
        min_value=config.acid_min,
        max_value=config.acid_max,
        drift_scale=config.acid_drift_scale,
        drift_clip=config.acid_amplitude,
    )
    dt = jnp.array(config.dt)
    target_ph = jnp.array(config.target_ph)
    pump_max_dose = jnp.array(config.pump_max_dose)

    def reset(rng_key: jax.Array) -> tuple[PhPlantState, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        acid_drift = jnp.array(0.0)
        cstr_state = cstr_reset(k2)
        dp_state = actuator_reset(k3)
        initial_ph = compute_ph(cstr_state.concentration, ph_model_params)
        ph_state = ph_sensor_reset(initial_ph, k4)
        pi_state = pi_reset(k4)
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

        plant_state = PhPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            acid_drift=acid_drift,
            cstr_state=cstr_state,
            actuator_state=dp_state,
            ph_sensor_state=ph_state,
            pi_state=pi_state,
            last_dose=last_dose,
            disturbance_schedule=disturbance_schedule,
        )

        measured_ph = ph_state.held_value
        obs = jnp.array(
            [
                measured_ph / 14.0,
                last_dose / pump_max_dose,
                target_ph / 14.0,
                (target_ph - measured_ph) / 7.0,
            ]
        )
        return plant_state, obs

    def step(
        state: PhPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[PhPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k1b, k2, k3, k4 = jax.random.split(rng_key, 5)

        # 1. Source: inlet flow (L/min), subject to disturbance events (e.g. rain).
        new_source_state, transport, _flow, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )
        transport = apply_active(state.disturbance_schedule, transport, state.step_count)
        flow = transport.hydraulics.flow

        # 1b. Acid inlet concentration (mol/L strong acid): its own diurnal
        #    channel, decoupled from the influent flow/demand chemistry above
        #    (mirrors chem_p_dosing.py's influent PO4 channel).
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        acid_signal = jnp.sin(2.0 * jnp.pi * t)
        new_acid_drift, acid_concentration = channel_step(
            state.acid_drift,
            acid_signal,
            acid_channel_params,
            k1b,
        )

        # 2. Dose pump realizes the agent's requested base dose (mol/min)
        new_pump_state, realized_dose = actuator_step(
            state.actuator_state,
            action,
            pump_params,
            dt,
        )

        # 3. CSTR: acid inlet drives base_excess negative; dose drives it positive
        #    inlet_concentration = -acid_concentration (acid reduces base_excess)
        new_cstr_state = cstr_step(
            state.cstr_state,
            flow,
            -acid_concentration,
            realized_dose,
            cstr_params,
            dt,
        )

        # 4. Compute true pH from base excess in CSTR
        true_ph = compute_ph(new_cstr_state.concentration, ph_model_params)

        # 5. pH sensor
        new_ph_state, measured_ph = ph_sensor_step(
            state.ph_sensor_state,
            true_ph,
            ph_sensor_params,
            k2,
        )

        # 6. PI controller (for info/comparison)
        ph_error = target_ph - measured_ph
        new_pi_state, pi_dose = pi_step(
            state.pi_state,
            measured_ph,
            target_ph,
            pi_params,
            dt,
        )

        # 7. Observation: [normalized_ph, normalized_dose, normalized_target, normalized_error]
        obs = jnp.array(
            [
                measured_ph / 14.0,
                realized_dose / pump_max_dose,
                target_ph / 14.0,
                ph_error / 7.0,
            ]
        )

        # 8. Reward: from sensor reading (matches what an online agent would see)
        reward = -((measured_ph - target_ph) ** 2)

        new_state = PhPlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            acid_drift=new_acid_drift,
            cstr_state=new_cstr_state,
            actuator_state=new_pump_state,
            ph_sensor_state=new_ph_state,
            pi_state=new_pi_state,
            last_dose=realized_dose,
            disturbance_schedule=state.disturbance_schedule,
        )

        info: dict[str, jax.Array] = {
            "true_ph": true_ph,
            "measured_ph": measured_ph,
            "realized_dose": realized_dose,
            "pi_dose": pi_dose,
            "acid_concentration": acid_concentration,
            "flow": flow,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
