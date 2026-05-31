from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams, DosePumpState
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.chemistry.ph_model import PhModelParams, compute_ph
from process_control.controllers.pi_controller import PIControllerParams, PIControllerState
from process_control.controllers.pi_controller import reset as pi_reset
from process_control.controllers.pi_controller import step as pi_step
from process_control.disturbances.schedule import DisturbanceSchedule, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.ph_sensor import PhSensorParams, PhSensorState
from process_control.sensors.ph_sensor import reset as ph_sensor_reset
from process_control.sensors.ph_sensor import step as ph_sensor_step
from process_control.units.cstr import CSTRParams, CSTRState
from process_control.units.cstr import reset as cstr_reset
from process_control.units.cstr import step as cstr_step


@dataclass(frozen=True)
class PhNeutralizationBenchmarkConfig:
    target_ph: float = 7.0
    dt: float = 1.0  # minutes

    # CSTR
    cstr_volume: float = 500.0  # L

    # Dose pump (base dosing, mol/min)
    pump_max_dose: float = 10.0
    pump_min_dose: float = 0.0
    pump_max_ramp_rate: float = 20.0

    # PI controller (operates in pH space → outputs dose in mol/min)
    pi_kp: float = 0.5
    pi_ki: float = 0.05
    pi_ff: float = 7.5  # near steady-state dose at default acid load
    pi_output_min: float = 0.0
    pi_output_max: float = 10.0
    pi_max_integral: float = 20.0

    # pH sensor
    ph_noise_std: float = 0.05
    ph_lag_coefficient: float = 0.3

    # pH model
    ph_sensitivity: float = 0.01  # mol/L half-transition width

    # Acid inlet source (repurposes DiurnalSourceState)
    # flow → inlet flow (L/min), demand → acid concentration (mol/L proxy)
    mean_flow: float = 50.0
    diurnal_amplitude: float = 15.0
    min_flow: float = 30.0
    max_flow: float = 80.0
    demand_offset: float = 0.0
    flow_demand_coefficient: float = 0.003  # acid_conc ≈ flow * coeff → ~0.15 mol/L at mean
    demand_noise_std: float = 0.01
    drift_scale: float = 0.1
    steps_per_day: int = 96

    max_disturbance_events: int = 16


@jax_dataclass
class PhPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    cstr_state: CSTRState
    dose_pump_state: DosePumpState
    ph_sensor_state: PhSensorState
    pi_state: PIControllerState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


jax.tree_util.register_dataclass(
    PhPlantState,
    data_fields=[
        "step_count",
        "source_state",
        "cstr_state",
        "dose_pump_state",
        "ph_sensor_state",
        "pi_state",
        "last_dose",
        "disturbance_schedule",
    ],
    meta_fields=[],
)


def make_ph_neutralization_benchmark(
    config: PhNeutralizationBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[PhPlantState, jax.Array]],
    Callable[[PhPlantState, jax.Array, jax.Array], tuple[PhPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    cstr_params = CSTRParams(volume=config.cstr_volume)
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
    ph_sensor_params = PhSensorParams(
        noise_std=config.ph_noise_std,
        lag_coefficient=config.ph_lag_coefficient,
    )
    ph_model_params = PhModelParams(sensitivity=config.ph_sensitivity)
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
    target_ph = jnp.array(config.target_ph)
    pump_max_dose = jnp.array(config.pump_max_dose)

    def reset(rng_key: jax.Array) -> tuple[PhPlantState, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        cstr_state = cstr_reset(k2)
        dp_state = dose_pump_reset(k3)
        ph_state = ph_sensor_reset(k4)
        pi_state = pi_reset(k4)
        last_dose = jnp.array(0.0)

        plant_state = PhPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            cstr_state=cstr_state,
            dose_pump_state=dp_state,
            ph_sensor_state=ph_state,
            pi_state=pi_state,
            last_dose=last_dose,
            disturbance_schedule=create_empty(config.max_disturbance_events),
        )

        measured_ph = jnp.array(7.0)
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
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Source: acid inlet (flow in L/min, demand used as acid concentration)
        new_source_state, transport, flow, acid_concentration = source_step(
            state.source_state, state.step_count, source_params, k1,
        )

        # 2. Dose pump realizes the agent's requested base dose (mol/min)
        new_pump_state, realized_dose = dose_pump_step(
            state.dose_pump_state, action, pump_params, dt,
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
            cstr_state=new_cstr_state,
            dose_pump_state=new_pump_state,
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
