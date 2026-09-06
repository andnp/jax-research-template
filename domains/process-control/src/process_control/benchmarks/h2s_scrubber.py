"""H₂S scrubber benchmark: multi-loop supervisory control of a chemical scrubber.

Models a caustic/bleach scrubber for removing H₂S from a gas stream.
The RL agent controls setpoints for three dosing loops; plant-side PI
controllers chase those setpoints with realistic imperfections.

Action space (3D):
  action[0]: pH setpoint for caustic dosing loop
  action[1]: ORP setpoint for bleach dosing loop (mV)
  action[2]: softened-water makeup flow setpoint (L/min)

Observation space (12D):
  [0]  sensed pH           (from caustic loop sensor)
  [1]  sensed ORP          (from bleach loop sensor, mV)
  [2]  sump level           (m³)
  [3]  inlet H₂S           (ppmv, direct measurement)
  [4]  outlet H₂S          (ppmv, direct measurement)
  [5]  gas flow             (m³/h, direct measurement)
  [6]  removal efficiency   (0–1)
  [7]  caustic pump output  (mL/min)
  [8]  bleach pump output   (mL/min)
  [9]  makeup pump output   (L/min)
  [10] sump temperature     (°C)
  [11] step fraction of day (0–1, diurnal context)

Reward: minimize OPEX subject to removal efficiency ≥ target.
  reward = -efficiency_penalty - cost_weight × (
      caustic_cost × caustic_pump + bleach_cost × bleach_pump + makeup_cost × makeup_pump
  )
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.dosing_system import (
    SUPERVISORY,
    DosingSystemParams,
    DosingSystemState,
)
from process_control.actuators.dosing_system import reset as dosing_reset
from process_control.actuators.dosing_system import step as dosing_step
from process_control.scenarios.gas_source import (
    GasSourceParams,
    GasSourceState,
    GasSourceTrajectory,
    trajectory_step,
)
from process_control.scenarios.gas_source import reset as gas_reset
from process_control.scenarios.gas_source import step as gas_step
from process_control.units.gas_liquid_contactor import (
    ContactorParams,
    GasInlet,
    compute_removal,
)
from process_control.units.scrubber_sump import (
    ScrubberSumpParams,
    ScrubberSumpState,
    SumpInputs,
    compute_orp,
    compute_ph,
)
from process_control.units.scrubber_sump import step as sump_step

# ── Configuration ─────────────────────────────────────────────────────


class H2SScenario(StrEnum):
    NOMINAL = "nominal"
    LOAD_SHIFT = "load_shift"


@dataclass(frozen=True, slots=True)
class H2SScenarioContract:
    """Immutable construction-time contract for a named H₂S evaluation regime."""

    name: H2SScenario
    load_shift_step: int | None
    load_shift_multiplier: float


H2S_SCENARIO_CONTRACTS = MappingProxyType({
    H2SScenario.NOMINAL: H2SScenarioContract(H2SScenario.NOMINAL, None, 1.0),
    H2SScenario.LOAD_SHIFT: H2SScenarioContract(H2SScenario.LOAD_SHIFT, 2_000, 2.0),
})


@dataclass(frozen=True)
class H2SScrubberConfig:
    dt: float = 1.0 / 12.0  # hours (5-min steps)
    scenario: H2SScenario = H2SScenario.NOMINAL

    # ── Action ranges (setpoint bounds) ───────────────────────────
    ph_sp_min: float = 8.0
    ph_sp_max: float = 9.8
    ph_sp_default: float = 9.5

    orp_sp_min: float = 400.0  # mV
    orp_sp_max: float = 800.0
    orp_sp_default: float = 650.0

    makeup_flow_sp_min: float = 10.0  # L/min
    makeup_flow_sp_max: float = 30.0
    makeup_flow_sp_default: float = 20.0

    # ── Caustic dosing loop (pH control) ──────────────────────────
    caustic_control_mode: int = SUPERVISORY
    caustic_sensor_noise: float = 0.05
    caustic_sensor_lag: float = 0.85
    caustic_sensor_drift: float = 0.001
    caustic_kp: float = 5.0
    caustic_ki: float = 1.0
    caustic_ff: float = 15.0  # resting pump speed (mL/min)
    caustic_pump_min: float = 0.0
    caustic_pump_max: float = 60.0  # mL/min
    caustic_max_integral: float = 100.0
    caustic_max_ramp: float = 30.0  # mL/min per hour

    # ── Bleach dosing loop (ORP control) ──────────────────────────
    bleach_control_mode: int = SUPERVISORY
    bleach_sensor_noise: float = 5.0  # mV
    bleach_sensor_lag: float = 0.80
    bleach_sensor_drift: float = 0.5
    bleach_kp: float = 0.1  # mV → mL/min (small gain, ORP range is large)
    bleach_ki: float = 0.02
    bleach_ff: float = 20.0  # resting pump speed
    bleach_pump_min: float = 0.0
    bleach_pump_max: float = 80.0  # mL/min
    bleach_max_integral: float = 500.0
    bleach_max_ramp: float = 40.0

    # ── Makeup flow loop (level control) ──────────────────────────
    makeup_control_mode: int = SUPERVISORY
    makeup_sensor_noise: float = 0.02  # m³
    makeup_sensor_lag: float = 0.7
    makeup_sensor_drift: float = 0.0005
    makeup_kp: float = 10.0
    makeup_ki: float = 2.0
    makeup_ff: float = 5.0  # resting flow (L/min)
    makeup_pump_min: float = 0.0
    makeup_pump_max: float = 40.0  # L/min
    makeup_max_integral: float = 50.0
    makeup_max_ramp: float = 20.0

    # ── Sump chemistry ────────────────────────────────────────────
    sump_volume_min: float = 1.0
    sump_volume_max: float = 10.0
    sump_nominal_volume: float = 10.0
    sump_bleach_yield: float = 150.0
    sump_initial_oxidant: float = 200.0
    sump_initial_alkalinity: float = 20.0
    sump_oxidation_rate: float = 0.5
    sump_alkalinity_consumption: float = 1.0
    sump_caustic_yield: float = 0.5
    sump_ph_amplitude: float = 3.0

    # ── Contactor ─────────────────────────────────────────────────
    contactor_base_efficiency: float = 0.99
    contactor_max_efficiency: float = 0.99
    contactor_recirc_weight: float = 0.02
    contactor_recirc_half_sat: float = 5.0
    contactor_gas_flow_exponent: float = -0.005
    recirc_flow: float = 50.0  # m³/h, fixed recirculation rate

    # ── Gas source ────────────────────────────────────────────────
    mean_gas_flow: float = 500.0
    gas_flow_amplitude: float = 100.0
    mean_h2s_ppm: float = 50.0
    h2s_amplitude: float = 20.0
    steps_per_day: int = 288

    # ── Reward ────────────────────────────────────────────────────
    target_efficiency: float = 0.95  # compliance target
    efficiency_penalty_weight: float = 100.0  # penalty for η < target
    cost_weight: float = 0.01
    caustic_unit_cost: float = 0.4536  # $/mL
    bleach_unit_cost: float = 1.355  # $/mL (3× caustic)
    makeup_unit_cost: float = 0.084  # $/L


# ── Plant State ───────────────────────────────────────────────────────


@jax_dataclass
class H2SScrubberState:
    step_count: jax.Array
    gas_source: GasSourceState
    sump: ScrubberSumpState
    caustic_loop: DosingSystemState
    bleach_loop: DosingSystemState
    makeup_loop: DosingSystemState


@jax_dataclass
class H2SDynamicParams:
    """The subset of H2SScrubberConfig calibration can vary, bundled as one pytree.

    Passing this explicitly to `reset`/`step` (instead of relying on the
    values `make_h2s_scrubber_benchmark` closed over from `config`) lets a
    caller that varies these values across many calls — e.g. a calibration
    loop — JIT-compile `reset`/`step` once and reuse that compilation, rather
    than rebuilding the benchmark (and retracing from scratch) on every call.
    """

    sump: ScrubberSumpParams
    contactor: ContactorParams
    caustic: DosingSystemParams
    bleach: DosingSystemParams
    makeup: DosingSystemParams
    ph_amplitude: jax.Array


# ── Benchmark Factory ─────────────────────────────────────────────────


def make_h2s_scrubber_benchmark(
    config: H2SScrubberConfig,
    gas_trajectory: GasSourceTrajectory | None = None,
) -> tuple[
    Callable[[jax.Array], tuple[H2SScrubberState, jax.Array]],
    Callable[[H2SScrubberState, jax.Array, jax.Array], tuple[H2SScrubberState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    scenario = H2S_SCENARIO_CONTRACTS[config.scenario]
    dt = jnp.array(config.dt)
    recirc_flow = jnp.array(config.recirc_flow)
    target_eff = jnp.array(config.target_efficiency)
    eff_penalty_w = jnp.array(config.efficiency_penalty_weight)
    cost_w = jnp.array(config.cost_weight)

    # Build sub-module params
    gas_params = GasSourceParams(
        mean_gas_flow=config.mean_gas_flow,
        gas_flow_amplitude=config.gas_flow_amplitude,
        mean_h2s_ppm=config.mean_h2s_ppm,
        h2s_amplitude=config.h2s_amplitude,
        steps_per_day=config.steps_per_day,
        load_shift_step=scenario.load_shift_step,
        load_shift_multiplier=scenario.load_shift_multiplier,
    )

    sump_params = ScrubberSumpParams(
        volume_min=config.sump_volume_min,
        volume_max=config.sump_volume_max,
        nominal_volume=config.sump_nominal_volume,
        bleach_yield=config.sump_bleach_yield,
        oxidation_rate=config.sump_oxidation_rate,
        alkalinity_consumption=config.sump_alkalinity_consumption,
        caustic_yield=config.sump_caustic_yield,
    )

    contactor_params = ContactorParams(
        base_efficiency=config.contactor_base_efficiency,
        max_efficiency=config.contactor_max_efficiency,
        recirc_weight=config.contactor_recirc_weight,
        recirc_half_sat=config.contactor_recirc_half_sat,
        gas_flow_exponent=config.contactor_gas_flow_exponent,
    )

    caustic_params = DosingSystemParams(
        control_mode=config.caustic_control_mode,
        base_setpoint=config.ph_sp_default,
        sensor_noise_std=config.caustic_sensor_noise,
        sensor_lag=config.caustic_sensor_lag,
        sensor_drift_rate=config.caustic_sensor_drift,
        kp=config.caustic_kp,
        ki=config.caustic_ki,
        ff=config.caustic_ff,
        output_min=config.caustic_pump_min,
        output_max=config.caustic_pump_max,
        max_integral=config.caustic_max_integral,
        max_ramp_up=config.caustic_max_ramp,
        max_ramp_down=config.caustic_max_ramp,
    )

    bleach_params = DosingSystemParams(
        control_mode=config.bleach_control_mode,
        base_setpoint=config.orp_sp_default,
        sensor_noise_std=config.bleach_sensor_noise,
        sensor_lag=config.bleach_sensor_lag,
        sensor_drift_rate=config.bleach_sensor_drift,
        kp=config.bleach_kp,
        ki=config.bleach_ki,
        ff=config.bleach_ff,
        output_min=config.bleach_pump_min,
        output_max=config.bleach_pump_max,
        max_integral=config.bleach_max_integral,
        max_ramp_up=config.bleach_max_ramp,
        max_ramp_down=config.bleach_max_ramp,
    )

    makeup_params = DosingSystemParams(
        control_mode=config.makeup_control_mode,
        base_setpoint=config.makeup_flow_sp_default,
        sensor_noise_std=config.makeup_sensor_noise,
        sensor_lag=config.makeup_sensor_lag,
        sensor_drift_rate=config.makeup_sensor_drift,
        kp=config.makeup_kp,
        ki=config.makeup_ki,
        ff=config.makeup_ff,
        output_min=config.makeup_pump_min,
        output_max=config.makeup_pump_max,
        max_integral=config.makeup_max_integral,
        max_ramp_up=config.makeup_max_ramp,
        max_ramp_down=config.makeup_max_ramp,
    )

    default_dynamic = H2SDynamicParams(
        sump=sump_params,
        contactor=contactor_params,
        caustic=caustic_params,
        bleach=bleach_params,
        makeup=makeup_params,
        ph_amplitude=jnp.array(config.sump_ph_amplitude),
    )

    # Cost per pump unit per step
    caustic_cost = jnp.array(config.caustic_unit_cost)
    bleach_cost = jnp.array(config.bleach_unit_cost)
    makeup_cost = jnp.array(config.makeup_unit_cost)

    # Observation normalization constants
    ph_scale = jnp.array(config.ph_sp_max - config.ph_sp_min)
    orp_scale = jnp.array(config.orp_sp_max - config.orp_sp_min)
    level_scale = jnp.array(config.sump_volume_max)
    h2s_scale = jnp.array(config.mean_h2s_ppm * 2.0)
    flow_scale = jnp.array(config.mean_gas_flow)
    steps_per_day = jnp.array(config.steps_per_day, dtype=jnp.float32)

    def reset(
        rng_key: jax.Array,
        dynamic: H2SDynamicParams = default_dynamic,
    ) -> tuple[H2SScrubberState, jax.Array]:
        k1, k3, k4, k5 = jax.random.split(rng_key, 4)

        gas_state = gas_reset(k1)
        sump_state = ScrubberSumpState.create(
            oxidant=config.sump_initial_oxidant * scenario.load_shift_multiplier,
            alkalinity=config.sump_initial_alkalinity,
            volume=config.sump_nominal_volume,
        )
        init_ph = compute_ph(sump_state, dynamic.ph_amplitude)
        init_orp = compute_orp(sump_state)
        caustic_state = dosing_reset(
            init_ph,
            dynamic.caustic.ff,
            k3,
        )
        bleach_state = dosing_reset(
            init_orp,
            dynamic.bleach.ff,
            k4,
        )
        makeup_state = dosing_reset(
            dynamic.makeup.ff,
            dynamic.makeup.ff,
            k5,
        )

        plant = H2SScrubberState(
            step_count=jnp.array(0, dtype=jnp.int32),
            gas_source=gas_state,
            sump=sump_state,
            caustic_loop=caustic_state,
            bleach_loop=bleach_state,
            makeup_loop=makeup_state,
        )

        obs = _build_obs(plant, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))
        return plant, obs

    def step(
        state: H2SScrubberState,
        action: jax.Array,
        rng_key: jax.Array,
        dynamic: H2SDynamicParams = default_dynamic,
    ) -> tuple[H2SScrubberState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k_gas, k_caustic, k_bleach, k_makeup = jax.random.split(rng_key, 4)

        # The public action-normalization wrapper converts [-1, 1] to setpoints.
        ph_sp, orp_sp, makeup_flow_sp = action

        # ── 2. Gas source ─────────────────────────────────────────
        if gas_trajectory is None:
            new_gas, gas_flow, h2s_ppm = gas_step(
                state.gas_source,
                state.step_count,
                gas_params,
                k_gas,
            )
        else:
            new_gas = state.gas_source
            gas_flow, h2s_ppm = trajectory_step(gas_trajectory, state.step_count)

        # ── 3. Contactor: removal based on current sump chemistry ─
        gas_inlet = GasInlet(
            gas_flow=gas_flow,
            h2s_ppm=h2s_ppm,
            temperature=state.sump.temperature,
        )
        contactor_result = compute_removal(
            gas_inlet,
            oxidant=state.sump.oxidant,
            alkalinity=state.sump.alkalinity,
            recirc_flow=recirc_flow,
            params=dynamic.contactor,
        )

        # ── 4. Dosing loops: process true PVs from sump ──────────
        true_ph = compute_ph(state.sump, dynamic.ph_amplitude)
        true_orp = compute_orp(state.sump)
        true_level = state.sump.volume
        true_makeup_flow = state.makeup_loop.pump_output

        new_caustic, sensed_ph, caustic_pump, caustic_pi = dosing_step(
            state.caustic_loop,
            ph_sp,
            true_ph,
            dynamic.caustic,
            dt,
            k_caustic,
        )
        new_bleach, sensed_orp, bleach_pump, bleach_pi = dosing_step(
            state.bleach_loop,
            orp_sp,
            true_orp,
            dynamic.bleach,
            dt,
            k_bleach,
        )
        new_makeup, sensed_makeup_flow, makeup_pump, makeup_pi = dosing_step(
            state.makeup_loop,
            makeup_flow_sp,
            true_makeup_flow,
            dynamic.makeup,
            dt,
            k_makeup,
        )

        # ── 5. Sump chemistry: advance with dosing + sulfide load ─
        sump_inputs = SumpInputs(
            bleach_flow=bleach_pump,
            caustic_flow=caustic_pump,
            makeup_flow=makeup_pump,
            sulfide_load=contactor_result.sulfide_load,
        )
        new_sump = sump_step(state.sump, sump_inputs, dynamic.sump, dt)

        # ── 6. Observation ────────────────────────────────────────
        obs = _build_obs(
            H2SScrubberState(
                step_count=state.step_count + 1,
                gas_source=new_gas,
                sump=new_sump,
                caustic_loop=new_caustic,
                bleach_loop=new_bleach,
                makeup_loop=new_makeup,
            ),
            h2s_ppm,
            contactor_result.outlet_h2s_ppm,
            gas_flow,
            contactor_result.removal_efficiency,
        )

        # ── 7. Reward ─────────────────────────────────────────────
        # Efficiency penalty (soft constraint)
        eff_deficit = jnp.maximum(target_eff - contactor_result.removal_efficiency, 0.0)
        eff_penalty = eff_penalty_w * eff_deficit

        # OPEX cost (per-step, proportional to pump outputs)
        opex = caustic_cost * caustic_pump + bleach_cost * bleach_pump + makeup_cost * makeup_pump
        reward = -(eff_penalty + cost_w * opex)

        # ── 8. New state ──────────────────────────────────────────
        new_state = H2SScrubberState(
            step_count=state.step_count + 1,
            gas_source=new_gas,
            sump=new_sump,
            caustic_loop=new_caustic,
            bleach_loop=new_bleach,
            makeup_loop=new_makeup,
        )

        done = jnp.array(False)
        info: dict[str, jax.Array] = {
            "removal_efficiency": contactor_result.removal_efficiency,
            "outlet_h2s_ppm": contactor_result.outlet_h2s_ppm,
            "inlet_h2s_ppm": h2s_ppm,
            "gas_flow": gas_flow,
            "true_ph": true_ph,
            "true_orp": true_orp,
            "true_level": true_level,
            "sensed_makeup_flow": sensed_makeup_flow,
            "caustic_pump": caustic_pump,
            "bleach_pump": bleach_pump,
            "makeup_pump": makeup_pump,
            "caustic_pi": caustic_pi,
            "bleach_pi": bleach_pi,
            "makeup_pi": makeup_pi,
            "opex": opex,
            "eff_penalty": eff_penalty,
        }

        return new_state, obs, reward, done, info

    def _build_obs(
        plant: H2SScrubberState,
        inlet_h2s: jax.Array,
        outlet_h2s: jax.Array,
        gas_flow: jax.Array,
        removal_eff: jax.Array,
    ) -> jax.Array:
        day_frac = (plant.step_count % config.steps_per_day) / steps_per_day
        return jnp.array(
            [
                plant.caustic_loop.sensor_value / ph_scale,  # sensed pH (normalised)
                plant.bleach_loop.sensor_value / orp_scale,  # sensed ORP (normalised)
                plant.sump.volume / level_scale,  # sump level (normalised)
                inlet_h2s / h2s_scale,  # inlet H₂S
                outlet_h2s / h2s_scale,  # outlet H₂S
                gas_flow / flow_scale,  # gas flow
                removal_eff,  # removal efficiency (0–1)
                plant.caustic_loop.pump_output / config.caustic_pump_max,
                plant.bleach_loop.pump_output / config.bleach_pump_max,
                plant.makeup_loop.pump_output / config.makeup_pump_max,
                plant.sump.temperature / 50.0,  # temperature (normalised)
                day_frac,  # diurnal phase
            ]
        )

    return reset, step
