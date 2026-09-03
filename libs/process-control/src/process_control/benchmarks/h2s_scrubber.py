"""H₂S scrubber benchmark: multi-loop supervisory control of a chemical scrubber.

Models a caustic/bleach scrubber for removing H₂S from a gas stream.
The RL agent controls setpoints for three dosing loops; plant-side PI
controllers chase those setpoints with realistic imperfections.

Action space (3D):
  action[0]: pH setpoint for caustic dosing loop
  action[1]: ORP setpoint for bleach dosing loop (mV)
  action[2]: sump level setpoint for makeup flow (m³)

Observation space (12D):
  [0]  sensed pH           (from caustic loop sensor)
  [1]  sensed ORP          (from bleach loop sensor, mV)
  [2]  sensed sump level   (from makeup loop sensor, m³)
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
from process_control.scenarios.gas_source import GasSourceParams, GasSourceState
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
from process_control.units.scrubber_sump import reset as sump_reset
from process_control.units.scrubber_sump import step as sump_step

# ── Configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class H2SScrubberConfig:
    dt: float = 1.0 / 12.0  # hours (5-min steps)

    # ── Action ranges (setpoint bounds) ───────────────────────────
    ph_sp_min: float = 8.0
    ph_sp_max: float = 11.0
    ph_sp_default: float = 9.5

    orp_sp_min: float = 400.0  # mV
    orp_sp_max: float = 800.0
    orp_sp_default: float = 650.0

    level_sp_min: float = 3.0  # m³
    level_sp_max: float = 8.0
    level_sp_default: float = 5.0

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
    sump_nominal_volume: float = 5.0

    # ── Contactor ─────────────────────────────────────────────────
    contactor_base_efficiency: float = 0.90
    contactor_max_efficiency: float = 0.99
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


# ── Benchmark Factory ─────────────────────────────────────────────────


def make_h2s_scrubber_benchmark(
    config: H2SScrubberConfig,
) -> tuple[
    Callable[[jax.Array], tuple[H2SScrubberState, jax.Array]],
    Callable[[H2SScrubberState, jax.Array, jax.Array], tuple[H2SScrubberState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
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
    )

    sump_params = ScrubberSumpParams(
        volume_min=config.sump_volume_min,
        volume_max=config.sump_volume_max,
        nominal_volume=config.sump_nominal_volume,
    )

    contactor_params = ContactorParams(
        base_efficiency=config.contactor_base_efficiency,
        max_efficiency=config.contactor_max_efficiency,
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
        base_setpoint=config.level_sp_default,
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

    # Cost per pump unit per step
    caustic_cost = jnp.array(config.caustic_unit_cost)
    bleach_cost = jnp.array(config.bleach_unit_cost)
    makeup_cost = jnp.array(config.makeup_unit_cost)

    # Observation normalization constants
    ph_scale = jnp.array(config.ph_sp_max - config.ph_sp_min)
    orp_scale = jnp.array(config.orp_sp_max - config.orp_sp_min)
    level_scale = jnp.array(config.level_sp_max - config.level_sp_min)
    h2s_scale = jnp.array(config.mean_h2s_ppm * 2.0)
    flow_scale = jnp.array(config.mean_gas_flow)
    steps_per_day = jnp.array(config.steps_per_day, dtype=jnp.float32)

    # Pre-compute initial sensor values from default sump state (outside JIT)
    _init_sump = ScrubberSumpState.create(volume=config.sump_nominal_volume)
    _init_ph = float(compute_ph(_init_sump))
    _init_orp = float(compute_orp(_init_sump))

    def reset(rng_key: jax.Array) -> tuple[H2SScrubberState, jax.Array]:
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        gas_state = gas_reset(k1)
        sump_state = sump_reset(k2, sump_params)
        caustic_state = dosing_reset(
            _init_ph,
            config.caustic_ff,
            k3,
        )
        bleach_state = dosing_reset(
            _init_orp,
            config.bleach_ff,
            k4,
        )
        makeup_state = dosing_reset(
            config.sump_nominal_volume,
            config.makeup_ff,
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
    ) -> tuple[H2SScrubberState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k_gas, k_caustic, k_bleach, k_makeup = jax.random.split(rng_key, 4)

        # ── 1. Scale actions to setpoint ranges ───────────────────
        # Actions arrive in [-1, 1] → scale to physical setpoint ranges
        ph_sp = _scale_action(action[0], config.ph_sp_min, config.ph_sp_max)
        orp_sp = _scale_action(action[1], config.orp_sp_min, config.orp_sp_max)
        level_sp = _scale_action(action[2], config.level_sp_min, config.level_sp_max)

        # ── 2. Gas source ─────────────────────────────────────────
        new_gas, gas_flow, h2s_ppm = gas_step(
            state.gas_source,
            state.step_count,
            gas_params,
            k_gas,
        )

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
            params=contactor_params,
        )

        # ── 4. Dosing loops: process true PVs from sump ──────────
        true_ph = compute_ph(state.sump)
        true_orp = compute_orp(state.sump)
        true_level = state.sump.volume

        new_caustic, sensed_ph, caustic_pump, caustic_pi = dosing_step(
            state.caustic_loop,
            ph_sp,
            true_ph,
            caustic_params,
            dt,
            k_caustic,
        )
        new_bleach, sensed_orp, bleach_pump, bleach_pi = dosing_step(
            state.bleach_loop,
            orp_sp,
            true_orp,
            bleach_params,
            dt,
            k_bleach,
        )
        new_makeup, sensed_level, makeup_pump, makeup_pi = dosing_step(
            state.makeup_loop,
            level_sp,
            true_level,
            makeup_params,
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
        new_sump = sump_step(state.sump, sump_inputs, sump_params, dt)

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
        eff_penalty = eff_penalty_w * eff_deficit**2

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
                plant.makeup_loop.sensor_value / level_scale,  # sensed level (normalised)
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


def _scale_action(action: jax.Array, low: float, high: float) -> jax.Array:
    """Scale action from [-1, 1] to [low, high]."""
    clamped = jnp.clip(action, -1.0, 1.0)
    return jnp.array(low) + (clamped + 1.0) * 0.5 * jnp.array(high - low)
