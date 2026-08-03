"""Chemical dosing subsystem: sensor + PI controller + ramp-limited pump.

Models a complete dosing loop as found on real plants.  Supports three
RL integration modes via ``control_mode``:

  DIRECT (0)       – RL action IS the pump command (bypass PI).
  SUPERVISORY (1)  – RL action IS the setpoint; the inner PI drives the pump.
  FEEDFORWARD (2)  – PI runs at a fixed base setpoint; RL action is a
                     delta added to the PI output.

In all modes, the sensor and PI controller always run — the sensor provides
the observation, and the PI provides a baseline for comparison or feedforward.

Configurable PID tuning lets the benchmark reproduce real-world behavior
like integrator windup, slow response, and overshoot from poorly-tuned
plant controllers.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.controllers.pi_controller import (
    PIControllerParams,
    PIControllerState,
)
from process_control.controllers.pi_controller import (
    step_with_diagnostics as pi_step,
)
from process_control.controllers.pid_controller import (
    PIDControllerParams,
    PIDControllerState,
)
from process_control.controllers.pid_controller import (
    step_with_diagnostics as pid_step_with_diagnostics,
)

# ── Control integration modes ────────────────────────────────────────
DIRECT: int = 0
SUPERVISORY: int = 1
FEEDFORWARD: int = 2


@dataclass(frozen=True)
class DosingSystemParams:
    # ── Control mode ──────────────────────────────────────────────
    control_mode: int = 1  # DIRECT=0, SUPERVISORY=1, FEEDFORWARD=2
    base_setpoint: float = 0.0  # fixed setpoint used in FEEDFORWARD mode

    # ── Sensor (continuous probe: noise + lag + drift) ────────────
    sensor_noise_std: float = 0.05
    sensor_lag: float = 0.9  # first-order smoothing (higher = more lag)
    sensor_drift_rate: float = 0.001  # random-walk drift per step

    # ── PI controller ─────────────────────────────────────────────
    kp: float = 2.0
    ki: float = 0.5
    kd: float = 0.0
    ff: float = 50.0  # feed-forward bias (resting pump speed)
    output_min: float = 0.0  # min pump command (% or dose units)
    output_max: float = 125.0  # max pump command
    max_integral: float = 200.0  # anti-windup clamp

    # ── Pump actuator ─────────────────────────────────────────────
    max_ramp_up: float = 50.0  # max output increase per hour
    max_ramp_down: float = 50.0  # max output decrease per hour (coast-down)
    startup_delay: float = 0.0  # hours of VFD init when going from off to on


@jax_dataclass
class DosingSystemState:
    sensor_value: jax.Array
    sensor_drift: jax.Array
    pi_integral: jax.Array
    pi_previous_measurement: jax.Array
    pi_initialized: jax.Array
    pump_output: jax.Array
    startup_remaining: jax.Array

    @staticmethod
    def create(initial_pv: float = 0.0, initial_pump: float = 50.0) -> "DosingSystemState":
        return DosingSystemState(
            sensor_value=jnp.array(initial_pv),
            sensor_drift=jnp.array(0.0),
            pi_integral=jnp.array(0.0),
            pi_previous_measurement=jnp.array(0.0),
            pi_initialized=jnp.array(False),
            pump_output=jnp.array(initial_pump),
            startup_remaining=jnp.array(0.0),
        )


def reset(initial_pv: float, initial_pump: float, rng_key: jax.Array) -> DosingSystemState:
    return DosingSystemState.create(initial_pv, initial_pump)


def step(
    state: DosingSystemState,
    action: jax.Array,
    true_pv: jax.Array,
    params: DosingSystemParams,
    dt: jax.Array,
    rng_key: jax.Array,
) -> tuple[DosingSystemState, jax.Array, jax.Array, jax.Array]:
    """Run one cycle of the dosing loop.

    The ``action`` parameter is interpreted based on ``params.control_mode``:
      - DIRECT (0):      action is the raw pump command
      - SUPERVISORY (1): action is the setpoint for the PI controller
      - FEEDFORWARD (2): action is a delta added to the PI output

    Returns:
        (new_state, sensed_pv, pump_output, pi_output)
        - sensed_pv:   what the sensor reports (noisy, lagged, drifted)
        - pump_output: actual pump command after ramp limiting
        - pi_output:   raw PI controller output (before ramp limiting)
    """
    k_drift, k_noise = jax.random.split(rng_key)

    # ── 1. Sensor: noise + drift + first-order lag ────────────────
    drift_step = jax.random.normal(k_drift) * params.sensor_drift_rate
    new_drift = state.sensor_drift + drift_step

    noise = jax.random.normal(k_noise) * params.sensor_noise_std
    raw = true_pv + noise + new_drift

    sensed_pv = params.sensor_lag * state.sensor_value + (1.0 - params.sensor_lag) * raw

    # ── 2. PI controller (always runs) ────────────────────────────
    # In SUPERVISORY mode: action is the setpoint
    # In DIRECT/FEEDFORWARD mode: use fixed base_setpoint
    pi_setpoint = jnp.where(
        params.control_mode == SUPERVISORY,
        action,
        jnp.array(params.base_setpoint),
    )
    if params.kd == 0.0:
        pi_state, pi_result = pi_step(
            PIControllerState(integral=state.pi_integral),
            sensed_pv,
            pi_setpoint,
            PIControllerParams(
                kp=params.kp,
                ki=params.ki,
                ff=params.ff,
                output_min=params.output_min,
                output_max=params.output_max,
                max_integral=params.max_integral,
            ),
            dt,
        )
        new_integral = pi_state.integral
    else:
        pid_state, pi_result = pid_step_with_diagnostics(
            PIDControllerState(
                integral=state.pi_integral,
                previous_measurement=state.pi_previous_measurement,
                initialized=state.pi_initialized,
            ),
            sensed_pv,
            pi_setpoint,
            PIDControllerParams(
                kp=params.kp,
                ki=params.ki,
                kd=params.kd,
                ff=params.ff,
                output_min=params.output_min,
                output_max=params.output_max,
                max_integral=params.max_integral,
            ),
            dt,
        )
        new_integral = pid_state.integral
    pi_output = pi_result.saturated

    # ── 3. Mode-dependent pump command ────────────────────────────
    # DIRECT:      pump = action (raw command)
    # SUPERVISORY: pump = PI output
    # FEEDFORWARD: pump = PI output + action (delta)
    command = jnp.where(
        params.control_mode == DIRECT,
        action,
        jnp.where(
            params.control_mode == SUPERVISORY,
            pi_output,
            pi_output + action,  # FEEDFORWARD
        ),
    )
    command = jnp.clip(command, params.output_min, params.output_max)

    # ── 4. Startup delay (VFD init when going off → on) ─────────
    was_off = state.pump_output <= params.output_min
    wants_on = command > params.output_min
    timer_expired = state.startup_remaining <= 0.0
    new_startup = jnp.where(
        was_off & wants_on & timer_expired,
        jnp.array(params.startup_delay),
        jnp.maximum(state.startup_remaining - dt, 0.0),
    )
    in_startup = new_startup > 0.0
    effective_command = jnp.where(in_startup, params.output_min, command)

    # ── 5. Asymmetric ramp-rate limiting ──────────────────────────
    delta = effective_command - state.pump_output
    max_up = params.max_ramp_up * dt
    max_down = params.max_ramp_down * dt
    clamped_delta = jnp.where(
        delta > 0,
        jnp.minimum(delta, max_up),
        jnp.maximum(delta, -max_down),
    )
    new_pump = jnp.clip(state.pump_output + clamped_delta, params.output_min, params.output_max)

    new_state = DosingSystemState(
        sensor_value=sensed_pv,
        sensor_drift=new_drift,
        pi_integral=new_integral,
        pi_previous_measurement=sensed_pv,
        pi_initialized=jnp.array(True),
        pump_output=new_pump,
        startup_remaining=new_startup,
    )
    return new_state, sensed_pv, new_pump, pi_output
