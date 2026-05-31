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

# ── Control integration modes ────────────────────────────────────────
DIRECT: int = 0
SUPERVISORY: int = 1
FEEDFORWARD: int = 2


@dataclass(frozen=True)
class DosingSystemParams:
    # ── Control mode ──────────────────────────────────────────────
    control_mode: int = 1         # DIRECT=0, SUPERVISORY=1, FEEDFORWARD=2
    base_setpoint: float = 0.0    # fixed setpoint used in FEEDFORWARD mode

    # ── Sensor (continuous probe: noise + lag + drift) ────────────
    sensor_noise_std: float = 0.05
    sensor_lag: float = 0.9       # first-order smoothing (higher = more lag)
    sensor_drift_rate: float = 0.001  # random-walk drift per step

    # ── PI controller ─────────────────────────────────────────────
    kp: float = 2.0
    ki: float = 0.5
    ff: float = 50.0              # feed-forward bias (resting pump speed)
    output_min: float = 0.0       # min pump command (% or dose units)
    output_max: float = 125.0     # max pump command
    max_integral: float = 200.0   # anti-windup clamp

    # ── Pump actuator ─────────────────────────────────────────────
    max_ramp_rate: float = 50.0   # output units per hour


@dataclass(frozen=True)
class DosingSystemState:
    sensor_value: jax.Array
    sensor_drift: jax.Array
    pi_integral: jax.Array
    pump_output: jax.Array

    @staticmethod
    def create(initial_pv: float = 0.0, initial_pump: float = 50.0) -> "DosingSystemState":
        return DosingSystemState(
            sensor_value=jnp.array(initial_pv),
            sensor_drift=jnp.array(0.0),
            pi_integral=jnp.array(0.0),
            pump_output=jnp.array(initial_pump),
        )


jax.tree_util.register_dataclass(
    DosingSystemState,
    data_fields=["sensor_value", "sensor_drift", "pi_integral", "pump_output"],
    meta_fields=[],
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
    error = pi_setpoint - sensed_pv
    new_integral = jnp.clip(
        state.pi_integral + error * dt,
        -params.max_integral,
        params.max_integral,
    )
    pi_output = params.kp * error + params.ki * new_integral + params.ff
    pi_output = jnp.clip(pi_output, params.output_min, params.output_max)

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

    # ── 4. Pump: ramp-rate limiting ───────────────────────────────
    max_change = params.max_ramp_rate * dt
    delta = jnp.clip(command - state.pump_output, -max_change, max_change)
    new_pump = jnp.clip(state.pump_output + delta, params.output_min, params.output_max)

    new_state = DosingSystemState(
        sensor_value=sensed_pv,
        sensor_drift=new_drift,
        pi_integral=new_integral,
        pump_output=new_pump,
    )
    return new_state, sensed_pv, new_pump, pi_output
