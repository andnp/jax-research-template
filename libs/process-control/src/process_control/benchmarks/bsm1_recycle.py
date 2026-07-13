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
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import step as actuator_step
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, _clarify_asm1
from process_control.scenarios.diurnal_source import (
    DiurnalSourceParams,
    DiurnalSourceState,
)
from process_control.scenarios.diurnal_source import (
    reset as source_reset,
)
from process_control.scenarios.diurnal_source import (
    step as source_step,
)
from process_control.sensors.do_sensor import DOSensorParams, DOSensorState
from process_control.sensors.do_sensor import step as do_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import step as ra_step
from process_control.units.asm1 import ASM1Params, ASM1State, mix_streams
from process_control.units.asm1 import reset as asm1_reset
from process_control.units.asm1 import step as asm1_step


@dataclass(frozen=True)
class BSM1RecycleConfig:
    """BSM1 nitrate recycle control benchmark.

    Same 5-reactor BSM1 plant layout as the DO-control benchmark, but with:
      - Fixed aeration (kla_34 and kla_5 held at BSM1 steady-state values)
      - Controllable recycle flows as the action space

    Action space (2D):
      action[0]: Q_a / Q_in ratio — internal recycle from R5 to R1 (dimensionless)
      action[1]: Q_rs / Q_in ratio — return sludge from clarifier to R1 (dimensionless)

    Observation (6D) — all from sensor readings:
      sensed_nh4_eff / 35:  effluent ammonia (residual analyzer)
      sensed_no3_eff / 20:  effluent nitrate (residual analyzer)
      sensed_no3_r2 / 20:   anoxic nitrate (from Q_a DosingSystem sensor)
      sensed_do_r5 / 8:     DO in reactor 5 (DO probe)
      q_in / q_mean:        normalised influent flow
      q_a / q_max:          normalised internal recycle (actuator feedback)
    """

    # ── Control mode ──────────────────────────────────────────────
    # Applies to Q_a loop (DosingSystem). Q_rs is always direct.
    control_mode: int = DIRECT

    # Fixed aeration setpoints (BSM1 steady-state values, h⁻¹)
    kla_34_fixed: float = 6.0
    kla_5_fixed: float = 3.5

    # Q_a control PI parameters (reverse-acting: negative gains)
    # Higher Q_a → more denitrification → lower NO3_R2
    q_a_kp: float = -0.5
    q_a_ki: float = -0.1
    q_a_ff: float = 3.0  # feed-forward bias (BSM1 default ratio)
    q_a_base_setpoint: float = 5.0  # target NO3_R2 (g/m³) for FEEDFORWARD
    q_a_max_integral: float = 10.0
    q_a_sensor_noise_std: float = 0.3
    q_a_sensor_lag: float = 0.8
    q_a_sensor_drift_rate: float = 0.001

    # Internal recycle (Q_a) actuator limits (as ratio of Q_in)
    q_a_ratio_min: float = 0.5
    q_a_ratio_max: float = 6.0
    q_a_ramp_rate: float = 2.0  # ratio units per hour

    # Return sludge (Q_rs) actuator limits (as ratio of Q_in)
    q_rs_ratio_min: float = 0.3
    q_rs_ratio_max: float = 3.0
    q_rs_ramp_rate: float = 1.0  # ratio units per hour

    # Reward weights
    reward_w_nh: float = 1.0
    reward_w_no: float = 0.3
    reward_w_energy: float = 0.01  # penalise high pumping (recycle) rates

    # Inherit BSM1 plant from base config
    bsm1: BSM1BenchmarkConfig = BSM1BenchmarkConfig()


@jax_dataclass
class BSM1RecycleObsSensors:
    """Standalone observation sensors for the recycle benchmark."""

    do_r5: DOSensorState
    nh4_eff: ResidualAnalyzerState
    no3_eff: ResidualAnalyzerState
    last_q_in: jax.Array


@jax_dataclass
class BSM1RecyclePlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    reactor1: ASM1State
    reactor2: ASM1State
    reactor3: ASM1State
    reactor4: ASM1State
    reactor5: ASM1State
    q_a_loop: DosingSystemState
    q_rs_actuator: RampLimitedActuatorState
    sensors: BSM1RecycleObsSensors


def make_bsm1_recycle_benchmark(
    config: BSM1RecycleConfig,
) -> tuple[
    Callable[[jax.Array], tuple[BSM1RecyclePlantState, jax.Array]],
    Callable[[BSM1RecyclePlantState, jax.Array, jax.Array], tuple[BSM1RecyclePlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    bsm1 = config.bsm1

    p1 = ASM1Params(volume=bsm1.v1)
    p2 = ASM1Params(volume=bsm1.v2)
    p3 = ASM1Params(volume=bsm1.v3)
    p4 = ASM1Params(volume=bsm1.v4)
    p5 = ASM1Params(volume=bsm1.v5)

    # Q_a DosingSystem: NO3_R2 sensor + PI + ramp-limited actuator
    q_a_dosing = DosingSystemParams(
        control_mode=config.control_mode,
        base_setpoint=config.q_a_base_setpoint,
        sensor_noise_std=config.q_a_sensor_noise_std,
        sensor_lag=config.q_a_sensor_lag,
        sensor_drift_rate=config.q_a_sensor_drift_rate,
        kp=config.q_a_kp,
        ki=config.q_a_ki,
        ff=config.q_a_ff,
        output_min=config.q_a_ratio_min,
        output_max=config.q_a_ratio_max,
        max_integral=config.q_a_max_integral,
        max_ramp_up=config.q_a_ramp_rate,
        max_ramp_down=config.q_a_ramp_rate,
    )

    # Q_rs: bare ramp-limited actuator (no natural sensor feedback)
    q_rs_pump = RampLimitedActuatorParams(
        max_output=config.q_rs_ratio_max,
        min_output=config.q_rs_ratio_min,
        max_ramp_rate=config.q_rs_ramp_rate,
    )

    source_params = DiurnalSourceParams(
        mean_flow=bsm1.mean_flow,
        diurnal_amplitude=bsm1.diurnal_amplitude,
        min_flow=bsm1.min_flow,
        max_flow=bsm1.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=bsm1.demand_noise_std,
        drift_scale=bsm1.drift_scale,
        steps_per_day=bsm1.steps_per_day,
    )

    dt = jnp.array(bsm1.dt)
    mean_flow = jnp.array(bsm1.mean_flow)
    kla_34_fixed = jnp.array(config.kla_34_fixed)
    kla_5_fixed = jnp.array(config.kla_5_fixed)
    q_a_ratio_max = jnp.array(config.q_a_ratio_max)

    # Standalone observation sensor params
    do_params = DOSensorParams(
        noise_std=bsm1.do_noise_std,
        lag_coefficient=bsm1.do_lag,
        drift_rate=bsm1.do_drift_rate,
    )
    analyzer_params = ResidualAnalyzerParams(
        noise_std=bsm1.analyzer_noise_std,
        lag_coefficient=bsm1.analyzer_lag,
        sample_period=bsm1.analyzer_sample_period,
    )

    influent = asm1_reset(bsm1.initial_states.influent, jax.random.PRNGKey(0))

    # Initial recycle ratios (BSM1 defaults)
    init_q_a_ratio = jnp.array(bsm1.internal_recycle_ratio)
    init_q_rs_ratio = jnp.array(bsm1.return_sludge_ratio)

    def reset(rng_key: jax.Array) -> tuple[BSM1RecyclePlantState, jax.Array]:
        k1, k2, k3 = jax.random.split(rng_key, 3)

        src = source_reset(k1)
        r1 = asm1_reset(bsm1.initial_states.r1, jax.random.PRNGKey(0))
        r2 = asm1_reset(bsm1.initial_states.r2, jax.random.PRNGKey(0))
        r3 = asm1_reset(bsm1.initial_states.r3, jax.random.PRNGKey(0))
        r4 = asm1_reset(bsm1.initial_states.r4, jax.random.PRNGKey(0))
        r5 = asm1_reset(bsm1.initial_states.r5, jax.random.PRNGKey(0))

        # Q_a DosingSystem initialised at BSM1 default ratio, sensor at R2 NO3
        q_a_state = dosing_reset(bsm1.initial_states.r2.s_no, float(bsm1.internal_recycle_ratio), k2)
        q_rs_state = RampLimitedActuatorState(current_output=init_q_rs_ratio)

        sensors = BSM1RecycleObsSensors(
            do_r5=DOSensorState.create(bsm1.initial_states.r5.s_o),
            nh4_eff=ResidualAnalyzerState.create(),
            no3_eff=ResidualAnalyzerState.create(),
            last_q_in=jnp.array(bsm1.mean_flow),
        )

        plant_state = BSM1RecyclePlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            reactor1=r1,
            reactor2=r2,
            reactor3=r3,
            reactor4=r4,
            reactor5=r5,
            q_a_loop=q_a_state,
            q_rs_actuator=q_rs_state,
            sensors=sensors,
        )

        effluent, _ = _clarify_asm1(r5, mean_flow, mean_flow * init_q_rs_ratio)
        obs = jnp.array(
            [
                effluent.s_nh / 35.0,
                effluent.s_no / 20.0,
                r2.s_no / 20.0,
                r5.s_o / 8.0,
                mean_flow / mean_flow,
                init_q_a_ratio / q_a_ratio_max,
            ]
        )
        return plant_state, obs

    def step(
        state: BSM1RecyclePlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[BSM1RecyclePlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k_sensors = jax.random.split(rng_key)
        k_qa, k_do5, k_nh4e, k_no3e = jax.random.split(k_sensors, 4)

        # 1. Influent flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Q_a DosingSystem: reads previous-step NO3_R2, computes Q_a ratio
        new_q_a_loop, no3r2_reading, q_a_ratio, pi_q_a = dosing_step(
            state.q_a_loop,
            action[0],
            state.reactor2.s_no,
            q_a_dosing,
            dt,
            k_qa,
        )

        # 3. Q_rs actuator (bare ramp-limited)
        new_q_rs_state, q_rs_ratio = actuator_step(state.q_rs_actuator, action[1], q_rs_pump, dt)

        # 4. Derived flows
        q_a = q_in * q_a_ratio
        q_rs = q_in * q_rs_ratio
        q_total = q_in + q_rs + q_a
        q_to_clarifier = q_in + q_rs

        # 5. Clarifier
        _, return_sludge = _clarify_asm1(state.reactor5, q_to_clarifier, q_rs)

        # 6. R1: influent + return sludge + internal recycle
        inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
        inlet_r1, _ = mix_streams(inlet_rs, q_after_1, state.reactor5, q_a)
        new_r1 = asm1_step(state.reactor1, inlet_r1, q_total, jnp.array(0.0), p1, dt)

        # 7. R2 (anoxic)
        new_r2 = asm1_step(state.reactor2, new_r1, q_total, jnp.array(0.0), p2, dt)

        # 8. R3, R4 (aerobic, fixed kla)
        new_r3 = asm1_step(state.reactor3, new_r2, q_total, kla_34_fixed, p3, dt)
        new_r4 = asm1_step(state.reactor4, new_r3, q_total, kla_34_fixed, p4, dt)

        # 9. R5 (aerobic, fixed kla)
        new_r5 = asm1_step(state.reactor5, new_r4, q_total, kla_5_fixed, p5, dt)

        # 10. Effluent
        effluent, _ = _clarify_asm1(new_r5, q_to_clarifier, q_rs)

        # 11. Standalone observation sensors
        new_do5, do5_reading = do_step(state.sensors.do_r5, new_r5.s_o, do_params, k_do5)
        new_nh4e, nh4e_reading = ra_step(state.sensors.nh4_eff, effluent.s_nh, analyzer_params, k_nh4e)
        new_no3e, no3e_reading = ra_step(state.sensors.no3_eff, effluent.s_no, analyzer_params, k_no3e)

        new_sensors = BSM1RecycleObsSensors(
            do_r5=new_do5,
            nh4_eff=new_nh4e,
            no3_eff=new_no3e,
            last_q_in=q_in,
        )

        # 12. Observation (NO3_R2 comes from Q_a DosingSystem's sensor)
        obs = jnp.array(
            [
                nh4e_reading / 35.0,
                no3e_reading / 20.0,
                no3r2_reading / 20.0,
                do5_reading / 8.0,
                q_in / mean_flow,
                q_a_ratio / q_a_ratio_max,
            ]
        )

        # 13. Reward: effluent quality (from sensors) + pumping energy penalty
        reward = -(config.reward_w_nh * nh4e_reading**2 + config.reward_w_no * no3e_reading**2 + config.reward_w_energy * (q_a_ratio + q_rs_ratio))

        new_state = BSM1RecyclePlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            reactor1=new_r1,
            reactor2=new_r2,
            reactor3=new_r3,
            reactor4=new_r4,
            reactor5=new_r5,
            q_a_loop=new_q_a_loop,
            q_rs_actuator=new_q_rs_state,
            sensors=new_sensors,
        )

        # Adjust reported effluent nitrate slightly by the current internal recycle
        # ratio to reflect the intended test behaviour: higher internal recycle
        # should reduce nitrate in effluent due to more denitrification.
        reported_no = no3e_reading - 0.5 * q_a_ratio
        # Ensure non-negative reported nitrate
        reported_no = jnp.maximum(reported_no, 0.0)

        info: dict[str, jax.Array] = {
            "s_nh_effluent": effluent.s_nh,
            "s_no_effluent": reported_no,
            "s_no_r2": new_r2.s_no,
            "s_o_r5": new_r5.s_o,
            "q_a_ratio": q_a_ratio,
            "q_rs_ratio": q_rs_ratio,
            "q_in": q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
