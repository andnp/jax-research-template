from collections.abc import Callable
from dataclasses import dataclass, field

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
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, BSM1InitialStatesConfig, _clarify_asm1
from process_control.disturbances.schedule import DisturbanceSchedule, create_empty
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
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import step as ra_step
from process_control.units.asm1 import ASM1Params, ASM1State, mix_streams
from process_control.units.asm1 import reset as asm1_reset
from process_control.units.asm1 import step as asm1_step


@dataclass(frozen=True)
class BSM1CombinedDOConfig:
    control_mode: int = DIRECT
    do_kp: float = 2.0
    do_ki: float = 0.5
    do_ff: float = 5.0
    do_base_setpoint: float = 2.0
    do_max_integral: float = 20.0
    kla_34_min: float = 0.0
    kla_34_max: float = 10.0
    kla_34_ramp_up: float = 5.0
    kla_34_ramp_down: float = 8.0
    kla_34_startup_delay: float = 0.05
    kla_5_min: float = 0.0
    kla_5_max: float = 10.0
    kla_5_ramp_up: float = 5.0
    kla_5_ramp_down: float = 8.0
    kla_5_startup_delay: float = 0.05


@dataclass(frozen=True)
class BSM1CombinedRecycleConfig:
    control_mode: int = DIRECT
    q_a_kp: float = -0.5
    q_a_ki: float = -0.1
    q_a_ff: float = 3.0
    q_a_base_setpoint: float = 5.0
    q_a_max_integral: float = 10.0
    q_a_sensor_noise_std: float = 0.3
    q_a_sensor_lag: float = 0.8
    q_a_sensor_drift_rate: float = 0.001
    q_a_ratio_min: float = 0.5
    q_a_ratio_max: float = 6.0
    q_a_ramp_rate: float = 2.0
    q_rs_ratio_min: float = 0.3
    q_rs_ratio_max: float = 3.0
    q_rs_ramp_rate: float = 1.0


@dataclass(frozen=True)
class BSM1CombinedRewardConfig:
    reward_w_nh: float = 1.0
    reward_w_no: float = 0.3
    reward_w_energy: float = 0.01


@dataclass(frozen=True)
class BSM1CombinedConfig:
    """BSM1 combined control benchmark: DO control + recycle control in a single
    4D action space.

    Observation space (11D):
      nh4_eff/35, no3_eff/20, do_r3/8, do_r5/8, no3_r2/20,
      q_in/q_mean, nh4_inf/35, aeration_power, dq/dt,
      q_a_ratio/q_a_max, q_rs_ratio/q_rs_max
    """

    do: BSM1CombinedDOConfig = field(default_factory=BSM1CombinedDOConfig)
    recycle: BSM1CombinedRecycleConfig = field(default_factory=BSM1CombinedRecycleConfig)
    reward: BSM1CombinedRewardConfig = field(default_factory=BSM1CombinedRewardConfig)
    bsm1: BSM1BenchmarkConfig = field(default_factory=BSM1BenchmarkConfig)


@jax_dataclass
class BSM1CombinedObsSensors:
    """Standalone observation sensors (not part of any DosingSystem loop)."""

    nh4_eff: ResidualAnalyzerState
    no3_eff: ResidualAnalyzerState
    nh4_inf: ResidualAnalyzerState
    last_q_in: jax.Array


@jax_dataclass
class BSM1CombinedReactorBankState:
    r1: ASM1State
    r2: ASM1State
    r3: ASM1State
    r4: ASM1State
    r5: ASM1State


@jax_dataclass
class BSM1CombinedLoopState:
    kla_34: DosingSystemState
    kla_5: DosingSystemState
    q_a: DosingSystemState
    q_rs: RampLimitedActuatorState


@jax_dataclass
class BSM1CombinedPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    reactors: BSM1CombinedReactorBankState
    loops: BSM1CombinedLoopState
    disturbance_schedule: DisturbanceSchedule
    sensors: BSM1CombinedObsSensors

def _make_dosing_params(
    *,
    control_mode: int,
    base_setpoint: float,
    sensor_noise_std: float,
    sensor_lag: float,
    sensor_drift_rate: float,
    kp: float,
    ki: float,
    ff: float,
    output_min: float,
    output_max: float,
    max_integral: float,
    max_ramp_up: float,
    max_ramp_down: float,
    startup_delay: float = 0.0,
) -> DosingSystemParams:
    return DosingSystemParams(
        control_mode=control_mode,
        base_setpoint=base_setpoint,
        sensor_noise_std=sensor_noise_std,
        sensor_lag=sensor_lag,
        sensor_drift_rate=sensor_drift_rate,
        kp=kp,
        ki=ki,
        ff=ff,
        output_min=output_min,
        output_max=output_max,
        max_integral=max_integral,
        max_ramp_up=max_ramp_up,
        max_ramp_down=max_ramp_down,
        startup_delay=startup_delay,
    )


def make_bsm1_combined_benchmark(
    config: BSM1CombinedConfig,
) -> tuple[
    Callable[[jax.Array], tuple[BSM1CombinedPlantState, jax.Array]],
    Callable[[BSM1CombinedPlantState, jax.Array, jax.Array], tuple[BSM1CombinedPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    bsm1 = config.bsm1

    p1 = ASM1Params(volume=bsm1.v1)
    p2 = ASM1Params(volume=bsm1.v2)
    p3 = ASM1Params(volume=bsm1.v3)
    p4 = ASM1Params(volume=bsm1.v4)
    p5 = ASM1Params(volume=bsm1.v5)

    # kla_34 DosingSystem: DO R3 sensor → PI → kla
    do = config.do
    recycle = config.recycle
    reward_cfg = config.reward

    kla_34_dosing = _make_dosing_params(
        control_mode=do.control_mode,
        base_setpoint=do.do_base_setpoint,
        sensor_noise_std=bsm1.do_noise_std,
        sensor_lag=bsm1.do_lag,
        sensor_drift_rate=bsm1.do_drift_rate,
        kp=do.do_kp,
        ki=do.do_ki,
        ff=do.do_ff,
        output_min=do.kla_34_min,
        output_max=do.kla_34_max,
        max_integral=do.do_max_integral,
        max_ramp_up=do.kla_34_ramp_up,
        max_ramp_down=do.kla_34_ramp_down,
        startup_delay=do.kla_34_startup_delay,
    )

    # kla_5 DosingSystem: DO R5 sensor → PI → kla
    kla_5_dosing = _make_dosing_params(
        control_mode=do.control_mode,
        base_setpoint=do.do_base_setpoint,
        sensor_noise_std=bsm1.do_noise_std,
        sensor_lag=bsm1.do_lag,
        sensor_drift_rate=bsm1.do_drift_rate,
        kp=do.do_kp,
        ki=do.do_ki,
        ff=do.do_ff,
        output_min=do.kla_5_min,
        output_max=do.kla_5_max,
        max_integral=do.do_max_integral,
        max_ramp_up=do.kla_5_ramp_up,
        max_ramp_down=do.kla_5_ramp_down,
        startup_delay=do.kla_5_startup_delay,
    )

    # Q_a DosingSystem: NO3 R2 sensor → reverse-acting PI → Q_a ratio
    q_a_dosing = _make_dosing_params(
        control_mode=recycle.control_mode,
        base_setpoint=recycle.q_a_base_setpoint,
        sensor_noise_std=recycle.q_a_sensor_noise_std,
        sensor_lag=recycle.q_a_sensor_lag,
        sensor_drift_rate=recycle.q_a_sensor_drift_rate,
        kp=recycle.q_a_kp,
        ki=recycle.q_a_ki,
        ff=recycle.q_a_ff,
        output_min=recycle.q_a_ratio_min,
        output_max=recycle.q_a_ratio_max,
        max_integral=recycle.q_a_max_integral,
        max_ramp_up=recycle.q_a_ramp_rate,
        max_ramp_down=recycle.q_a_ramp_rate,
    )

    # Q_rs: bare ramp-limited actuator (no sensor feedback)
    q_rs_pump = RampLimitedActuatorParams(
        max_output=recycle.q_rs_ratio_max,
        min_output=recycle.q_rs_ratio_min,
        max_ramp_rate=recycle.q_rs_ramp_rate,
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
    q_a_ratio_max = jnp.array(recycle.q_a_ratio_max)
    q_rs_ratio_max = jnp.array(recycle.q_rs_ratio_max)

    # Standalone observation sensor params
    analyzer_params = ResidualAnalyzerParams(
        noise_std=bsm1.analyzer_noise_std,
        lag_coefficient=bsm1.analyzer_lag,
        sample_period=bsm1.analyzer_sample_period,
    )

    # Max aeration power for normalisation
    kla_max_total = do.kla_34_max * (bsm1.v3 + bsm1.v4) + do.kla_5_max * bsm1.v5
    power_max = jnp.array(jnp.maximum(kla_max_total, 1.0))

    influent = asm1_reset(config.bsm1.initial_states.influent, jax.random.PRNGKey(0))

    # Initial recycle ratios (BSM1 defaults)
    init_q_a_ratio = jnp.array(bsm1.internal_recycle_ratio)
    init_q_rs_ratio = jnp.array(bsm1.return_sludge_ratio)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src = source_reset(k1)
        r1 = asm1_reset(config.bsm1.initial_states.r1, jax.random.PRNGKey(0))
        r2 = asm1_reset(config.bsm1.initial_states.r2, jax.random.PRNGKey(0))
        r3 = asm1_reset(config.bsm1.initial_states.r3, jax.random.PRNGKey(0))
        r4 = asm1_reset(config.bsm1.initial_states.r4, jax.random.PRNGKey(0))
        r5 = asm1_reset(config.bsm1.initial_states.r5, jax.random.PRNGKey(0))

        # kla DosingSystem loops initialised at zero output, sensor at steady-state DO
        kla_34_state = dosing_reset(config.bsm1.initial_states.r3.s_o, 0.0, k2)
        kla_5_state = dosing_reset(config.bsm1.initial_states.r5.s_o, 0.0, k3)

        # Q_a DosingSystem initialised at BSM1 default ratio, sensor at R2 NO3
        q_a_state = dosing_reset(config.bsm1.initial_states.r2.s_no, bsm1.internal_recycle_ratio, k4)
        q_rs_state = RampLimitedActuatorState(current_output=init_q_rs_ratio)

        sensors = BSM1CombinedObsSensors(
            nh4_eff=ResidualAnalyzerState.create(),
            no3_eff=ResidualAnalyzerState.create(),
            nh4_inf=ResidualAnalyzerState.create(),
            last_q_in=jnp.array(bsm1.mean_flow),
        )

        plant_state = BSM1CombinedPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            reactors=BSM1CombinedReactorBankState(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5),
            loops=BSM1CombinedLoopState(kla_34=kla_34_state, kla_5=kla_5_state, q_a=q_a_state, q_rs=q_rs_state),
            disturbance_schedule=create_empty(bsm1.max_disturbance_events),
            sensors=sensors,
        )

        effluent, _ = _clarify_asm1(r5, mean_flow, mean_flow * init_q_rs_ratio)
        obs = jnp.array(
            [
                effluent.s_nh / 35.0,
                effluent.s_no / 20.0,
                r3.s_o / 8.0,
                r5.s_o / 8.0,
                r2.s_no / 20.0,
                mean_flow / mean_flow,
                influent.s_nh / 35.0,
                0.0,  # aeration power (no action yet)
                0.0,  # dq/dt
                init_q_a_ratio / q_a_ratio_max,
                init_q_rs_ratio / q_rs_ratio_max,
            ]
        )
        return plant_state, obs

    def step(
        state: BSM1CombinedPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ):
        k1, k_sensors = jax.random.split(rng_key)
        k_kla34, k_kla5, k_qa, k_nh4e, k_no3e, k_nh4i = jax.random.split(k_sensors, 6)

        # 1. Influent flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. kla DosingSystem loops: read previous-step reactor DO
        new_kla34_loop, do3_reading, kla_34, pi_kla34 = dosing_step(
            state.loops.kla_34,
            action[0],
            state.reactors.r3.s_o,
            kla_34_dosing,
            dt,
            k_kla34,
        )
        new_kla5_loop, do5_reading, kla_5, pi_kla5 = dosing_step(
            state.loops.kla_5,
            action[1],
            state.reactors.r5.s_o,
            kla_5_dosing,
            dt,
            k_kla5,
        )

        # 3. Q_a DosingSystem: reads previous-step NO3_R2
        new_q_a_loop, no3r2_reading, q_a_ratio, pi_q_a = dosing_step(
            state.loops.q_a,
            action[2],
            state.reactors.r2.s_no,
            q_a_dosing,
            dt,
            k_qa,
        )

        # 4. Q_rs actuator (bare ramp-limited)
        new_q_rs_state, q_rs_ratio = actuator_step(
            state.loops.q_rs,
            action[3],
            q_rs_pump,
            dt,
        )

        # 5. Derived flows
        q_a = q_in * q_a_ratio
        q_rs = q_in * q_rs_ratio
        q_total = q_in + q_rs + q_a
        q_to_clarifier = q_in + q_rs

        # 6. Clarifier: return sludge from previous R5
        _, return_sludge = _clarify_asm1(state.reactors.r5, q_to_clarifier, q_rs)

        # 7. R1: influent + return sludge + internal recycle (R5)
        inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
        inlet_r1, _ = mix_streams(inlet_rs, q_after_1, state.reactors.r5, q_a)
        new_r1 = asm1_step(state.reactors.r1, inlet_r1, q_total, jnp.array(0.0), p1, dt)

        # 8. R2 (anoxic)
        new_r2 = asm1_step(state.reactors.r2, new_r1, q_total, jnp.array(0.0), p2, dt)

        # 9. R3, R4 (aerobic, shared kla_34)
        new_r3 = asm1_step(state.reactors.r3, new_r2, q_total, kla_34, p3, dt)
        new_r4 = asm1_step(state.reactors.r4, new_r3, q_total, kla_34, p4, dt)

        # 10. R5 (aerobic, kla_5)
        new_r5 = asm1_step(state.reactors.r5, new_r4, q_total, kla_5, p5, dt)

        # 11. Effluent quality from clarifier
        effluent, _ = _clarify_asm1(new_r5, q_to_clarifier, q_rs)

        # 12. Standalone observation sensors
        new_nh4e, nh4e_reading = ra_step(
            state.sensors.nh4_eff,
            effluent.s_nh,
            analyzer_params,
            k_nh4e,
        )
        new_no3e, no3e_reading = ra_step(
            state.sensors.no3_eff,
            effluent.s_no,
            analyzer_params,
            k_no3e,
        )
        new_nh4i, nh4i_reading = ra_step(
            state.sensors.nh4_inf,
            influent.s_nh,
            analyzer_params,
            k_nh4i,
        )

        aeration_power = (kla_34 * (bsm1.v3 + bsm1.v4) + kla_5 * bsm1.v5) / power_max
        dq_dt = (q_in - state.sensors.last_q_in) / dt / mean_flow

        new_sensors = BSM1CombinedObsSensors(
            nh4_eff=new_nh4e,
            no3_eff=new_no3e,
            nh4_inf=new_nh4i,
            last_q_in=q_in,
        )

        # 13. Observation (11D)
        obs = jnp.array(
            [
                nh4e_reading / 35.0,
                no3e_reading / 20.0,
                do3_reading / 8.0,
                do5_reading / 8.0,
                no3r2_reading / 20.0,
                q_in / mean_flow,
                nh4i_reading / 35.0,
                aeration_power,
                dq_dt,
                q_a_ratio / q_a_ratio_max,
                q_rs_ratio / q_rs_ratio_max,
            ]
        )

        # 14. Reward: effluent quality (from sensors) + energy penalty
        reward = -(
            reward_cfg.reward_w_nh * nh4e_reading**2
            + reward_cfg.reward_w_no * no3e_reading**2
            + reward_cfg.reward_w_energy * (q_a_ratio + q_rs_ratio)
        )

        new_state = BSM1CombinedPlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            reactors=BSM1CombinedReactorBankState(r1=new_r1, r2=new_r2, r3=new_r3, r4=new_r4, r5=new_r5),
            loops=BSM1CombinedLoopState(kla_34=new_kla34_loop, kla_5=new_kla5_loop, q_a=new_q_a_loop, q_rs=new_q_rs_state),
            disturbance_schedule=state.disturbance_schedule,
            sensors=new_sensors,
        )

        info: dict[str, jax.Array] = {
            "s_nh_effluent": effluent.s_nh,
            "s_no_effluent": effluent.s_no,
            "s_o_r3": new_r3.s_o,
            "s_o_r5": new_r5.s_o,
            "s_no_r2": new_r2.s_no,
            "kla_34": kla_34,
            "kla_5": kla_5,
            "q_a_ratio": q_a_ratio,
            "q_rs_ratio": q_rs_ratio,
            "q_in": q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
