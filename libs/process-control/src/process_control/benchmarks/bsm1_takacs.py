"""BSM1 with Takács 10-layer secondary clarifier.

Replaces the perfect settler in the standard BSM1 with the Takács
double-exponential settling model. This adds realistic sludge dynamics:
- sludge blanket that rises under high hydraulic load
- solids washout risk during storms
- SRT management through waste sludge control

Action (3D): [kla_34, kla_5, Q_w / Q_w_max]
Observation (12D):
  NH₄ effluent / 35
  NO₃ effluent / 20
  DO R3 / 8
  DO R5 / 8
  NO₃ R2 / 20
  Q_in / Q_mean
  NH₄ influent / 35
  aeration_power
  dQ/dt
  blanket_height / depth
  effluent_TSS / 100
  underflow_TSS / 10000
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.dosing_system import DosingSystemParams, DosingSystemState
from process_control.actuators.dosing_system import reset as dosing_reset
from process_control.actuators.dosing_system import step as dosing_step
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import reset as pump_reset
from process_control.actuators.ramp_limited import step as pump_step
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig
from process_control.disturbances.schedule import DisturbanceSchedule, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import step as ra_step
from process_control.units.asm1 import ASM1Params, ASM1State, mix_streams
from process_control.units.asm1 import reset as asm1_reset
from process_control.units.asm1 import step as asm1_step
from process_control.units.takacs_settler import (
    TakacsSettlerParams,
    TakacsSettlerState,
    compute_blanket_height,
    get_effluent_tss,
    get_underflow_tss,
)
from process_control.units.takacs_settler import reset as settler_reset
from process_control.units.takacs_settler import step as settler_step

# COD-to-TSS conversion factor (BSM1 convention)
COD_TO_TSS = 0.75


def _compute_tss(state: ASM1State):
    """Total suspended solids from ASM1 particulate components."""
    return COD_TO_TSS * (state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p)


def _return_sludge_from_tss(
    r5: ASM1State,
    feed_tss: jax.Array,
    underflow_tss: jax.Array,
):
    """Create ASM1 return sludge state by scaling particulates proportionally."""
    ratio = underflow_tss / (feed_tss + 1e-10)
    return ASM1State(
        s_i=r5.s_i,
        s_s=r5.s_s,
        x_i=r5.x_i * ratio,
        x_s=r5.x_s * ratio,
        x_bh=r5.x_bh * ratio,
        x_ba=r5.x_ba * ratio,
        x_p=r5.x_p * ratio,
        s_o=r5.s_o,
        s_no=r5.s_no,
        s_nh=r5.s_nh,
        s_nd=r5.s_nd,
        x_nd=r5.x_nd * ratio,
        s_alk=r5.s_alk,
    )


def _effluent_from_tss(
    r5: ASM1State,
    feed_tss: jax.Array,
    effluent_tss: jax.Array,
):
    """Create ASM1 effluent state with reduced particulates based on effluent TSS."""
    ratio = effluent_tss / (feed_tss + 1e-10)
    return ASM1State(
        s_i=r5.s_i,
        s_s=r5.s_s,
        x_i=r5.x_i * ratio,
        x_s=r5.x_s * ratio,
        x_bh=r5.x_bh * ratio,
        x_ba=r5.x_ba * ratio,
        x_p=r5.x_p * ratio,
        s_o=r5.s_o,
        s_no=r5.s_no,
        s_nh=r5.s_nh,
        s_nd=r5.s_nd,
        x_nd=r5.x_nd * ratio,
        s_alk=r5.s_alk,
    )


@jax_dataclass
class BSM1TakacsSensors:
    nh4_eff: ResidualAnalyzerState
    no3_eff: ResidualAnalyzerState
    no3_r2: ResidualAnalyzerState
    nh4_inf: ResidualAnalyzerState
    blanket: ResidualAnalyzerState
    eff_tss: ResidualAnalyzerState
    und_tss: ResidualAnalyzerState
    last_q_in: jax.Array


@jax_dataclass
class BSM1TakacsPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    reactor1: ASM1State
    reactor2: ASM1State
    reactor3: ASM1State
    reactor4: ASM1State
    reactor5: ASM1State
    settler_state: TakacsSettlerState
    kla_34_loop: DosingSystemState
    kla_5_loop: DosingSystemState
    waste_pump: RampLimitedActuatorState
    disturbance_schedule: DisturbanceSchedule
    sensors: BSM1TakacsSensors


@dataclass(frozen=True)
class BSM1TakacsConfig:
    bsm1: BSM1BenchmarkConfig = BSM1BenchmarkConfig()

    # Settler
    settler: TakacsSettlerParams = TakacsSettlerParams()

    # Waste sludge
    q_w_min: float = 0.0  # m³/h
    q_w_max: float = 50.0  # m³/h
    q_w_ramp_rate: float = 20.0

    # Sludge blanket sensor
    blanket_threshold: float = 1500.0
    blanket_noise_std: float = 0.1  # m
    blanket_lag: float = 0.8
    blanket_sample_period: int = 5

    # TSS sensors
    tss_noise_std: float = 5.0
    tss_lag: float = 0.7
    tss_sample_period: int = 8

    # Reward additions
    reward_w_tss: float = 0.5
    reward_w_blanket: float = 1.0
    eff_tss_limit: float = 30.0
    blanket_alarm: float = 3.0  # m — blanket height alarm


def make_bsm1_takacs_benchmark(config: BSM1TakacsConfig):
    c = config.bsm1
    p1 = ASM1Params(volume=c.v1)
    p2 = ASM1Params(volume=c.v2)
    p3 = ASM1Params(volume=c.v3)
    p4 = ASM1Params(volume=c.v4)
    p5 = ASM1Params(volume=c.v5)

    kla_34_dosing = DosingSystemParams(
        control_mode=c.control_mode,
        base_setpoint=c.do_base_setpoint,
        sensor_noise_std=c.do_noise_std,
        sensor_lag=c.do_lag,
        sensor_drift_rate=c.do_drift_rate,
        kp=c.do_kp,
        ki=c.do_ki,
        ff=c.do_ff,
        output_min=c.kla_34_min,
        output_max=c.kla_34_max,
        max_integral=c.do_max_integral,
    )
    kla_5_dosing = DosingSystemParams(
        control_mode=c.control_mode,
        base_setpoint=c.do_base_setpoint,
        sensor_noise_std=c.do_noise_std,
        sensor_lag=c.do_lag,
        sensor_drift_rate=c.do_drift_rate,
        kp=c.do_kp,
        ki=c.do_ki,
        ff=c.do_ff,
        output_min=c.kla_5_min,
        output_max=c.kla_5_max,
        max_integral=c.do_max_integral,
    )
    waste_pump_params = RampLimitedActuatorParams(
        max_output=config.q_w_max,
        min_output=config.q_w_min,
        max_ramp_rate=config.q_w_ramp_rate,
    )

    source_params = DiurnalSourceParams(
        mean_flow=c.mean_flow,
        diurnal_amplitude=c.diurnal_amplitude,
        min_flow=c.min_flow,
        max_flow=c.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=0.0,
        drift_scale=c.drift_scale,
        steps_per_day=c.steps_per_day,
    )

    analyzer_params = ResidualAnalyzerParams(
        noise_std=c.analyzer_noise_std,
        lag_coefficient=c.analyzer_lag,
        sample_period=c.analyzer_sample_period,
    )
    blanket_analyzer = ResidualAnalyzerParams(
        noise_std=config.blanket_noise_std,
        lag_coefficient=config.blanket_lag,
        sample_period=config.blanket_sample_period,
    )
    tss_analyzer = ResidualAnalyzerParams(
        noise_std=config.tss_noise_std,
        lag_coefficient=config.tss_lag,
        sample_period=config.tss_sample_period,
    )

    influent = ASM1State.create(
        c.inf_s_i,
        c.inf_s_s,
        c.inf_x_i,
        c.inf_x_s,
        c.inf_x_bh,
        c.inf_x_ba,
        c.inf_x_p,
        c.inf_s_o,
        c.inf_s_no,
        c.inf_s_nh,
        c.inf_s_nd,
        c.inf_x_nd,
        c.inf_s_alk,
    )
    dt = jnp.array(c.dt)
    mean_flow = jnp.array(c.mean_flow)
    return_sludge_ratio = jnp.array(c.return_sludge_ratio)
    internal_recycle_ratio = jnp.array(c.internal_recycle_ratio)
    power_max = c.kla_34_max * (c.v3 + c.v4) + c.kla_5_max * c.v5
    settler_depth = jnp.array(config.settler.depth)
    eff_tss_limit = jnp.array(config.eff_tss_limit)
    blanket_alarm = jnp.array(config.blanket_alarm)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

        src = source_reset(k1)

        r1 = asm1_reset(
            c.r1_s_i,
            c.r1_s_s,
            c.r1_x_i,
            c.r1_x_s,
            c.r1_x_bh,
            c.r1_x_ba,
            c.r1_x_p,
            c.r1_s_o,
            c.r1_s_no,
            c.r1_s_nh,
            c.r1_s_nd,
            c.r1_x_nd,
            c.r1_s_alk,
            k1,
        )
        r2 = asm1_reset(
            c.r2_s_i,
            c.r2_s_s,
            c.r2_x_i,
            c.r2_x_s,
            c.r2_x_bh,
            c.r2_x_ba,
            c.r2_x_p,
            c.r2_s_o,
            c.r2_s_no,
            c.r2_s_nh,
            c.r2_s_nd,
            c.r2_x_nd,
            c.r2_s_alk,
            k1,
        )
        r3 = asm1_reset(
            c.r3_s_i,
            c.r3_s_s,
            c.r3_x_i,
            c.r3_x_s,
            c.r3_x_bh,
            c.r3_x_ba,
            c.r3_x_p,
            c.r3_s_o,
            c.r3_s_no,
            c.r3_s_nh,
            c.r3_s_nd,
            c.r3_x_nd,
            c.r3_s_alk,
            k1,
        )
        r4 = asm1_reset(
            c.r4_s_i,
            c.r4_s_s,
            c.r4_x_i,
            c.r4_x_s,
            c.r4_x_bh,
            c.r4_x_ba,
            c.r4_x_p,
            c.r4_s_o,
            c.r4_s_no,
            c.r4_s_nh,
            c.r4_s_nd,
            c.r4_x_nd,
            c.r4_s_alk,
            k1,
        )
        r5 = asm1_reset(
            c.r5_s_i,
            c.r5_s_s,
            c.r5_x_i,
            c.r5_x_s,
            c.r5_x_bh,
            c.r5_x_ba,
            c.r5_x_p,
            c.r5_s_o,
            c.r5_s_no,
            c.r5_s_nh,
            c.r5_s_nd,
            c.r5_x_nd,
            c.r5_s_alk,
            k1,
        )

        # Initialize settler with typical MLSS
        feed_tss_init = float(_compute_tss(r5))
        settler = settler_reset(feed_tss_init, config.settler, k2)

        kla34_loop = dosing_reset(2.0, c.do_ff, k3)
        kla5_loop = dosing_reset(2.0, c.do_ff, k4)
        waste = pump_reset(k5)

        sensors = BSM1TakacsSensors(
            nh4_eff=ResidualAnalyzerState.create(),
            no3_eff=ResidualAnalyzerState.create(),
            no3_r2=ResidualAnalyzerState.create(),
            nh4_inf=ResidualAnalyzerState.create(),
            blanket=ResidualAnalyzerState.create(),
            eff_tss=ResidualAnalyzerState.create(),
            und_tss=ResidualAnalyzerState.create(),
            last_q_in=jnp.array(c.mean_flow),
        )

        state = BSM1TakacsPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            reactor1=r1,
            reactor2=r2,
            reactor3=r3,
            reactor4=r4,
            reactor5=r5,
            settler_state=settler,
            kla_34_loop=kla34_loop,
            kla_5_loop=kla5_loop,
            waste_pump=waste,
            disturbance_schedule=create_empty(c.max_disturbance_events),
            sensors=sensors,
        )
        obs = jnp.zeros(12)
        return state, obs

    def step(state: BSM1TakacsPlantState, action: jax.Array, rng_key: jax.Array):
        k1, k_sensors = jax.random.split(rng_key)
        k_kla34, k_kla5, k_nh4e, k_no3e, k_no3r2, k_nh4i, k_bl, k_etss, k_utss = jax.random.split(k_sensors, 9)

        # 1. Influent flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. DosingSystem loops: kla_34, kla_5
        new_kla34_loop, do3_reading, kla_34, _ = dosing_step(
            state.kla_34_loop,
            action[0],
            state.reactor3.s_o,
            kla_34_dosing,
            dt,
            k_kla34,
        )
        new_kla5_loop, do5_reading, kla_5, _ = dosing_step(
            state.kla_5_loop,
            action[1],
            state.reactor5.s_o,
            kla_5_dosing,
            dt,
            k_kla5,
        )

        # 3. Waste pump
        new_waste, q_w = pump_step(state.waste_pump, action[2], waste_pump_params, dt)

        # 4. Flows
        q_a = q_in * internal_recycle_ratio
        q_rs = q_in * return_sludge_ratio
        q_total = q_in + q_rs + q_a
        q_to_clarifier = q_in + q_rs
        q_underflow = q_rs + q_w  # total underflow = return sludge + waste

        # 5. Return sludge from Takács settler
        feed_tss = _compute_tss(state.reactor5)
        underflow_tss = get_underflow_tss(state.settler_state)
        return_sludge = _return_sludge_from_tss(state.reactor5, feed_tss, underflow_tss)

        # 6. Reactor train
        inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
        inlet_r1, _ = mix_streams(inlet_rs, q_after_1, state.reactor5, q_a)
        new_r1 = asm1_step(state.reactor1, inlet_r1, q_total, jnp.array(0.0), p1, dt)
        new_r2 = asm1_step(state.reactor2, new_r1, q_total, jnp.array(0.0), p2, dt)
        new_r3 = asm1_step(state.reactor3, new_r2, q_total, kla_34, p3, dt)
        new_r4 = asm1_step(state.reactor4, new_r3, q_total, kla_34, p4, dt)
        new_r5 = asm1_step(state.reactor5, new_r4, q_total, kla_5, p5, dt)

        # 7. Takács settler step
        new_feed_tss = _compute_tss(new_r5)
        new_settler = settler_step(
            state.settler_state,
            new_feed_tss,
            q_to_clarifier,
            q_underflow,
            config.settler,
            dt,
        )

        # 8. Effluent quality
        eff_tss_val = get_effluent_tss(new_settler)
        effluent = _effluent_from_tss(new_r5, new_feed_tss, eff_tss_val)
        new_und_tss = get_underflow_tss(new_settler)
        blanket_h = compute_blanket_height(new_settler, config.settler, config.blanket_threshold)

        # 9. Sensors
        new_nh4e, nh4e = ra_step(state.sensors.nh4_eff, effluent.s_nh, analyzer_params, k_nh4e)
        new_no3e, no3e = ra_step(state.sensors.no3_eff, effluent.s_no, analyzer_params, k_no3e)
        new_no3r2, no3r2 = ra_step(state.sensors.no3_r2, new_r2.s_no, analyzer_params, k_no3r2)
        new_nh4i, nh4i = ra_step(state.sensors.nh4_inf, influent.s_nh, analyzer_params, k_nh4i)
        new_bl, bl_reading = ra_step(state.sensors.blanket, blanket_h, blanket_analyzer, k_bl)
        new_etss, etss_reading = ra_step(state.sensors.eff_tss, eff_tss_val, tss_analyzer, k_etss)
        new_utss, utss_reading = ra_step(state.sensors.und_tss, new_und_tss, tss_analyzer, k_utss)

        aeration_power = (kla_34 * (c.v3 + c.v4) + kla_5 * c.v5) / power_max
        dq = (q_in - state.sensors.last_q_in) / dt / mean_flow

        obs = jnp.array(
            [
                nh4e / 35.0,
                no3e / 20.0,
                do3_reading / 8.0,
                do5_reading / 8.0,
                no3r2 / 20.0,
                q_in / mean_flow,
                nh4i / 35.0,
                aeration_power,
                dq,
                bl_reading / settler_depth,
                etss_reading / 100.0,
                utss_reading / 10000.0,
            ]
        )

        # Reward: BSM1 N penalty + TSS penalty + blanket alarm
        tss_violation = jnp.maximum(etss_reading - eff_tss_limit, 0.0)
        blanket_violation = jnp.maximum(bl_reading - blanket_alarm, 0.0)
        reward = -(c.reward_w_nh * nh4e**2 + c.reward_w_no * no3e**2 + config.reward_w_tss * (tss_violation / eff_tss_limit) ** 2 + config.reward_w_blanket * blanket_violation**2)

        new_sensors = BSM1TakacsSensors(
            nh4_eff=new_nh4e,
            no3_eff=new_no3e,
            no3_r2=new_no3r2,
            nh4_inf=new_nh4i,
            blanket=new_bl,
            eff_tss=new_etss,
            und_tss=new_utss,
            last_q_in=q_in,
        )

        new_state = BSM1TakacsPlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            reactor1=new_r1,
            reactor2=new_r2,
            reactor3=new_r3,
            reactor4=new_r4,
            reactor5=new_r5,
            settler_state=new_settler,
            kla_34_loop=new_kla34_loop,
            kla_5_loop=new_kla5_loop,
            waste_pump=new_waste,
            disturbance_schedule=state.disturbance_schedule,
            sensors=new_sensors,
        )

        info: dict[str, jax.Array] = {
            "s_nh_effluent": effluent.s_nh,
            "s_no_effluent": effluent.s_no,
            "eff_tss": eff_tss_val,
            "und_tss": new_und_tss,
            "blanket_height": blanket_h,
            "q_w": q_w,
            "kla_34": kla_34,
            "kla_5": kla_5,
            "q_in": q_in,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
