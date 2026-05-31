from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams
from process_control.actuators.dose_pump import DosePumpState
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig
from process_control.benchmarks.bsm1 import BSM1PlantState
from process_control.benchmarks.bsm1 import _clarify_asm1
from process_control.disturbances.schedule import create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.units.asm1 import ASM1Params
from process_control.units.asm1 import ASM1State
from process_control.units.asm1 import mix_streams
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

    Observation (6D):
      s_nh_eff / 35:   normalised effluent ammonia
      s_no_eff / 20:   normalised effluent nitrate
      s_no_r2 / 20:    normalised nitrate in anoxic zone (denitrification feedback)
      s_o_r5 / 8:      normalised DO in reactor 5 (aerobic indicator)
      q_in / q_mean:   normalised influent flow
      q_a / q_max:     normalised internal recycle (actuator feedback)
    """
    # Fixed aeration setpoints (BSM1 steady-state values, h⁻¹)
    kla_34_fixed: float = 6.0
    kla_5_fixed: float = 3.5

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


def make_bsm1_recycle_benchmark(
    config: BSM1RecycleConfig,
) -> tuple[
    Callable[[jax.Array], tuple[BSM1PlantState, jax.Array]],
    Callable[[BSM1PlantState, jax.Array, jax.Array], tuple[BSM1PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    bsm1 = config.bsm1

    p1 = ASM1Params(volume=bsm1.v1)
    p2 = ASM1Params(volume=bsm1.v2)
    p3 = ASM1Params(volume=bsm1.v3)
    p4 = ASM1Params(volume=bsm1.v4)
    p5 = ASM1Params(volume=bsm1.v5)

    # Recycle ratio actuators (use DosePump for ramp-rate limiting)
    q_a_pump = DosePumpParams(
        max_dose=config.q_a_ratio_max, min_dose=config.q_a_ratio_min,
        max_ramp_rate=config.q_a_ramp_rate,
    )
    q_rs_pump = DosePumpParams(
        max_dose=config.q_rs_ratio_max, min_dose=config.q_rs_ratio_min,
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

    influent = ASM1State(
        s_i=jnp.array(bsm1.inf_s_i),
        s_s=jnp.array(bsm1.inf_s_s),
        x_i=jnp.array(bsm1.inf_x_i),
        x_s=jnp.array(bsm1.inf_x_s),
        x_bh=jnp.array(bsm1.inf_x_bh),
        x_ba=jnp.array(bsm1.inf_x_ba),
        x_p=jnp.array(bsm1.inf_x_p),
        s_o=jnp.array(bsm1.inf_s_o),
        s_no=jnp.array(bsm1.inf_s_no),
        s_nh=jnp.array(bsm1.inf_s_nh),
        s_nd=jnp.array(bsm1.inf_s_nd),
        x_nd=jnp.array(bsm1.inf_x_nd),
        s_alk=jnp.array(bsm1.inf_s_alk),
    )

    # Initial recycle ratios (BSM1 defaults)
    init_q_a_ratio = jnp.array(bsm1.internal_recycle_ratio)
    init_q_rs_ratio = jnp.array(bsm1.return_sludge_ratio)

    def reset(rng_key: jax.Array) -> tuple[BSM1PlantState, jax.Array]:
        k1, k2, k3 = jax.random.split(rng_key, 3)

        src = source_reset(k1)
        r1 = asm1_reset(
            bsm1.r1_s_i,
            bsm1.r1_s_s,
            bsm1.r1_x_i,
            bsm1.r1_x_s,
            bsm1.r1_x_bh,
            bsm1.r1_x_ba,
            bsm1.r1_x_p,
            bsm1.r1_s_o,
            bsm1.r1_s_no,
            bsm1.r1_s_nh,
            bsm1.r1_s_nd,
            bsm1.r1_x_nd,
            bsm1.r1_s_alk,
            k1,
        )
        r2 = asm1_reset(
            bsm1.r2_s_i,
            bsm1.r2_s_s,
            bsm1.r2_x_i,
            bsm1.r2_x_s,
            bsm1.r2_x_bh,
            bsm1.r2_x_ba,
            bsm1.r2_x_p,
            bsm1.r2_s_o,
            bsm1.r2_s_no,
            bsm1.r2_s_nh,
            bsm1.r2_s_nd,
            bsm1.r2_x_nd,
            bsm1.r2_s_alk,
            k1,
        )
        r3 = asm1_reset(
            bsm1.r3_s_i,
            bsm1.r3_s_s,
            bsm1.r3_x_i,
            bsm1.r3_x_s,
            bsm1.r3_x_bh,
            bsm1.r3_x_ba,
            bsm1.r3_x_p,
            bsm1.r3_s_o,
            bsm1.r3_s_no,
            bsm1.r3_s_nh,
            bsm1.r3_s_nd,
            bsm1.r3_x_nd,
            bsm1.r3_s_alk,
            k1,
        )
        r4 = asm1_reset(
            bsm1.r4_s_i,
            bsm1.r4_s_s,
            bsm1.r4_x_i,
            bsm1.r4_x_s,
            bsm1.r4_x_bh,
            bsm1.r4_x_ba,
            bsm1.r4_x_p,
            bsm1.r4_s_o,
            bsm1.r4_s_no,
            bsm1.r4_s_nh,
            bsm1.r4_s_nd,
            bsm1.r4_x_nd,
            bsm1.r4_s_alk,
            k1,
        )
        r5 = asm1_reset(
            bsm1.r5_s_i,
            bsm1.r5_s_s,
            bsm1.r5_x_i,
            bsm1.r5_x_s,
            bsm1.r5_x_bh,
            bsm1.r5_x_ba,
            bsm1.r5_x_p,
            bsm1.r5_s_o,
            bsm1.r5_s_no,
            bsm1.r5_s_nh,
            bsm1.r5_s_nd,
            bsm1.r5_x_nd,
            bsm1.r5_s_alk,
            k1,
        )

        # Initialise actuators at BSM1 default ratios
        q_a_state = DosePumpState(current_output=init_q_a_ratio)
        q_rs_state = DosePumpState(current_output=init_q_rs_ratio)

        plant_state = BSM1PlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            reactor1=r1,
            reactor2=r2,
            reactor3=r3,
            reactor4=r4,
            reactor5=r5,
            kla_34_actuator=q_a_state,   # repurposed: stores Q_a ratio
            kla_5_actuator=q_rs_state,    # repurposed: stores Q_rs ratio
            disturbance_schedule=create_empty(bsm1.max_disturbance_events),
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
        state: BSM1PlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[BSM1PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, _k2 = jax.random.split(rng_key)

        # 1. Influent flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state, state.step_count, source_params, k1,
        )

        # 2. Recycle ratio actuators (ramp-rate limited)
        new_q_a_state, q_a_ratio = dose_pump_step(state.kla_34_actuator, action[0], q_a_pump, dt)
        new_q_rs_state, q_rs_ratio = dose_pump_step(state.kla_5_actuator, action[1], q_rs_pump, dt)

        # 3. Derived flows
        q_a = q_in * q_a_ratio
        q_rs = q_in * q_rs_ratio
        q_total = q_in + q_rs + q_a
        q_to_clarifier = q_in + q_rs

        # 4. Clarifier
        _, return_sludge = _clarify_asm1(state.reactor5, q_to_clarifier, q_rs)

        # 5. R1: influent + return sludge + internal recycle
        inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
        inlet_r1, _ = mix_streams(inlet_rs, q_after_1, state.reactor5, q_a)
        new_r1 = asm1_step(state.reactor1, inlet_r1, q_total, jnp.array(0.0), p1, dt)

        # 6. R2 (anoxic)
        new_r2 = asm1_step(state.reactor2, new_r1, q_total, jnp.array(0.0), p2, dt)

        # 7. R3, R4 (aerobic, fixed kla)
        new_r3 = asm1_step(state.reactor3, new_r2, q_total, kla_34_fixed, p3, dt)
        new_r4 = asm1_step(state.reactor4, new_r3, q_total, kla_34_fixed, p4, dt)

        # 8. R5 (aerobic, fixed kla)
        new_r5 = asm1_step(state.reactor5, new_r4, q_total, kla_5_fixed, p5, dt)

        # 9. Effluent
        effluent, _ = _clarify_asm1(new_r5, q_to_clarifier, q_rs)

        # 10. Observation
        obs = jnp.array([
            effluent.s_nh / 35.0,
            effluent.s_no / 20.0,
            new_r2.s_no / 20.0,
            new_r5.s_o / 8.0,
            q_in / mean_flow,
            q_a_ratio / q_a_ratio_max,
        ])

        # 11. Reward: effluent quality + pumping energy penalty
        reward = -(
            config.reward_w_nh * effluent.s_nh ** 2
            + config.reward_w_no * effluent.s_no ** 2
            + config.reward_w_energy * (q_a_ratio + q_rs_ratio)
        )

        new_state = BSM1PlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            reactor1=new_r1, reactor2=new_r2, reactor3=new_r3,
            reactor4=new_r4, reactor5=new_r5,
            kla_34_actuator=new_q_a_state,
            kla_5_actuator=new_q_rs_state,
            disturbance_schedule=state.disturbance_schedule,
        )

        info: dict[str, jax.Array] = {
            "s_nh_effluent": effluent.s_nh,
            "s_no_effluent": effluent.s_no,
            "s_no_r2": new_r2.s_no,
            "s_o_r5": new_r5.s_o,
            "q_a_ratio": q_a_ratio,
            "q_rs_ratio": q_rs_ratio,
            "q_in": q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
