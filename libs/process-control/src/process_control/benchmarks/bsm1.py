from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams, DosePumpState
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.disturbances.schedule import DisturbanceSchedule, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.do_sensor import DOSensorParams, DOSensorState
from process_control.sensors.do_sensor import reset as do_reset
from process_control.sensors.do_sensor import step as do_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as ra_reset
from process_control.sensors.residual_analyzer import step as ra_step
from process_control.units.asm1 import ASM1Params, ASM1State, compute_tss, mix_streams
from process_control.units.asm1 import reset as asm1_reset
from process_control.units.asm1 import step as asm1_step


@dataclass(frozen=True)
class BSM1BenchmarkConfig:
    """Full BSM1-layout wastewater treatment benchmark using complete ASM1 kinetics.

    Follows the BSM1 specification (Copp et al., 2002):
      5 reactors in series: 2 anoxic (1000 m³ each) + 3 aerobic (1333 m³ each).
      Flow: 18 446 m³/day dry-weather mean ≈ 769 m³/h.
      Internal recycle: Q_a (from reactor 5 to reactor 1), default = 3 × Q_in.
      External recycle: Q_r (return sludge from clarifier to reactor 1), = 1 × Q_in.

    Control actions (2D):
      action[0]: kla for aerobic reactors 3 and 4 (h⁻¹)
      action[1]: kla for aerobic reactor 5 (h⁻¹)

    Observation (6D):
      s_nh_eff / 35:   normalised effluent ammonia
      s_no_eff / 20:   normalised effluent nitrate
      s_o_r3 / 8:      normalised DO in reactor 3
      s_o_r5 / 8:      normalised DO in reactor 5
      s_no_r2 / 20:    normalised nitrate in reactor 2 (anoxic feedback signal)
      Q_in / Q_mean:   normalised influent flow
    """
    dt: float = 0.02  # hours (~1.2 min)

    # Reactor volumes (m³) — BSM1 standard
    v1: float = 1000.0   # anoxic 1
    v2: float = 1000.0   # anoxic 2
    v3: float = 1333.0   # aerobic 1
    v4: float = 1333.0   # aerobic 2
    v5: float = 1333.0   # aerobic 3

    # Aeration actuators for zones 3-4 and zone 5
    kla_34_min: float = 0.0
    kla_34_max: float = 10.0
    kla_34_ramp_rate: float = 5.0
    kla_5_min: float = 0.0
    kla_5_max: float = 10.0
    kla_5_ramp_rate: float = 5.0

    # Fixed flow fractions (ratios of Q_in)
    internal_recycle_ratio: float = 3.0  # Q_a = ratio × Q_in
    return_sludge_ratio: float = 1.0     # Q_r = ratio × Q_in

    # Influent composition — BSM1 dry-weather values (g/m³ or mol/m³)
    inf_s_i: float = 30.0
    inf_s_s: float = 69.5
    inf_x_i: float = 51.2
    inf_x_s: float = 202.32
    inf_x_bh: float = 28.17
    inf_x_ba: float = 0.0
    inf_x_p: float = 0.0
    inf_s_o: float = 0.0
    inf_s_no: float = 0.0
    inf_s_nh: float = 31.56
    inf_s_nd: float = 6.95
    inf_x_nd: float = 10.59
    inf_s_alk: float = 7.0

    # Influent flow variation
    mean_flow: float = 769.0
    diurnal_amplitude: float = 150.0
    min_flow: float = 500.0
    max_flow: float = 1050.0
    demand_noise_std: float = 1.0
    drift_scale: float = 0.05
    steps_per_day: int = 1200

    # Reactor initial states — approximate BSM1 dry-weather steady state
    # Reactor 1 (anoxic)
    r1_s_i: float = 30.0
    r1_s_s: float = 2.81
    r1_x_i: float = 1100.0
    r1_x_s: float = 82.1
    r1_x_bh: float = 2552.0
    r1_x_ba: float = 148.0
    r1_x_p: float = 449.0
    r1_s_o: float = 0.0
    r1_s_no: float = 8.7
    r1_s_nh: float = 7.92
    r1_s_nd: float = 1.22
    r1_x_nd: float = 5.29
    r1_s_alk: float = 5.08

    # Reactor 2 (anoxic)
    r2_s_i: float = 30.0
    r2_s_s: float = 1.46
    r2_x_i: float = 1100.0
    r2_x_s: float = 52.1
    r2_x_bh: float = 2553.0
    r2_x_ba: float = 148.6
    r2_x_p: float = 450.0
    r2_s_o: float = 0.0
    r2_s_no: float = 6.8
    r2_s_nh: float = 7.26
    r2_s_nd: float = 0.883
    r2_x_nd: float = 3.53
    r2_s_alk: float = 5.43

    # Reactor 3 (aerobic)
    r3_s_i: float = 30.0
    r3_s_s: float = 1.15
    r3_x_i: float = 1100.0
    r3_x_s: float = 41.2
    r3_x_bh: float = 2557.0
    r3_x_ba: float = 148.9
    r3_x_p: float = 451.0
    r3_s_o: float = 1.72
    r3_s_no: float = 9.88
    r3_s_nh: float = 4.25
    r3_s_nd: float = 0.734
    r3_x_nd: float = 2.97
    r3_s_alk: float = 4.82

    # Reactor 4 (aerobic)
    r4_s_i: float = 30.0
    r4_s_s: float = 0.995
    r4_x_i: float = 1100.0
    r4_x_s: float = 37.1
    r4_x_bh: float = 2559.0
    r4_x_ba: float = 149.0
    r4_x_p: float = 452.0
    r4_s_o: float = 2.43
    r4_s_no: float = 10.3
    r4_s_nh: float = 2.57
    r4_s_nd: float = 0.655
    r4_x_nd: float = 2.80
    r4_s_alk: float = 4.65

    # Reactor 5 (aerobic)
    r5_s_i: float = 30.0
    r5_s_s: float = 0.889
    r5_x_i: float = 1100.0
    r5_x_s: float = 35.4
    r5_x_bh: float = 2559.0
    r5_x_ba: float = 149.1
    r5_x_p: float = 452.0
    r5_s_o: float = 2.97
    r5_s_no: float = 10.4
    r5_s_nh: float = 1.73
    r5_s_nd: float = 0.612
    r5_x_nd: float = 2.77
    r5_s_alk: float = 4.58

    # Reward weights
    reward_w_nh: float = 1.0
    reward_w_no: float = 0.3

    max_disturbance_events: int = 16

    # Sensor parameters (set all to zero for ideal/pure sensors)
    # DO probes (fast response, ~30s lag)
    do_noise_std: float = 0.05   # g O₂/m³
    do_lag: float = 0.9          # first-order lag coefficient
    do_drift_rate: float = 0.001 # drift per step
    # Concentration analyzers (NH₄, NO₃: 10-min sample period ≈ 8 steps at dt=0.02h)
    analyzer_noise_std: float = 0.3  # g/m³
    analyzer_lag: float = 0.8        # lag coefficient
    analyzer_sample_period: int = 8  # steps between samples (~10 min)


@dataclass(frozen=True)
class BSM1SensorState:
    """Sensor states for realistic instrumentation mode."""
    do_r3: DOSensorState
    do_r5: DOSensorState
    nh4_eff: ResidualAnalyzerState
    no3_eff: ResidualAnalyzerState
    no3_r2: ResidualAnalyzerState
    nh4_inf: ResidualAnalyzerState
    last_q_in: jax.Array


jax.tree_util.register_dataclass(
    BSM1SensorState,
    data_fields=[
        "do_r3", "do_r5",
        "nh4_eff", "no3_eff", "no3_r2", "nh4_inf",
        "last_q_in",
    ],
    meta_fields=[],
)


def _create_default_sensors(mean_flow: float, r3_do: float, r5_do: float) -> BSM1SensorState:
    """Create initial sensor states (used in both pure and realistic modes)."""
    return BSM1SensorState(
        do_r3=DOSensorState.create(r3_do),
        do_r5=DOSensorState.create(r5_do),
        nh4_eff=ResidualAnalyzerState.create(),
        no3_eff=ResidualAnalyzerState.create(),
        no3_r2=ResidualAnalyzerState.create(),
        nh4_inf=ResidualAnalyzerState.create(),
        last_q_in=jnp.array(mean_flow),
    )


@jax_dataclass
class BSM1PlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    reactor1: ASM1State
    reactor2: ASM1State
    reactor3: ASM1State
    reactor4: ASM1State
    reactor5: ASM1State
    kla_34_actuator: DosePumpState
    kla_5_actuator: DosePumpState
    disturbance_schedule: DisturbanceSchedule
    sensors: BSM1SensorState


jax.tree_util.register_dataclass(
    BSM1PlantState,
    data_fields=[
        "step_count",
        "source_state",
        "reactor1",
        "reactor2",
        "reactor3",
        "reactor4",
        "reactor5",
        "kla_34_actuator",
        "kla_5_actuator",
        "disturbance_schedule",
        "sensors",
    ],
    meta_fields=[],
)


def _clarify_asm1(
    r5: ASM1State,
    q_aerobic: jax.Array,
    q_rs: jax.Array,
) -> tuple[ASM1State, ASM1State]:
    """Perfect-settler clarifier: effluent has no particulates; return sludge
    concentrates particulate matter by mass conservation.

    Returns: (effluent_state, return_sludge_state)
    """
    concentration_factor = q_aerobic / jnp.maximum(q_rs, 1.0)

    effluent = ASM1State(
        s_i=r5.s_i,
        s_s=r5.s_s,
        x_i=jnp.array(0.0),
        x_s=jnp.array(0.0),
        x_bh=jnp.array(0.0),
        x_ba=jnp.array(0.0),
        x_p=jnp.array(0.0),
        s_o=r5.s_o,
        s_no=r5.s_no,
        s_nh=r5.s_nh,
        s_nd=r5.s_nd,
        x_nd=jnp.array(0.0),
        s_alk=r5.s_alk,
    )
    return_sludge = ASM1State(
        s_i=r5.s_i,
        s_s=r5.s_s,
        x_i=r5.x_i * concentration_factor,
        x_s=r5.x_s * concentration_factor,
        x_bh=r5.x_bh * concentration_factor,
        x_ba=r5.x_ba * concentration_factor,
        x_p=r5.x_p * concentration_factor,
        s_o=r5.s_o,
        s_no=r5.s_no,
        s_nh=r5.s_nh,
        s_nd=r5.s_nd,
        x_nd=r5.x_nd * concentration_factor,
        s_alk=r5.s_alk,
    )
    return effluent, return_sludge


def make_bsm1_benchmark(
    config: BSM1BenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[BSM1PlantState, jax.Array]],
    Callable[[BSM1PlantState, jax.Array, jax.Array], tuple[BSM1PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    p1 = ASM1Params(volume=config.v1)
    p2 = ASM1Params(volume=config.v2)
    p3 = ASM1Params(volume=config.v3)
    p4 = ASM1Params(volume=config.v4)
    p5 = ASM1Params(volume=config.v5)

    kla_34_pump = DosePumpParams(
        max_dose=config.kla_34_max, min_dose=config.kla_34_min,
        max_ramp_rate=config.kla_34_ramp_rate,
    )
    kla_5_pump = DosePumpParams(
        max_dose=config.kla_5_max, min_dose=config.kla_5_min,
        max_ramp_rate=config.kla_5_ramp_rate,
    )

    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=config.demand_noise_std,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )

    dt = jnp.array(config.dt)
    mean_flow = jnp.array(config.mean_flow)
    internal_recycle_ratio = jnp.array(config.internal_recycle_ratio)
    return_sludge_ratio = jnp.array(config.return_sludge_ratio)

    # Sensor params
    do_params = DOSensorParams(
        noise_std=config.do_noise_std,
        lag_coefficient=config.do_lag,
        drift_rate=config.do_drift_rate,
    )
    analyzer_params = ResidualAnalyzerParams(
        noise_std=config.analyzer_noise_std,
        lag_coefficient=config.analyzer_lag,
        sample_period=config.analyzer_sample_period,
    )

    # Max aeration power for normalisation
    kla_max_total = config.kla_34_max * (config.v3 + config.v4) + config.kla_5_max * config.v5
    power_max = jnp.array(jnp.maximum(kla_max_total, 1.0))

    influent = ASM1State(
        s_i=jnp.array(config.inf_s_i),
        s_s=jnp.array(config.inf_s_s),
        x_i=jnp.array(config.inf_x_i),
        x_s=jnp.array(config.inf_x_s),
        x_bh=jnp.array(config.inf_x_bh),
        x_ba=jnp.array(config.inf_x_ba),
        x_p=jnp.array(config.inf_x_p),
        s_o=jnp.array(config.inf_s_o),
        s_no=jnp.array(config.inf_s_no),
        s_nh=jnp.array(config.inf_s_nh),
        s_nd=jnp.array(config.inf_s_nd),
        x_nd=jnp.array(config.inf_x_nd),
        s_alk=jnp.array(config.inf_s_alk),
    )

    def reset(rng_key: jax.Array) -> tuple[BSM1PlantState, jax.Array]:
        k1, k2, k3 = jax.random.split(rng_key, 3)

        src = source_reset(k1)
        r1 = asm1_reset(
            config.r1_s_i,
            config.r1_s_s,
            config.r1_x_i,
            config.r1_x_s,
            config.r1_x_bh,
            config.r1_x_ba,
            config.r1_x_p,
            config.r1_s_o,
            config.r1_s_no,
            config.r1_s_nh,
            config.r1_s_nd,
            config.r1_x_nd,
            config.r1_s_alk,
            k1,
        )
        r2 = asm1_reset(
            config.r2_s_i,
            config.r2_s_s,
            config.r2_x_i,
            config.r2_x_s,
            config.r2_x_bh,
            config.r2_x_ba,
            config.r2_x_p,
            config.r2_s_o,
            config.r2_s_no,
            config.r2_s_nh,
            config.r2_s_nd,
            config.r2_x_nd,
            config.r2_s_alk,
            k1,
        )
        r3 = asm1_reset(
            config.r3_s_i,
            config.r3_s_s,
            config.r3_x_i,
            config.r3_x_s,
            config.r3_x_bh,
            config.r3_x_ba,
            config.r3_x_p,
            config.r3_s_o,
            config.r3_s_no,
            config.r3_s_nh,
            config.r3_s_nd,
            config.r3_x_nd,
            config.r3_s_alk,
            k1,
        )
        r4 = asm1_reset(
            config.r4_s_i,
            config.r4_s_s,
            config.r4_x_i,
            config.r4_x_s,
            config.r4_x_bh,
            config.r4_x_ba,
            config.r4_x_p,
            config.r4_s_o,
            config.r4_s_no,
            config.r4_s_nh,
            config.r4_s_nd,
            config.r4_x_nd,
            config.r4_s_alk,
            k1,
        )
        r5 = asm1_reset(
            config.r5_s_i,
            config.r5_s_s,
            config.r5_x_i,
            config.r5_x_s,
            config.r5_x_bh,
            config.r5_x_ba,
            config.r5_x_p,
            config.r5_s_o,
            config.r5_s_no,
            config.r5_s_nh,
            config.r5_s_nd,
            config.r5_x_nd,
            config.r5_s_alk,
            k1,
        )
        kla_34_state = dose_pump_reset(k2)
        kla_5_state = dose_pump_reset(k3)

        sensors = _create_default_sensors(config.mean_flow, config.r3_s_o, config.r5_s_o)

        plant_state = BSM1PlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            reactor1=r1,
            reactor2=r2,
            reactor3=r3,
            reactor4=r4,
            reactor5=r5,
            kla_34_actuator=kla_34_state,
            kla_5_actuator=kla_5_state,
            disturbance_schedule=create_empty(config.max_disturbance_events),
            sensors=sensors,
        )

        effluent, _ = _clarify_asm1(r5, mean_flow, mean_flow * return_sludge_ratio)
        obs = jnp.array([
            effluent.s_nh / 35.0,
            effluent.s_no / 20.0,
            r3.s_o / 8.0,
            r5.s_o / 8.0,
            r2.s_no / 20.0,
            mean_flow / mean_flow,
            influent.s_nh / 35.0,
            0.0,  # aeration power (no action yet)
            0.0,  # dq/dt
        ])
        return plant_state, obs

    def step(
        state: BSM1PlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[BSM1PlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k_sensors = jax.random.split(rng_key)
        k_do3, k_do5, k_nh4e, k_no3e, k_no3r2, k_nh4i = jax.random.split(k_sensors, 6)

        # 1. Influent flow (composition fixed at BSM1 dry-weather values)
        new_source, _transport, q_in, _demand = source_step(
            state.source_state, state.step_count, source_params, k1,
        )

        # 2. Actuators
        new_kla_34_state, kla_34 = dose_pump_step(state.kla_34_actuator, action[0], kla_34_pump, dt)
        new_kla_5_state, kla_5 = dose_pump_step(state.kla_5_actuator, action[1], kla_5_pump, dt)

        # 3. Derived flows
        q_a = q_in * internal_recycle_ratio   # internal recycle (R5 → R1)
        q_rs = q_in * return_sludge_ratio      # external recycle (clarifier → R1)
        q_total = q_in + q_rs + q_a           # flow through all 5 reactors
        q_to_clarifier = q_in + q_rs          # flow from R5 into clarifier

        # 4. Clarifier: return sludge from previous R5
        _, return_sludge = _clarify_asm1(state.reactor5, q_to_clarifier, q_rs)

        # 5. R1 inlet: influent + return sludge + internal recycle (R5)
        inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
        inlet_r1, _ = mix_streams(inlet_rs, q_after_1, state.reactor5, q_a)
        new_r1 = asm1_step(state.reactor1, inlet_r1, q_total, jnp.array(0.0), p1, dt)

        # 6. R2: fed by R1
        new_r2 = asm1_step(state.reactor2, new_r1, q_total, jnp.array(0.0), p2, dt)

        # 7. R3, R4: aerobic zone 1 (shared kla_34)
        new_r3 = asm1_step(state.reactor3, new_r2, q_total, kla_34, p3, dt)
        new_r4 = asm1_step(state.reactor4, new_r3, q_total, kla_34, p4, dt)

        # 8. R5: aerobic zone 2 (kla_5)
        new_r5 = asm1_step(state.reactor5, new_r4, q_total, kla_5, p5, dt)

        # 9. Effluent quality from clarifier
        effluent, _ = _clarify_asm1(new_r5, q_to_clarifier, q_rs)

        # 10. Sensors and observation
        new_do3, do3_reading = do_step(state.sensors.do_r3, new_r3.s_o, do_params, k_do3)
        new_do5, do5_reading = do_step(state.sensors.do_r5, new_r5.s_o, do_params, k_do5)
        new_nh4e, nh4e_reading = ra_step(state.sensors.nh4_eff, effluent.s_nh, analyzer_params, k_nh4e)
        new_no3e, no3e_reading = ra_step(state.sensors.no3_eff, effluent.s_no, analyzer_params, k_no3e)
        new_no3r2, no3r2_reading = ra_step(state.sensors.no3_r2, new_r2.s_no, analyzer_params, k_no3r2)
        new_nh4i, nh4i_reading = ra_step(state.sensors.nh4_inf, influent.s_nh, analyzer_params, k_nh4i)

        aeration_power = (kla_34 * (config.v3 + config.v4) + kla_5 * config.v5) / power_max
        dq_dt = (q_in - state.sensors.last_q_in) / dt / mean_flow

        new_sensors = BSM1SensorState(
            do_r3=new_do3, do_r5=new_do5,
            nh4_eff=new_nh4e, no3_eff=new_no3e, no3_r2=new_no3r2,
            nh4_inf=new_nh4i, last_q_in=q_in,
        )

        obs = jnp.array([
            nh4e_reading / 35.0,
            no3e_reading / 20.0,
            do3_reading / 8.0,
            do5_reading / 8.0,
            no3r2_reading / 20.0,
            q_in / mean_flow,
            nh4i_reading / 35.0,
            aeration_power,
            dq_dt,
        ])

        reward = -(
            config.reward_w_nh * nh4e_reading ** 2
            + config.reward_w_no * no3e_reading ** 2
        )

        new_state = BSM1PlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            reactor1=new_r1, reactor2=new_r2, reactor3=new_r3,
            reactor4=new_r4, reactor5=new_r5,
            kla_34_actuator=new_kla_34_state,
            kla_5_actuator=new_kla_5_state,
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
            "q_in": q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
