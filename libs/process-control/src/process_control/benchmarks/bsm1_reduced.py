from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.blower import BlowerParams, BlowerState
from process_control.actuators.blower import reset as blower_reset
from process_control.actuators.blower import step as blower_step
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import reset as actuator_reset
from process_control.actuators.ramp_limited import step as actuator_step
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
from process_control.units.biological_reactor import BiologicalReactorParams, BiologicalReactorState, mix_streams
from process_control.units.biological_reactor import reset as reactor_reset
from process_control.units.biological_reactor import step as reactor_step


@dataclass(frozen=True)
class BSM1ReducedBenchmarkConfig:
    """Reduced-order BSM1-inspired wastewater treatment benchmark.

    Simplified to 2 reactors (1 anoxic + 1 aerobic) plus a perfect settler.
    Kinetics follow BSM1 standard (Henze et al., 2000). Time unit: hours.

    Control actions (2D):
      - action[0]: aeration rate kla (h⁻¹) for the aerobic reactor
      - action[1]: internal recycle ratio (Q_int / Q_in)

    Observation (4D) — all from sensor readings:
      - sensed_nh4_eff / 35: effluent ammonia (residual analyzer)
      - sensed_no3_eff / 20: effluent nitrate (residual analyzer)
      - sensed_do / 8:       aerobic DO (DO probe)
      - Q_in / mean_flow:    normalised influent flow
    """
    dt: float = 0.02  # hours (~1.2 min per step)

    # Reactor volumes (m³)
    anoxic_volume: float = 1000.0
    aerobic_volume: float = 1333.0

    # Blower params
    kla_max: float = 10.0
    kla_ramp_up: float = 5.0
    kla_ramp_down: float = 8.0
    kla_startup_delay: float = 0.05

    # Internal recycle actuator (ratio = Q_int / Q_in)
    recycle_min: float = 0.5
    recycle_max: float = 5.0
    recycle_ramp_rate: float = 3.0

    # Return sludge (fixed fraction of Q_in)
    return_sludge_ratio: float = 1.0

    # Influent composition (BSM1 dry weather, g/m³)
    influent_s_s: float = 69.5
    influent_s_nh: float = 31.56
    influent_x_bh: float = 28.17

    # Influent flow variation (m³/h)
    mean_flow: float = 769.0
    diurnal_amplitude: float = 150.0
    min_flow: float = 500.0
    max_flow: float = 1050.0
    demand_noise_std: float = 1.0
    drift_scale: float = 0.05
    steps_per_day: int = 1200  # 24 h / dt = 1200 at dt=0.02 h

    # Anoxic reactor initial state (approximate BSM1 steady state)
    anoxic_init_s_s: float = 2.81
    anoxic_init_s_o: float = 0.0
    anoxic_init_s_no: float = 5.37
    anoxic_init_s_nh: float = 7.92
    anoxic_init_x_bh: float = 2880.0
    anoxic_init_x_ba: float = 149.0

    # Aerobic reactor initial state (approximate BSM1 steady state)
    aerobic_init_s_s: float = 0.89
    aerobic_init_s_o: float = 2.0
    aerobic_init_s_no: float = 10.4
    aerobic_init_s_nh: float = 1.73
    aerobic_init_x_bh: float = 2559.0
    aerobic_init_x_ba: float = 148.0

    # Reward weights
    reward_w_nh: float = 1.0
    reward_w_no: float = 0.3

    max_disturbance_events: int = 16

    # Sensor parameters (set all to zero for ideal/pure sensors)
    do_noise_std: float = 0.05
    do_lag: float = 0.9
    do_drift_rate: float = 0.001
    analyzer_noise_std: float = 0.3
    analyzer_lag: float = 0.8
    analyzer_sample_period: int = 8


@dataclass(frozen=True)
class BSM1ReducedSensorState:
    do_aerobic: DOSensorState
    nh4_eff: ResidualAnalyzerState
    no3_eff: ResidualAnalyzerState


jax.tree_util.register_dataclass(
    BSM1ReducedSensorState,
    data_fields=["do_aerobic", "nh4_eff", "no3_eff"],
    meta_fields=[],
)


@dataclass(frozen=True)
class BSM1ReducedPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    anoxic_state: BiologicalReactorState
    aerobic_state: BiologicalReactorState
    kla_blower: BlowerState
    recycle_actuator: RampLimitedActuatorState
    disturbance_schedule: DisturbanceSchedule
    sensors: BSM1ReducedSensorState


jax.tree_util.register_dataclass(
    BSM1ReducedPlantState,
    data_fields=[
        "step_count",
        "source_state",
        "anoxic_state",
        "aerobic_state",
        "kla_blower",
        "recycle_actuator",
        "disturbance_schedule",
        "sensors",
    ],
    meta_fields=[],
)


def _clarify(
    aerobic_state: BiologicalReactorState,
    aerobic_flow: jax.Array,
    return_fraction: jax.Array,
) -> tuple[BiologicalReactorState, BiologicalReactorState]:
    """Split aerobic outlet into clarified effluent and return sludge.

    Uses a perfect settler: effluent has zero biomass, return sludge
    concentrates biomass by mass conservation.

    Returns: (effluent_state, return_sludge_state)
    """
    return_flow = aerobic_flow * return_fraction
    concentration_factor = aerobic_flow / jnp.maximum(return_flow, 1.0)

    effluent = BiologicalReactorState(
        s_s=aerobic_state.s_s,
        s_o=aerobic_state.s_o,
        s_no=aerobic_state.s_no,
        s_nh=aerobic_state.s_nh,
        x_bh=jnp.array(0.0),  # perfect settling
        x_ba=jnp.array(0.0),
    )
    return_sludge = BiologicalReactorState(
        s_s=aerobic_state.s_s,
        s_o=aerobic_state.s_o,
        s_no=aerobic_state.s_no,
        s_nh=aerobic_state.s_nh,
        x_bh=aerobic_state.x_bh * concentration_factor,
        x_ba=aerobic_state.x_ba * concentration_factor,
    )
    return effluent, return_sludge


def make_bsm1_reduced_benchmark(
    config: BSM1ReducedBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[BSM1ReducedPlantState, jax.Array]],
    Callable[[BSM1ReducedPlantState, jax.Array, jax.Array], tuple[BSM1ReducedPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    anoxic_params = BiologicalReactorParams(volume=config.anoxic_volume)
    aerobic_params = BiologicalReactorParams(volume=config.aerobic_volume)

    blower_params = BlowerParams(
        max_kla=config.kla_max,
        max_ramp_up=config.kla_ramp_up,
        max_ramp_down=config.kla_ramp_down,
        startup_delay=config.kla_startup_delay,
    )
    recycle_pump_params = RampLimitedActuatorParams(
        max_output=config.recycle_max,
        min_output=config.recycle_min,
        max_ramp_rate=config.recycle_ramp_rate,
    )
    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,  # not used for wastewater influent
        demand_noise_std=config.demand_noise_std,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )

    dt = jnp.array(config.dt)
    return_sludge_ratio = jnp.array(config.return_sludge_ratio)
    mean_flow = jnp.array(config.mean_flow)

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

    # Pre-built influent composition (constant, only flow varies)
    influent_composition = BiologicalReactorState(
        s_s=jnp.array(config.influent_s_s),
        s_o=jnp.array(0.0),
        s_no=jnp.array(0.0),
        s_nh=jnp.array(config.influent_s_nh),
        x_bh=jnp.array(config.influent_x_bh),
        x_ba=jnp.array(0.0),
    )

    def reset(rng_key: jax.Array) -> tuple[BSM1ReducedPlantState, jax.Array]:
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        src_state = source_reset(k1)
        anoxic_state = reactor_reset(
            config.anoxic_init_s_s, config.anoxic_init_s_o,
            config.anoxic_init_s_no, config.anoxic_init_s_nh,
            config.anoxic_init_x_bh, config.anoxic_init_x_ba, k2,
        )
        aerobic_state = reactor_reset(
            config.aerobic_init_s_s, config.aerobic_init_s_o,
            config.aerobic_init_s_no, config.aerobic_init_s_nh,
            config.aerobic_init_x_bh, config.aerobic_init_x_ba, k3,
        )
        kla_state = blower_reset(0.0, k4)
        recycle_state = actuator_reset(k5)

        sensors = BSM1ReducedSensorState(
            do_aerobic=DOSensorState.create(config.aerobic_init_s_o),
            nh4_eff=ResidualAnalyzerState.create(),
            no3_eff=ResidualAnalyzerState.create(),
        )

        plant_state = BSM1ReducedPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            anoxic_state=anoxic_state,
            aerobic_state=aerobic_state,
            kla_blower=kla_state,
            recycle_actuator=recycle_state,
            disturbance_schedule=create_empty(config.max_disturbance_events),
            sensors=sensors,
        )

        effluent, _ = _clarify(aerobic_state, mean_flow, return_sludge_ratio)
        obs = jnp.array([
            effluent.s_nh / 35.0,
            effluent.s_no / 20.0,
            aerobic_state.s_o / 8.0,
            mean_flow / mean_flow,
        ])
        return plant_state, obs

    def step(
        state: BSM1ReducedPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[BSM1ReducedPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k_sensors = jax.random.split(rng_key)
        k_do, k_nh4, k_no3 = jax.random.split(k_sensors, 3)

        # 1. Influent flow (composition is fixed; only flow varies)
        new_source_state, _transport, Q_in, _demand = source_step(
            state.source_state, state.step_count, source_params, k1,
        )

        # 2. Actuators: kla and recycle ratio
        kla_action = action[0]
        recycle_action = action[1]
        new_kla_state, kla = blower_step(state.kla_blower, kla_action, blower_params, dt)
        new_recycle_state, recycle_ratio = actuator_step(state.recycle_actuator, recycle_action, recycle_pump_params, dt)

        # 3. Stream flows
        Q_rs = Q_in * return_sludge_ratio
        Q_int = Q_in * recycle_ratio
        Q_anoxic = Q_in + Q_rs + Q_int
        Q_aerobic = Q_in + Q_int

        # 4. Return sludge from clarifier (using current aerobic state)
        _, return_sludge = _clarify(state.aerobic_state, Q_aerobic, Q_rs / jnp.maximum(Q_aerobic, 1.0))

        # 5. Mix anoxic reactor inlet: influent + return sludge + internal recycle
        after_influent_rs, Q_after_1 = mix_streams(influent_composition, Q_in, return_sludge, Q_rs)
        anoxic_inlet, _ = mix_streams(after_influent_rs, Q_after_1, state.aerobic_state, Q_int)

        # 6. Anoxic reactor (kla=0)
        new_anoxic_state = reactor_step(
            state.anoxic_state, anoxic_inlet, Q_anoxic, jnp.array(0.0), anoxic_params, dt,
        )

        # 7. Aerobic reactor inlet: anoxic outlet only (internal recycle already accounted for in anoxic inlet)
        new_aerobic_state = reactor_step(
            state.aerobic_state, new_anoxic_state, Q_aerobic, kla, aerobic_params, dt,
        )

        # 8. Clarifier: effluent quality
        effluent, _ = _clarify(new_aerobic_state, Q_aerobic, Q_rs / jnp.maximum(Q_aerobic, 1.0))

        # 9. Sensors
        new_do, do_reading = do_step(state.sensors.do_aerobic, new_aerobic_state.s_o, do_params, k_do)
        new_nh4, nh4_reading = ra_step(state.sensors.nh4_eff, effluent.s_nh, analyzer_params, k_nh4)
        new_no3, no3_reading = ra_step(state.sensors.no3_eff, effluent.s_no, analyzer_params, k_no3)

        new_sensors = BSM1ReducedSensorState(
            do_aerobic=new_do,
            nh4_eff=new_nh4,
            no3_eff=new_no3,
        )

        # 10. Observation (from sensor readings)
        obs = jnp.array([
            nh4_reading / 35.0,
            no3_reading / 20.0,
            do_reading / 8.0,
            Q_in / mean_flow,
        ])

        # 11. Reward (from sensor readings)
        reward = -(
            config.reward_w_nh * nh4_reading ** 2
            + config.reward_w_no * no3_reading ** 2
        )

        new_state = BSM1ReducedPlantState(
            step_count=state.step_count + 1,
            source_state=new_source_state,
            anoxic_state=new_anoxic_state,
            aerobic_state=new_aerobic_state,
            kla_blower=new_kla_state,
            recycle_actuator=new_recycle_state,
            disturbance_schedule=state.disturbance_schedule,
            sensors=new_sensors,
        )

        info: dict[str, jax.Array] = {
            "s_nh_effluent": effluent.s_nh,
            "s_no_effluent": effluent.s_no,
            "s_o_aerobic": new_aerobic_state.s_o,
            "kla": kla,
            "recycle_ratio": recycle_ratio,
            "Q_in": Q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
