"""Configurable BSM1 benchmark on the coupled activated-sludge flowsheet.

Alex et al. (2008), Benchmark Simulation Model no. 1:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf

Each control handle of Table 15 is driven by the agent, by a PI controller on a sensed plant quantity
(whose setpoint may itself be an agent action), or held fixed. Sensors, controllers, actuators, the
influent and the plant advance every control interval; the agent acts every `intervals_per_step`
intervals. Times are in hours, flows in m³/h and kLa in h⁻¹ throughout, actions included; the
performance indices in `info` keep the BSM1 per-day units.

The benchmark needs float64 (`jax_enable_x64`): in float32 the slow particulate states stop moving after
weeks of simulated time. Observations and rewards are returned as float32.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators import lagged
from process_control.actuators.lagged import BSM1_KLA_ACTUATOR, LaggedActuatorState
from process_control.benchmarks.bsm1_measurements import Measure, PlantSnapshot, asm1_state, measurements, stream_qualities, take_snapshot
from process_control.benchmarks.bsm1_metrics import aeration_energy, effluent_quality_index, mixing_energy, operational_cost_index, pumping_energy
from process_control.controllers import pi_controller
from process_control.controllers.pi_controller import BSM1_NITRATE_CONTROLLER, BSM1_OXYGEN_CONTROLLER, PIControllerParams, PIControllerState
from process_control.plants.activated_sludge import (
    BSM1_LAYOUT,
    BSM1_OPEN_LOOP_LAYER_TSS,
    BSM1_OPEN_LOOP_REACTORS,
    ActivatedSludgeLayout,
    activated_sludge_flowsheet,
    settled_state,
)
from process_control.scenarios import tabular_influent
from process_control.scenarios.tabular_influent import (
    DEFAULT_RANDOMIZATION,
    NO_RANDOMIZATION,
    InfluentRandomization,
    TabularInfluentState,
    bsm1_influent,
    bsm1_influent_profiles,
)
from process_control.sensors import response_time
from process_control.sensors.response_time import (
    BSM1_CLASS_A,
    BSM1_CLASS_B0,
    BSM1_NITRATE_SENSOR,
    BSM1_OXYGEN_SENSOR,
    SensorParams,
    SensorState,
)
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Params

_DAY = 24.0
Q0_STAB = 18446.0 / _DAY

HANDLES = ("kla_1", "kla_2", "kla_3", "kla_4", "kla_5", "q_a", "q_r", "q_w")
KLA_HANDLES = HANDLES[:5]
# Table 15, converted to hours.
HANDLE_LIMITS = {
    **dict.fromkeys(KLA_HANDLES, (0.0, 360.0 / _DAY)),
    "q_a": (0.0, 92230.0 / _DAY),
    "q_r": (0.0, 36892.0 / _DAY),
    "q_w": (0.0, 1844.6 / _DAY),
}


@dataclass(frozen=True)
class Fixed:
    """The handle is held at `value`."""

    value: float


@dataclass(frozen=True)
class Agent:
    """The handle follows the agent action `action`, which every handle naming it shares; by default the handle's own name."""

    action: str | None = None


@dataclass(frozen=True)
class AgentSetpoint:
    """A controller setpoint the agent chooses within [low, high]."""

    low: float
    high: float


@dataclass(frozen=True)
class PIControl:
    """The handle follows a PI controller acting on a sensed plant quantity toward a fixed or agent-chosen setpoint."""

    measurement: str
    setpoint: float | AgentSetpoint
    controller: PIControllerParams
    sensor: SensorParams


type HandleControl = Fixed | Agent | PIControl

# The BSM1 default controllers of Sec. 5: DO in reactor 5 by kLa5, nitrate in reactor 2 by Qa.
BSM1_OXYGEN_CONTROL = PIControl("s_o_5", 2.0, BSM1_OXYGEN_CONTROLLER, BSM1_OXYGEN_SENSOR)
BSM1_NITRATE_CONTROL = PIControl("s_no_2", 1.0, BSM1_NITRATE_CONTROLLER, BSM1_NITRATE_SENSOR)


@dataclass(frozen=True)
class BSM1Handles:
    """How each control handle of Table 15 is driven. Every kLa command passes through the BSM1 aeration actuator.

    The default lets the agent set a shared kLa for reactors 3 and 4, kLa5 and the wastage Qw, with Qa on
    the BSM1 nitrate controller, Qr at Q0,stab and the anoxic reactors unaerated.
    """

    kla_1: HandleControl = Fixed(0.0)
    kla_2: HandleControl = Fixed(0.0)
    kla_3: HandleControl = Agent("kla_34")
    kla_4: HandleControl = Agent("kla_34")
    kla_5: HandleControl = Agent()
    q_a: HandleControl = BSM1_NITRATE_CONTROL
    q_r: HandleControl = Fixed(Q0_STAB)
    q_w: HandleControl = Agent()

    def items(self) -> tuple[tuple[str, HandleControl], ...]:
        controls = (self.kla_1, self.kla_2, self.kla_3, self.kla_4, self.kla_5, self.q_a, self.q_r, self.q_w)
        return tuple(zip(HANDLES, controls, strict=True))

    def __post_init__(self) -> None:
        for name, control in self.items():
            low, high = HANDLE_LIMITS[name]
            if isinstance(control, Fixed) and not low <= control.value <= high:
                raise ValueError(f"fixed {name} = {control.value} lies outside its limits [{low}, {high}]")


@dataclass(frozen=True)
class ActionSpace:
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]


def action_space(handles: BSM1Handles) -> ActionSpace:
    """The agent's actions in order of first use: shared handle commands, then any supervised setpoints."""
    limits: dict[str, tuple[float, float]] = {}
    for name, control in handles.items():
        if not isinstance(control, Agent):
            continue
        action = control.action or name
        if limits.setdefault(action, HANDLE_LIMITS[name]) != HANDLE_LIMITS[name]:
            raise ValueError(f"action {action!r} drives handles with different limits")
    for name, control in handles.items():
        if not isinstance(control, PIControl) or not isinstance(control.setpoint, AgentSetpoint):
            continue
        action = f"{name}_setpoint"
        if action in limits:
            raise ValueError(f"action {action!r} is already a handle command")
        limits[action] = (control.setpoint.low, control.setpoint.high)
    return ActionSpace(
        names=tuple(limits), low=tuple(low for low, _ in limits.values()), high=tuple(high for _, high in limits.values())
    )


@dataclass(frozen=True)
class InfluentTemperature:
    """The reactors run at the temperature recorded with the influent profile."""


@dataclass(frozen=True)
class SeasonalTemperature:
    """A sinusoidal annual temperature, °C, starting at `phase` of its period, or a random phase each episode.

    The period is the 364-day (52-week) year of the BSM1_LT influent, whose evaluation spans its final
    364 days: Alex et al., Long Term Benchmark Simulation Model no. 1 (BSM1_LT), Secs. 2.6 and 6.
    """

    mean: float = 15.0
    amplitude: float = 5.0
    period: float = 364.0 * _DAY
    phase: float | None = None


type TemperatureModel = InfluentTemperature | SeasonalTemperature


@dataclass(frozen=True)
class Observation:
    """A plant quantity, read through `sensor` (exactly when unset) and divided by `scale`."""

    quantity: str
    scale: float
    sensor: SensorParams | None = None


# Table 14, converted to hours.
AMMONIA_SENSOR = SensorParams(BSM1_CLASS_B0, low=0.0, high=20.0, noise_std=0.5)
INFLUENT_AMMONIA_SENSOR = SensorParams(BSM1_CLASS_B0, low=0.0, high=50.0, noise_std=1.25)
FLOW_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=100000.0 / _DAY, noise_std=2500.0 / _DAY)
SOLIDS_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=10000.0, noise_std=250.0)
EFFLUENT_SOLIDS_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=200.0, noise_std=5.0)
BLANKET_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=5.0, noise_std=0.125)
TEMPERATURE_SENSOR = SensorParams(BSM1_CLASS_A, low=5.0, high=25.0, noise_std=0.5)

# Concentrations are scaled by their sensor's range, flows by the handle limits or Q0,stab, so each
# reads roughly within [0, 1] at dry weather.
DEFAULT_OBSERVATIONS = (
    Observation("s_o_3", 10.0, BSM1_OXYGEN_SENSOR),
    Observation("s_o_4", 10.0, BSM1_OXYGEN_SENSOR),
    Observation("s_o_5", 10.0, BSM1_OXYGEN_SENSOR),
    Observation("s_no_2", 20.0, BSM1_NITRATE_SENSOR),
    Observation("s_no_5", 20.0, BSM1_NITRATE_SENSOR),
    Observation("s_nh_5", 20.0, AMMONIA_SENSOR),
    Observation("effluent_s_nh", 20.0, AMMONIA_SENSOR),
    Observation("effluent_tss", 200.0, EFFLUENT_SOLIDS_SENSOR),
    Observation("influent_flow", 4.0 * Q0_STAB, FLOW_SENSOR),
    Observation("influent_s_nh", 50.0, INFLUENT_AMMONIA_SENSOR),
    Observation("mlss", 10000.0, SOLIDS_SENSOR),
    Observation("underflow_tss", 10000.0, SOLIDS_SENSOR),
    Observation("blanket_height", 4.0, BLANKET_SENSOR),
    Observation("temperature", 25.0, TEMPERATURE_SENSOR),
    Observation("kla_3", HANDLE_LIMITS["kla_3"][1]),
    Observation("kla_5", HANDLE_LIMITS["kla_5"][1]),
    Observation("q_a", HANDLE_LIMITS["q_a"][1]),
    Observation("q_w", HANDLE_LIMITS["q_w"][1]),
)

# Table 9 effluent limits, g/m³.
EFFLUENT_LIMITS = {"n_tot": 18.0, "cod": 100.0, "s_nh": 4.0, "tss": 30.0, "bod5": 10.0}


@dataclass(frozen=True)
class RewardWeights:
    """The reward is −scale · Σ weight · term, each term averaged over the agent step.

    Terms: `effluent_quality` is the BSM1 EQ (kg pollution units/d); `aeration_energy`, `pumping_energy`
    and `mixing_energy` are kWh/d; `sludge_production` is the BSM1 SP, the change in plant solids plus
    the wasted solids (kg SS/d); `sludge_wastage` the wasted solids alone (kg SS/d); `limit_violation`
    sums the relative exceedance of each Table 9 effluent limit.

    The defaults price effluent quality against the operational cost index OCI = AE + PE + ME + 5·SP one
    unit for one, the way BSM1 ranks strategies on both, so aerating harder pays only while it buys
    more effluent quality than it costs in energy. A violation of 10 % over a limit costs as much as
    200 pollution units/d. The scale brings a dry-weather reward near −2.
    """

    effluent_quality: float = 1.0
    aeration_energy: float = 1.0
    pumping_energy: float = 1.0
    mixing_energy: float = 1.0
    sludge_production: float = 5.0
    sludge_wastage: float = 0.0
    limit_violation: float = 2000.0
    scale: float = 1e-4


@dataclass(frozen=True)
class OperatingPoint:
    """Handle values the plant starts at (h⁻¹, m³/h); by default the open-loop point of Tables 6-8."""

    kla_1: float = 0.0
    kla_2: float = 0.0
    kla_3: float = 240.0 / _DAY
    kla_4: float = 240.0 / _DAY
    kla_5: float = 84.0 / _DAY
    q_a: float = 55338.0 / _DAY
    q_r: float = Q0_STAB
    q_w: float = 385.0 / _DAY

    def values(self) -> dict[str, float]:
        values = (self.kla_1, self.kla_2, self.kla_3, self.kla_4, self.kla_5, self.q_a, self.q_r, self.q_w)
        return dict(zip(HANDLES, values, strict=True))


@dataclass(frozen=True)
class BSM1Config:
    """The configurable BSM1 benchmark.

    The plant starts from `initial_reactors` (flow order, ASM1 schema) and settler layer TSS bottom first,
    with handles, actuators and controllers at `operating_point` and every sensor settled, so a plant at
    that steady state starts without a transient. `substeps` RK4 steps per control interval keep the
    anoxic oxygen balance stable, which needs steps of at most about 20 s.
    """

    handles: BSM1Handles = field(default_factory=BSM1Handles)
    observations: tuple[Observation, ...] = DEFAULT_OBSERVATIONS
    reward: RewardWeights = field(default_factory=RewardWeights)
    temperature: TemperatureModel = field(default_factory=InfluentTemperature)
    influent: InfluentRandomization = DEFAULT_RANDOMIZATION
    layout: ActivatedSludgeLayout[ASM1Params] = BSM1_LAYOUT
    initial_reactors: tuple[tuple[float, ...], ...] = BSM1_OPEN_LOOP_REACTORS
    initial_layer_tss: tuple[float, ...] = BSM1_OPEN_LOOP_LAYER_TSS
    operating_point: OperatingPoint = field(default_factory=OperatingPoint)
    blanket_threshold: float = 1500.0
    control_interval: float = 1.0 / 60.0
    intervals_per_step: int = 15
    substeps: int = 4

    def __post_init__(self) -> None:
        if len(self.layout.volumes) != len(KLA_HANDLES):
            raise ValueError(f"BSM1 aerates {len(KLA_HANDLES)} reactors, the layout has {len(self.layout.volumes)}")
        if self.control_interval <= 0 or self.intervals_per_step < 1 or self.substeps < 1:
            raise ValueError("the control interval must be positive, with at least one interval per step and one substep")


@jax_dataclass
class BSM1State:
    plant: dict[str, jax.Array]
    influent: TabularInfluentState
    temperature_phase: jax.Array
    time: jax.Array
    handles: dict[str, jax.Array]
    setpoints: dict[str, jax.Array]
    actuators: dict[str, LaggedActuatorState]
    controllers: dict[str, PIControllerState]
    controller_sensors: dict[str, SensorState]
    observation_sensors: dict[str, SensorState]


type ResetFn = Callable[[jax.Array], tuple[BSM1State, jax.Array]]
type StepFn = Callable[[BSM1State, jax.Array, jax.Array], tuple[BSM1State, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]]


def _arrays(values: Mapping[str, object]) -> dict[str, jax.Array]:
    arrays = {name: value for name, value in values.items() if isinstance(value, jax.Array)}
    if len(arrays) != len(values):
        raise TypeError(f"expected arrays, got {values}")
    return arrays


def require_float64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "the BSM1 benchmark needs float64: set jax_enable_x64 before building or tracing it, "
            "since in float32 its slow particulate states stop moving"
        )


def bsm1_action_bounds(config: BSM1Config) -> tuple[tuple[float, ...], tuple[float, ...]]:
    space = action_space(config.handles)
    return space.low, space.high


def make_bsm1_benchmark(config: BSM1Config) -> tuple[ResetFn, StepFn]:
    require_float64()
    layout = config.layout
    flowsheet = activated_sludge_flowsheet(layout)
    influent_params = bsm1_influent(config.influent)
    space = action_space(config.handles)
    controls = config.handles.items()
    supervised = {name: control for name, control in controls if isinstance(control, PIControl)}
    catalog = measurements(layout, config.blanket_threshold, HANDLES, tuple(supervised))

    def lookup(quantity: str) -> Measure:
        if quantity not in catalog:
            raise ValueError(f"unknown plant quantity {quantity!r}; available: {sorted(catalog)}")
        return catalog[quantity]

    controller_measures = {name: lookup(control.measurement) for name, control in supervised.items()}
    observation_measures = [lookup(observation.quantity) for observation in config.observations]
    sensed = {str(index): observation.sensor for index, observation in enumerate(config.observations) if observation.sensor is not None}
    scales = jnp.array([observation.scale for observation in config.observations])
    action_low, action_high = jnp.array(space.low), jnp.array(space.high)
    volumes = jnp.array(layout.volumes)
    settler = layout.settler
    dt = jnp.array(config.control_interval)
    step_days = config.control_interval * config.intervals_per_step / _DAY

    def temperature_at(time: jax.Array, phase: jax.Array, profile_temperature: jax.Array) -> jax.Array:
        model = config.temperature
        if isinstance(model, InfluentTemperature):
            return profile_temperature
        return model.mean + model.amplitude * jnp.sin(2.0 * jnp.pi * (time / model.period + phase))

    def signals_of(handles: Mapping[str, jax.Array], temperature: jax.Array) -> dict[str, jax.Array]:
        return {
            **{name: handles[name] for name in KLA_HANDLES},
            "q_a": handles["q_a"],
            "q_r": handles["q_r"],
            "q_underflow": handles["q_r"] + handles["q_w"],
            "temperature": temperature,
        }

    def snapshot(state: BSM1State, setpoints: Mapping[str, jax.Array]) -> PlantSnapshot:
        stream, profile_temperature = tabular_influent.influent(state.influent, influent_params)
        temperature = temperature_at(state.time, state.temperature_phase, profile_temperature)
        signals = signals_of(state.handles, temperature)
        return take_snapshot(flowsheet, layout, state.plant, stream, signals, state.handles, setpoints)

    def initial_phase(rng_key: jax.Array) -> jax.Array:
        model = config.temperature
        if not isinstance(model, SeasonalTemperature):
            return jnp.zeros(())
        return jax.random.uniform(rng_key) if model.phase is None else jnp.asarray(model.phase)

    def solids_inventory(plant: Mapping[str, jax.Array]) -> jax.Array:
        reactors = jnp.stack([plant[name] for name in layout.reactors])
        layer_volume = settler.area * settler.height / settler.n_layers
        return jnp.sum(ASM1_SCHEMA.tss(reactors) * volumes) + layer_volume * jnp.sum(ASM1_SCHEMA.tss(plant["settler"]))

    def setpoints_from(actions: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        return {
            name: actions[f"{name}_setpoint"] if isinstance(control.setpoint, AgentSetpoint) else jnp.asarray(control.setpoint)
            for name, control in supervised.items()
        }

    def reset(rng_key: jax.Array) -> tuple[BSM1State, jax.Array]:
        require_float64()
        influent_key, phase_key, controller_key, observation_key = jax.random.split(rng_key, 4)
        operating = config.operating_point.values()
        handles = {name: jnp.asarray(control.value if isinstance(control, Fixed) else operating[name]) for name, control in controls}
        setpoints = {
            name: jnp.asarray((control.setpoint.low + control.setpoint.high) / 2 if isinstance(control.setpoint, AgentSetpoint) else control.setpoint)
            for name, control in supervised.items()
        }
        unsensed = BSM1State(
            plant=settled_state(layout, jnp.array(config.initial_reactors), jnp.array(config.initial_layer_tss)),
            influent=tabular_influent.reset(influent_params, influent_key),
            temperature_phase=initial_phase(phase_key),
            time=jnp.zeros(()),
            handles=handles,
            setpoints=setpoints,
            actuators={name: lagged.reset(handles[name], BSM1_KLA_ACTUATOR) for name in KLA_HANDLES},
            controllers={},
            controller_sensors={},
            observation_sensors={},
        )
        plant_now = snapshot(unsensed, setpoints)

        controller_sensors = {
            name: response_time.reset(controller_measures[name](plant_now), control.sensor, jax.random.fold_in(controller_key, index))
            for index, (name, control) in enumerate(supervised.items())
        }
        controllers = {
            name: pi_controller.track_output(
                PIControllerState(integral=jnp.zeros(())), controller_sensors[name].reading, setpoints[name], handles[name], control.controller
            )
            for name, control in supervised.items()
        }
        readings = jnp.stack([measure(plant_now) for measure in observation_measures])
        observation_sensors = {
            key: response_time.reset(readings[int(key)], sensor, jax.random.fold_in(observation_key, int(key))) for key, sensor in sensed.items()
        }
        state = dataclasses.replace(unsensed, controllers=controllers, controller_sensors=controller_sensors, observation_sensors=observation_sensors)
        return state, (readings / scales).astype(jnp.float32)

    def interval(state: BSM1State, actions: Mapping[str, jax.Array], rng_key: jax.Array) -> tuple[BSM1State, dict[str, jax.Array]]:
        setpoints = setpoints_from(actions)
        plant_now = snapshot(state, setpoints)
        temperature = plant_now.temperature

        commands: dict[str, jax.Array] = {}
        controller_sensors: dict[str, SensorState] = {}
        controllers: dict[str, PIControllerState] = {}
        for name, control in controls:
            if isinstance(control, Fixed):
                commands[name] = jnp.asarray(control.value)
            elif isinstance(control, Agent):
                commands[name] = actions[control.action or name]
            else:
                controller_sensors[name], reading = response_time.step(
                    state.controller_sensors[name], controller_measures[name](plant_now), control.sensor, dt
                )
                controllers[name], commands[name] = pi_controller.step(state.controllers[name], reading, setpoints[name], control.controller, dt)

        actuators: dict[str, LaggedActuatorState] = {}
        handles: dict[str, jax.Array] = {}
        for name in KLA_HANDLES:
            actuators[name], handles[name] = lagged.step(state.actuators[name], commands[name], BSM1_KLA_ACTUATOR, dt)
        for name in HANDLES[len(KLA_HANDLES) :]:
            handles[name] = jnp.clip(commands[name], *HANDLE_LIMITS[name])

        plant = flowsheet.step(state.plant, {"influent": plant_now.influent}, signals_of(handles, temperature), dt, config.substeps)
        observation_sensors: dict[str, SensorState] = {}
        readings = []
        for index, measure in enumerate(observation_measures):
            true_value = measure(plant_now)
            key = str(index)
            if key in sensed:
                observation_sensors[key], true_value = response_time.step(state.observation_sensors[key], true_value, sensed[key], dt)
            readings.append(true_value)

        effluent = plant_now.effluent
        record = {
            "readings": jnp.stack(readings),
            "effluent": effluent.components,
            "effluent_flow": effluent.flow,
            "waste_solids": plant_now.waste.flow * ASM1_SCHEMA.tss(plant_now.waste.components),
            "kla": jnp.stack([plant_now.handles[name] for name in KLA_HANDLES]),
            "q_in": plant_now.influent.flow,
            "q_a": plant_now.handles["q_a"],
            "q_r": plant_now.handles["q_r"],
            "q_w": plant_now.handles["q_w"],
            "temperature": temperature,
            "split_deficit": plant_now.split_deficit,
        }
        advanced = BSM1State(
            plant=_arrays(plant),
            influent=tabular_influent.advance(state.influent, influent_params, dt, rng_key),
            temperature_phase=state.temperature_phase,
            time=state.time + dt,
            handles=handles,
            setpoints=setpoints,
            actuators=actuators,
            controllers=controllers,
            controller_sensors=controller_sensors,
            observation_sensors=observation_sensors,
        )
        return advanced, record

    def summarize(record: Mapping[str, jax.Array], start: BSM1State, end: BSM1State) -> dict[str, jax.Array]:
        params = layout.kinetics_params
        effluent_flow = record["effluent_flow"]
        load = effluent_flow[:, None] * record["effluent"]
        mean_flow = jnp.mean(effluent_flow)
        concentrations = stream_qualities(jnp.mean(load, axis=0) / jnp.maximum(mean_flow, 1e-9), params)
        per_interval = stream_qualities(record["effluent"], params)
        kla_per_day = record["kla"] * _DAY

        wastage = jnp.mean(record["waste_solids"]) * _DAY / 1000.0
        inventory_change = (solids_inventory(end.plant) - solids_inventory(start.plant)) / 1000.0 / step_days
        info = {
            "effluent_flow": mean_flow,
            **{f"effluent_{quality}": concentrations[quality] for quality in ("cod", "bod5", "tkn", "n_tot", "tss", "s_nh", "s_no")},
            "effluent_load": jnp.mean(load, axis=0),
            "effluent_quality": effluent_quality_index(asm1_state(record["effluent"]), params, effluent_flow * _DAY),
            "aeration_energy": aeration_energy(kla_per_day, volumes),
            "pumping_energy": pumping_energy(record["q_a"] * _DAY, record["q_r"] * _DAY, record["q_w"] * _DAY),
            "mixing_energy": mixing_energy(volumes, kla_per_day),
            "sludge_production": inventory_change + wastage,
            "sludge_wastage": wastage,
            "solids_inventory": solids_inventory(end.plant) / 1000.0,
            "limit_violation": jnp.mean(
                sum((jnp.maximum(per_interval[quality] / limit - 1.0, 0.0) for quality, limit in EFFLUENT_LIMITS.items()), jnp.zeros(()))
            ),
            "kla": jnp.mean(record["kla"], axis=0),
            "q_in": jnp.mean(record["q_in"]),
            "q_a": jnp.mean(record["q_a"]),
            "q_r": jnp.mean(record["q_r"]),
            "q_w": jnp.mean(record["q_w"]),
            "temperature": jnp.mean(record["temperature"]),
            "split_deficit": jnp.max(record["split_deficit"]),
            "infeasible_hydraulics": jnp.any(record["split_deficit"] > 0.0),
        }
        info["operational_cost"] = operational_cost_index(
            info["aeration_energy"], info["pumping_energy"], sp=info["sludge_production"], me=info["mixing_energy"]
        )
        return info

    def reward_of(info: Mapping[str, jax.Array]) -> jax.Array:
        weights = config.reward
        cost = (
            weights.effluent_quality * info["effluent_quality"]
            + weights.aeration_energy * info["aeration_energy"]
            + weights.pumping_energy * info["pumping_energy"]
            + weights.mixing_energy * info["mixing_energy"]
            + weights.sludge_production * info["sludge_production"]
            + weights.sludge_wastage * info["sludge_wastage"]
            + weights.limit_violation * info["limit_violation"]
        )
        return -weights.scale * cost

    def step(state: BSM1State, action: jax.Array, rng_key: jax.Array):
        require_float64()
        clipped = jnp.clip(jnp.asarray(action, dtype=jnp.float64), action_low, action_high)
        actions = {name: clipped[index] for index, name in enumerate(space.names)}

        def scan_interval(current: BSM1State, key: jax.Array) -> tuple[BSM1State, dict[str, jax.Array]]:
            return interval(current, actions, key)

        end, record = jax.lax.scan(scan_interval, state, jax.random.split(rng_key, config.intervals_per_step))
        info = summarize(record, state, end)
        observation = (record["readings"][-1] / scales).astype(jnp.float32)
        return end, observation, reward_of(info).astype(jnp.float32), jnp.array(False), info

    return reset, step


def _without_noise(control: PIControl) -> PIControl:
    return dataclasses.replace(control, sensor=dataclasses.replace(control.sensor, noise_std=0.0))


def bsm1_reference_closed_loop(weather: str = "dry", sensor_noise: bool = True) -> BSM1Config:
    """The BSM1 default control strategy of Sec. 5 on one unrandomized weather profile, with no agent actions.

    kLa3 and kLa4 stay at 240 d⁻¹, the oxygen controller holds reactor 5 at 2 g/m³, the nitrate
    controller holds reactor 2 at 1 g N/m³, and Qr and Qw stay at 18446 and 385 m³/d. `weather` is one
    of the profiles `constant`, `dry`, `rain` or `storm`; the constant profile with `sensor_noise` off is
    the stabilisation run.
    """
    profiles = tuple(bsm1_influent_profiles())
    if weather not in profiles:
        raise ValueError(f"weather must be one of {profiles}, got {weather!r}")
    oxygen, nitrate = BSM1_OXYGEN_CONTROL, BSM1_NITRATE_CONTROL
    if not sensor_noise:
        oxygen, nitrate = _without_noise(oxygen), _without_noise(nitrate)
    handles = BSM1Handles(
        kla_3=Fixed(240.0 / _DAY), kla_4=Fixed(240.0 / _DAY), kla_5=oxygen, q_a=nitrate, q_r=Fixed(Q0_STAB), q_w=Fixed(385.0 / _DAY)
    )
    weights = tuple(float(name == weather) for name in profiles)
    return BSM1Config(handles=handles, influent=dataclasses.replace(NO_RANDOMIZATION, profile_weights=weights))


def bsm1_recycle_control() -> BSM1Config:
    """The agent sets the internal recycle Qa and the return sludge Qr; aeration follows the BSM1 default strategy."""
    handles = BSM1Handles(kla_3=Fixed(240.0 / _DAY), kla_4=Fixed(240.0 / _DAY), kla_5=BSM1_OXYGEN_CONTROL, q_a=Agent(), q_r=Agent(), q_w=Fixed(385.0 / _DAY))
    return BSM1Config(handles=handles, observations=(*DEFAULT_OBSERVATIONS, Observation("q_r", HANDLE_LIMITS["q_r"][1])))


def bsm1_combined_control() -> BSM1Config:
    """The agent sets the shared kLa of reactors 3 and 4, kLa5, Qa and Qr; wastage stays at 385 m³/d."""
    handles = BSM1Handles(q_a=Agent(), q_r=Agent(), q_w=Fixed(385.0 / _DAY))
    return BSM1Config(handles=handles, observations=(*DEFAULT_OBSERVATIONS, Observation("q_r", HANDLE_LIMITS["q_r"][1])))


def bsm1_long_term() -> BSM1Config:
    """The default control problem under a seasonal temperature cycle, starting each episode at a random time of year."""
    return BSM1Config(temperature=SeasonalTemperature())


def bsm1_sludge_control() -> BSM1Config:
    """The agent manages the sludge inventory through Qr and Qw; aeration and Qa follow the BSM1 default strategy."""
    handles = BSM1Handles(kla_3=Fixed(240.0 / _DAY), kla_4=Fixed(240.0 / _DAY), kla_5=BSM1_OXYGEN_CONTROL, q_a=BSM1_NITRATE_CONTROL, q_r=Agent(), q_w=Agent())
    observations = (*DEFAULT_OBSERVATIONS, Observation("q_r", HANDLE_LIMITS["q_r"][1]), Observation("plant_sludge_age", 20.0 * _DAY))
    return BSM1Config(handles=handles, observations=observations)


def bsm1_supervisory_control() -> BSM1Config:
    """The agent chooses the setpoints of the BSM1 oxygen and nitrate controllers; kLa3 and kLa4 stay at 240 d⁻¹."""
    handles = BSM1Handles(
        kla_3=Fixed(240.0 / _DAY),
        kla_4=Fixed(240.0 / _DAY),
        kla_5=dataclasses.replace(BSM1_OXYGEN_CONTROL, setpoint=AgentSetpoint(0.5, 4.0)),
        q_a=dataclasses.replace(BSM1_NITRATE_CONTROL, setpoint=AgentSetpoint(0.0, 4.0)),
        q_w=Fixed(385.0 / _DAY),
    )
    observations = (*DEFAULT_OBSERVATIONS, Observation("kla_5_setpoint", 4.0), Observation("q_a_setpoint", 4.0))
    return BSM1Config(handles=handles, observations=observations)
