"""A versioned, instrumented direct-dose chlorine control environment.

The first public task is deliberately small: the controller sends a bounded
chlorine dose command every simulation interval.  The benchmark remains the
plant model; this module assembles it with a scenario and an instrumentation
profile, then computes feedback and reward from the profile's observed frame.
Supervisory and feed-forward tasks are intentionally left for a later task
version rather than silently changing the meaning of the action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.benchmarks.chlorine import (
    DIRECT,
    ChlorineBenchmarkConfig,
    make_chlorine_benchmark,
)
from process_control.benchmarks.chlorine import (
    PlantState as BenchmarkPlantState,
)
from process_control.disturbances.types import (
    DISTURBANCE_DEMAND_SLUG,
    DISTURBANCE_RAIN_STORM,
)
from process_control.environment import (
    ResetOptions,
    ResetResult,
    RuntimeMetadata,
    StepResult,
    StepTiming,
)
from process_control.environment_specs import (
    ActionSpec,
    ControlTaskSpec,
    EnvironmentSpec,
    ModelSpec,
    ScenarioSpec,
    SignalSpec,
    SpecIdentity,
)
from process_control.instrumentation import (
    InstrumentationChannel,
    InstrumentationProfile,
    ObservedSignals,
)
from process_control.scenarios import (
    EventKind,
    InitialCondition,
    InitialValue,
    RandomStream,
    ScenarioPack,
    SeedBundle,
    TimedEvent,
)

_DOSE = "dose.realized"
_RESIDUAL_MEASURED = "residual.measured"
_RESIDUAL_TRUE = "residual.true"
_TARGET = "target.residual"
_FLOW_MEASURED = "flow.measured"
_FLOW_TRUE = "flow.true"
_DEMAND_TRUE = "demand.true"

_SIGNAL_SPECS: Final[tuple[SignalSpec, ...]] = (
    SignalSpec(_DOSE, "mg/L"),
    SignalSpec(_RESIDUAL_MEASURED, "mg/L"),
    SignalSpec(_RESIDUAL_TRUE, "mg/L"),
    SignalSpec(_TARGET, "mg/L"),
    SignalSpec(_FLOW_MEASURED, "m3/h"),
    SignalSpec(_FLOW_TRUE, "m3/h"),
    SignalSpec(_DEMAND_TRUE, "mg/L"),
)
_SIGNAL_BY_NAME: Final[dict[str, SignalSpec]] = {signal.name: signal for signal in _SIGNAL_SPECS}
_SIGNAL_UNITS: Final[dict[str, str]] = {signal.name: signal.units for signal in _SIGNAL_SPECS}
_ACTION = ActionSpec("chlorine.dose", "mg/L", 0.0, 5.0)
_REWARD_INPUTS: Final[tuple[str, ...]] = (_RESIDUAL_MEASURED, _TARGET)
_EVALUATION_INPUTS: Final[tuple[str, ...]] = (
    _RESIDUAL_TRUE,
    _FLOW_TRUE,
    _DEMAND_TRUE,
    _DOSE,
)


def default_chlorine_scenario(config: ChlorineBenchmarkConfig | None = None) -> ScenarioPack:
    """Return a nominal seeded scenario matching the benchmark time step."""
    benchmark_config = config or ChlorineBenchmarkConfig()
    return ScenarioPack(
        spec=ScenarioSpec(SpecIdentity("chlorine.nominal", "1.0.0"), "1.0.0"),
        horizon_steps=benchmark_config.steps_per_day,
        simulation_time_step=benchmark_config.dt,
        signals=(
            SignalSpec("influent.flow", "m3/h"),
            SignalSpec("influent.demand", "mg/L"),
        ),
        initial_conditions=(
            InitialCondition(
                "nominal",
                (
                    InitialValue("residual", "mg/L", 0.0),
                    InitialValue("flow", "m3/h", benchmark_config.mean_flow),
                ),
            ),
        ),
    )


def default_chlorine_instrumentation_profile() -> InstrumentationProfile:
    """Return ideal online sensors for the default chlorine task."""
    channels = tuple(
        # Explicit channels keep the signal names stable while allowing a
        # later profile to map these outputs to a different latent source.
        InstrumentationChannel(source=signal, output=signal)
        for signal in (
            _SIGNAL_BY_NAME[_DOSE],
            _SIGNAL_BY_NAME[_RESIDUAL_MEASURED],
            _SIGNAL_BY_NAME[_TARGET],
            _SIGNAL_BY_NAME[_FLOW_MEASURED],
        )
    )
    return InstrumentationProfile.from_channels(
        channels,
        name="chlorine.ideal",
        version="1.0.0",
        observation_schema_version="1.0.0",
        latent_signals=tuple(_SIGNAL_BY_NAME[name] for name in (_RESIDUAL_TRUE, _FLOW_TRUE, _DEMAND_TRUE)),
    )


def _disturbance_type(event: TimedEvent) -> int:
    target = event.target.lower()
    if target in {"influent.demand", "demand", _DEMAND_TRUE}:
        return DISTURBANCE_DEMAND_SLUG
    if target in {"influent.flow", "flow", _FLOW_TRUE}:
        return DISTURBANCE_RAIN_STORM
    raise ValueError(f"chlorine disturbance {event.name!r} targets unsupported signal {event.target!r}")


def _benchmark_disturbances(events: tuple[TimedEvent, ...]) -> tuple[tuple[int, int, float, int], ...]:
    """Translate supported scenario events to the benchmark's fixed schedule."""
    translated: list[tuple[int, int, float, int]] = []
    for event in events:
        if event.kind is not EventKind.DISTURBANCE:
            continue
        type_id = _disturbance_type(event)
        period = event.repeat_every_steps or 0
        for occurrence in range(event.repeat_count):
            start = event.start_step + occurrence * period
            translated.append((start, start + event.duration_steps, event.magnitude, type_id))
    return tuple(translated)


@jax_dataclass
class ChlorinePlantState:
    """Plant state plus deterministic roots and latent history for sensors."""

    benchmark_state: BenchmarkPlantState
    process_key: jax.Array
    sensor_key: jax.Array
    fault_key: jax.Array
    scenario_key: jax.Array
    controller_key: jax.Array
    latent_history: jax.Array


@jax_dataclass
class ChlorineObservation:
    """Controller-visible values in profile order and availability mask."""

    values: jax.Array
    available: jax.Array


class ChlorineEnvironment:
    """Compose the chlorine benchmark, one scenario, and one profile."""

    def __init__(
        self,
        benchmark_config: ChlorineBenchmarkConfig,
        scenario: ScenarioPack,
        instrumentation: InstrumentationProfile,
    ) -> None:
        if benchmark_config.control_mode != DIRECT:
            raise ValueError("the chlorine direct-dose environment requires control_mode=DIRECT; supervisory and feed-forward tasks need separate task versions")
        if abs(benchmark_config.dt - scenario.simulation_time_step) > 1e-9:
            raise ValueError("scenario simulation_time_step must match ChlorineBenchmarkConfig.dt")
        if scenario.exogenous_signals:
            raise ValueError("the current chlorine benchmark accepts TimedEvent disturbances; custom exogenous trajectories need a model adapter")
        unsupported = tuple(event.name for event in scenario.events if event.kind not in (EventKind.DISTURBANCE, EventKind.SENSOR_FAULT))
        if unsupported:
            raise ValueError(f"chlorine environment does not yet support parameter, equipment, or operating events: {', '.join(unsupported)}")
        disturbance_events = _benchmark_disturbances(scenario.events)
        if len(disturbance_events) > benchmark_config.max_disturbance_events:
            raise ValueError("scenario disturbance events exceed benchmark schedule capacity")

        self._benchmark_config = replace(
            benchmark_config,
            disturbance_events=disturbance_events,
            # The environment owns instrumentation.  Keep the benchmark's
            # internal probes ideal so sensor noise is never applied twice.
            flow_noise_std=0.0,
            flow_bias=0.0,
            flow_dropout_probability=0.0,
            residual_noise_std=0.0,
            residual_lag_coefficient=0.0,
            residual_drift_rate=0.0,
        )
        self.scenario = scenario
        self.instrumentation = instrumentation
        self._reset_fn, self._step_fn = make_chlorine_benchmark(self._benchmark_config)
        self._observed_names = instrumentation.observed_names
        self._latent_names = tuple(_SIGNAL_BY_NAME)
        self._validate_instrumentation()
        self._spec = self._build_spec()
        self._runtime = RuntimeMetadata(
            self._spec.identity,
            observation_schema_version=self._spec.instrumentation.observation_schema_version,
            action_schema_version=self._spec.model.action_schema_version,
            simulation_time_step=benchmark_config.dt,
            control_interval=benchmark_config.dt,
            horizon_steps=scenario.horizon_steps,
        )

    def _validate_instrumentation(self) -> None:
        self.instrumentation.validate_reward_inputs(_REWARD_INPUTS)
        unknown = tuple(channel.source.name for channel in self.instrumentation.channels if channel.source.name not in _SIGNAL_BY_NAME)
        if unknown:
            raise ValueError("chlorine instrumentation references unknown latent signals: " + ", ".join(unknown))
        unknown_outputs = tuple(name for name in self.instrumentation.observed_names if name not in _SIGNAL_BY_NAME)
        if unknown_outputs:
            raise ValueError("chlorine instrumentation exposes unknown signals: " + ", ".join(unknown_outputs))

    def _build_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            identity=SpecIdentity("chlorine.direct-dose", "1.0.0"),
            model=ModelSpec(
                identity=SpecIdentity("chlorine.contact-basin", "1.0.0"),
                signal_schema_version="1.0.0",
                action_schema_version="1.0.0",
                signals=_SIGNAL_SPECS,
                actions=(
                    replace(
                        _ACTION,
                        minimum=self._benchmark_config.pump_min_dose,
                        maximum=self._benchmark_config.pump_max_dose,
                    ),
                ),
            ),
            scenario=self.scenario.spec,
            instrumentation=self.instrumentation.spec,
            task=ControlTaskSpec(
                identity=SpecIdentity("chlorine.direct-dose", "1.0.0"),
                schema_version="1.0.0",
                action_names=("chlorine.dose",),
                reward_input_names=_REWARD_INPUTS,
                evaluation_input_names=_EVALUATION_INPUTS,
            ),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        """Return the immutable four-component environment specification."""
        return self._spec

    @property
    def runtime(self) -> RuntimeMetadata:
        """Return timing, horizon, and schema metadata."""
        return self._runtime

    @property
    def observation_names(self) -> tuple[str, ...]:
        """Return observed signal names in controller vector order."""
        return self._observed_names

    @property
    def observation_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`observation_names`."""
        return tuple(_SIGNAL_UNITS[name] for name in self._observed_names)

    @property
    def action_limits(self) -> tuple[float, float]:
        """Return inclusive direct-dose action limits."""
        action = self.spec.model.actions[0]
        return action.minimum, action.maximum

    @property
    def run_metadata(self) -> Mapping[str, str | float | int]:
        """Return stable identifiers and timing useful for run provenance."""
        return {
            "environment": self.spec.identity.name,
            "environment_version": self.spec.identity.version,
            "model": self.spec.model.identity.name,
            "scenario": self.spec.scenario.identity.name,
            "scenario_version": self.spec.scenario.identity.version,
            "instrumentation": self.spec.instrumentation.identity.name,
            "instrumentation_version": self.spec.instrumentation.identity.version,
            "task": self.spec.task.identity.name,
            "simulation_time_step": self.runtime.simulation_time_step,
            "control_interval": self.runtime.control_interval,
            "horizon_steps": self.runtime.horizon_steps,
        }

    def reset(self, seed: int, *, options: ResetOptions | None = None) -> ResetResult[ChlorinePlantState, ChlorineObservation]:
        """Reset the benchmark and instantiate the selected seeded scenario."""
        if options is not None and options != ResetOptions():
            raise ValueError("chlorine environment currently supports cold reset only")
        episode = self.scenario.instantiate(seed)
        seeds = episode.seeds
        benchmark_state, raw_observation = self._reset_fn(seeds.key(RandomStream.PROCESS))
        latent = self._latent_frame(raw_observation, {})
        history = jnp.zeros((len(self._latent_names), self.runtime.horizon_steps + 1), dtype=jnp.float32)
        history = history.at[:, 0].set(jnp.asarray([latent[name] for name in self._latent_names]))
        state = ChlorinePlantState(
            benchmark_state=benchmark_state,
            process_key=seeds.process,
            sensor_key=seeds.sensor,
            fault_key=seeds.fault,
            scenario_key=seeds.scenario,
            controller_key=seeds.controller,
            latent_history=history,
        )
        observation, _frame = self._observe(state, 0, episode.events)
        return ResetResult(
            plant_state=state,
            observation=observation,
            timing=StepTiming(0, 0.0),
            evaluation_metrics={name: latent[name] for name in _EVALUATION_INPUTS},
            info={
                "seed": seed,
                "reset_mode": "cold",
                "scenario_identity": episode.spec.identity.name,
                "initial_condition": episode.initial_condition.name,
                "run_metadata": self.run_metadata,
            },
        )

    def step(self, plant_state: ChlorinePlantState, action: jax.Array | float) -> StepResult[ChlorinePlantState, ChlorineObservation]:
        """Advance one fixed control interval with observed-only reward."""
        previous_step = int(plant_state.benchmark_state.step_count)
        if previous_step >= self.runtime.horizon_steps:
            raise ValueError("cannot step a finished chlorine episode; call reset first")
        action_array = jnp.asarray(action, dtype=jnp.float32)
        if action_array.ndim != 0:
            raise ValueError("direct chlorine dose action must be scalar")
        clipped_action = jnp.clip(action_array, *self.action_limits)
        next_benchmark, raw_observation, _benchmark_reward, _done, info = self._step_fn(
            plant_state.benchmark_state,
            clipped_action,
            jax.random.fold_in(plant_state.process_key, previous_step + 1),
        )
        latent = self._latent_frame(raw_observation, info)
        next_history = plant_state.latent_history.at[:, previous_step + 1].set(jnp.asarray([latent[name] for name in self._latent_names]))
        next_state = replace(plant_state, benchmark_state=next_benchmark, latent_history=next_history)
        episode_events = self.scenario.events
        observation, frame = self._observe(next_state, previous_step + 1, episode_events)
        reward_inputs = self.instrumentation.reward_inputs(frame, _REWARD_INPUTS)
        reward = -jnp.square(reward_inputs[_RESIDUAL_MEASURED] - reward_inputs[_TARGET])
        evaluation_metrics = {name: latent[name] for name in _EVALUATION_INPUTS}
        active_events = tuple(event.name for event in episode_events if event.is_active(previous_step))
        observed_events = tuple(event.name for event in episode_events if event.is_active(previous_step + 1))
        return StepResult(
            plant_state=next_state,
            observation=observation,
            reward=reward,
            reward_inputs=reward_inputs,
            terminated=False,
            truncated=previous_step + 1 >= self.runtime.horizon_steps,
            timing=StepTiming(previous_step + 1, (previous_step + 1) * self.runtime.control_interval),
            evaluation_metrics=evaluation_metrics,
            info={
                "action_requested": action_array,
                "action_clipped": clipped_action,
                "active_events": active_events,
                "observed_step_events": observed_events,
                "observation_available": frame.available,
                "raw_benchmark_reward": _benchmark_reward,
                "run_metadata": self.run_metadata,
            },
        )

    def _latent_frame(self, raw_observation: jax.Array, info: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        return {
            _DOSE: raw_observation[0],
            _RESIDUAL_MEASURED: raw_observation[1],
            _RESIDUAL_TRUE: info.get("outlet_residual", raw_observation[1]),
            _TARGET: raw_observation[2],
            _FLOW_MEASURED: raw_observation[3],
            _FLOW_TRUE: info.get("flow", raw_observation[3]),
            _DEMAND_TRUE: info.get("demand", jnp.asarray(0.0)),
        }

    def _observe(
        self,
        state: ChlorinePlantState,
        step: int,
        events: tuple[TimedEvent, ...],
    ) -> tuple[ChlorineObservation, ObservedSignals]:
        bundle = SeedBundle(
            root_seed=0,
            process=state.process_key,
            sensor=state.sensor_key,
            fault=state.fault_key,
            scenario=state.scenario_key,
            controller=state.controller_key,
        )
        latent = {name: state.latent_history[index] for index, name in enumerate(self._latent_names)}
        batch = self.instrumentation.observe_trajectory(
            latent,
            seed_bundle=bundle,
            simulation_time_step=self.runtime.simulation_time_step,
            events=events,
        )
        frame = batch.at(step)
        values = jnp.asarray([frame.values[name] for name in self._observed_names])
        available = jnp.asarray([frame.available[name] for name in self._observed_names])
        return ChlorineObservation(values=values, available=available), frame


def make_chlorine_environment(
    benchmark_config: ChlorineBenchmarkConfig | None = None,
    *,
    scenario: ScenarioPack | None = None,
    instrumentation: InstrumentationProfile | None = None,
) -> ChlorineEnvironment:
    """Build the direct-dose chlorine environment from versioned components."""
    config = benchmark_config or ChlorineBenchmarkConfig()
    selected_scenario = scenario or default_chlorine_scenario(config)
    selected_instrumentation = instrumentation or default_chlorine_instrumentation_profile()
    return ChlorineEnvironment(config, selected_scenario, selected_instrumentation)


__all__ = [
    "ChlorineEnvironment",
    "ChlorineObservation",
    "ChlorinePlantState",
    "default_chlorine_instrumentation_profile",
    "default_chlorine_scenario",
    "make_chlorine_environment",
]
