"""Runtime contract for versioned process-control environments."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from jax import Array

from process_control.environment_specs import EnvironmentSpec, SpecIdentity

PlantStateT = TypeVar("PlantStateT")
ObservationT = TypeVar("ObservationT", covariant=True)
ActionT = TypeVar("ActionT", contravariant=True)
MetricValue = object


class ResetMode(StrEnum):
    """How state is initialized at episode start."""

    COLD = "cold"
    WARM_START = "warm_start"


@dataclass(frozen=True, slots=True)
class PersistenceSpec:
    """State paths retained by each supported reset mode."""

    supported_reset_modes: tuple[ResetMode, ...] = (ResetMode.COLD,)
    warm_start_state_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.supported_reset_modes:
            raise ValueError("at least one reset mode must be supported")
        if len(set(self.supported_reset_modes)) != len(self.supported_reset_modes):
            raise ValueError("reset modes must be unique")
        if ResetMode.COLD not in self.supported_reset_modes:
            raise ValueError("cold reset must be supported")
        if self.warm_start_state_paths and ResetMode.WARM_START not in self.supported_reset_modes:
            raise ValueError("warm-start paths require warm-start support")


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Timing, schema, and reset behavior associated with an environment."""

    environment_identity: SpecIdentity
    observation_schema_version: str
    action_schema_version: str
    simulation_time_step: float
    control_interval: float
    horizon_steps: int
    persistence: PersistenceSpec = PersistenceSpec()

    def __post_init__(self) -> None:
        if self.simulation_time_step <= 0.0:
            raise ValueError("simulation_time_step must be positive")
        if self.control_interval <= 0.0:
            raise ValueError("control_interval must be positive")
        ratio = self.control_interval / self.simulation_time_step
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_interval must be an integer multiple of simulation_time_step")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")

    def validate_spec(self, spec: EnvironmentSpec) -> None:
        """Verify that runtime metadata describes the supplied environment spec."""
        if self.environment_identity != spec.identity:
            raise ValueError("runtime environment identity does not match environment spec")
        if self.observation_schema_version != spec.instrumentation.observation_schema_version:
            raise ValueError("runtime observation schema does not match environment spec")
        if self.action_schema_version != spec.model.action_schema_version:
            raise ValueError("runtime action schema does not match environment spec")


@dataclass(frozen=True, slots=True)
class StepTiming:
    """Episode clock after reset or a control step."""

    step_index: int
    elapsed_time: float


@dataclass(frozen=True, slots=True)
class ResetOptions:
    """Requested reset behavior and optional state to warm-start from."""

    mode: ResetMode = ResetMode.COLD
    previous_plant_state: object | None = None

    def __post_init__(self) -> None:
        if self.mode is ResetMode.COLD and self.previous_plant_state is not None:
            raise ValueError("cold reset cannot retain previous plant state")
        if self.mode is ResetMode.WARM_START and self.previous_plant_state is None:
            raise ValueError("warm start requires previous plant state")


@dataclass(frozen=True, slots=True)
class ResetResult[PlantStateT, ObservationT]:
    """Initial physical state and controller-visible output."""

    plant_state: PlantStateT
    observation: ObservationT
    timing: StepTiming
    evaluation_metrics: Mapping[str, MetricValue]
    info: Mapping[str, MetricValue]


@dataclass(frozen=True, slots=True)
class StepResult[PlantStateT, ObservationT]:
    """Output of one controller interval."""

    plant_state: PlantStateT
    observation: ObservationT
    reward: float | Array
    reward_inputs: Mapping[str, MetricValue]
    terminated: bool
    truncated: bool
    timing: StepTiming
    evaluation_metrics: Mapping[str, MetricValue]
    info: Mapping[str, MetricValue]


@runtime_checkable
class ProcessControlEnvironment(Protocol[PlantStateT, ObservationT, ActionT]):
    """Common runtime interface implemented by environments and adapters."""

    @property
    def spec(self) -> EnvironmentSpec: ...

    @property
    def runtime(self) -> RuntimeMetadata: ...

    def reset(
        self, seed: int, *, options: ResetOptions | None = None
    ) -> ResetResult[PlantStateT, ObservationT]: ...

    def step(
        self, plant_state: PlantStateT, action: ActionT
    ) -> StepResult[PlantStateT, ObservationT]: ...


__all__ = [
    "MetricValue",
    "PersistenceSpec",
    "ProcessControlEnvironment",
    "ResetMode",
    "ResetOptions",
    "ResetResult",
    "RuntimeMetadata",
    "StepResult",
    "StepTiming",
]
