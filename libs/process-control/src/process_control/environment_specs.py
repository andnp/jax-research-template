"""Versioned contracts for composing process-control environments."""

from dataclasses import dataclass
from enum import StrEnum


def _require_text(value: str, field: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty and have no surrounding whitespace")


def _require_unique(values: tuple[str, ...], kind: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {kind}: {', '.join(duplicates)}")


@dataclass(frozen=True, slots=True)
class SpecIdentity:
    """Stable component identity and behavior version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.version, "version")


@dataclass(frozen=True, slots=True)
class SignalTiming:
    """Availability timing for a controller-visible signal."""

    sample_period: float
    delay: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_period <= 0.0:
            raise ValueError("sample_period must be positive")
        if self.delay < 0.0:
            raise ValueError("delay must be non-negative")


class SignalSource(StrEnum):
    """How an instrumentation signal becomes available."""

    SENSOR = "sensor"
    CALCULATED = "calculated"
    MEASURED_DISTURBANCE = "measured_disturbance"
    FORECAST = "forecast"
    SOFT_SENSOR = "soft_sensor"
    LATENT = "latent"


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """A signal with a stable name and units."""

    name: str
    units: str

    def __post_init__(self) -> None:
        _require_text(self.name, "signal name")
        _require_text(self.units, f"units for signal {self.name}")


@dataclass(frozen=True, slots=True)
class InstrumentedSignalSpec:
    """A controller-visible signal and its availability metadata."""

    signal: SignalSpec
    source: SignalSource
    timing: SignalTiming

    def __post_init__(self) -> None:
        if self.source is SignalSource.LATENT:
            raise ValueError(f"observed signal {self.name} cannot have latent source")

    @property
    def name(self) -> str:
        """Return the stable signal name."""
        return self.signal.name

    @property
    def units(self) -> str:
        """Return the signal units."""
        return self.signal.units


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """A bounded action with a stable name and units."""

    name: str
    units: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        _require_text(self.name, "action name")
        _require_text(self.units, f"units for action {self.name}")
        if self.minimum >= self.maximum:
            raise ValueError(f"action {self.name} minimum must be below maximum")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Versioned physical-model interface."""

    identity: SpecIdentity
    signal_schema_version: str
    action_schema_version: str
    signals: tuple[SignalSpec, ...]
    actions: tuple[ActionSpec, ...]

    def __post_init__(self) -> None:
        _require_text(self.signal_schema_version, "signal_schema_version")
        _require_text(self.action_schema_version, "action_schema_version")
        _require_unique(tuple(signal.name for signal in self.signals), "model signal names")
        _require_unique(tuple(action.name for action in self.actions), "model action names")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Versioned run-condition and disturbance contract."""

    identity: SpecIdentity
    schema_version: str

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "scenario schema_version")


@dataclass(frozen=True, slots=True)
class InstrumentationProfileSpec:
    """Versioned separation of controller-visible and latent signals."""

    identity: SpecIdentity
    observation_schema_version: str
    observed_signals: tuple[InstrumentedSignalSpec, ...]
    latent_signals: tuple[SignalSpec, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.observation_schema_version, "observation_schema_version")
        observed = tuple(signal.name for signal in self.observed_signals)
        latent = tuple(signal.name for signal in self.latent_signals)
        _require_unique(observed, "observed signal names")
        _require_unique(latent, "latent signal names")
        overlap = sorted(set(observed) & set(latent))
        if overlap:
            raise ValueError(f"signals cannot be both observed and latent: {', '.join(overlap)}")


@dataclass(frozen=True, slots=True)
class ControlTaskSpec:
    """Versioned action, learning-reward, and evaluation contract."""

    identity: SpecIdentity
    schema_version: str
    action_names: tuple[str, ...]
    reward_input_names: tuple[str, ...]
    evaluation_input_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "control-task schema_version")
        for kind, names in (
            ("task action names", self.action_names),
            ("reward input names", self.reward_input_names),
            ("evaluation input names", self.evaluation_input_names),
        ):
            for name in names:
                _require_text(name, kind)
            _require_unique(names, kind)


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Validated composition of the four versioned environment components."""

    identity: SpecIdentity
    model: ModelSpec
    scenario: ScenarioSpec
    instrumentation: InstrumentationProfileSpec
    task: ControlTaskSpec

    def __post_init__(self) -> None:
        model_actions = {action.name for action in self.model.actions}
        missing_actions = sorted(set(self.task.action_names) - model_actions)
        if missing_actions:
            raise ValueError(f"task actions missing from model: {', '.join(missing_actions)}")

        observed = {signal.name for signal in self.instrumentation.observed_signals}
        missing_reward = sorted(set(self.task.reward_input_names) - observed)
        if missing_reward:
            raise ValueError(
                "reward inputs unavailable through instrumentation: " + ", ".join(missing_reward)
            )

        available_for_evaluation = observed | {
            signal.name for signal in self.instrumentation.latent_signals
        }
        missing_evaluation = sorted(
            set(self.task.evaluation_input_names) - available_for_evaluation
        )
        if missing_evaluation:
            raise ValueError(
                "evaluation inputs missing from instrumentation: "
                + ", ".join(missing_evaluation)
            )

        model_units = {signal.name: signal.units for signal in self.model.signals}
        for signal in (
            *self.instrumentation.observed_signals,
            *self.instrumentation.latent_signals,
        ):
            expected_units = model_units.get(signal.name)
            if expected_units is not None and signal.units != expected_units:
                raise ValueError(
                    f"signal {signal.name} units {signal.units!r} do not match model "
                    f"units {expected_units!r}"
                )


__all__ = [
    "ActionSpec",
    "ControlTaskSpec",
    "EnvironmentSpec",
    "InstrumentedSignalSpec",
    "InstrumentationProfileSpec",
    "ModelSpec",
    "ScenarioSpec",
    "SignalSource",
    "SignalSpec",
    "SignalTiming",
    "SpecIdentity",
]
