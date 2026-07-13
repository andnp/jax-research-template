import dataclasses
from collections.abc import Callable

import pytest
from process_control.environment_specs import (
    ActionSpec,
    ControlTaskSpec,
    EnvironmentSpec,
    InstrumentationProfileSpec,
    InstrumentedSignalSpec,
    ModelSpec,
    ScenarioSpec,
    SignalSource,
    SignalSpec,
    SignalTiming,
    SpecIdentity,
)


def _signal(
    name: str,
    *,
    units: str = "mg/L",
) -> SignalSpec:
    return SignalSpec(name, units)


def _observed(name: str, *, units: str = "mg/L") -> InstrumentedSignalSpec:
    return InstrumentedSignalSpec(
        _signal(name, units=units), SignalSource.SENSOR, SignalTiming(1.0, delay=0.25)
    )


def _environment(
    *,
    reward_inputs: tuple[str, ...] = ("residual.measured",),
    evaluation_inputs: tuple[str, ...] = ("residual.true",),
) -> EnvironmentSpec:
    model = ModelSpec(
        SpecIdentity("chlorine.contact_basin", "1.0.0"),
        signal_schema_version="1.0.0",
        action_schema_version="1.0.0",
        signals=(
            _signal("residual.measured"),
            _signal("residual.true"),
        ),
        actions=(ActionSpec("chlorine.dose", "mg/L", 0.0, 10.0),),
    )
    instrumentation = InstrumentationProfileSpec(
        SpecIdentity("chlorine.standard", "1.0.0"),
        observation_schema_version="1.0.0",
        observed_signals=(_observed("residual.measured"),),
        latent_signals=(_signal("residual.true"),),
    )
    task = ControlTaskSpec(
        SpecIdentity("chlorine.direct", "1.0.0"),
        schema_version="1.0.0",
        action_names=("chlorine.dose",),
        reward_input_names=reward_inputs,
        evaluation_input_names=evaluation_inputs,
    )
    return EnvironmentSpec(
        SpecIdentity("chlorine-standard-direct", "1.0.0"),
        model,
        ScenarioSpec(SpecIdentity("chlorine.nominal", "1.0.0"), "1.0.0"),
        instrumentation,
        task,
    )


def test_composes_versioned_environment() -> None:
    environment = _environment()

    assert environment.model.signal_schema_version == "1.0.0"
    assert environment.instrumentation.observation_schema_version == "1.0.0"
    assert environment.task.schema_version == "1.0.0"
    assert environment.instrumentation.observed_signals[0].timing == SignalTiming(1.0, 0.25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        environment.identity.version = "2.0.0"  # type: ignore[misc]


def test_rejects_reward_input_missing_from_observed_instrumentation() -> None:
    with pytest.raises(ValueError, match="reward inputs unavailable.*residual.true"):
        _environment(reward_inputs=("residual.true",))


def test_allows_latent_truth_only_for_evaluation() -> None:
    environment = _environment(evaluation_inputs=("residual.measured", "residual.true"))

    assert environment.task.evaluation_input_names == (
        "residual.measured",
        "residual.true",
    )
    assert environment.instrumentation.latent_signals[0].name == "residual.true"


def test_rejects_signal_present_as_observed_and_latent() -> None:
    with pytest.raises(ValueError, match="both observed and latent"):
        InstrumentationProfileSpec(
            SpecIdentity("invalid", "1"),
            "1",
            observed_signals=(_observed("residual"),),
            latent_signals=(_signal("residual"),),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ModelSpec(
                SpecIdentity("model", "1"),
                "1",
                "1",
                (_signal("flow"), _signal("flow")),
                (),
            ),
            "duplicate model signal names",
        ),
        (
            lambda: ModelSpec(
                SpecIdentity("model", "1"),
                "1",
                "1",
                (),
                (
                    ActionSpec("dose", "mg/L", 0.0, 1.0),
                    ActionSpec("dose", "mg/L", 0.0, 1.0),
                ),
            ),
            "duplicate model action names",
        ),
        (
            lambda: ControlTaskSpec(
                SpecIdentity("task", "1"),
                "1",
                (),
                ("residual", "residual"),
            ),
            "duplicate reward input names",
        ),
    ],
)
def test_rejects_duplicate_names(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("version", ["", " 1.0.0", "1.0.0 "])
def test_rejects_invalid_schema_versions(version: str) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ScenarioSpec(SpecIdentity("scenario", "1.0.0"), version)


def test_rejects_latent_source_for_observed_signal() -> None:
    with pytest.raises(ValueError, match="cannot have latent source"):
        InstrumentedSignalSpec(
            SignalSpec("residual", "mg/L"), SignalSource.LATENT, SignalTiming(1.0)
        )
