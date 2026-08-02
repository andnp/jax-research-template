from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from process_control.benchmarks.chlorine import ChlorineBenchmarkConfig
from process_control.environment_conformance import assert_environment_conforms
from process_control.environment_specs import ScenarioSpec, SignalSpec, SpecIdentity
from process_control.environments.chlorine import (
    default_chlorine_instrumentation_profile,
    make_chlorine_environment,
)
from process_control.instrumentation import InstrumentationChannel, InstrumentationProfile
from process_control.scenarios import (
    EventKind,
    EventSeverity,
    InitialCondition,
    InitialValue,
    ScenarioPack,
    TimedEvent,
)


def _scenario(*events: TimedEvent, horizon_steps: int = 4) -> ScenarioPack:
    return ScenarioPack(
        spec=ScenarioSpec(SpecIdentity("chlorine.test", "1.0.0"), "1.0.0"),
        horizon_steps=horizon_steps,
        simulation_time_step=0.25,
        signals=(
            SignalSpec("influent.flow", "m3/h"),
            SignalSpec("influent.demand", "mg/L"),
        ),
        initial_conditions=(InitialCondition("nominal", (InitialValue("flow", "m3/h", 75.0),)),),
        events=tuple(sorted(events, key=lambda event: (event.start_step, event.name))),
    )


def test_default_chlorine_environment_conforms() -> None:
    environment = make_chlorine_environment(
        ChlorineBenchmarkConfig(steps_per_day=3),
    )

    assert_environment_conforms(
        environment,
        lambda _step, _reset: jnp.asarray(2.0),
        seed=7,
    )
    assert environment.observation_names == (
        "dose.realized",
        "residual.measured",
        "target.residual",
        "flow.measured",
    )
    assert environment.observation_units == ("mg/L", "mg/L", "mg/L", "m3/h")
    assert environment.action_limits == (0.0, 5.0)


def test_scenario_disturbance_is_visible_in_step_metadata_and_metrics() -> None:
    event = TimedEvent(
        name="storm",
        kind=EventKind.DISTURBANCE,
        target="influent.flow",
        units="m3/h",
        start_step=0,
        end_step=1,
        magnitude=2.0,
        severity=EventSeverity.HIGH,
    )
    environment = make_chlorine_environment(
        ChlorineBenchmarkConfig(steps_per_day=4),
        scenario=_scenario(event),
    )
    reset = environment.reset(4)
    result = environment.step(reset.plant_state, 2.0)

    assert result.info["active_events"] == ("storm",)
    assert float(cast(float, result.evaluation_metrics["flow.true"])) > 75.0
    assert "flow.true" not in result.reward_inputs


def test_reward_inputs_are_observed_even_when_profile_uses_latent_source() -> None:
    base = default_chlorine_instrumentation_profile()
    channels = tuple(
        InstrumentationChannel(
            source=(SignalSpec("residual.true", "mg/L") if channel.name == "residual.measured" else channel.source),
            output=channel.output,
            timing=channel.timing,
        )
        for channel in base.channels
    )
    profile = InstrumentationProfile.from_channels(
        channels,
        name="chlorine.latent-source-test",
        latent_signals=base.latent_signals,
    )
    environment = make_chlorine_environment(
        ChlorineBenchmarkConfig(steps_per_day=2),
        scenario=_scenario(horizon_steps=2),
        instrumentation=profile,
    )
    result = environment.step(environment.reset(2).plant_state, 2.0)

    assert set(result.reward_inputs) == {"residual.measured", "target.residual"}
    assert "residual.true" not in result.reward_inputs
    with pytest.raises(ValueError, match="latent"):
        profile.validate_reward_inputs(("residual.true",))


def test_horizon_is_time_limit_truncation() -> None:
    environment = make_chlorine_environment(
        ChlorineBenchmarkConfig(steps_per_day=2),
        scenario=_scenario(horizon_steps=2),
    )
    reset = environment.reset(0)
    first = environment.step(reset.plant_state, 2.0)
    second = environment.step(first.plant_state, 2.0)

    assert not first.truncated
    assert second.truncated
    assert not second.terminated
    assert second.timing.step_index == 2


def test_same_seed_reproduces_scenario_observation_and_reward() -> None:
    environment = make_chlorine_environment(
        ChlorineBenchmarkConfig(steps_per_day=3),
        scenario=_scenario(horizon_steps=3),
    )

    def run() -> tuple[list[np.ndarray], list[float]]:
        reset = environment.reset(123)
        observations = [np.asarray(reset.observation.values)]
        rewards: list[float] = []
        state = reset.plant_state
        for _ in range(3):
            result = environment.step(state, 2.0)
            observations.append(np.asarray(result.observation.values))
            rewards.append(float(result.reward))
            state = result.plant_state
        return observations, rewards

    first, first_rewards = run()
    second, second_rewards = run()
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first_rewards, second_rewards)
