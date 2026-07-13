from dataclasses import replace
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from process_control.environment import (
    PersistenceSpec,
    ProcessControlEnvironment,
    ResetMode,
    ResetOptions,
    ResetResult,
    RuntimeMetadata,
    StepResult,
    StepTiming,
)
from process_control.environment_conformance import assert_environment_conforms
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


def _spec() -> EnvironmentSpec:
    measured = SignalSpec("level.measured", "m")
    true = SignalSpec("level.true", "m")
    return EnvironmentSpec(
        SpecIdentity("toy-level", "1.0.0"),
        ModelSpec(
            SpecIdentity("toy-tank", "1.0.0"),
            "1.0.0",
            "1.0.0",
            (measured, true),
            (ActionSpec("outflow", "m3/h", 0.0, 2.0),),
        ),
        ScenarioSpec(SpecIdentity("seeded-level", "1.0.0"), "1.0.0"),
        InstrumentationProfileSpec(
            SpecIdentity("level-sensor", "1.0.0"),
            "1.0.0",
            (
                InstrumentedSignalSpec(
                    measured, SignalSource.SENSOR, SignalTiming(0.5)
                ),
            ),
            (true,),
        ),
        ControlTaskSpec(
            SpecIdentity("hold-level", "1.0.0"),
            "1.0.0",
            ("outflow",),
            ("level.measured",),
            ("level.true",),
        ),
    )


class ToyLevelEnvironment:
    """Small stochastic environment used to exercise the public contract."""

    def __init__(self) -> None:
        self._spec = _spec()
        self._runtime = RuntimeMetadata(
            self._spec.identity,
            observation_schema_version="1.0.0",
            action_schema_version="1.0.0",
            simulation_time_step=0.1,
            control_interval=0.5,
            horizon_steps=3,
            persistence=PersistenceSpec(
                (ResetMode.COLD, ResetMode.WARM_START), ("inventory.level",)
            ),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    @property
    def runtime(self) -> RuntimeMetadata:
        return self._runtime

    def reset(
        self, seed: int, *, options: ResetOptions | None = None
    ) -> ResetResult[Array, Array]:
        options = options or ResetOptions()
        if options.mode not in self.runtime.persistence.supported_reset_modes:
            raise ValueError("unsupported reset mode")
        if options.mode is ResetMode.WARM_START:
            level = jnp.asarray(options.previous_plant_state)[0]
        else:
            level = jax.random.uniform(jax.random.key(seed), (), minval=0.8, maxval=1.2)
        state = jnp.asarray([level, 0.0], dtype=jnp.float32)
        observation = state[:1] + jnp.float32(0.05)
        return ResetResult(
            state,
            observation,
            StepTiming(0, 0.0),
            {"level.true": level},
            {"reset_mode": options.mode.value},
        )

    def step(self, plant_state: Array, action: Array) -> StepResult[Array, Array]:
        level, previous_step = plant_state
        step_index = int(previous_step) + 1
        next_level = level + jnp.float32(0.1) - jnp.float32(0.05) * action[0]
        next_state = jnp.asarray([next_level, step_index], dtype=jnp.float32)
        measured = next_state[:1] + jnp.float32(0.05)
        return StepResult(
            next_state,
            measured,
            -float(jnp.square(measured[0] - 1.0)),
            {"level.measured": measured[0]},
            terminated=False,
            truncated=step_index >= self.runtime.horizon_steps,
            timing=StepTiming(step_index, step_index * self.runtime.control_interval),
            evaluation_metrics={"level.true": next_level},
            info={"integration_substeps": 5},
        )


def test_toy_environment_satisfies_runtime_protocol() -> None:
    environment = ToyLevelEnvironment()

    assert isinstance(environment, ProcessControlEnvironment)
    assert_environment_conforms(
        environment,
        lambda _step, _reset: jnp.asarray([1.0], dtype=jnp.float32),
        seed=7,
    )


def test_seeded_reset_is_deterministic_and_warm_start_is_explicit() -> None:
    environment = ToyLevelEnvironment()
    first = environment.reset(11)
    repeated = environment.reset(11)
    different = environment.reset(12)

    assert jnp.array_equal(first.plant_state, repeated.plant_state)
    assert jnp.array_equal(first.observation, repeated.observation)
    assert not jnp.array_equal(first.plant_state, different.plant_state)
    warm = environment.reset(
        999,
        options=ResetOptions(ResetMode.WARM_START, previous_plant_state=first.plant_state),
    )
    assert warm.plant_state[0] == first.plant_state[0]
    assert warm.info["reset_mode"] == "warm_start"


def test_step_separates_feedback_from_latent_evaluation() -> None:
    environment = ToyLevelEnvironment()
    initial = environment.reset(0)
    result = environment.step(initial.plant_state, jnp.asarray([1.0], dtype=jnp.float32))

    assert set(result.reward_inputs) == {"level.measured"}
    assert set(result.evaluation_metrics) == {"level.true"}
    assert "level.true" not in result.reward_inputs
    assert result.timing == StepTiming(1, 0.5)
    assert not result.terminated
    assert not result.truncated


def test_distinguishes_task_termination_from_time_limit_truncation() -> None:
    environment = ToyLevelEnvironment()
    initial = environment.reset(0)
    terminal = replace(
        environment.step(initial.plant_state, jnp.asarray([1.0], dtype=jnp.float32)),
        terminated=True,
    )

    assert terminal.terminated
    assert not terminal.truncated
    state = initial.plant_state
    horizon = environment.step(state, jnp.asarray([1.0], dtype=jnp.float32))
    state = horizon.plant_state
    for _ in range(1, environment.runtime.horizon_steps):
        horizon = environment.step(state, jnp.asarray([1.0], dtype=jnp.float32))
        state = horizon.plant_state
    assert not horizon.terminated
    assert horizon.truncated


def test_conformance_rejects_observation_schema_drift() -> None:
    class DriftingObservationEnvironment(ToyLevelEnvironment):
        @override
        def step(self, plant_state: Array, action: Array) -> StepResult[Array, Array]:
            result = super().step(plant_state, action)
            return replace(result, observation=jnp.append(result.observation, 0.0))

    with pytest.raises(AssertionError, match="shape and dtype"):
        assert_environment_conforms(
            DriftingObservationEnvironment(),
            lambda _step, _reset: jnp.asarray([1.0], dtype=jnp.float32),
        )


def test_runtime_metadata_rejects_invalid_control_interval() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="integer multiple"):
        RuntimeMetadata(spec.identity, "1.0.0", "1.0.0", 0.3, 0.5, 10)
