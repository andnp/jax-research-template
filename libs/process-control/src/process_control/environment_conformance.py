"""Reusable conformance checks for process-control environments."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from process_control.environment import ProcessControlEnvironment, ResetResult, StepResult


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    leaves = jax.tree.leaves(tree)
    return tuple((tuple(jnp.shape(leaf)), str(jnp.asarray(leaf).dtype)) for leaf in leaves)


def _trees_equal(left: object, right: object) -> bool:
    left_structure = jax.tree.structure(left)
    if left_structure != jax.tree.structure(right):
        return False
    return all(
        bool(jnp.array_equal(left_leaf, right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree.leaves(left), jax.tree.leaves(right), strict=True
        )
    )


def _check_timing[PlantStateT, ObservationT](
    result: ResetResult[PlantStateT, ObservationT] | StepResult[PlantStateT, ObservationT],
    *,
    step_index: int,
    control_interval: float,
) -> None:
    assert result.timing.step_index == step_index, "step index must advance once per control step"
    expected_time = step_index * control_interval
    assert abs(result.timing.elapsed_time - expected_time) <= 1e-9, (
        "elapsed time must equal step_index * control_interval"
    )


def _check_named_outputs[PlantStateT, ObservationT](
    result: StepResult[PlantStateT, ObservationT],
    *,
    observed_names: set[str],
    reward_names: set[str],
    evaluation_names: set[str],
) -> None:
    assert set(result.reward_inputs) == reward_names, (
        "reward_inputs must exactly identify the task's declared reward inputs"
    )
    assert set(result.reward_inputs) <= observed_names, "reward inputs must be controller-visible"
    assert set(result.evaluation_metrics) == evaluation_names, (
        "evaluation_metrics must exactly match the task's declared evaluation inputs"
    )
    assert not (set(result.reward_inputs) & (evaluation_names - observed_names)), (
        "latent evaluation signals cannot feed learning reward"
    )


def assert_environment_conforms[PlantStateT, ObservationT, ActionT](
    environment: ProcessControlEnvironment[PlantStateT, ObservationT, ActionT],
    action_for_step: Callable[[int, ResetResult[PlantStateT, ObservationT]], ActionT],
    *,
    seed: int = 0,
) -> None:
    """Assert core reset, timing, schema, and signal-separation invariants."""
    environment.runtime.validate_spec(environment.spec)
    first = environment.reset(seed)
    repeated = environment.reset(seed)
    _check_timing(first, step_index=0, control_interval=environment.runtime.control_interval)
    assert _trees_equal(first.plant_state, repeated.plant_state), (
        "reset with the same seed must reproduce physical state"
    )
    assert _trees_equal(first.observation, repeated.observation), (
        "reset with the same seed must reproduce observations"
    )
    assert _tree_signature(first.observation) == _tree_signature(repeated.observation)

    observed_names = {
        signal.name for signal in environment.spec.instrumentation.observed_signals
    }
    reward_names = set(environment.spec.task.reward_input_names)
    evaluation_names = set(environment.spec.task.evaluation_input_names)
    assert set(first.evaluation_metrics) == evaluation_names, (
        "reset evaluation_metrics must match the task's declared evaluation inputs"
    )
    reset_signature = _tree_signature(first.observation)
    reward_signature: tuple[tuple[tuple[int, ...], str], ...] | None = None
    state = first.plant_state
    for step_index in range(1, environment.runtime.horizon_steps + 1):
        result = environment.step(state, action_for_step(step_index, first))
        assert _tree_signature(result.observation) == reset_signature, (
            "observation shape and dtype must remain stable"
        )
        current_reward_signature = _tree_signature(result.reward)
        assert len(current_reward_signature) == 1 and current_reward_signature[0][0] == (), (
            "reward must be scalar"
        )
        if reward_signature is None:
            reward_signature = current_reward_signature
        else:
            assert current_reward_signature == reward_signature, "reward dtype must remain stable"
        assert type(result.terminated) is bool, "terminated must be a bool"
        assert type(result.truncated) is bool, "truncated must be a bool"
        _check_timing(
            result,
            step_index=step_index,
            control_interval=environment.runtime.control_interval,
        )
        _check_named_outputs(
            result,
            observed_names=observed_names,
            reward_names=reward_names,
            evaluation_names=evaluation_names,
        )
        if step_index < environment.runtime.horizon_steps:
            assert not result.truncated, "time-limit truncation occurred before the declared horizon"
        if result.terminated or result.truncated:
            assert step_index == environment.runtime.horizon_steps or result.terminated, (
                "episodes may end early only through task/plant termination"
            )
            return
        state = result.plant_state

    raise AssertionError("environment did not terminate or truncate by its declared horizon")


__all__ = ["assert_environment_conforms"]
