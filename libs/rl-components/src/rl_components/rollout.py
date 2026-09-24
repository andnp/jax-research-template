"""Generic JAX mechanics for batched stateful-controller rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple, Protocol

import chex
import jax

from rl_components.env_protocol import EnvReset, EnvStep

ResetPolicy = Literal["ignore", "auto_reset"]
BatchedRNGMode = Literal["two-way", "three-way"]

type Telemetry = Mapping[str, jax.Array]


class BatchedEnvironment[ObservationT, EnvStateT, ActionT](Protocol):
    """Environment methods required by the batched rollout kernel."""

    def reset(self, key: chex.PRNGKey) -> EnvReset[ObservationT, EnvStateT]: ...

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvStateT,
        action: ActionT,
    ) -> EnvStep[ObservationT, EnvStateT]: ...


class ActionStep[ObservationT, ControllerStateT, ActionT](Protocol):
    """Stateful action callback used by one scan iteration."""

    def __call__(
        self,
        observations: ObservationT,
        controller_state: ControllerStateT,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[ActionT, ControllerStateT]: ...


class ControllerReset[ControllerStateT](Protocol):
    """Reset callback used by explicit episodic auto-reset."""

    def __call__(self, key: jax.Array) -> ControllerStateT: ...


class TelemetryHook(Protocol):
    """Callback for adding static-key traced telemetry."""

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        previous_action: jax.Array,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        info: Telemetry,
    ) -> Telemetry: ...


class BatchedRolloutCarry[EnvStateT, ControllerStateT](NamedTuple):
    """JAX-compatible state for a seed-batched rollout."""

    observations: jax.Array
    environment_state: EnvStateT
    controller_state: ControllerStateT
    rng: jax.Array
    previous_action: jax.Array


class BatchedRolloutResult[EnvStateT, ControllerStateT](NamedTuple):
    """Final carry, generic metrics, boundaries, and environment diagnostics."""

    final_carry: BatchedRolloutCarry[EnvStateT, ControllerStateT]
    metrics: dict[str, jax.Array]
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]

    @property
    def done(self) -> jax.Array:
        """Return the combined environment boundary signal."""
        return self.terminated | self.truncated


class _RolloutFrame(NamedTuple):
    metrics: dict[str, jax.Array]
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]


ActionStepFn = Callable[
    [jax.Array, object, jax.Array, jax.Array, jax.Array | None],
    tuple[jax.Array, object],
]


def _normalize_environment_action(
    action: jax.Array,
    batch_size: int,
    action_shape: tuple[int, ...],
) -> jax.Array:
    """Normalize callback actions to the environment's batched shape."""
    array = jax.numpy.asarray(action)
    target_shape = (batch_size, *action_shape)
    if array.ndim == 0 or array.shape == action_shape:
        return jax.numpy.broadcast_to(array, target_shape)
    if array.shape == target_shape:
        return array
    if action_shape == () and array.shape == (batch_size, 1):
        return array[:, 0]
    if action_shape == (1,) and array.shape == (batch_size,):
        return array[:, None]
    raise ValueError(
        f"action shape {array.shape} cannot represent batched action shape "
        f"{target_shape}"
    )


def _normalize_callback_previous_action(
    previous_action: jax.Array,
    batch_size: int,
    action_shape: tuple[int, ...],
) -> jax.Array:
    """Normalize previous actions while preserving scalar callback inputs."""
    array = jax.numpy.asarray(previous_action)
    if action_shape in ((), (1,)):
        if array.ndim == 0 or array.shape == action_shape:
            return jax.numpy.broadcast_to(array, (batch_size,))
        if array.shape == (batch_size,):
            return array
        if array.shape == (batch_size, 1):
            return array[:, 0]
        raise ValueError(
            f"previous action shape {array.shape} cannot represent "
            f"batch size {batch_size}"
        )
    return _normalize_environment_action(array, batch_size, action_shape)


def _metric_action(actions: jax.Array, action_shape: tuple[int, ...]) -> jax.Array:
    if action_shape == (1,):
        return actions[:, 0]
    return actions


def make_fresh_batched_carry(
    environment: BatchedEnvironment,
    seed_keys: jax.Array,
    *,
    previous_action: jax.Array,
    controller_state: object = None,
    action_shape: tuple[int, ...],
) -> BatchedRolloutCarry:
    """Create fresh state using separate reset and rollout RNG streams."""

    def split_two(key: jax.Array) -> jax.Array:
        return jax.random.split(key, 2)

    split_keys = jax.vmap(split_two)(seed_keys)
    resets = jax.vmap(environment.reset)(split_keys[:, 0])
    batch_size = seed_keys.shape[0]
    return BatchedRolloutCarry(
        resets.observation,
        resets.state,
        controller_state,
        split_keys[:, 1],
        _normalize_callback_previous_action(previous_action, batch_size, action_shape),
    )


def run_batched_rollout(
    environment: BatchedEnvironment,
    initial_carry: BatchedRolloutCarry,
    *,
    steps: int,
    action_step: ActionStep,
    rng_mode: BatchedRNGMode,
    capture_action: bool,
    mask_step_zero_delta: bool,
    telemetry_hook: TelemetryHook | None = None,
    telemetry_keys: tuple[str, ...] | None = None,
    action_shape: tuple[int, ...],
    step_offset: int | jax.Array,
    reset_policy: ResetPolicy = "ignore",
    controller_reset: ControllerReset | None = None,
) -> BatchedRolloutResult:
    """Scan a fixed horizon while preserving boundary and info sidecars."""
    if steps < 0:
        raise ValueError("steps must not be negative")
    if rng_mode not in ("two-way", "three-way"):
        raise ValueError(f"unsupported RNG mode: {rng_mode}")
    if reset_policy not in ("ignore", "auto_reset"):
        raise ValueError(f"unsupported reset policy: {reset_policy}")
    if reset_policy == "auto_reset" and controller_reset is None:
        raise ValueError("auto_reset requires controller_reset")

    def reset_controller_state(key: jax.Array):
        if controller_reset is None:
            raise ValueError("auto_reset requires controller_reset")
        return controller_reset(key)

    batch_size = initial_carry.observations.shape[0]
    initial_carry = initial_carry._replace(
        previous_action=_normalize_callback_previous_action(
            initial_carry.previous_action,
            batch_size,
            action_shape,
        )
    )

    def step(
        carry: BatchedRolloutCarry,
        step_index: jax.Array,
    ) -> tuple[BatchedRolloutCarry, _RolloutFrame]:
        observations, environment_state, controller_state, rng, previous_action = carry
        previous_action = _normalize_callback_previous_action(
            previous_action,
            batch_size,
            action_shape,
        )
        previous_environment_action = _normalize_environment_action(
            previous_action,
            batch_size,
            action_shape,
        )
        if rng_mode == "three-way":
            split_count = 4 if reset_policy == "auto_reset" else 3

            def split_n(key: jax.Array, split_count: int = split_count) -> jax.Array:
                return jax.random.split(key, split_count)

            split_keys = jax.vmap(split_n)(rng)
            action_keys = split_keys[:, 0]
            environment_keys = split_keys[:, 1]
            reset_keys = split_keys[:, 2] if reset_policy == "auto_reset" else None
            next_rng = split_keys[:, -1]
        else:
            split_count = 3 if reset_policy == "auto_reset" else 2

            def split_n(key: jax.Array, split_count: int = split_count) -> jax.Array:
                return jax.random.split(key, split_count)

            split_keys = jax.vmap(split_n)(rng)
            action_keys = None
            environment_keys = split_keys[:, 0]
            reset_keys = split_keys[:, 1] if reset_policy == "auto_reset" else None
            next_rng = split_keys[:, -1]

        actions, next_controller_state = action_step(
            observations,
            controller_state,
            previous_action,
            step_index + step_offset,
            action_keys,
        )
        actions = _normalize_environment_action(actions, batch_size, action_shape)
        env_step = jax.vmap(environment.step)(
            environment_keys,
            environment_state,
            actions,
        )
        if reset_policy == "auto_reset":
            assert reset_keys is not None
            def split_two(key: jax.Array) -> jax.Array:
                return jax.random.split(key, 2)

            reset_split_keys = jax.vmap(split_two)(reset_keys)
            reset_results = jax.vmap(environment.reset)(reset_split_keys[:, 0])
            reset_states = jax.vmap(reset_controller_state)(reset_split_keys[:, 1])
            done = env_step.terminated | env_step.truncated

            def select_rows(
                reset_value: jax.Array,
                continuation_value: jax.Array,
            ) -> jax.Array:
                mask = done.reshape((batch_size, *([1] * (reset_value.ndim - 1))))
                return jax.numpy.where(mask, reset_value, continuation_value)

            next_observations = select_rows(
                reset_results.observation,
                env_step.observation,
            )
            next_environment_state = jax.tree_util.tree_map(
                select_rows,
                reset_results.state,
                env_step.state,
            )
            next_controller_state = jax.tree_util.tree_map(
                select_rows,
                reset_states,
                next_controller_state,
            )
        else:
            next_observations = env_step.observation
            next_environment_state = env_step.state
        action_delta = actions - previous_environment_action
        action_delta_norm = (
            jax.numpy.abs(action_delta)
            if action_shape == ()
            else jax.numpy.linalg.norm(action_delta, axis=tuple(range(1, actions.ndim)))
        )
        action_delta_norm = jax.numpy.where(
            jax.numpy.asarray(mask_step_zero_delta) & (step_index == 0),
            0.0,
            action_delta_norm,
        )
        metrics = {
            "reward": jax.numpy.asarray(env_step.reward),
            "observation": env_step.observation,
            "action_delta_norm": action_delta_norm,
        }
        if capture_action:
            metrics["action"] = _metric_action(actions, action_shape)
        telemetry = {} if telemetry_hook is None else dict(
            telemetry_hook(
                env_step.observation,
                actions,
                previous_environment_action,
                env_step.reward,
                env_step.terminated,
                env_step.truncated,
                env_step.info,
            )
        )
        if telemetry_keys is not None and set(telemetry) != set(telemetry_keys):
            raise ValueError(
                f"telemetry keys {sorted(telemetry)} do not match declared keys "
                f"{sorted(telemetry_keys)}"
            )
        collisions = set(metrics).intersection(telemetry)
        if collisions:
            raise ValueError(
                f"telemetry collides with rollout metrics: {sorted(collisions)}"
            )
        metrics.update(telemetry)
        return (
            BatchedRolloutCarry(
                next_observations,
                next_environment_state,
                next_controller_state,
                next_rng,
                _normalize_callback_previous_action(actions, batch_size, action_shape),
            ),
            _RolloutFrame(
                metrics,
                jax.numpy.asarray(env_step.terminated, dtype=bool),
                jax.numpy.asarray(env_step.truncated, dtype=bool),
                dict(env_step.info),
            ),
        )

    final_carry, frames = jax.lax.scan(step, initial_carry, jax.numpy.arange(steps))
    return BatchedRolloutResult(
        final_carry,
        frames.metrics,
        frames.terminated,
        frames.truncated,
        frames.info,
    )
