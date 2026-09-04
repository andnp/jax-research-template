"""JAX-native execution of process-control traces through replay adapters."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from process_control.sysid.replay import (
    ReplayCase,
    ReplayEnvironmentFactory,
    ReplayResult,
    ReplayResultArrays,
)
from process_control.sysid.trace import SignalSpec


def _column_indices(specs: tuple[SignalSpec, ...], names: Sequence[str]) -> tuple[int, ...]:
    indices = {spec.name: index for index, spec in enumerate(specs)}
    return tuple(indices[name] for name in names)


def _select_columns(
    values: jax.Array,
    specs: tuple[SignalSpec, ...],
    names: Sequence[str],
    rows: slice,
) -> jax.Array:
    selected = values[rows]
    indices = _column_indices(specs, names)
    if not indices:
        return jnp.zeros((selected.shape[0], 0), dtype=selected.dtype)
    return selected[:, jnp.asarray(indices)]


def _require_valid(
    case: ReplayCase,
    specs: tuple[SignalSpec, ...],
    names: Sequence[str],
    offset: int,
) -> None:
    indices = _column_indices(specs, names)
    if not indices:
        return
    valid = case.trace.arrays.valid[case.rows, jnp.asarray(indices) + offset]
    if not bool(jnp.all(valid)):
        raise ValueError("replay inputs contain invalid required samples")


def run_replay[StateT](
    case: ReplayCase,
    environment_factory: ReplayEnvironmentFactory[StateT],
) -> ReplayResult[StateT]:
    """Run one canonical trace through a benchmark-specific replay adapter."""
    control_specs = case.trace.schema.controls
    disturbance_specs = case.trace.schema.disturbances
    if case.required_controls is None or case.required_disturbances is None:
        raise RuntimeError("replay requirements were not initialized")
    required_controls = case.required_controls
    required_disturbances = case.required_disturbances
    _require_valid(case, control_specs, required_controls, 0)
    _require_valid(case, disturbance_specs, required_disturbances, len(control_specs))

    controls = _select_columns(
        case.trace.arrays.controls,
        control_specs,
        required_controls,
        case.rows,
    )
    disturbances = _select_columns(
        case.trace.arrays.disturbances,
        disturbance_specs,
        required_disturbances,
        case.rows,
    )
    environment = environment_factory(case)
    keys = jax.random.split(jax.random.key(case.seed), case.horizon + 1)
    state, _ = environment.reset(keys[0])

    def scan_step(
        state: StateT,
        item: tuple[jax.Array, jax.Array],
    ) -> tuple[StateT, tuple[jax.Array, jax.Array, jax.Array, Mapping[str, jax.Array]]]:
        action, key = item
        next_state, observation, reward, terminated, diagnostics = environment.step(
            state,
            action,
            key,
        )
        return next_state, (observation, reward, terminated, diagnostics)

    final_state, (observations, rewards, terminated, diagnostics) = jax.lax.scan(
        scan_step,
        state,
        (controls, keys[1:]),
    )
    arrays = ReplayResultArrays(
        time=case.trace.arrays.time[case.rows],
        controls=controls,
        disturbances=disturbances,
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=jnp.zeros_like(terminated),
        diagnostics=diagnostics,
    )
    return ReplayResult(case=case, arrays=arrays, final_state=final_state)
