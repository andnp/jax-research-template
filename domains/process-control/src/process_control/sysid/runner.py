"""JAX-native execution of process-control traces through replay adapters."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from process_control.sysid.replay import (
    ReplayCase,
    ReplayEnvironmentFactory,
    ReplayResult,
    ReplayResultArrays,
    column_indices,
)
from process_control.sysid.trace import SignalSpec


def _select_columns(
    values: jax.Array,
    specs: tuple[SignalSpec, ...],
    names: Sequence[str],
    rows: slice,
) -> jax.Array:
    selected = values[rows]
    indices = column_indices(specs, tuple(names))
    if not indices:
        return jnp.zeros((selected.shape[0], 0), dtype=selected.dtype)
    return selected[:, jnp.asarray(indices)]


def run_replay[StateT](
    case: ReplayCase,
    environment_factory: ReplayEnvironmentFactory[StateT],
) -> ReplayResult[StateT]:
    """Run one canonical trace through a benchmark-specific replay adapter.

    Required-sample validity and segment-boundary checks already happened at
    `ReplayCase` construction (eager, since they depend only on the static
    trace, never on this call's `environment_factory`) so this stays safe to
    call from inside a `jax.jit`-traced function.
    """
    control_specs = case.trace.schema.controls
    disturbance_specs = case.trace.schema.disturbances
    if case.required_controls is None or case.required_disturbances is None:
        raise RuntimeError("replay requirements were not initialized")
    required_controls = case.required_controls
    required_disturbances = case.required_disturbances

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
        carry: tuple[StateT, jax.Array],
        item: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[StateT, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array, Mapping[str, jax.Array]],
    ]:
        state, active = carry
        action, key = item
        next_state, observation, reward, terminated, diagnostics = environment.step(
            state,
            action,
            key,
        )
        def select_active_state(new: jax.Array, old: jax.Array) -> jax.Array:
            return jnp.where(active, new, old)

        next_state = jax.tree.map(select_active_state, next_state, state)
        observation = jnp.where(active, observation, jnp.zeros_like(observation))
        reward = jnp.where(active, reward, jnp.zeros_like(reward))
        terminated = active & terminated

        def zero_when_inactive(value: jax.Array) -> jax.Array:
            return jnp.where(active, value, jnp.zeros_like(value))

        diagnostics = jax.tree.map(zero_when_inactive, diagnostics)
        return (next_state, active & ~terminated), (observation, reward, terminated, diagnostics)

    (final_state, _), (observations, rewards, terminated, diagnostics) = jax.lax.scan(
        scan_step,
        (state, jnp.array(True)),
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
