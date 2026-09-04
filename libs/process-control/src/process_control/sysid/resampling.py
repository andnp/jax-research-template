"""Causal alignment of native-rate process signals onto a uniform grid."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp

from process_control.sysid.trace import (
    SignalRole,
    SignalSeries,
    TraceSchema,
    UniformTrace,
    UniformTraceArrays,
)


class ResamplingMethod(StrEnum):
    """Supported deterministic signal-alignment methods."""

    LAST = "last"
    MEAN = "mean"
    LINEAR = "linear"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ResamplingPolicy:
    """Alignment method and optional age/span limit for source samples."""

    method: ResamplingMethod
    max_gap: float | None = None

    def __post_init__(self) -> None:
        if self.max_gap is not None and self.max_gap <= 0.0:
            raise ValueError("max_gap must be positive when provided")
        if self.method is ResamplingMethod.LINEAR and self.max_gap is None:
            raise ValueError("linear resampling requires max_gap")


def default_policy(role: SignalRole) -> ResamplingPolicy:
    """Return the conservative policy associated with one process role."""
    if role is SignalRole.CONTROL:
        return ResamplingPolicy(ResamplingMethod.LAST)
    return ResamplingPolicy(ResamplingMethod.MEAN)


def canonicalize(
    signals: tuple[SignalSeries, ...],
    grid: jax.Array,
    sample_period: float,
    policies: Mapping[str, ResamplingPolicy] | None = None,
) -> UniformTrace:
    """Align native-rate signals onto a validated fixed-step trace."""
    if not signals:
        raise ValueError("at least one signal is required")
    grid = jnp.asarray(grid)
    if grid.ndim != 1 or grid.shape[0] == 0:
        raise ValueError("grid must be a non-empty vector")
    if not bool(jnp.all(jnp.isfinite(grid))):
        raise ValueError("grid timestamps must be finite")
    if not bool(
        jnp.all(
            jnp.isclose(
                jnp.diff(grid),
                sample_period,
                rtol=0.0,
                atol=sample_period * 1e-9,
            )
        )
    ):
        raise ValueError("grid timestamps must use sample_period spacing")

    resolved_policies = policies or {}
    unknown = set(resolved_policies) - {signal.spec.name for signal in signals}
    if unknown:
        raise ValueError(f"policies refer to unknown signals: {sorted(unknown)}")

    groups: dict[SignalRole, list[SignalSeries]] = {
        role: [] for role in SignalRole
    }
    matrices: dict[SignalRole, list[jax.Array]] = {role: [] for role in SignalRole}
    validity: dict[SignalRole, list[jax.Array]] = {role: [] for role in SignalRole}
    source_counts: dict[SignalRole, list[jax.Array]] = {role: [] for role in SignalRole}
    for signal in signals:
        groups[signal.spec.role].append(signal)
        policy = resolved_policies.get(signal.spec.name, default_policy(signal.spec.role))
        values, valid, count = resample_signal(signal, grid, sample_period, policy)
        matrices[signal.spec.role].append(values[:, None])
        validity[signal.spec.role].append(valid[:, None])
        source_counts[signal.spec.role].append(count[:, None])

    schema = TraceSchema(
        controls=tuple(signal.spec for signal in groups[SignalRole.CONTROL]),
        disturbances=tuple(signal.spec for signal in groups[SignalRole.DISTURBANCE]),
        measurements=tuple(signal.spec for signal in groups[SignalRole.MEASUREMENT]),
        actuators=tuple(signal.spec for signal in groups[SignalRole.ACTUATOR]),
    )
    role_order = (
        SignalRole.CONTROL,
        SignalRole.DISTURBANCE,
        SignalRole.MEASUREMENT,
        SignalRole.ACTUATOR,
    )
    valid = jnp.concatenate(
        [column for role in role_order for column in validity[role]],
        axis=1,
    )
    source_count = jnp.concatenate(
        [column for role in role_order for column in source_counts[role]],
        axis=1,
    )
    complete = jnp.all(valid, axis=1)
    starts = ~complete & jnp.concatenate((jnp.array([True]), complete[:-1]))
    segment_id = jnp.cumsum(starts, dtype=jnp.int32)
    arrays = UniformTraceArrays(
        time=grid,
        controls=_stack_group(matrices[SignalRole.CONTROL], grid.shape[0]),
        disturbances=_stack_group(matrices[SignalRole.DISTURBANCE], grid.shape[0]),
        measurements=_stack_group(matrices[SignalRole.MEASUREMENT], grid.shape[0]),
        actuators=_stack_group(matrices[SignalRole.ACTUATOR], grid.shape[0]),
        valid=valid,
        source_count=source_count,
        segment_id=segment_id,
    )
    return UniformTrace(arrays=arrays, schema=schema, sample_period=sample_period)


def resample_signal(
    signal: SignalSeries,
    grid: jax.Array,
    sample_period: float,
    policy: ResamplingPolicy,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return aligned values, validity, and source counts for one signal."""
    if policy.method is ResamplingMethod.LAST:
        return _last(signal, grid, policy.max_gap)
    if policy.method is ResamplingMethod.MEAN:
        return _mean(signal, grid, sample_period)
    if policy.method is ResamplingMethod.LINEAR:
        return _linear(signal, grid, sample_period, policy.max_gap)
    return _invalid(signal, grid)


def _last(
    signal: SignalSeries,
    grid: jax.Array,
    max_gap: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = jnp.searchsorted(signal.time, grid, side="right") - 1
    has_source = indices >= 0
    safe_indices = jnp.maximum(indices, 0)
    age = grid - signal.time[safe_indices]
    valid = has_source & signal.validity[safe_indices]
    if max_gap is not None:
        valid &= age <= max_gap
    values = jnp.where(valid, signal.values[safe_indices], jnp.nan)
    count = jnp.where(valid, 1, 0).astype(jnp.int32)
    return values, valid, count


def _mean(
    signal: SignalSeries,
    grid: jax.Array,
    sample_period: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    starts = jnp.searchsorted(signal.time, grid, side="left")
    ends = jnp.searchsorted(signal.time, grid + sample_period, side="left")
    valid_values = jnp.where(signal.validity, signal.values, 0.0)
    cumulative_values = jnp.concatenate((jnp.zeros(1), jnp.cumsum(valid_values)))
    cumulative_counts = jnp.concatenate(
        (jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(signal.validity, dtype=jnp.int32))
    )
    counts = cumulative_counts[ends] - cumulative_counts[starts]
    sums = cumulative_values[ends] - cumulative_values[starts]
    valid = counts > 0
    values = jnp.where(valid, sums / jnp.maximum(counts, 1), jnp.nan)
    return values, valid, counts


def _linear(
    signal: SignalSeries,
    grid: jax.Array,
    sample_period: float,
    max_gap: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    right = jnp.searchsorted(signal.time, grid, side="left")
    right_exists = right < signal.time.shape[0]
    left = jnp.maximum(right - 1, 0)
    safe_right = jnp.minimum(right, signal.time.shape[0] - 1)
    left_time = signal.time[left]
    right_time = signal.time[safe_right]
    span = right_time - left_time
    bracketed = (
        right_exists
        & signal.validity[left]
        & signal.validity[safe_right]
        & (right_time <= grid + sample_period)
        & (span > 0.0)
    )
    if max_gap is not None:
        bracketed &= span <= max_gap
    exact = right_exists & jnp.isclose(
        right_time,
        grid,
        rtol=0.0,
        atol=sample_period * 1e-9,
    )
    valid = (bracketed | exact) & signal.validity[safe_right]
    weight = jnp.where(span > 0.0, (grid - left_time) / span, 0.0)
    interpolated = signal.values[left] + weight * (
        signal.values[safe_right] - signal.values[left]
    )
    values = jnp.where(valid, jnp.where(exact, signal.values[safe_right], interpolated), jnp.nan)
    count = jnp.where(valid, jnp.where(exact, 1, 2), 0).astype(jnp.int32)
    return values, valid, count


def _invalid(
    signal: SignalSeries,
    grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = jnp.searchsorted(signal.time, grid, side="left")
    exists = indices < signal.time.shape[0]
    safe_indices = jnp.minimum(indices, signal.time.shape[0] - 1)
    valid = (
        exists
        & jnp.isclose(
            signal.time[safe_indices],
            grid,
            rtol=0.0,
            atol=1e-9,
        )
        & signal.validity[safe_indices]
    )
    values = jnp.where(valid, signal.values[safe_indices], jnp.nan)
    return values, valid, jnp.where(valid, 1, 0).astype(jnp.int32)


def _stack_group(groups: list[jax.Array], steps: int) -> jax.Array:
    if not groups:
        return jnp.empty((steps, 0), dtype=jnp.float32)
    return jnp.concatenate(groups, axis=1)
