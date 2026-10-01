"""Run one batch of configs as a single compiled ``jit(vmap(train))``."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from experiment_definition.metric import MetricSpec
from research_instrument.thinning import thin_metric

from .config import _merge_traced, _split_traced, _walk

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type _Train[C] = Callable[[C, jax.Array], tuple[object, Mapping[str, jax.Array]]]

_DTYPES = {bool: jnp.bool_, int: jnp.int32, float: jnp.float32}
_RUNNER_METRICS = ("runner/compile_seconds", "runner/execute_seconds")


def _run_batch[C: DataclassInstance](
    train: _Train[C], metrics: Sequence[MetricSpec], configs: Sequence[C], seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """Run every config with its seed in one ``jit(vmap(train))``; return each declared metric, run-major.

    The batch's own ``runner/compile_seconds`` and ``runner/execute_seconds`` are returned as
    scalars when declared.
    """
    if not configs:
        raise ValueError("a batch needs at least one config")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"seeds must be integers, got {list(seeds)}")
    if len(configs) != len(seeds):
        raise ValueError(f"a batch needs one seed per config, got {len(configs)} configs and {len(seeds)} seeds")
    static, _ = _split_traced(configs[0])
    values = []
    for config in configs:
        other, traced = _split_traced(config)
        if other != static:
            raise ValueError(f"configs in one batch must differ only in traced fields; {_first_difference(static, other)} differs")
        values.append(traced)
    stacked = {path: jnp.asarray([v[path] for v in values], _DTYPES[type(field.default)]) for path, field in _walk(static)}
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    for metric in metrics:
        if metric.name in _RUNNER_METRICS and (metric.every != 1 or metric.ema is not None):
            raise ValueError(f"metric {metric.name!r} is one value per batch, so it takes neither every nor ema")

    def one(traced: dict[str, jax.Array], key: jax.Array) -> dict[str, jax.Array]:
        _, emitted = train(_merge_traced(static, traced), key)
        return {metric.name: _record(metric, emitted) for metric in metrics if metric.name not in _RUNNER_METRICS}

    start = time.perf_counter()
    compiled = jax.jit(jax.vmap(one)).lower(stacked, keys).compile()
    compiled_at = time.perf_counter()
    outputs = {name: np.asarray(value) for name, value in jax.device_get(compiled(stacked, keys)).items()}
    timings = {"runner/compile_seconds": compiled_at - start, "runner/execute_seconds": time.perf_counter() - compiled_at}
    return outputs | {metric.name: np.asarray(timings[metric.name]) for metric in metrics if metric.name in timings}


def _record(metric: MetricSpec, emitted: Mapping[str, jax.Array]) -> jax.Array:
    """One run's declared metric: smoothed and thinned if per-step, as emitted if per-run."""
    if metric.name not in emitted:
        raise ValueError(f"metric {metric.name!r} is declared but not emitted; emitted metrics are {sorted([*emitted, *_RUNNER_METRICS])}")
    value = emitted[metric.name]
    if value.ndim > 0:
        return thin_metric(value, ema=metric.ema, every=metric.every)
    if metric.every != 1 or metric.ema is not None:
        raise ValueError(f"metric {metric.name!r} is one value per run, so it takes neither every nor ema")
    return value


def _first_difference(a: object, b: object, path: str = "") -> str | None:
    """The dotted path of the first field where two configs differ, or None if they are equal."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type) and type(a) is type(b):
        for field in dataclasses.fields(a):
            found = _first_difference(getattr(a, field.name), getattr(b, field.name), f"{path}{field.name}.")
            if found is not None:
                return found
        return None
    return None if a == b else path.rstrip(".") or type(a).__name__
