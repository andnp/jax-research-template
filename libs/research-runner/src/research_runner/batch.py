"""Run one batch of configs as a single compiled ``jit(vmap(train))``."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from experiment_definition.metric import MetricSpec

from .config import _merge_traced, _split_traced, _walk

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type _Train[C] = Callable[[C, jax.Array], tuple[object, Mapping[str, jax.Array]]]

_DTYPES = {bool: jnp.bool_, int: jnp.int32, float: jnp.float32}


def _run_batch[C: DataclassInstance](
    train: _Train[C], metrics: Sequence[MetricSpec], configs: Sequence[C], seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """Run every config with its seed in one ``jit(vmap(train))``; return each declared metric, run-major."""
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

    def one(traced: dict[str, jax.Array], key: jax.Array) -> dict[str, jax.Array]:
        _, emitted = train(_merge_traced(static, traced), key)
        return {metric.name: _require(emitted, metric.name) for metric in metrics}

    return {name: np.asarray(value) for name, value in jax.device_get(jax.jit(jax.vmap(one))(stacked, keys)).items()}


def _require(emitted: Mapping[str, jax.Array], name: str) -> jax.Array:
    if name not in emitted:
        raise ValueError(f"metric {name!r} is declared but not emitted; emitted metrics are {sorted(emitted)}")
    return emitted[name]


def _first_difference(a: object, b: object, path: str = "") -> str | None:
    """The dotted path of the first field where two configs differ, or None if they are equal."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type) and type(a) is type(b):
        for field in dataclasses.fields(a):
            found = _first_difference(getattr(a, field.name), getattr(b, field.name), f"{path}{field.name}.")
            if found is not None:
                return found
        return None
    return None if a == b else path.rstrip(".") or type(a).__name__
