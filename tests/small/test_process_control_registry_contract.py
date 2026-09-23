import importlib

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkEntry,
    get_action_bounds,
)


def _make_benchmark(entry: BenchmarkEntry):
    module = importlib.import_module(entry.module)
    config = getattr(module, entry.config_cls)()
    return getattr(module, entry.make_fn)(config), config


def _mid_range_action(entry: BenchmarkEntry, config: object):
    low, high = get_action_bounds(entry, config)
    if low is None or high is None:
        values = jnp.full((entry.action_dim,), 0.5)
    elif isinstance(low, tuple) and isinstance(high, tuple):
        values = jnp.asarray([(left + right) / 2 for left, right in zip(low, high, strict=True)])
    else:
        assert isinstance(low, float) and isinstance(high, float)
        values = jnp.asarray([(low + high) / 2])
    return values[0] if entry.scalar_action else values


@pytest.mark.parametrize("entry", BENCHMARK_REGISTRY, ids=lambda entry: entry.name)
def test_registry_resets_are_jit_traceable(entry: BenchmarkEntry):
    """Every registered reset compiles, returns its declared observation, and batches."""
    (reset, _), _config = _make_benchmark(entry)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    state, obs = jax.jit(reset)(keys[0])
    batched_states, batched_obs = jax.vmap(reset)(keys)

    assert obs.shape == (entry.obs_dim,)
    assert batched_obs.shape == (2, entry.obs_dim)
    assert jax.tree.leaves(state)
    assert jax.tree.leaves(batched_states)
