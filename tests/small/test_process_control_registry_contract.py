import importlib

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkEntry,
    get_action_bounds,
    get_action_dim,
    get_observation_dim,
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
    with jax.enable_x64(entry.requires_x64):
        (reset, _), _config = _make_benchmark(entry)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)

        state, obs = jax.jit(reset)(keys[0])
        batched_states, batched_obs = jax.vmap(reset)(keys)

        obs_dim = get_observation_dim(reset)

    assert obs.shape == (obs_dim,)
    assert batched_obs.shape == (2, obs_dim)
    assert jax.tree.leaves(state)
    assert jax.tree.leaves(batched_states)


@pytest.mark.parametrize("entry", BENCHMARK_REGISTRY, ids=lambda entry: entry.name)
def test_registry_entries_declare_action_bounds(entry: BenchmarkEntry) -> None:
    """Every registered benchmark exposes finite action bounds, one pair per action."""
    module = importlib.import_module(entry.module)
    config = getattr(module, entry.config_cls)()
    low, high = get_action_bounds(entry, config)

    assert get_action_dim(entry, config) == entry.action_dim
    assert low is not None
    assert high is not None
    if isinstance(low, tuple) and isinstance(high, tuple):
        assert all(left < right for left, right in zip(low, high, strict=True))
    else:
        assert isinstance(low, float) and isinstance(high, float)
        assert low < high


@pytest.mark.parametrize("entry", BENCHMARK_REGISTRY, ids=lambda entry: entry.name)
def test_registry_rollouts_stay_finite(entry: BenchmarkEntry):
    """Every registered benchmark stays finite during a compiled short rollout."""
    with jax.enable_x64(entry.requires_x64):
        _assert_rollout_finite(entry)


def _assert_rollout_finite(entry: BenchmarkEntry) -> None:
    (reset, step), config = _make_benchmark(entry)
    action = _mid_range_action(entry, config)
    state, _ = reset(jax.random.PRNGKey(1))

    def scan_step(carry: tuple[object, jax.Array], step_index: jax.Array):
        current_state, key = carry
        next_key = jax.random.fold_in(key, step_index)
        next_state, obs, reward, _done, _info = step(current_state, action, next_key)
        return (next_state, key), (obs, reward)

    def rollout(s: object):
        return jax.lax.scan(scan_step, (s, jax.random.PRNGKey(2)), jnp.arange(200))

    (_, _), (obs, reward) = jax.jit(rollout)(state)

    assert obs.shape == (200, get_observation_dim(reset))
    assert jnp.all(jnp.isfinite(obs))
    assert jnp.all(jnp.isfinite(reward))


@pytest.mark.parametrize("name", ["membrane_fouling", "combined_np", "bsm1_takacs", "dewatering", "bsm1", "bsm1_lt", "bsm1_recycle", "bsm1_combined"])
def test_selected_registry_resets_report_initial_observation(name: str) -> None:
    """Selected benchmark resets expose their initialized observation."""
    entry = next(item for item in BENCHMARK_REGISTRY if item.name == name)
    with jax.enable_x64(entry.requires_x64):
        (reset, _), _config = _make_benchmark(entry)
        _state, obs = reset(jax.random.PRNGKey(3))

    assert jnp.any(obs != 0.0)
