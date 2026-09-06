"""Medium transform-compatibility tests for jax-eval replay diagnostics."""

import jax
import jax.numpy as jnp
from jax_eval.replay import priority_utilization, sample_staleness


def _staleness_pair(current: jax.Array, sampled: jax.Array, mask: jax.Array) -> jax.Array:
    return sample_staleness(current, sampled, mask)


def test_priority_utilization_runs_under_jit_and_vmap() -> None:
    """Compile independent priority distributions under jit and vmap."""
    priorities = jnp.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

    result = jax.jit(jax.vmap(priority_utilization))(priorities)

    assert jnp.allclose(result, jnp.array([1.0, 0.5]))


def test_priority_utilization_runs_inside_lax_scan() -> None:
    """Evaluate one priority utilization scalar per scan step under jit."""
    priorities = jnp.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )

    def step(_: jax.Array, values: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = priority_utilization(values)
        return jnp.float32(0.0), value

    def scan_priorities(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), values)

    _, result = jax.jit(scan_priorities)(priorities)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result[0], 1.0)
    assert result[1] > 0.0
    assert result[2] == 0.0


def test_sample_staleness_runs_under_jit_and_vmap() -> None:
    """Compile independent sample-age reductions under jit and vmap."""
    current = jnp.array([10, 20], dtype=jnp.uint32)
    sampled = jnp.array([[9, 11], [15, 25]], dtype=jnp.uint32)
    masks = jnp.array([[True, False], [True, False]])

    result = jax.jit(jax.vmap(_staleness_pair))(current, sampled, masks)

    assert jnp.allclose(result, jnp.array([1.0, 5.0]))


def test_sample_staleness_runs_inside_lax_scan() -> None:
    """Evaluate one masked sample-age scalar per scan step under jit."""
    current = jnp.array([10, 20, 30], dtype=jnp.int32)
    sampled = jnp.array([[9, 11], [18, 22], [30, 30]], dtype=jnp.int32)
    masks = jnp.array([[True, True], [True, False], [False, False]])

    def step(_: jax.Array, values: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value = sample_staleness(values[0], values[1], values[2])
        return jnp.float32(0.0), value

    def scan_staleness(
        steps: jax.Array, samples: jax.Array, valid: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), (steps, samples, valid))

    _, result = jax.jit(scan_staleness)(current, sampled, masks)

    assert jnp.allclose(result, jnp.array([0.0, 2.0, 0.0]))
