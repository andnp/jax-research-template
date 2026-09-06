"""Medium transform-compatibility tests for jax-eval replay diagnostics."""

import jax
import jax.numpy as jnp
from jax_eval.replay import priority_utilization


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
