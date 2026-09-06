"""Medium transform-compatibility tests for jax-eval optimization diagnostics."""

import jax
import jax.numpy as jnp
from jax_eval.optimization import gradient_snr, update_ratio


def test_gradient_snr_runs_under_jit_and_vmap() -> None:
    """Compile independent gradient batches under jit and vmap."""
    grads = {"weight": jnp.array([[[1.0], [3.0]], [[2.0], [2.0]]])}

    result = jax.jit(jax.vmap(gradient_snr))(grads)

    assert jnp.allclose(result[0], 4.0)
    assert jnp.isfinite(result[1])
    assert result[1] > 1e6


def test_gradient_snr_runs_inside_lax_scan() -> None:
    """Evaluate one gradient-SNR scalar per scan step under jit."""
    grads = jnp.array([[[1.0], [3.0]], [[2.0], [2.0]], [[0.0], [0.0]]])

    def step(_: jax.Array, batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = gradient_snr({"weight": batch})
        return jnp.float32(0.0), value

    def scan_batches(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), values)

    _, result = jax.jit(scan_batches)(grads)

    assert jnp.allclose(result, jnp.array([4.0, 4e8, 0.0]), rtol=1e-5)


def test_update_ratio_runs_under_jit_and_vmap() -> None:
    """Compile independent update ratios under jit and vmap."""
    params = {"weight": jnp.array([[3.0, 4.0], [0.0, 0.0]])}
    updates = {"weight": jnp.array([[0.0, 3.0], [1.0, 0.0]])}

    result = jax.jit(jax.vmap(update_ratio))(params, updates)

    assert jnp.allclose(result[0], 0.6)
    assert jnp.isfinite(result[1])


def test_update_ratio_runs_inside_lax_scan() -> None:
    """Evaluate one update ratio per scan step under jit."""
    params = jnp.array([[[3.0, 4.0]], [[0.0, 0.0]], [[1.0, 0.0]]])
    updates = jnp.array([[[0.0, 3.0]], [[1.0, 0.0]], [[0.0, 1.0]]])

    def step(_: jax.Array, values: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value = update_ratio({"weight": values[0]}, {"weight": values[1]})
        return jnp.float32(0.0), value

    def scan_updates(
        parameter_values: jax.Array, update_values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), (parameter_values, update_values))

    _, result = jax.jit(scan_updates)(params, updates)

    assert jnp.allclose(result, jnp.array([0.6, 1e8, 1.0]), rtol=1e-5)
