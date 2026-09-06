"""Medium transform-compatibility tests for jax-eval optimization diagnostics."""

import jax
import jax.numpy as jnp
from jax_eval.optimization import gradient_snr


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
