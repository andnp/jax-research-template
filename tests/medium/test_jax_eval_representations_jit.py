"""Medium transform-compatibility tests for jax-eval representations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_eval.representations import dead_neurons, feature_drift


def test_representation_kernels_run_under_jit() -> None:
    """Compile both representation kernels without Python-side evaluation."""
    activations = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    old = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    new = jnp.array([[1.0, 0.0], [1.0, 0.0]])

    def evaluate(a: jax.Array, x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        return dead_neurons(a), feature_drift(x, y)

    dead_result, drift_result = jax.jit(evaluate)(activations, old, new)

    assert jnp.allclose(dead_result, 0.5)
    assert jnp.allclose(drift_result, 0.5)


def test_representation_kernels_vmap_over_independent_probes() -> None:
    """Map each kernel over independent leading probe batches."""
    activations = jnp.array([[[0.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]]])
    old = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
    new = jnp.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])

    dead_result = jax.jit(jax.vmap(dead_neurons))(activations)
    drift_result = jax.jit(jax.vmap(feature_drift))(old, new)

    assert jnp.allclose(dead_result, jnp.array([0.5, 0.0]))
    assert jnp.allclose(drift_result, jnp.array([0.5, 0.5]))


def test_feature_drift_runs_inside_lax_scan() -> None:
    """Accumulate per-step feature distances inside a compiled scan."""
    old = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    new = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    def step(total: jax.Array, pair: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        distance = feature_drift(pair[0], pair[1])
        return total + distance, distance

    def scan_features(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), (x, y))

    total, distances = jax.jit(scan_features)(old, new)

    assert jnp.allclose(distances, jnp.array([0.0, 1.0, 0.0]))
    assert jnp.allclose(total, 1.0)


def test_dead_neurons_runs_inside_lax_scan() -> None:
    """Evaluate dead-neuron fractions for each scan step under jit."""
    activations = jnp.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )

    def step(_: jax.Array, values: jax.Array) -> tuple[jax.Array, jax.Array]:
        fraction = dead_neurons(values)
        return jnp.float32(0.0), fraction

    def scan_activations(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.scan(step, jnp.float32(0.0), x)

    _, fractions = jax.jit(scan_activations)(activations)

    assert jnp.allclose(fractions, jnp.array([0.5, 1.0, 0.0]))
