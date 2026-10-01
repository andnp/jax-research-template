"""Small tests for research_instrument.thinning — on-device metric thinning."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from research_instrument.thinning import thin_metric


def reference_subsample(x: np.ndarray, every: int) -> np.ndarray:
    """ml_instrumentation.Subsample: emit on steps 0, every, 2*every, ..."""
    return np.array([v for t, v in enumerate(x) if t % every == 0])


@pytest.mark.parametrize("length", [1, 6, 7, 10])
@pytest.mark.parametrize("every", [1, 2, 3, 20])
def test_every_keeps_steps_zero_then_every_nth(length: int, every: int) -> None:
    x = np.arange(length, dtype=np.float32) * 1.5

    out = thin_metric(jnp.asarray(x), every=every)

    np.testing.assert_array_equal(np.asarray(out), reference_subsample(x, every))


def reference_moving_average(x: np.ndarray, decay: float) -> np.ndarray:
    """ml_instrumentation.MovingAverage: z_0 = 0, no bias correction."""
    z = 0.0
    out = []
    for v in x:
        z = decay * z + (1.0 - decay) * v
        out.append(z)
    return np.array(out)


@pytest.mark.parametrize("ema", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("every", [1, 3])
def test_ema_smooths_before_thinning(ema: float, every: int) -> None:
    x = np.random.default_rng(0).normal(size=10).astype(np.float32)

    out = thin_metric(jnp.asarray(x), ema=ema, every=every)

    expected = reference_subsample(reference_moving_average(x, ema), every)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-6)


def test_ema_accepts_integer_input_without_truncating() -> None:
    x = np.array([1, 2, 3, 4])

    out = thin_metric(jnp.asarray(x), ema=0.5)

    np.testing.assert_allclose(np.asarray(out), reference_moving_average(x, 0.5), rtol=1e-5)


def test_thins_a_batch_of_runs_under_jit_and_vmap() -> None:
    x = np.random.default_rng(1).normal(size=(4, 11)).astype(np.float32)

    thin = jax.jit(jax.vmap(partial(thin_metric, ema=0.8, every=4)))
    out = thin(jnp.asarray(x))

    expected = np.stack([reference_subsample(reference_moving_average(r, 0.8), 4) for r in x])
    assert out.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-6)
