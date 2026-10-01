"""Small tests for research_instrument.thinning — on-device metric thinning."""

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
