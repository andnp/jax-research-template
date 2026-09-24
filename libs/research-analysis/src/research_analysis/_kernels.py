from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def bootstrap_resample_means(
    data: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_resamples, n_draws = indices.shape
    n_steps = data.shape[1]
    boot_means = np.empty((n_resamples, n_steps), dtype=np.float64)

    for resample_index in range(n_resamples):
        for step_index in range(n_steps):
            total = 0.0
            for draw_index in range(n_draws):
                total += data[indices[resample_index, draw_index], step_index]
            boot_means[resample_index, step_index] = total / n_draws

    return boot_means
