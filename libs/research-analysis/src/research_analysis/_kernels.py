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


@numba.njit(cache=True)
def mean_axis0(data: NDArray[np.float64]) -> NDArray[np.float64]:
    n_rows, n_cols = data.shape
    mean = np.empty(n_cols, dtype=np.float64)

    for column_index in range(n_cols):
        total = 0.0
        for row_index in range(n_rows):
            total += data[row_index, column_index]
        mean[column_index] = total / n_rows

    return mean


@numba.njit(cache=True)
def percentile_axis0(
    sorted_values: NDArray[np.float64],
    percentile: float,
) -> NDArray[np.float64]:
    n_rows, n_cols = sorted_values.shape
    quantile = percentile / 100.0
    position = (n_rows - 1) * quantile
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    interpolation = position - lower_index

    percentile_values = np.empty(n_cols, dtype=np.float64)
    for column_index in range(n_cols):
        lower_value = sorted_values[lower_index, column_index]
        upper_value = sorted_values[upper_index, column_index]
        percentile_values[column_index] = (
            lower_value + (upper_value - lower_value) * interpolation
        )

    return percentile_values


@numba.njit(cache=True)
def mann_whitney_rank_sum(
    sorted_order: NDArray[np.int64],
    sorted_values: NDArray[np.float64],
    n_a: int,
) -> tuple[float, float]:
    rank_sum_a = 0.0
    tie_term = 0.0
    n_total = sorted_values.shape[0]
    start_index = 0

    while start_index < n_total:
        end_index = start_index + 1
        while end_index < n_total and sorted_values[end_index] == sorted_values[start_index]:
            end_index += 1

        average_rank = 0.5 * (start_index + end_index + 1)
        tie_count = end_index - start_index
        if tie_count > 1:
            tie_term += float(tie_count * tie_count * tie_count - tie_count)

        for sorted_index in range(start_index, end_index):
            if sorted_order[sorted_index] < n_a:
                rank_sum_a += average_rank

        start_index = end_index

    return rank_sum_a, tie_term