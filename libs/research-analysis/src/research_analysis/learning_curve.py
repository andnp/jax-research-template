"""Learning-curve primitives for step-weighted RL analysis.

This module implements the copy-forward interpolation described in the
science guide: episodic returns are converted into a per-step curve by
assigning each episode's return to every step until the next episode ends.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_step_array(cumulative_steps: NDArray[np.integer]) -> NDArray[np.int64]:
    steps = np.asarray(cumulative_steps, dtype=np.int64)
    if steps.ndim != 1:
        raise ValueError("cumulative_steps must be a 1-D array")
    if steps.size == 0:
        raise ValueError("cumulative_steps must not be empty")
    if np.any(steps <= 0):
        raise ValueError("cumulative_steps must be strictly positive")

    step_differences = np.diff(steps)
    if np.any(step_differences == 0):
        raise ValueError("cumulative_steps must not contain duplicates")
    if np.any(step_differences < 0):
        raise ValueError("cumulative_steps must be strictly increasing")

    return steps


def _as_return_array(episodic_returns: NDArray[np.floating]) -> NDArray[np.float64]:
    returns = np.asarray(episodic_returns, dtype=np.float64)
    if returns.ndim != 1:
        raise ValueError("episodic_returns must be a 1-D array")
    if returns.size == 0:
        raise ValueError("episodic_returns must not be empty")
    return returns


def step_weighted_returns(
    cumulative_steps: NDArray[np.integer],
    episodic_returns: NDArray[np.floating],
    *,
    end_step: int | None = None,
) -> NDArray[np.float64]:
    """Expand episodic returns into a step-weighted copy-forward curve.

    Args:
        cumulative_steps: Monotonically increasing cumulative episode end
            steps. Each value marks the exclusive upper bound of an episode's
            copy-forward segment.
        episodic_returns: Episodic returns aligned with ``cumulative_steps``.
        end_step: Optional total output length. When provided, the final return
            is copied forward through ``end_step``.

    Returns:
        A ``float64`` array of length ``end_step`` (or the last cumulative
        step when ``end_step`` is omitted) containing per-step returns.

    Raises:
        ValueError: If inputs are empty, misaligned, not strictly positive,
            not strictly increasing, or if ``end_step`` is smaller than the
            last cumulative step.
    """
    steps = _as_step_array(cumulative_steps)
    returns = _as_return_array(episodic_returns)

    if steps.shape[0] != returns.shape[0]:
        raise ValueError("cumulative_steps and episodic_returns must have the same length")

    final_step = int(steps[-1])
    if end_step is None:
        output_length = final_step
    else:
        output_length = int(end_step)
        if output_length < final_step:
            raise ValueError("end_step must be at least the last cumulative step")

    curve = np.empty(output_length, dtype=np.float64)
    start = 0
    for stop, episode_return in zip(steps, returns, strict=True):
        curve[start:stop] = episode_return
        start = int(stop)

    if output_length > final_step:
        curve[final_step:output_length] = returns[-1]

    return curve
