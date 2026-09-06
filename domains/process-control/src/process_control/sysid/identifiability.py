"""Finite-difference identifiability diagnostics for bounded calibration parameters."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from process_control.sysid.calibration import CalibrationParameterSpace

FloatArray = NDArray[np.float64]
ResidualFn = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class ParameterSensitivity:
    """One parameter's finite-difference residual sensitivity at one reference point."""

    name: str
    reference_index: int
    sensitivity: float


def parameter_sensitivities(
    residual_fn: ResidualFn,
    space: CalibrationParameterSpace,
    reference_points: tuple[FloatArray, ...],
    indices: NDArray[np.int32] | None = None,
    relative_step: float = 1e-3,
) -> tuple[ParameterSensitivity, ...]:
    """Score each parameter's local residual sensitivity at each reference point.

    Central finite differences are computed directly from ``residual_fn``, independent
    of any optimizer's internal Jacobian or robust-loss reweighting. A parameter with
    zero sensitivity at every supplied reference point reflects a real excitation gap
    in the data rather than an artifact of one optimizer's converged point.
    """
    if not reference_points:
        raise ValueError("at least one reference point is required")
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    selected = (
        np.arange(len(space.parameters), dtype=np.int32)
        if indices is None
        else np.asarray(indices, dtype=np.int32)
    )
    lower = np.asarray(space.lower_bounds, dtype=np.float64)
    upper = np.asarray(space.upper_bounds, dtype=np.float64)
    names = space.names
    results = []
    for point_index, point in enumerate(reference_points):
        base = np.asarray(point, dtype=np.float64)
        if base.shape != (len(names),):
            raise ValueError("reference point must match the parameter space width")
        for index in selected:
            index = int(index)
            step = relative_step * (upper[index] - lower[index])
            plus = np.array(base, copy=True)
            minus = np.array(base, copy=True)
            plus[index] = min(base[index] + step, upper[index])
            minus[index] = max(base[index] - step, lower[index])
            span = plus[index] - minus[index]
            sensitivity = (
                0.0
                if span <= 0.0
                else float(np.linalg.norm(residual_fn(plus) - residual_fn(minus)) / span)
            )
            results.append(ParameterSensitivity(names[index], point_index, sensitivity))
    return tuple(results)


def weakly_identified_parameters(
    sensitivities: tuple[ParameterSensitivity, ...],
    threshold: float,
) -> tuple[str, ...]:
    """Return, in first-seen order, parameter names at or under threshold at every reference point."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    peak_by_name: dict[str, float] = {}
    for item in sensitivities:
        peak_by_name[item.name] = max(peak_by_name.get(item.name, 0.0), item.sensitivity)
    ordered_names = dict.fromkeys(item.name for item in sensitivities)
    return tuple(name for name in ordered_names if peak_by_name[name] <= threshold)
