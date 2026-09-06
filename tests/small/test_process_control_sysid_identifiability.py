"""Small contracts for finite-difference calibration identifiability diagnostics."""

import numpy as np
import pytest
from process_control.sysid.calibration import CalibrationParameter, CalibrationParameterSpace
from process_control.sysid.identifiability import (
    parameter_sensitivities,
    weakly_identified_parameters,
)


def _space() -> CalibrationParameterSpace:
    return CalibrationParameterSpace(
        parameters=(
            CalibrationParameter("gain", 1.0, 0.0, 4.0, 1.0, "loop"),
            CalibrationParameter("kink", 1.0, -2.0, 4.0, 1.0, "loop"),
            CalibrationParameter("dead", 1.0, 0.0, 4.0, 1.0, "process"),
        )
    )


def _residual(parameters: np.ndarray) -> np.ndarray:
    """A residual with a locally flat parameter that is sensitive elsewhere, and one always flat."""
    gain, kink, _dead = parameters
    return np.array([gain, kink**2])


def test_sensitivity_is_computed_per_reference_point() -> None:
    """Score each selected parameter independently at every supplied reference point."""
    space = _space()

    results = parameter_sensitivities(
        _residual,
        space,
        reference_points=(np.array([1.0, 0.0, 1.0]), np.array([1.0, 3.0, 1.0])),
    )

    by_point = {(item.name, item.reference_index): item.sensitivity for item in results}
    assert by_point[("gain", 0)] == pytest.approx(1.0, abs=1e-3)
    assert by_point[("gain", 1)] == pytest.approx(1.0, abs=1e-3)
    assert by_point[("kink", 0)] == pytest.approx(0.0, abs=1e-6)
    assert by_point[("kink", 1)] == pytest.approx(6.0, rel=1e-2)
    assert by_point[("dead", 0)] == pytest.approx(0.0, abs=1e-9)
    assert by_point[("dead", 1)] == pytest.approx(0.0, abs=1e-9)


def test_sensitivity_selects_a_subset_of_parameters_by_index() -> None:
    """Restrict the scan to a staged fit's active parameter indices."""
    space = _space()

    results = parameter_sensitivities(
        _residual,
        space,
        reference_points=(np.array([1.0, 0.0, 1.0]),),
        indices=np.array([0], dtype=np.int32),
    )

    assert {item.name for item in results} == {"gain"}


def test_weakly_identified_parameters_requires_flat_at_every_reference_point() -> None:
    """Flag only parameters that never showed sensitivity, not ones flat at one point only."""
    space = _space()

    results = parameter_sensitivities(
        _residual,
        space,
        reference_points=(np.array([1.0, 0.0, 1.0]), np.array([1.0, 3.0, 1.0])),
    )

    assert weakly_identified_parameters(results, threshold=1e-6) == ("dead",)


def test_sensitivity_rejects_empty_reference_points_and_wrong_shape() -> None:
    """Reject inputs that would silently skip the identifiability scan or misalign parameters."""
    space = _space()

    with pytest.raises(ValueError, match="at least one reference point"):
        parameter_sensitivities(_residual, space, reference_points=())
    with pytest.raises(ValueError, match="must match the parameter space width"):
        parameter_sensitivities(_residual, space, reference_points=(np.array([1.0, 0.0]),))
