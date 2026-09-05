"""Small contracts for generic calibration parameter spaces."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.calibration import (
    CalibrationParameter,
    CalibrationParameterSpace,
)


def _space() -> CalibrationParameterSpace:
    return CalibrationParameterSpace(
        parameters=(
            CalibrationParameter("gain", 2.0, 0.0, 4.0, 2.0, "loop"),
            CalibrationParameter("lag", 3.0, 1.0, 8.0, 1.0, "process"),
        )
    )


def test_parameter_space_exposes_stable_ordered_numeric_views() -> None:
    """Keep metadata order aligned with initial values, bounds, and scales."""
    space = _space()

    assert space.names == ("gain", "lag")
    assert space.groups == ("loop", "process")
    assert jnp.array_equal(space.initial_vector, jnp.array([2.0, 3.0]))
    assert jnp.array_equal(space.lower_bounds, jnp.array([0.0, 1.0]))
    assert jnp.array_equal(space.upper_bounds, jnp.array([4.0, 8.0]))
    assert jnp.array_equal(space.scales, jnp.array([2.0, 1.0]))


def test_parameter_space_selects_groups_without_reordering_parameters() -> None:
    """Select staged-fit parameters while retaining their declaration order."""
    space = _space()

    indices = space.indices_for_groups(("process",))

    assert jnp.array_equal(indices, jnp.array([1], dtype=jnp.int32))


def test_parameter_contract_rejects_invalid_bounds_and_initial_values() -> None:
    """Reject parameter metadata that could create an invalid optimizer domain."""
    with pytest.raises(ValueError, match="within its bounds"):
        CalibrationParameter("gain", 5.0, 0.0, 4.0, 1.0, "loop")
    with pytest.raises(ValueError, match="lower bound"):
        CalibrationParameter("gain", 2.0, 4.0, 0.0, 1.0, "loop")
    with pytest.raises(ValueError, match="scale must be positive"):
        CalibrationParameter("gain", 2.0, 0.0, 4.0, 0.0, "loop")


def test_parameter_space_rejects_duplicate_names_and_unknown_groups() -> None:
    """Fail early when staged calibration metadata cannot be resolved safely."""
    parameter = CalibrationParameter("gain", 2.0, 0.0, 4.0, 1.0, "loop")
    with pytest.raises(ValueError, match="names must be unique"):
        CalibrationParameterSpace((parameter, parameter))

    with pytest.raises(ValueError, match="unknown calibration parameter groups"):
        _space().indices_for_groups(("missing",))


def test_parameter_views_work_inside_jit() -> None:
    """Keep ordered parameter metadata usable by compiled numeric consumers."""
    space = _space()
    def scale_values(values: jax.Array):
        return values / space.scales

    scaled = jax.jit(scale_values)(space.initial_vector)

    assert jnp.allclose(scaled, jnp.array([1.0, 3.0]))
