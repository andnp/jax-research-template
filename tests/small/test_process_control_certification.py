"""Focused checks for process-control certification helpers."""

from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.chlorine import (
    ChlorineBenchmarkConfig,
    make_chlorine_benchmark,
)
from process_control.benchmarks.equalization_tank import (
    EqualizationTankBenchmarkConfig,
    make_equalization_tank_benchmark,
)
from process_control.benchmarks.ph_neutralization import (
    PhNeutralizationBenchmarkConfig,
    make_ph_neutralization_benchmark,
)
from process_control.benchmarks.primary_clarifier import (
    PrimaryClarifierConfig,
    make_primary_clarifier_benchmark,
)
from process_control.certification import (
    certify_reset_step,
    check_mass_balance,
    check_timestep_refinement,
)


def _reset(_key: jax.Array) -> tuple[jax.Array]:
    return (jnp.asarray(0.0),)


def _step(
    state: object,
    action: object,
    _key: jax.Array,
) -> tuple[jax.Array]:
    return (jnp.asarray(state) + jnp.asarray(action),)


def test_certify_reset_step_checks_finite_deterministic_outputs() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
        steps=2,
    )

    assert report.passed
    assert report.failed == ()


def test_certify_reset_step_checks_declared_action_bounds() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
        action_low=0.0,
        action_high=1.0,
    )

    assert report.passed


def test_certify_reset_step_reports_action_outside_declared_bounds() -> None:
    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=_step,
        action_for_step=lambda _step_index: jnp.asarray(2.0),
        action_low=0.0,
        action_high=1.0,
    )

    assert not report.passed
    assert report.failed[0].name == "bounded_action_no_nan"
    assert "above action_high" in report.failed[0].message


def test_certify_reset_step_reports_nonfinite_outputs() -> None:
    def nonfinite_step(
        state: object,
        _action: object,
        _key: jax.Array,
    ) -> tuple[jax.Array]:
        return (jnp.asarray(state) + jnp.nan,)

    report = certify_reset_step(
        module="synthetic",
        reset=_reset,
        step=nonfinite_step,
        action_for_step=lambda _step_index: jnp.asarray(1.0),
    )

    assert not report.passed
    assert report.failed[0].name == "finite_reset_and_step"
    with pytest.raises(AssertionError, match="certification failed"):
        report.raise_for_failure()


def test_timestep_refinement_and_mass_balance_checks() -> None:
    refinement = check_timestep_refinement(
        name="linear",
        initial_state=jnp.asarray(0.0),
        coarse_step=lambda state, dt: jnp.asarray(state) + dt,
        fine_step=lambda state, dt: jnp.asarray(state) + dt,
        state_value=lambda state: jnp.asarray(state),
        coarse_dt=1.0,
        fine_dt=0.5,
        tolerance=1e-6,
    )
    balance = check_mass_balance(
        name="inventory",
        initial_inventory=10.0,
        final_inventory=11.0,
        inlet_flow=jnp.asarray([2.0, 2.0]),
        realized_outlet_flow=jnp.asarray([1.0, 1.0]),
        overflow_flow=jnp.asarray([0.0, 0.0]),
        dt=0.5,
    )

    assert refinement.passed
    assert balance.passed


def test_certify_chlorine_benchmark_factory() -> None:
    reset, step = make_chlorine_benchmark(ChlorineBenchmarkConfig())

    report = certify_reset_step(
        module="chlorine",
        reset=reset,
        step=cast(Callable[[object, object, jax.Array], Sequence[object]], step),
        action_for_step=lambda _step_index: jnp.asarray(1.0),
        action_low=0.0,
        action_high=5.0,
        steps=3,
    )

    assert report.passed


def test_certify_ph_neutralization_benchmark_factory() -> None:
    reset, step = make_ph_neutralization_benchmark(PhNeutralizationBenchmarkConfig())

    report = certify_reset_step(
        module="ph_neutralization",
        reset=reset,
        step=cast(Callable[[object, object, jax.Array], Sequence[object]], step),
        action_for_step=lambda _step_index: jnp.asarray(7.5),
        steps=3,
    )

    assert report.passed


def test_certify_equalization_tank_benchmark_factory() -> None:
    reset, step = make_equalization_tank_benchmark(EqualizationTankBenchmarkConfig())

    report = certify_reset_step(
        module="equalization_tank",
        reset=reset,
        step=cast(Callable[[object, object, jax.Array], Sequence[object]], step),
        action_for_step=lambda _step_index: jnp.asarray(75.0),
        steps=3,
    )

    assert report.passed


def test_certify_primary_clarifier_benchmark_factory() -> None:
    reset, step = make_primary_clarifier_benchmark(PrimaryClarifierConfig())

    report = certify_reset_step(
        module="primary_clarifier",
        reset=reset,
        step=cast(Callable[[object, object, jax.Array], Sequence[object]], step),
        action_for_step=lambda _step_index: jnp.asarray([20.0]),
        steps=3,
    )

    assert report.passed
