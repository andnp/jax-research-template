import jax.numpy as jnp
from process_control.metrics import (
    actuator_metrics,
    constraint_metrics,
    event_metrics,
    integrated_cost,
    movement_metrics,
    nonfinite_count,
    tracking_metrics,
)


def test_tracking_metrics_weight_error_by_control_interval() -> None:
    """
    Compute IAE, ISE, and overshoot per trailing sample axis.

    The control interval must scale both integral metrics consistently.
    """
    result = tracking_metrics(
        jnp.array([[-1.0, 2.0], [1.0, -3.0]], dtype=jnp.float32),
        dt=0.5,
        settling_tolerance=0.5,
    )

    assert jnp.allclose(result.iae, jnp.array([1.0, 2.5]))
    assert jnp.allclose(result.ise, jnp.array([1.0, 6.5]))
    assert jnp.allclose(result.maximum_overshoot, jnp.array([1.0, 2.0]))


def test_tracking_metrics_reports_settling_from_last_violation() -> None:
    """
    Identify the first index after which every sample stays within tolerance.

    A trace that never settles must carry an explicit false status.
    """
    result = tracking_metrics(
        jnp.array([[1.0, 0.0], [0.2, 0.1], [0.1, 0.8], [0.0, 0.8]]),
        dt=1.0,
        settling_tolerance=0.25,
    )

    assert jnp.array_equal(result.settled, jnp.array([True, False]))
    assert jnp.array_equal(result.settling_index, jnp.array([1, 4]))


def test_movement_metrics_uses_zero_initial_delta_by_default() -> None:
    """
    Compute scalar action movement without inventing a startup transition.

    The first delta is zero when no previous action is supplied.
    """
    result = movement_metrics(jnp.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]))

    assert jnp.array_equal(result.delta, jnp.array([[0.0, 0.0], [2.0, -1.0], [-1.0, 3.0]]))
    assert jnp.allclose(result.l1_total, jnp.array([3.0, 4.0]))
    assert jnp.allclose(result.l1_mean, jnp.array([1.0, 4.0 / 3.0]))
    assert jnp.allclose(result.l2_max, jnp.array([2.0, 3.0]))


def test_movement_metrics_sums_multidimensional_action_components() -> None:
    """
    Aggregate movement across the final action dimension.

    Batch and action axes must remain distinct.
    """
    actions = jnp.array(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 3.0], [0.0, 1.0]],
        ]
    )
    result = movement_metrics(actions)

    assert result.delta.shape == actions.shape
    assert jnp.allclose(result.l1_total, jnp.array([3.0, 1.0]))
    assert jnp.allclose(result.l2_max, jnp.array([jnp.sqrt(5.0), 1.0]))


def test_actuator_metrics_distinguishes_saturation_and_rate_limiting() -> None:
    """
    Detect output-bound saturation separately from requested-realized mismatch.

    A multidimensional action is one actuator event per time and batch sample.
    """
    result = actuator_metrics(
        requested=jnp.array(
            [[[0.0, 2.0]], [[2.0, 2.0]], [[1.0, 1.0]]]
        ),
        realized=jnp.array(
            [[[0.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]
        ),
        output_min=0.0,
        output_max=1.0,
    )

    assert jnp.array_equal(result.saturated[:, 0], jnp.array([True, True, True]))
    assert jnp.allclose(result.saturation_fraction, jnp.array([1.0]))
    assert jnp.array_equal(result.maximum_saturation_run, jnp.array([3]))
    assert jnp.array_equal(result.rate_limited[:, 0], jnp.array([True, True, False]))
    assert jnp.allclose(result.rate_limit_fraction, jnp.array([2.0 / 3.0]))


def test_event_and_constraint_metrics_report_runs_and_excess() -> None:
    """
    Summarize boolean events and explicit signal-bound violations.

    Constraint excess must retain the largest physical excursion.
    """
    events = event_metrics(jnp.array([[True, False], [True, True], [False, True]]))
    constraints = constraint_metrics(
        jnp.array([[0.0, 2.0], [1.0, 3.0], [2.0, 1.0]]),
        lower=0.5,
        upper=2.5,
    )

    assert jnp.array_equal(events.event_count, jnp.array([2, 2]))
    assert jnp.array_equal(events.maximum_run, jnp.array([2, 2]))
    assert jnp.array_equal(constraints.event_count, jnp.array([1, 1]))
    assert jnp.allclose(constraints.maximum_excess, jnp.array([0.5, 0.5]))


def test_integrated_cost_and_nonfinite_count_use_explicit_inputs() -> None:
    """
    Integrate supplied costs and count invalid values without assuming reward meaning.

    These helpers must remain independent of any benchmark-specific schema.
    """
    assert jnp.allclose(integrated_cost(jnp.array([[1.0, 2.0], [3.0, 4.0]]), dt=0.25), jnp.array([1.0, 1.5]))
    assert int(nonfinite_count(jnp.array([1.0, jnp.nan]), jnp.array([jnp.inf, 2.0]))) == 2
