import jax
import jax.numpy as jnp
from process_control.sensors.lagged import LaggedSensorParams, reset, step


def test_reset_reports_initial_value() -> None:
    """Reset preserves the physical quantity measured before the first step."""
    state = reset(7.2, jax.random.key(0))

    assert state.held_value == 7.2


def test_sample_period_holds_until_sample() -> None:
    """A sampled sensor holds its reading between sampling steps."""
    params = LaggedSensorParams(noise_std=0.0, lag_coefficient=0.0, sample_period=2)
    state = reset(1.0, jax.random.key(0))
    state, first = step(state, jnp.array(2.0), params, jax.random.key(1))
    state, held = step(state, jnp.array(4.0), params, jax.random.key(2))
    _, sampled = step(state, jnp.array(4.0), params, jax.random.key(3))

    assert first == 1.0
    assert held == 1.0
    assert sampled == 4.0


def test_lag_smooths_measurement() -> None:
    """A continuously sampled sensor applies its configured first-order lag."""
    params = LaggedSensorParams(noise_std=0.0, lag_coefficient=0.75)
    state = reset(0.0, jax.random.key(0))
    _, reading = step(state, jnp.array(4.0), params, jax.random.key(1))

    assert reading == 1.0


def test_drift_updates_bias_before_measurement() -> None:
    """A drifting sensor carries its random-walk bias into the reading."""
    params = LaggedSensorParams(noise_std=0.0, lag_coefficient=0.0, drift_rate=1.0)
    state = reset(0.0, jax.random.key(0))
    new_state, reading = step(state, jnp.array(0.0), params, jax.random.key(1))

    assert new_state.drift_bias != 0.0
    assert reading == new_state.drift_bias
