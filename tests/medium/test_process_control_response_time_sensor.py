import dataclasses

import jax
import jax.numpy as jnp
import pytest
from process_control.sensors.response_time import (
    BSM1_CLASS_A,
    BSM1_CLASS_B0,
    BSM1_CLASS_B1,
    BSM1_CLASS_D,
    BSM1_OXYGEN_SENSOR,
    SensorClass,
    SensorParams,
    SensorState,
    reset,
    step,
)

MINUTE = 1.0 / 60.0


def _noiseless(sensor_class: SensorClass) -> SensorParams:
    return SensorParams(sensor_class, low=0.0, high=100.0, noise_std=0.0)


def _run(params: SensorParams, values: jax.Array, dt: float, initial: float = 0.0, seed: int = 0) -> jax.Array:
    def advance(state: SensorState, value: jax.Array) -> tuple[SensorState, jax.Array]:
        return step(state, value, params, jnp.array(dt))

    _, readings = jax.lax.scan(advance, reset(jnp.array(initial), params, jax.random.key(seed)), values)
    return readings


class TestContinuousSensors:
    @pytest.mark.parametrize("sensor_class", [BSM1_CLASS_A, BSM1_CLASS_B0])
    def test_step_is_ninety_percent_through_after_t90(self, sensor_class: SensorClass) -> None:
        """A noiseless sensor's reading crosses 90 % of a step exactly at its class T90."""
        dt = sensor_class.t90 / 20
        readings = _run(_noiseless(sensor_class), jnp.full(20, 10.0), dt)

        assert jnp.allclose(readings[-1], 9.0, atol=1e-4)
        assert jnp.all(jnp.diff(readings) >= 0.0)

    def test_settled_sensor_reports_the_true_value(self) -> None:
        """A sensor reset to the process value and fed that value keeps reporting it."""
        readings = _run(_noiseless(BSM1_CLASS_B0), jnp.full(5, 3.0), MINUTE, initial=3.0)

        assert jnp.allclose(readings, 3.0)

    def test_readings_are_clipped_to_the_range(self) -> None:
        """Noise and out-of-range values never leave the measuring range."""
        params = SensorParams(BSM1_CLASS_A, low=0.0, high=10.0, noise_std=5.0)

        readings = _run(params, jnp.full(200, 9.5), MINUTE, initial=9.5)

        assert jnp.all((readings >= 0.0) & (readings <= 10.0))
        assert jnp.any(readings == 10.0)

    def test_noise_has_the_configured_spread_at_its_knots(self) -> None:
        """Read once per noise period, a steady value scatters with the configured standard deviation."""
        readings = _run(BSM1_OXYGEN_SENSOR, jnp.full(4000, 5.0), MINUTE, initial=5.0)

        assert jnp.mean(readings) == pytest.approx(5.0, abs=0.02)
        assert jnp.std(readings) == pytest.approx(0.25, rel=0.05)

    def test_noise_is_linear_between_knots(self) -> None:
        """Read faster than the noise period, the noise moves on straight lines between its one-minute knots."""
        readings = _run(BSM1_OXYGEN_SENSOR, jnp.full(60, 5.0), MINUTE / 4, initial=5.0)

        within_minute = readings.reshape(15, 4)
        slopes = jnp.diff(within_minute, axis=1)
        assert jnp.allclose(slopes, slopes[:, :1], atol=1e-5)
        assert jnp.std(slopes[:, 0]) > 0.0

    def test_noise_path_does_not_depend_on_the_read_rate(self) -> None:
        """The same key gives the same noise at a shared time whether read every minute or every quarter minute."""
        coarse = _run(BSM1_OXYGEN_SENSOR, jnp.full(10, 5.0), MINUTE, initial=5.0)
        fine = _run(BSM1_OXYGEN_SENSOR, jnp.full(40, 5.0), MINUTE / 4, initial=5.0)

        assert jnp.allclose(coarse, fine[3::4], atol=1e-5)


class TestSampledSensors:
    def test_sample_and_hold_updates_once_per_interval(self) -> None:
        """A B1 sensor stepped every minute changes its reading only at five-minute samples."""
        readings = _run(_noiseless(BSM1_CLASS_B1), jnp.full(30, 10.0), MINUTE)

        changed = jnp.flatnonzero(jnp.diff(readings) != 0.0) + 1
        assert jnp.array_equal(changed, jnp.array([4, 9, 14, 19, 24, 29]))

    def test_class_d_reports_the_previous_sample(self) -> None:
        """A class D sensor reports at each 30-minute sample what it measured one interval earlier."""
        values = jnp.arange(90, dtype=jnp.float32)

        readings = _run(_noiseless(BSM1_CLASS_D), values, MINUTE, initial=-1.0)

        assert jnp.allclose(readings[29], -1.0)
        assert jnp.allclose(readings[59], 29.0)
        assert jnp.allclose(readings[89], 59.0)

    def test_steps_longer_than_the_interval_sample_every_step(self) -> None:
        """When the control step exceeds the sampling interval each step takes a fresh sample."""
        readings = _run(_noiseless(dataclasses.replace(BSM1_CLASS_D, report_previous_sample=False)), jnp.arange(4.0), 1.0)

        assert jnp.allclose(readings, jnp.arange(4.0))

    def test_previous_sample_needs_a_sampling_sensor(self) -> None:
        """A continuous sensor has no previous sample to report."""
        with pytest.raises(ValueError, match="previous sample"):
            SensorClass(t90=1.0, order=2, report_previous_sample=True)
