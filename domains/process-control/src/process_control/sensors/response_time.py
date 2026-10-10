"""Sensors with a T90 response time, white noise, a measuring range and optional sample-and-hold.

Times are in hours. The BSM1 presets follow Alex et al. (2008), Sec. 7 and Tables 11-14:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.lag_chain import advance_lag_chain, time_constant_for_t90

_MINUTE = 1.0 / 60.0


@dataclass(frozen=True)
class SensorClass:
    """Sensor dynamics: a chain of `order` lags reaching 90 % of a step after `t90`, then optionally a
    sample-and-hold every `sample_interval` (continuous when zero) that can report the previous sample.
    """

    t90: float
    order: int
    sample_interval: float = 0.0
    report_previous_sample: bool = False

    def __post_init__(self) -> None:
        if self.order < 0 or self.sample_interval < 0:
            raise ValueError("sensor order and sample interval must be non-negative")
        if self.report_previous_sample and not self.sample_interval:
            raise ValueError("only a sampling sensor can report its previous sample")

    @property
    def time_constant(self) -> float:
        return time_constant_for_t90(self.t90, self.order) if self.order else 0.0


BSM1_CLASS_A = SensorClass(t90=1.0 * _MINUTE, order=2)
BSM1_CLASS_B0 = SensorClass(t90=10.0 * _MINUTE, order=8)
BSM1_CLASS_B1 = SensorClass(t90=10.0 * _MINUTE, order=8, sample_interval=5.0 * _MINUTE)
BSM1_CLASS_C0 = SensorClass(t90=20.0 * _MINUTE, order=8)
BSM1_CLASS_C1 = SensorClass(t90=20.0 * _MINUTE, order=8, sample_interval=5.0 * _MINUTE)
BSM1_CLASS_D = SensorClass(t90=0.0, order=0, sample_interval=30.0 * _MINUTE, report_previous_sample=True)


@dataclass(frozen=True)
class SensorParams:
    """A sensor class with its measuring range and noise standard deviation.

    BSM1 sets the noise to 2.5 % of the range.
    """

    sensor_class: SensorClass
    low: float
    high: float
    noise_std: float


BSM1_OXYGEN_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=10.0, noise_std=0.25)
BSM1_NITRATE_SENSOR = SensorParams(BSM1_CLASS_B0, low=0.0, high=20.0, noise_std=0.5)


@jax_dataclass
class SensorState:
    stages: jax.Array
    reading: jax.Array
    pending: jax.Array
    since_sample: jax.Array


def reset(value: jax.Array, params: SensorParams) -> SensorState:
    """A sensor settled at `value`, with a fresh sampling period."""
    value = jnp.asarray(value)
    return SensorState(
        stages=jnp.full(params.sensor_class.order, value),
        reading=value,
        pending=value,
        since_sample=jnp.zeros_like(value),
    )


def step(state: SensorState, true_value: jax.Array, params: SensorParams, dt: jax.Array, rng_key: jax.Array) -> tuple[SensorState, jax.Array]:
    """Advance the sensor over dt with the true value held, returning the reported measurement."""
    sensor = params.sensor_class
    stages = advance_lag_chain(state.stages, true_value, sensor.time_constant, dt)
    delayed = stages[-1] if sensor.order else jnp.asarray(true_value)
    measured = jnp.clip(delayed + params.noise_std * jax.random.normal(rng_key, delayed.shape, delayed.dtype), params.low, params.high)
    if not sensor.sample_interval:
        return SensorState(stages=stages, reading=measured, pending=measured, since_sample=state.since_sample), measured

    elapsed = state.since_sample + dt
    # Tolerance absorbs rounding when dt divides the interval exactly.
    sampling = elapsed >= sensor.sample_interval * (1.0 - 1e-6)
    released = state.pending if sensor.report_previous_sample else measured
    reading = jnp.where(sampling, released, state.reading)
    new_state = SensorState(
        stages=stages,
        reading=reading,
        pending=jnp.where(sampling, measured, state.pending),
        since_sample=jnp.where(sampling, jnp.clip(elapsed - sensor.sample_interval, 0.0, sensor.sample_interval), elapsed),
    )
    return new_state, reading
