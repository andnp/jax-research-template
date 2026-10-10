"""Sensors with a T90 response time, white noise, a measuring range and optional sample-and-hold.

Times are in hours. The BSM1 presets follow Alex et al. (2008), Sec. 7 and Tables 11-14:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

import dataclasses
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
    """A sensor class with its measuring range and noise.

    As in BSM1, the noise is white Gaussian samples drawn every `noise_period` and linearly
    interpolated in time, so the noise path depends only on the sensor's key, not on how often it is
    read. BSM1 sets the standard deviation to 2.5 % of the range and the period to one minute.
    """

    sensor_class: SensorClass
    low: float
    high: float
    noise_std: float
    noise_period: float = _MINUTE

    def __post_init__(self) -> None:
        if self.noise_period <= 0:
            raise ValueError(f"noise period must be positive, got {self.noise_period}")


BSM1_OXYGEN_SENSOR = SensorParams(BSM1_CLASS_A, low=0.0, high=10.0, noise_std=0.25)
BSM1_NITRATE_SENSOR = SensorParams(BSM1_CLASS_B0, low=0.0, high=20.0, noise_std=0.5)


@jax_dataclass
class SensorState:
    """Lag stages, the reported and pending samples, and the sampling and noise clocks.

    Clocks are kept as a knot index plus a phase within the period so they stay precise over long runs.
    """

    stages: jax.Array
    reading: jax.Array
    pending: jax.Array
    since_sample: jax.Array
    noise_knot: jax.Array
    noise_phase: jax.Array
    noise_key: jax.Array


def reset(value: jax.Array, params: SensorParams, rng_key: jax.Array) -> SensorState:
    """A sensor settled at `value`, with a fresh sampling period and a noise path drawn from `rng_key`."""
    value = jnp.asarray(value)
    return SensorState(
        stages=jnp.full(params.sensor_class.order, value),
        reading=value,
        pending=value,
        since_sample=jnp.zeros_like(value),
        noise_knot=jnp.zeros((), dtype=jnp.int32),
        noise_phase=jnp.zeros_like(value),
        noise_key=rng_key,
    )


def step(state: SensorState, true_value: jax.Array, params: SensorParams, dt: jax.Array) -> tuple[SensorState, jax.Array]:
    """Advance the sensor over dt with the true value held, returning the reported measurement."""
    sensor = params.sensor_class
    measured = _measure(state, true_value, params, dt)
    knots_passed = jnp.floor((state.noise_phase + dt) / params.noise_period)
    advanced = dataclasses.replace(
        state,
        stages=advance_lag_chain(state.stages, true_value, sensor.time_constant, dt),
        noise_knot=state.noise_knot + knots_passed.astype(jnp.int32),
        noise_phase=state.noise_phase + dt - knots_passed * params.noise_period,
    )
    if not sensor.sample_interval:
        return dataclasses.replace(advanced, reading=measured, pending=measured), measured

    elapsed = state.since_sample + dt
    # Tolerance absorbs rounding when dt divides the interval exactly.
    sampling = elapsed >= sensor.sample_interval * (1.0 - 1e-6)
    released = state.pending if sensor.report_previous_sample else measured
    reading = jnp.where(sampling, released, state.reading)
    sampled = dataclasses.replace(
        advanced,
        reading=reading,
        pending=jnp.where(sampling, measured, state.pending),
        since_sample=jnp.where(sampling, jnp.clip(elapsed - sensor.sample_interval, 0.0, sensor.sample_interval), elapsed),
    )
    return sampled, reading


def _measure(state: SensorState, true_value: jax.Array, params: SensorParams, offset: jax.Array) -> jax.Array:
    """What the sensor would report `offset` into the current step, before any sample-and-hold."""
    sensor = params.sensor_class
    delayed = advance_lag_chain(state.stages, true_value, sensor.time_constant, offset)[-1] if sensor.order else jnp.asarray(true_value)
    return jnp.clip(delayed + _noise(state, params, offset), params.low, params.high)


def _noise(state: SensorState, params: SensorParams, offset: jax.Array) -> jax.Array:
    position = (state.noise_phase + offset) / params.noise_period
    knot = jnp.floor(position)
    weight = position - knot
    index = state.noise_knot + knot.astype(jnp.int32)
    before = jax.random.normal(jax.random.fold_in(state.noise_key, index))
    after = jax.random.normal(jax.random.fold_in(state.noise_key, index + 1))
    return params.noise_std * ((1.0 - weight) * before + weight * after)
