"""Diurnal gas influent source for H₂S scrubber benchmarks.

Generates time-varying gas flow rate and H₂S concentration with:
  - sinusoidal diurnal pattern (peak during high-activity hours)
  - random-walk drift for slow baseline wander
  - per-step noise for turbulence / process variability
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step


@dataclass(frozen=True)
class GasSourceParams:
    # ── Gas flow ──────────────────────────────────────────────────
    mean_gas_flow: float = 500.0  # m³/h nominal
    gas_flow_amplitude: float = 100.0  # diurnal swing ± (m³/h)
    min_gas_flow: float = 200.0
    max_gas_flow: float = 800.0
    gas_flow_noise_std: float = 10.0  # per-step noise (m³/h)
    gas_flow_drift_scale: float = 0.05  # random walk scale

    # ── H₂S concentration ────────────────────────────────────────
    mean_h2s_ppm: float = 50.0  # ppmv nominal
    h2s_amplitude: float = 20.0  # diurnal swing ± (ppmv)
    min_h2s_ppm: float = 5.0
    max_h2s_ppm: float = 200.0
    h2s_noise_std: float = 3.0  # per-step noise (ppmv)
    h2s_drift_scale: float = 0.1  # random walk scale

    # ── Diurnal timing ────────────────────────────────────────────
    steps_per_day: int = 288  # 5-min steps → 288/day
    h2s_phase_shift: float = 0.0  # phase offset (0–1) for H₂S vs gas flow


@jax_dataclass
class GasSourceState:
    gas_flow_drift: jax.Array
    h2s_drift: jax.Array


def reset(rng_key: jax.Array) -> GasSourceState:
    return GasSourceState(
        gas_flow_drift=jnp.array(0.0),
        h2s_drift=jnp.array(0.0),
    )


def step(
    state: GasSourceState,
    step_count: jax.Array,
    params: GasSourceParams,
    rng_key: jax.Array,
) -> tuple[GasSourceState, jax.Array, jax.Array]:
    """Advance gas source by one step.

    Returns:
        (new_state, gas_flow, h2s_ppm)
    """
    k1, k2 = jax.random.split(rng_key)

    phase = (step_count % params.steps_per_day) / params.steps_per_day * 2.0 * jnp.pi

    flow_channel = ChannelParams(
        mean=params.mean_gas_flow,
        amplitude=params.gas_flow_amplitude,
        min_value=params.min_gas_flow,
        max_value=params.max_gas_flow,
        noise_std=params.gas_flow_noise_std,
        drift_scale=params.gas_flow_drift_scale,
        drift_clip=params.gas_flow_amplitude,
    )
    new_flow_drift, gas_flow = channel_step(
        state.gas_flow_drift,
        jnp.sin(phase),
        flow_channel,
        k1,
    )

    h2s_channel = ChannelParams(
        mean=params.mean_h2s_ppm,
        amplitude=params.h2s_amplitude,
        min_value=params.min_h2s_ppm,
        max_value=params.max_h2s_ppm,
        noise_std=params.h2s_noise_std,
        drift_scale=params.h2s_drift_scale,
        drift_clip=params.h2s_amplitude,
    )
    h2s_phase = phase + params.h2s_phase_shift * 2.0 * jnp.pi
    new_h2s_drift, h2s_ppm = channel_step(
        state.h2s_drift,
        jnp.sin(h2s_phase),
        h2s_channel,
        k2,
    )

    new_state = GasSourceState(
        gas_flow_drift=new_flow_drift,
        h2s_drift=new_h2s_drift,
    )
    return new_state, gas_flow, h2s_ppm
