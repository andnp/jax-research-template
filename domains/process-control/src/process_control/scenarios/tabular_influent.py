"""Influent replayed from tabulated, periodic profiles with per-episode randomization.

Times are in hours, flows in m³/h and temperatures in °C.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from process_control._jax_dataclass import jax_dataclass
from process_control.components import ComponentSchema
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA


class Sampler(Protocol):
    def sample(self, key: jax.Array) -> jax.Array: ...


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(key, minval=self.low, maxval=self.high)


@dataclass(frozen=True, eq=False)
class InfluentProfile:
    """A periodic influent record that repeats from its first sample after `period`."""

    times: np.ndarray
    components: np.ndarray
    flow: np.ndarray
    temperature: np.ndarray
    period: float

    def __post_init__(self) -> None:
        samples = self.times.shape
        if not samples[0] or self.components.shape[0] != samples[0] or self.flow.shape != samples or self.temperature.shape != samples:
            raise ValueError("a profile needs at least one sample and equal sample counts in every column")
        if np.any(np.diff(self.times) <= 0) or self.times[0] < 0 or self.times[-1] >= self.period:
            raise ValueError("profile sample times must increase within [0, period)")

    def at(self, time: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Components, flow and temperature linearly interpolated at a time, wrapping around the period."""
        table = np.column_stack([self.components, self.flow, self.temperature])
        knots = jnp.array(np.concatenate([[self.times[-1] - self.period], self.times, [self.times[0] + self.period]]))
        table = jnp.array(np.vstack([table[-1], table, table[0]]))

        phase = jnp.mod(time, self.period)
        index = jnp.clip(jnp.searchsorted(knots, phase, side="right") - 1, 0, knots.size - 2)
        weight = (phase - knots[index]) / (knots[index + 1] - knots[index])
        row = table[index] + weight * (table[index + 1] - table[index])
        return row[:-2], row[-2], row[-1]


@dataclass(frozen=True)
class InfluentRandomization:
    """How each episode draws from the profiles.

    An episode picks a profile by `profile_weights` (uniform when unset), starts at a fraction of its
    period, and scales its flow and the `scaled_components` (all when unset) by per-episode factors.
    Flow also carries multiplicative AR(1) noise with unit mean and the given stationary log-standard
    deviation and correlation time.
    """

    profile_weights: tuple[float, ...] | None = None
    start_fraction: Sampler = Uniform(0.0, 1.0)
    flow_scale: Sampler = Uniform(0.9, 1.1)
    load_scale: Sampler = Uniform(0.9, 1.1)
    scaled_components: frozenset[str] | None = None
    flow_noise_std: float = 0.05
    flow_noise_correlation_time: float = 2.0


DEFAULT_RANDOMIZATION = InfluentRandomization()
NO_RANDOMIZATION = InfluentRandomization(
    start_fraction=Uniform(0.0, 0.0), flow_scale=Uniform(1.0, 1.0), load_scale=Uniform(1.0, 1.0), flow_noise_std=0.0
)


@dataclass(frozen=True, eq=False)
class TabularInfluentParams:
    """Named influent profiles over one component schema, and their per-episode randomization."""

    schema: ComponentSchema
    profiles: Mapping[str, InfluentProfile]
    randomization: InfluentRandomization = DEFAULT_RANDOMIZATION

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("an influent needs at least one profile")
        width = len(self.schema.names)
        mismatched = [name for name, profile in self.profiles.items() if profile.components.shape[1:] != (width,)]
        if mismatched:
            raise ValueError(f"profiles {mismatched} do not carry the {width} schema components")
        weights = self.randomization.profile_weights
        if weights is not None and len(weights) != len(self.profiles):
            raise ValueError(f"{len(weights)} weights given for {len(self.profiles)} profiles")
        unknown = (self.randomization.scaled_components or frozenset()) - set(self.schema.names)
        if unknown:
            raise ValueError(f"scaled components {sorted(unknown)} are not in the schema")


@jax_dataclass
class TabularInfluentState:
    profile: jax.Array
    time: jax.Array
    flow_scale: jax.Array
    load_scale: jax.Array
    flow_noise: jax.Array


def reset(params: TabularInfluentParams, rng_key: jax.Array) -> TabularInfluentState:
    """Draw an episode: its profile, start time, scale factors and initial flow noise."""
    randomization = params.randomization
    profile_key, start_key, flow_key, load_key, noise_key = jax.random.split(rng_key, 5)
    weights = jnp.array(randomization.profile_weights or (1.0,) * len(params.profiles))
    profile = jax.random.choice(profile_key, len(params.profiles), p=weights / weights.sum())
    periods = jnp.array([candidate.period for candidate in params.profiles.values()])
    return TabularInfluentState(
        profile=profile,
        time=randomization.start_fraction.sample(start_key) * periods[profile],
        flow_scale=randomization.flow_scale.sample(flow_key),
        load_scale=randomization.load_scale.sample(load_key),
        flow_noise=randomization.flow_noise_std * jax.random.normal(noise_key),
    )


def influent(state: TabularInfluentState, params: TabularInfluentParams) -> tuple[Stream, jax.Array]:
    """The influent stream and its temperature at the episode's current time."""
    branches = [profile.at for profile in params.profiles.values()]
    components, flow, temperature = jax.lax.switch(state.profile, branches, state.time)

    randomization = params.randomization
    scaled = randomization.scaled_components
    mask = np.array([scaled is None or name in scaled for name in params.schema.names])
    components = components * jnp.where(mask, state.load_scale, 1.0)
    flow = flow * state.flow_scale * jnp.exp(state.flow_noise - 0.5 * randomization.flow_noise_std**2)
    return Stream(flow=flow, components=components, schema=params.schema), temperature


def advance(state: TabularInfluentState, params: TabularInfluentParams, dt: jax.Array, rng_key: jax.Array) -> TabularInfluentState:
    """Move the episode on by dt, evolving the flow noise exactly for that interval."""
    randomization = params.randomization
    persistence = jnp.exp(-dt / randomization.flow_noise_correlation_time)
    innovation = randomization.flow_noise_std * jnp.sqrt(1.0 - persistence**2) * jax.random.normal(rng_key)
    return dataclasses.replace(state, time=state.time + dt, flow_noise=persistence * state.flow_noise + innovation)


# Alex et al. (2008) constant influent used to stabilise the plant before the weather files.
_BSM1_STABILIZATION = np.array([30.0, 69.5, 51.2, 202.32, 28.17, 0.0, 0.0, 0.0, 0.0, 31.56, 6.95, 10.59, 7.0])


@cache
def bsm1_influent_profiles() -> Mapping[str, InfluentProfile]:
    """The BSM1 constant stabilisation influent and the dry, rain and storm weather files."""
    profiles = {
        "constant": InfluentProfile(
            times=np.zeros(1),
            components=_BSM1_STABILIZATION[None, :],
            flow=np.array([18446.0 / 24.0]),
            temperature=np.array([15.0]),
            period=24.0,
        )
    }
    for weather in ("dry", "rain", "storm"):
        with (files("process_control") / "data" / "bsm1" / f"{weather}influent.csv").open() as handle:
            table = np.loadtxt(handle, delimiter=",")
        times = table[:, 0] * 24.0
        profiles[weather] = InfluentProfile(
            times=times,
            components=table[:, 1:14],
            flow=table[:, 15] / 24.0,
            temperature=table[:, 16],
            period=float(times[-1] + times[1] - times[0]),
        )
    return profiles


def bsm1_influent(randomization: InfluentRandomization = DEFAULT_RANDOMIZATION) -> TabularInfluentParams:
    """The BSM1 profiles over the ASM1 schema, in the order constant, dry, rain, storm."""
    return TabularInfluentParams(schema=ASM1_SCHEMA, profiles=bsm1_influent_profiles(), randomization=randomization)
