"""Runtime instrumentation profiles for separating plant truth from observations.

The plant supplies named latent trajectories.  An :class:`InstrumentationProfile`
turns those trajectories into the signals a controller can actually see.  The
transformation is deterministic for a :class:`~process_control.scenarios.SeedBundle`
and keeps sensor randomness separate from process and equipment randomness.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

from process_control.environment_specs import (
    InstrumentationProfileSpec,
    InstrumentedSignalSpec,
    SignalSource,
    SignalSpec,
    SignalTiming,
    SpecIdentity,
)
from process_control.scenarios.scenario_pack import RandomStream, SeedBundle, TimedEvent

SignalInput: TypeAlias = float | Array
LatentSignals: TypeAlias = Mapping[str, SignalInput]


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")


def _step_count(value: float, simulation_time_step: float, field: str) -> int:
    """Convert a timing value in seconds to an exact control-step count."""
    if simulation_time_step <= 0.0 or not math.isfinite(float(simulation_time_step)):
        raise ValueError("simulation_time_step must be positive and finite")
    ratio = float(value) / float(simulation_time_step)
    rounded = round(ratio)
    if abs(ratio - rounded) > 1e-8:
        raise ValueError(f"{field} must be an integer multiple of simulation_time_step")
    return int(rounded)


@dataclass(frozen=True, slots=True)
class InstrumentationChannel:
    """One calibrated path from a latent signal to an observed signal."""

    source: SignalSpec
    output: SignalSpec
    timing: SignalTiming = SignalTiming(1.0)
    noise_std: float = 0.0
    bias: float = 0.0
    drift_per_step: float = 0.0
    dropout_probability: float = 0.0
    hold_last: bool = True
    scale: float = 1.0
    offset: float = 0.0
    provenance: SignalSource = SignalSource.SENSOR

    def __post_init__(self) -> None:
        if self.provenance is SignalSource.LATENT:
            raise ValueError("an observed channel cannot have latent provenance")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be between zero and one")
        for name, value in (
            ("noise_std", self.noise_std),
            ("bias", self.bias),
            ("drift_per_step", self.drift_per_step),
            ("dropout_probability", self.dropout_probability),
            ("scale", self.scale),
            ("offset", self.offset),
        ):
            _require_finite(value, name)
        if self.scale == 0.0:
            raise ValueError("scale must be non-zero")
        if self.source.units != self.output.units:
            raise ValueError(
                f"channel {self.output.name} units {self.output.units!r} do not match "
                f"source {self.source.name} units {self.source.units!r}"
            )

    @property
    def name(self) -> str:
        """Return the controller-visible signal name."""
        return self.output.name

    def observed_spec(self) -> InstrumentedSignalSpec:
        """Return the static observed-signal contract for this channel."""
        return InstrumentedSignalSpec(self.output, self.provenance, self.timing)


# Short aliases make the channel contract easy to discover without creating
# multiple implementations of the same concept.
SignalMapping = InstrumentationChannel
InstrumentationSignal = InstrumentationChannel


@dataclass(frozen=True, slots=True)
class ObservedSignals:
    """One controller-time frame containing observed values only."""

    values: Mapping[str, Array]
    available: Mapping[str, bool]
    timing: Mapping[str, SignalTiming]
    provenance: Mapping[str, SignalSource]

    def require(self, names: Iterable[str]) -> dict[str, Array]:
        """Return available observed values, rejecting missing or stale signals."""
        requested = tuple(names)
        result: dict[str, Array] = {}
        for name in requested:
            if name not in self.values:
                raise ValueError(f"signal {name!r} is not available in the instrumentation profile")
            if not self.available.get(name, False):
                raise ValueError(f"observed signal {name!r} is unavailable at this step")
            result[name] = self.values[name]
        return result

    @property
    def names(self) -> tuple[str, ...]:
        """Return observed names in stable profile order."""
        return tuple(self.values)


@dataclass(frozen=True, slots=True)
class InstrumentationBatch:
    """Observed trajectories and their per-step availability masks."""

    values: Mapping[str, Array]
    available: Mapping[str, Array]
    timing: Mapping[str, SignalTiming]
    provenance: Mapping[str, SignalSource]
    horizon_steps: int

    def at(self, step: int) -> ObservedSignals:
        """Return one controller-time frame from the batch."""
        if step < 0 or step >= self.horizon_steps:
            raise IndexError(f"step {step} is outside instrumentation horizon")
        return ObservedSignals(
            values={name: values[step] for name, values in self.values.items()},
            available={name: bool(available[step]) for name, available in self.available.items()},
            timing=self.timing,
            provenance=self.provenance,
        )

    def reward_inputs(self, names: Iterable[str], *, step: int) -> dict[str, Array]:
        """Return reward inputs from one observed frame only."""
        return self.at(step).require(names)


def _coerce_channels(signals: Iterable[SignalSpec | InstrumentationChannel]) -> tuple[InstrumentationChannel, ...]:
    channels: list[InstrumentationChannel] = []
    for signal in signals:
        if isinstance(signal, InstrumentationChannel):
            channels.append(signal)
        elif isinstance(signal, SignalSpec):
            channels.append(InstrumentationChannel(source=signal, output=signal))
        else:
            raise TypeError("profiles accept SignalSpec or InstrumentationChannel values")
    if not channels:
        raise ValueError("an instrumentation profile requires at least one channel")
    return tuple(channels)


@dataclass(frozen=True, slots=True)
class InstrumentationProfile:
    """Validated, composable mapping from latent trajectories to observations."""

    identity: SpecIdentity
    channels: tuple[InstrumentationChannel, ...]
    observation_schema_version: str = "1.0.0"
    latent_signals: tuple[SignalSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_schema_version or self.observation_schema_version.strip() != self.observation_schema_version:
            raise ValueError("observation_schema_version must be non-empty and have no surrounding whitespace")
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("an instrumentation profile requires at least one channel")
        names = tuple(channel.name for channel in channels)
        if len(set(names)) != len(names):
            raise ValueError("instrumentation output names must be unique")

        explicit_latent = {signal.name: signal for signal in self.latent_signals}
        if len(explicit_latent) != len(self.latent_signals):
            raise ValueError("instrumentation latent signal names must be unique")
        observed_names = set(names)
        inferred = dict(explicit_latent)
        for channel in channels:
            if channel.source.name not in observed_names:
                previous = inferred.get(channel.source.name)
                if previous is not None and previous.units != channel.source.units:
                    raise ValueError(f"latent signal {channel.source.name!r} has inconsistent units")
                inferred[channel.source.name] = channel.source
        overlap = sorted(observed_names & set(inferred))
        if overlap:
            raise ValueError("signals cannot be both observed and latent: " + ", ".join(overlap))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "latent_signals", tuple(inferred.values()))

    @property
    def spec(self) -> InstrumentationProfileSpec:
        """Return the static environment-spec view of this runtime profile."""
        return InstrumentationProfileSpec(
            identity=self.identity,
            observation_schema_version=self.observation_schema_version,
            observed_signals=tuple(channel.observed_spec() for channel in self.channels),
            latent_signals=self.latent_signals,
        )

    @property
    def observed_names(self) -> tuple[str, ...]:
        """Return controller-visible names in stable order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def latent_names(self) -> tuple[str, ...]:
        """Return names deliberately kept outside the observation path."""
        return tuple(signal.name for signal in self.latent_signals)

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[InstrumentationChannel],
        *,
        name: str = "custom",
        version: str = "1.0.0",
        observation_schema_version: str = "1.0.0",
        latent_signals: Iterable[SignalSpec] = (),
    ) -> "InstrumentationProfile":
        """Build a profile from explicit calibrated channels."""
        return cls(
            identity=SpecIdentity(name, version),
            channels=tuple(channels),
            observation_schema_version=observation_schema_version,
            latent_signals=tuple(latent_signals),
        )

    @classmethod
    def ideal(
        cls,
        signals: Iterable[SignalSpec | InstrumentationChannel],
        *,
        name: str = "ideal",
        version: str = "1.0.0",
    ) -> "InstrumentationProfile":
        """Build an exact, zero-noise profile for a set of signals."""
        channels = tuple(
            replace(channel, noise_std=0.0, bias=0.0, drift_per_step=0.0, dropout_probability=0.0)
            for channel in _coerce_channels(signals)
        )
        return cls.from_channels(channels, name=name, version=version)

    @classmethod
    def standard(
        cls,
        signals: Iterable[SignalSpec | InstrumentationChannel],
        *,
        noise_std: float = 0.01,
        name: str = "standard",
        version: str = "1.0.0",
    ) -> "InstrumentationProfile":
        """Build a conventional online-sensor profile with modest noise."""
        channels = tuple(replace(channel, noise_std=noise_std) for channel in _coerce_channels(signals))
        return cls.from_channels(channels, name=name, version=version)

    @classmethod
    def rich(
        cls,
        signals: Iterable[SignalSpec | InstrumentationChannel],
        *,
        name: str = "rich",
        version: str = "1.0.0",
    ) -> "InstrumentationProfile":
        """Build an ideal profile used as an instrumentation upper bound."""
        return cls.ideal(signals, name=name, version=version)

    @classmethod
    def degraded(
        cls,
        signals: Iterable[SignalSpec | InstrumentationChannel],
        *,
        noise_std: float = 0.05,
        dropout_probability: float = 0.1,
        name: str = "degraded",
        version: str = "1.0.0",
    ) -> "InstrumentationProfile":
        """Build a profile with noisy, intermittently missing measurements."""
        channels = tuple(
            replace(channel, noise_std=noise_std, dropout_probability=dropout_probability)
            for channel in _coerce_channels(signals)
        )
        return cls.from_channels(channels, name=name, version=version)

    def compose(
        self,
        other: "InstrumentationProfile",
        *,
        name: str | None = None,
        version: str | None = None,
    ) -> "InstrumentationProfile":
        """Compose profiles, with ``other`` overriding duplicate output names."""
        merged: dict[str, InstrumentationChannel] = {channel.name: channel for channel in self.channels}
        merged.update({channel.name: channel for channel in other.channels})
        latent: dict[str, SignalSpec] = {signal.name: signal for signal in self.latent_signals}
        latent.update({signal.name: signal for signal in other.latent_signals})
        observed_names = set(merged)
        return InstrumentationProfile.from_channels(
            tuple(merged.values()),
            name=name or f"{self.identity.name}+{other.identity.name}",
            version=version or other.identity.version,
            observation_schema_version=other.observation_schema_version,
            latent_signals=tuple(signal for name, signal in latent.items() if name not in observed_names),
        )

    def validate_reward_inputs(self, names: Iterable[str]) -> tuple[str, ...]:
        """Validate that reward names belong to observed profile outputs."""
        requested = tuple(names)
        latent = sorted(set(requested) & set(self.latent_names))
        if latent:
            raise ValueError("reward inputs cannot use latent signals: " + ", ".join(latent))
        missing = sorted(set(requested) - set(self.observed_names))
        if missing:
            raise ValueError("reward inputs unavailable through instrumentation: " + ", ".join(missing))
        return requested

    def reward_inputs(
        self,
        observation: ObservedSignals | InstrumentationBatch,
        names: Iterable[str],
        *,
        step: int | None = None,
    ) -> dict[str, Array]:
        """Retrieve reward inputs only from available observed outputs."""
        self.validate_reward_inputs(names)
        if isinstance(observation, InstrumentationBatch):
            if step is None:
                raise ValueError("step is required when retrieving reward inputs from a batch")
            frame = observation.at(step)
        else:
            frame = observation
        return frame.require(names)

    def observe_trajectory(
        self,
        latent_signals: LatentSignals,
        *,
        seed_bundle: SeedBundle | int,
        simulation_time_step: float = 1.0,
        events: Iterable[TimedEvent] = (),
    ) -> InstrumentationBatch:
        """Transform named latent trajectories into deterministic observations."""
        arrays: dict[str, Array] = {}
        for channel in self.channels:
            if channel.source.name not in latent_signals:
                raise ValueError(f"latent signal {channel.source.name!r} is unavailable to instrumentation")
            values = jnp.asarray(latent_signals[channel.source.name])
            if values.ndim != 1:
                raise ValueError(f"latent trajectory {channel.source.name!r} must be one-dimensional")
            if not jnp.issubdtype(values.dtype, jnp.number):
                raise ValueError(f"latent trajectory {channel.source.name!r} must be numeric")
            if not jnp.issubdtype(values.dtype, jnp.inexact):
                values = values.astype(jnp.float32)
            arrays[channel.source.name] = values
        lengths = {values.shape[0] for values in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("latent trajectories must share a non-empty horizon")

        seeds = SeedBundle.from_seed(seed_bundle) if isinstance(seed_bundle, int) else seed_bundle
        event_tuple = tuple(events)
        measured: dict[str, Array] = {}
        available: dict[str, Array] = {}
        timing = {channel.name: channel.timing for channel in self.channels}
        provenance = {channel.name: channel.provenance for channel in self.channels}
        for channel_index, channel in enumerate(self.channels):
            values, valid = self._measure_channel(
                arrays[channel.source.name],
                channel,
                channel_index=channel_index,
                seeds=seeds,
                simulation_time_step=simulation_time_step,
                events=event_tuple,
            )
            measured[channel.name] = values
            available[channel.name] = valid
        return InstrumentationBatch(measured, available, timing, provenance, next(iter(lengths)))

    def observe(
        self,
        latent_signals: LatentSignals,
        *,
        step: int = 0,
        seed_bundle: SeedBundle | int,
        simulation_time_step: float = 1.0,
        events: Iterable[TimedEvent] = (),
    ) -> ObservedSignals:
        """Return one observed frame from a latent trajectory at ``step``."""
        return self.observe_trajectory(
            latent_signals,
            seed_bundle=seed_bundle,
            simulation_time_step=simulation_time_step,
            events=events,
        ).at(step)

    transform = observe_trajectory

    def _measure_channel(
        self,
        source: Array,
        channel: InstrumentationChannel,
        *,
        channel_index: int,
        seeds: SeedBundle,
        simulation_time_step: float,
        events: Sequence[TimedEvent],
    ) -> tuple[Array, Array]:
        horizon = source.shape[0]
        delay_steps = _step_count(channel.timing.delay, simulation_time_step, f"delay for {channel.name}")
        cadence_steps = _step_count(
            channel.timing.sample_period, simulation_time_step, f"sample_period for {channel.name}"
        )
        if cadence_steps <= 0:
            raise ValueError(f"sample_period for {channel.name} must be at least one simulation step")
        indices = jnp.arange(horizon, dtype=jnp.int32)
        source_indices = indices - delay_steps
        safe_indices = jnp.clip(source_indices, 0, horizon - 1)
        delayed = source[safe_indices]
        delayed = jnp.where(source_indices >= 0, delayed, jnp.nan)
        sample_mask = (source_indices >= 0) & (source_indices % cadence_steps == 0)

        sensor_root = jax.random.fold_in(seeds.key(RandomStream.SENSOR, channel_index), channel_index)
        fault_root = jax.random.fold_in(seeds.key(RandomStream.FAULT, channel_index), channel_index)
        sensor_keys = jax.vmap(lambda step: jax.random.fold_in(sensor_root, step))(indices)
        fault_keys = jax.vmap(lambda step: jax.random.fold_in(fault_root, step))(indices)
        noise = jax.vmap(jax.random.normal)(sensor_keys) * channel.noise_std
        dropout = jax.vmap(jax.random.uniform)(fault_keys) < channel.dropout_probability
        fault_active = jnp.asarray(
            tuple(
                any(
                    event.kind.value == "sensor_fault"
                    and event.target in (channel.name, channel.source.name)
                    and event.is_active(int(step))
                    for event in events
                )
                for step in range(horizon)
            ),
            dtype=bool,
        )
        valid_sample = sample_mask & ~dropout & ~fault_active
        measured = channel.scale * delayed + channel.offset + channel.bias
        measured = measured + channel.drift_per_step * indices + noise
        measured = jnp.where(valid_sample, measured, jnp.nan)

        if channel.hold_last:
            def hold(
                carry: tuple[Array, Array], item: tuple[Array, Array]
            ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
                previous, seen = carry
                value, valid = item
                current = jnp.where(valid, value, previous)
                current_seen = seen | valid
                return (current, current_seen), (current, current_seen)

            (_, _), (held, held_available) = jax.lax.scan(
                hold,
                (jnp.asarray(jnp.nan, dtype=source.dtype), jnp.asarray(False)),
                (measured, valid_sample),
            )
            output = jnp.where(fault_active, jnp.nan, held)
            is_available = held_available & ~fault_active
        else:
            output = measured
            is_available = valid_sample
        return output, is_available


def validate_reward_inputs(profile: InstrumentationProfile, names: Iterable[str]) -> tuple[str, ...]:
    """Validate reward names against observed profile outputs."""
    return profile.validate_reward_inputs(names)


__all__ = [
    "InstrumentationBatch",
    "InstrumentationChannel",
    "InstrumentationProfile",
    "InstrumentationSignal",
    "LatentSignals",
    "ObservedSignals",
    "SignalMapping",
    "SignalInput",
    "validate_reward_inputs",
]
