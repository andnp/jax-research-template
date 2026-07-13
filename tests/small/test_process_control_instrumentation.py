from dataclasses import replace

import numpy as np
import pytest
from jax import numpy as jnp
from process_control.environment_specs import SignalSource, SignalSpec, SignalTiming
from process_control.instrumentation import InstrumentationChannel, InstrumentationProfile
from process_control.scenarios import EventKind, EventSeverity, SeedBundle, TimedEvent

LEVEL = SignalSpec("level", "m")


def test_ideal_profile_maps_latent_values_exactly() -> None:
    profile = InstrumentationProfile.ideal((LEVEL,))
    batch = profile.observe_trajectory(
        {"level": jnp.asarray([1.0, 2.0, 3.0])}, seed_bundle=SeedBundle.from_seed(4)
    )

    np.testing.assert_array_equal(batch.values["level"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(batch.available["level"], [True, True, True])
    assert profile.spec.observed_signals[0].source is SignalSource.SENSOR


def test_delay_and_cadence_hold_the_last_sample() -> None:
    profile = InstrumentationProfile.from_channels(
        (
            InstrumentationChannel(
                source=LEVEL,
                output=SignalSpec("level.measured", "m"),
                timing=SignalTiming(sample_period=2.0, delay=1.0),
            ),
        )
    )
    batch = profile.observe_trajectory(
        {"level": jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0])},
        seed_bundle=0,
    )

    assert not bool(batch.available["level.measured"][0])
    np.testing.assert_array_equal(batch.values["level.measured"][1:], [10.0, 10.0, 30.0, 30.0])
    np.testing.assert_array_equal(batch.available["level.measured"], [False, True, True, True, True])


def test_bias_and_drift_are_deterministic_and_noise_uses_sensor_stream() -> None:
    channel = InstrumentationChannel(
        source=LEVEL,
        output=SignalSpec("level.measured", "m"),
        noise_std=0.2,
        bias=1.0,
        drift_per_step=0.5,
    )
    profile = InstrumentationProfile.from_channels((channel,))
    latent = {"level": jnp.zeros(5)}
    first = profile.observe_trajectory(latent, seed_bundle=17)
    repeated = profile.observe_trajectory(latent, seed_bundle=17)
    other = profile.observe_trajectory(latent, seed_bundle=18)

    np.testing.assert_array_equal(first.values["level.measured"], repeated.values["level.measured"])
    assert not np.array_equal(first.values["level.measured"], other.values["level.measured"])

    deterministic = InstrumentationProfile.from_channels((replace(channel, noise_std=0.0),))
    expected = deterministic.observe_trajectory(latent, seed_bundle=17)
    np.testing.assert_array_equal(expected.values["level.measured"], [1.0, 1.5, 2.0, 2.5, 3.0])


def test_dropout_is_deterministic_and_uses_fault_stream() -> None:
    channel = InstrumentationChannel(
        source=LEVEL,
        output=SignalSpec("level.measured", "m"),
        dropout_probability=1.0,
    )
    profile = InstrumentationProfile.from_channels((channel,))
    first = profile.observe_trajectory({"level": jnp.ones(4)}, seed_bundle=9)
    repeated = profile.observe_trajectory({"level": jnp.ones(4)}, seed_bundle=9)

    np.testing.assert_array_equal(first.available["level.measured"], [False] * 4)
    np.testing.assert_array_equal(first.available["level.measured"], repeated.available["level.measured"])
    assert np.isnan(np.asarray(first.values["level.measured"])).all()


def test_sensor_fault_event_masks_observation_then_recovers() -> None:
    profile = InstrumentationProfile.ideal((LEVEL,))
    event = TimedEvent(
        name="analyzer-failure",
        kind=EventKind.SENSOR_FAULT,
        target="level",
        units="m",
        start_step=1,
        end_step=3,
        magnitude=1.0,
        severity=EventSeverity.HIGH,
    )
    batch = profile.observe_trajectory(
        {"level": jnp.asarray([1.0, 2.0, 3.0, 4.0])}, seed_bundle=1, events=(event,)
    )

    np.testing.assert_array_equal(batch.available["level"], [True, False, False, True])
    np.testing.assert_array_equal(batch.values["level"][::3], [1.0, 4.0])
    assert np.isnan(np.asarray(batch.values["level"])[1:3]).all()


def test_reward_inputs_reject_latent_or_unavailable_signals() -> None:
    profile = InstrumentationProfile.from_channels(
        (
            InstrumentationChannel(
                source=SignalSpec("level.true", "m"),
                output=SignalSpec("level.measured", "m"),
                timing=SignalTiming(sample_period=1.0, delay=1.0),
            ),
        )
    )
    assert profile.latent_names == ("level.true",)
    with pytest.raises(ValueError, match="latent"):
        profile.validate_reward_inputs(("level.true",))
    with pytest.raises(ValueError, match="unavailable"):
        profile.validate_reward_inputs(("not-a-signal",))

    batch = profile.observe_trajectory({"level.true": jnp.ones(2)}, seed_bundle=2)
    with pytest.raises(ValueError, match="unavailable at this step"):
        profile.reward_inputs(batch, ("level.measured",), step=0)
    with pytest.raises(ValueError, match="unavailable"):
        profile.observe({"other": jnp.ones(2)}, seed_bundle=2)


def test_profiles_compose_with_later_channel_overrides() -> None:
    base = InstrumentationProfile.ideal((LEVEL,))
    extra = InstrumentationProfile.from_channels(
        (
            InstrumentationChannel(
                source=LEVEL,
                output=SignalSpec("level.measured", "m"),
            ),
        ),
        name="standard",
    )
    combined = base.compose(extra)

    assert combined.observed_names == ("level", "level.measured")
    assert combined.latent_names == ()
