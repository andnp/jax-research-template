import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.components import ComponentSchema
from process_control.scenarios.tabular_influent import (
    NO_RANDOMIZATION,
    InfluentProfile,
    InfluentRandomization,
    TabularInfluentParams,
    TabularInfluentState,
    Uniform,
    advance,
    bsm1_influent,
    influent,
    reset,
)

SCHEMA = ComponentSchema(names=("solute", "solids"), particulate=(False, True), tss_factor=(0.0, 0.75))
RAMP = InfluentProfile(
    times=np.array([0.0, 1.0, 3.0]),
    components=np.array([[10.0, 100.0], [20.0, 200.0], [40.0, 400.0]]),
    flow=np.array([1.0, 2.0, 4.0]),
    temperature=np.array([10.0, 12.0, 14.0]),
    period=4.0,
)
FLAT = InfluentProfile(
    times=np.zeros(1), components=np.array([[5.0, 50.0]]), flow=np.array([3.0]), temperature=np.array([20.0]), period=1.0
)


def _params(randomization: InfluentRandomization = NO_RANDOMIZATION) -> TabularInfluentParams:
    return TabularInfluentParams(schema=SCHEMA, profiles={"ramp": RAMP, "flat": FLAT}, randomization=randomization)


def _state(profile: int = 0, time: float = 0.0, flow_scale: float = 1.0, load_scale: float = 1.0) -> TabularInfluentState:
    return TabularInfluentState(
        profile=jnp.array(profile),
        time=jnp.array(time),
        flow_scale=jnp.array(flow_scale),
        load_scale=jnp.array(load_scale),
        flow_noise=jnp.array(0.0),
    )


def _episodes(params: TabularInfluentParams, count: int) -> TabularInfluentState:
    def draw(key: jax.Array) -> TabularInfluentState:
        return reset(params, key)

    return jax.jit(jax.vmap(draw))(jax.random.split(jax.random.key(0), count))


class TestInterpolation:
    @pytest.mark.parametrize(
        ("time", "flow", "solute", "temperature"),
        [(1.0, 2.0, 20.0, 12.0), (2.0, 3.0, 30.0, 13.0), (0.5, 1.5, 15.0, 11.0), (3.5, 2.5, 25.0, 12.0), (5.0, 2.0, 20.0, 12.0)],
    )
    def test_values_are_linear_between_samples_and_wrap_around(self, time: float, flow: float, solute: float, temperature: float) -> None:
        """Samples are reproduced, gaps are linear, the last sample blends into the first and time repeats."""
        stream, water_temperature = influent(_state(time=time), _params())

        assert jnp.allclose(stream.flow, flow)
        assert jnp.allclose(stream.components, jnp.array([solute, 10.0 * solute]))
        assert jnp.allclose(water_temperature, temperature)
        assert stream.schema == SCHEMA

    def test_single_sample_profile_is_constant(self) -> None:
        """A one-sample profile is a constant influent at every time."""
        stream, temperature = influent(_state(profile=1, time=7.3), _params())

        assert jnp.allclose(stream.flow, 3.0)
        assert jnp.allclose(stream.components, jnp.array([5.0, 50.0]))
        assert jnp.allclose(temperature, 20.0)


class TestEpisodeRandomization:
    def test_weights_select_the_profile(self) -> None:
        """Zero-weight profiles are never drawn."""
        episodes = _episodes(_params(dataclasses.replace(NO_RANDOMIZATION, profile_weights=(0.0, 1.0))), 64)

        assert jnp.all(episodes.profile == 1)

    def test_unweighted_episodes_use_every_profile(self) -> None:
        """Without weights each profile is drawn."""
        episodes = _episodes(_params(), 64)

        assert set(np.asarray(episodes.profile).tolist()) == {0, 1}

    def test_start_lies_within_the_chosen_profile_period(self) -> None:
        """Start offsets are fractions of the drawn profile's own period."""
        randomization = InfluentRandomization(profile_weights=(1.0, 0.0), start_fraction=Uniform(0.0, 1.0))

        episodes = _episodes(_params(randomization), 256)

        assert jnp.all((episodes.time >= 0.0) & (episodes.time < RAMP.period))
        assert jnp.std(episodes.time) > 0.5

    def test_scale_factors_follow_their_distributions(self) -> None:
        """Flow and load scales are drawn per episode from their configured ranges."""
        randomization = InfluentRandomization(flow_scale=Uniform(0.5, 0.6), load_scale=Uniform(2.0, 3.0))

        episodes = _episodes(_params(randomization), 128)

        assert jnp.all((episodes.flow_scale >= 0.5) & (episodes.flow_scale <= 0.6))
        assert jnp.all((episodes.load_scale >= 2.0) & (episodes.load_scale <= 3.0))

    def test_load_scale_applies_to_the_selected_components(self) -> None:
        """Only the components named for scaling change concentration; flow follows the flow scale."""
        params = _params(dataclasses.replace(NO_RANDOMIZATION, scaled_components=frozenset({"solids"})))

        stream, _ = influent(_state(time=1.0, flow_scale=0.5, load_scale=2.0), params)

        assert jnp.allclose(stream.components, jnp.array([20.0, 400.0]))
        assert jnp.allclose(stream.flow, 1.0)

    def test_no_randomization_replays_the_first_profile_from_its_start(self) -> None:
        """With every factor fixed at one, a single-profile episode is the raw record."""
        params = TabularInfluentParams(schema=SCHEMA, profiles={"ramp": RAMP}, randomization=NO_RANDOMIZATION)

        stream, _ = influent(reset(params, jax.random.key(3)), params)

        assert jnp.allclose(stream.flow, 1.0)
        assert jnp.allclose(stream.components, jnp.array([10.0, 100.0]))


class TestFlowNoise:
    def test_noise_has_unit_mean_and_the_configured_persistence(self) -> None:
        """AR(1) flow noise keeps the mean flow and decorrelates over its correlation time."""
        randomization = dataclasses.replace(NO_RANDOMIZATION, flow_noise_std=0.1, flow_noise_correlation_time=2.0)
        params = TabularInfluentParams(schema=SCHEMA, profiles={"flat": FLAT}, randomization=randomization)
        dt = jnp.array(0.5)

        def evolve(state: TabularInfluentState, key: jax.Array) -> tuple[TabularInfluentState, jax.Array]:
            state = advance(state, params, dt, key)
            return state, influent(state, params)[0].flow

        _, flows = jax.lax.scan(evolve, reset(params, jax.random.key(1)), jax.random.split(jax.random.key(2), 20000))

        log_noise = jnp.log(flows / 3.0)
        lag_correlation = jnp.corrcoef(log_noise[:-1], log_noise[1:])[0, 1]
        assert jnp.mean(flows) == pytest.approx(3.0, rel=0.01)
        assert jnp.std(log_noise) == pytest.approx(0.1, rel=0.1)
        assert lag_correlation == pytest.approx(np.exp(-0.25), abs=0.03)

    def test_advance_moves_the_profile_clock(self) -> None:
        """Advancing by dt moves the episode along its profile."""
        state = advance(_state(time=0.5), _params(), jnp.array(1.5), jax.random.key(0))

        stream, _ = influent(state, _params())

        assert jnp.allclose(state.time, 2.0)
        assert jnp.allclose(stream.flow, 3.0)


class TestValidation:
    def test_times_must_increase_within_the_period(self) -> None:
        """Unordered sample times or samples beyond the period cannot be interpolated periodically."""
        with pytest.raises(ValueError, match="increase within"):
            dataclasses.replace(RAMP, period=3.0)

    def test_columns_must_have_equal_sample_counts(self) -> None:
        """Every column needs one value per sample time."""
        with pytest.raises(ValueError, match="equal sample counts"):
            dataclasses.replace(RAMP, flow=np.array([1.0, 2.0]))

    def test_profiles_must_match_the_schema(self) -> None:
        """A profile's component columns must follow the schema."""
        other = ComponentSchema(names=("a",), particulate=(False,), tss_factor=(0.0,))

        with pytest.raises(ValueError, match="do not carry the 1 schema components"):
            TabularInfluentParams(schema=other, profiles={"ramp": RAMP})

    def test_weights_must_cover_every_profile(self) -> None:
        """One weight is needed per profile."""
        with pytest.raises(ValueError, match="1 weights given for 2 profiles"):
            _params(InfluentRandomization(profile_weights=(1.0,)))

    def test_scaled_components_must_exist(self) -> None:
        """Scaling a component the schema lacks is a configuration mistake."""
        with pytest.raises(ValueError, match="not in the schema"):
            _params(InfluentRandomization(scaled_components=frozenset({"missing"})))


class TestBSM1Preset:
    def test_bundles_the_stabilisation_and_weather_profiles(self) -> None:
        """The preset offers the constant influent and the three 14-day weather files in hours."""
        params = bsm1_influent()

        assert list(params.profiles) == ["constant", "dry", "rain", "storm"]
        for weather in ("dry", "rain", "storm"):
            profile = params.profiles[weather]
            assert profile.times.shape == (1344,)
            assert profile.period == pytest.approx(336.0, rel=1e-6)
            assert np.allclose(profile.temperature, 15.0)

    def test_records_are_converted_to_hourly_units(self) -> None:
        """The first dry-weather sample carries 21477 m³/d and the constant influent 18446 m³/d."""
        params = bsm1_influent(NO_RANDOMIZATION)

        dry, _ = influent(_state(profile=1), params)
        constant, _ = influent(_state(profile=0), params)

        assert jnp.allclose(dry.flow, 21477.0 / 24.0)
        assert jnp.allclose(dry.components[1], 63.63455)
        assert jnp.allclose(constant.flow, 18446.0 / 24.0)
        assert jnp.allclose(constant.components[3], 202.32)
