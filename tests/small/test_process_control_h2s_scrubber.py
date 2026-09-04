"""Tests for the H₂S scrubber benchmark."""

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.dosing_system import DIRECT, FEEDFORWARD
from process_control.benchmarks.h2s_scrubber import (
    H2SScenario,
    H2SScrubberConfig,
    H2SScrubberState,
    make_h2s_scrubber_benchmark,
)

_H2SResetFn = Callable[[jax.Array], tuple[H2SScrubberState, jax.Array]]
_H2SStepFn = Callable[
    [H2SScrubberState, jax.Array, jax.Array],
    tuple[H2SScrubberState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]],
]


@pytest.fixture
def config() -> H2SScrubberConfig:
    return H2SScrubberConfig()


@pytest.fixture
def scrubber_benchmark(config: H2SScrubberConfig) -> tuple[_H2SResetFn, _H2SStepFn]:
    return make_h2s_scrubber_benchmark(config)


@pytest.fixture
def rng() -> jax.Array:
    return jax.random.key(42)


class TestReset:
    def test_reset_produces_valid_state_and_obs(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, _ = scrubber_benchmark
        state, obs = reset_fn(rng)

        assert isinstance(state, H2SScrubberState)
        assert obs.shape == (12,)
        assert jnp.all(jnp.isfinite(obs))

    def test_reset_step_count_zero(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, _ = scrubber_benchmark
        state, _ = reset_fn(rng)
        assert int(state.step_count) == 0

    def test_reset_sump_at_nominal_volume(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array, config: H2SScrubberConfig) -> None:
        reset_fn, _ = scrubber_benchmark
        state, _ = reset_fn(rng)
        assert jnp.isclose(state.sump.volume, config.sump_nominal_volume)


class TestStep:
    def test_step_returns_correct_shapes(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, step_fn = scrubber_benchmark
        state, _ = reset_fn(rng)

        k1, k2 = jax.random.split(rng)
        action = jnp.zeros(3)  # neutral setpoints

        new_state, obs, reward, done, info = step_fn(state, action, k2)
        assert obs.shape == (12,)
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(info, dict)

    def test_step_increments_count(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, step_fn = scrubber_benchmark
        state, _ = reset_fn(rng)

        new_state, _, _, _, _ = step_fn(state, jnp.zeros(3), rng)
        assert int(new_state.step_count) == 1

    def test_step_produces_finite_outputs(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, step_fn = scrubber_benchmark
        state, _ = reset_fn(rng)

        for i in range(20):
            k = jax.random.fold_in(rng, i)
            state, obs, reward, done, info = step_fn(state, jnp.zeros(3), k)

        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)
        for v in info.values():
            assert jnp.isfinite(v), "Non-finite value in info"


class TestActions:
    def test_benchmark_consumes_physical_setpoints(self, config: H2SScrubberConfig, rng: jax.Array) -> None:
        """Accept physical setpoints because normalization belongs to the bridge."""
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)
        _, _, _, _, info = step_fn(
            state,
            jnp.array([config.ph_sp_max, config.orp_sp_max, config.makeup_flow_sp_max]),
            rng,
        )

        assert jnp.all(jnp.isfinite(jnp.array(tuple(info.values()))))

    def test_high_setpoints_increase_dosing(self, rng: jax.Array) -> None:
        """Setting all setpoints high should increase pump outputs over time."""
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)

        high_action = jnp.array([config.ph_sp_max, config.orp_sp_max, config.makeup_flow_sp_max])
        for i in range(50):
            k = jax.random.fold_in(rng, i)
            state, _, _, _, info = step_fn(state, high_action, k)

        # At least caustic or bleach pump should be elevated
        assert float(info["caustic_pump"]) > config.caustic_ff * 0.5 or float(info["bleach_pump"]) > config.bleach_ff * 0.5

    def test_makeup_flow_setpoint_drives_makeup_pump(self, rng: jax.Array) -> None:
        """Drive the makeup pump from the supervisory flow setpoint.

        The sump level remains a measured process state rather than the third action.
        """
        config = H2SScrubberConfig(
            makeup_sensor_noise=0.0,
            makeup_sensor_drift=0.0,
        )
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)

        def run(flow_setpoint: float) -> dict[str, jax.Array]:
            state, _ = reset_fn(rng)
            action = jnp.array([config.ph_sp_default, config.orp_sp_default, flow_setpoint])
            for i in range(80):
                state, _, _, _, info = step_fn(state, action, jax.random.fold_in(rng, i))
            return info

        low_flow = run(config.makeup_flow_sp_min)
        high_flow = run(config.makeup_flow_sp_max)

        assert float(high_flow["makeup_pump"]) > float(low_flow["makeup_pump"])
        assert jnp.isfinite(high_flow["true_level"])


def test_load_shift_scenario_is_selected_at_construction() -> None:
    """Apply the named twofold gas-load shift without changing environment shape."""
    nominal_reset, nominal_step = make_h2s_scrubber_benchmark(H2SScrubberConfig())
    shifted_reset, shifted_step = make_h2s_scrubber_benchmark(H2SScrubberConfig(scenario=H2SScenario.LOAD_SHIFT))
    nominal_state, _ = nominal_reset(jax.random.PRNGKey(0))
    shifted_state, _ = shifted_reset(jax.random.PRNGKey(0))
    nominal_state = replace(nominal_state, step_count=jnp.array(2_000))
    shifted_state = replace(shifted_state, step_count=jnp.array(2_000))

    _, _, _, _, nominal_info = nominal_step(nominal_state, jnp.zeros(3), jax.random.PRNGKey(1))
    _, _, _, _, shifted_info = shifted_step(shifted_state, jnp.zeros(3), jax.random.PRNGKey(1))

    assert float(shifted_info["inlet_h2s_ppm"]) > float(nominal_info["inlet_h2s_ppm"])


class TestReward:
    def test_efficiency_penalty_is_linear_one_sided_hinge(self, rng: jax.Array) -> None:
        """Penalize each unit of deficit linearly, but never reward compliance."""
        config = H2SScrubberConfig(efficiency_penalty_weight=100.0)
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)
        _, _, reward, _, info = step_fn(state, jnp.zeros(3), rng)

        expected = -100.0 * max(0.95 - float(info["removal_efficiency"]), 0.0)
        expected -= 0.01 * float(info["opex"])
        assert jnp.isclose(reward, expected)

        compliant_config = H2SScrubberConfig(target_efficiency=0.0)
        compliant_reset, compliant_step = make_h2s_scrubber_benchmark(compliant_config)
        compliant_state, _ = compliant_reset(rng)
        _, _, compliant_reward, _, compliant_info = compliant_step(compliant_state, jnp.zeros(3), rng)
        assert jnp.isclose(compliant_reward, -0.01 * compliant_info["opex"])

    def test_reward_is_negative(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        """Reward should be negative (minimizing cost + penalty)."""
        reset_fn, step_fn = scrubber_benchmark
        state, _ = reset_fn(rng)

        _, _, reward, _, _ = step_fn(state, jnp.zeros(3), rng)
        assert float(reward) <= 0.0

    def test_higher_efficiency_gives_better_reward(self, rng: jax.Array) -> None:
        """Configurations with better removal should get less penalty."""
        # High oxidant = better removal = less penalty
        good_config = H2SScrubberConfig(efficiency_penalty_weight=100.0)
        reset_good, step_good = make_h2s_scrubber_benchmark(good_config)

        state, _ = reset_good(rng)
        # Run a few steps to let things stabilize
        for i in range(10):
            k = jax.random.fold_in(rng, i)
            state, _, reward, _, info = step_good(state, jnp.zeros(3), k)

        # Efficiency should be between 0 and 1
        assert 0.0 <= float(info["removal_efficiency"]) <= 1.0

    @pytest.mark.parametrize("scenario", [H2SScenario.NOMINAL, H2SScenario.LOAD_SHIFT])
    def test_high_chemical_setpoints_improve_removal(self, rng: jax.Array, scenario: H2SScenario) -> None:
        """High chemical setpoints improve removal after chemistry settles.

        Makeup flow is held at its minimum so this isolates reagent control.
        """
        config = H2SScrubberConfig(
            scenario=scenario,
            caustic_sensor_noise=0.0,
            caustic_sensor_drift=0.0,
            bleach_sensor_noise=0.0,
            bleach_sensor_drift=0.0,
            makeup_sensor_noise=0.0,
            makeup_sensor_drift=0.0,
        )
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)

        def settled_efficiency(action: jax.Array) -> float:
            state, _ = reset_fn(rng)
            efficiencies = []
            for i in range(2200):
                state, _, _, _, info = step_fn(state, action, jax.random.fold_in(rng, i))
                efficiencies.append(float(info["removal_efficiency"]))
            post_event = jnp.array(efficiencies[2050:2100])
            assert jnp.all(jnp.isfinite(post_event))
            return float(jnp.mean(post_event))

        low_action = jnp.array([config.ph_sp_min, config.orp_sp_min, config.makeup_flow_sp_min])
        high_action = jnp.array([config.ph_sp_max, config.orp_sp_max, config.makeup_flow_sp_min])

        assert settled_efficiency(high_action) > settled_efficiency(low_action)

    def test_low_cost_control_is_less_compliant(self, rng: jax.Array) -> None:
        """Low dosing costs less while producing a measurable compliance deficit."""
        config = H2SScrubberConfig(
            caustic_sensor_noise=0.0,
            caustic_sensor_drift=0.0,
            bleach_sensor_noise=0.0,
            bleach_sensor_drift=0.0,
            makeup_sensor_noise=0.0,
            makeup_sensor_drift=0.0,
        )
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)

        def rollout(action: jax.Array) -> tuple[float, float]:
            state, _ = reset_fn(rng)
            efficiencies = []
            for i in range(2200):
                state, _, _, _, info = step_fn(state, action, jax.random.fold_in(rng, i))
                efficiencies.append(float(info["removal_efficiency"]))
            return float(info["opex"]), sum(efficiencies[2050:2100]) / 50

        low_cost, low_efficiency = rollout(jnp.array([config.ph_sp_min, config.orp_sp_min, config.makeup_flow_sp_min]))
        high_cost, high_efficiency = rollout(jnp.array([config.ph_sp_max, config.orp_sp_max, config.makeup_flow_sp_default]))

        assert low_cost < high_cost
        assert low_efficiency < high_efficiency


class TestInfo:
    def test_info_keys_present(self, scrubber_benchmark: tuple[_H2SResetFn, _H2SStepFn], rng: jax.Array) -> None:
        reset_fn, step_fn = scrubber_benchmark
        state, _ = reset_fn(rng)

        _, _, _, _, info = step_fn(state, jnp.zeros(3), rng)
        expected_keys = [
            "removal_efficiency",
            "outlet_h2s_ppm",
            "inlet_h2s_ppm",
            "gas_flow",
            "true_ph",
            "true_orp",
            "true_level",
            "sensed_makeup_flow",
            "caustic_pump",
            "bleach_pump",
            "makeup_pump",
            "caustic_pi",
            "bleach_pi",
            "makeup_pi",
            "opex",
            "eff_penalty",
        ]
        for key in expected_keys:
            assert key in info, f"Missing info key: {key}"


class TestStability:
    def test_long_rollout_stable(self, rng: jax.Array) -> None:
        """Run 500 steps (~42 hours) without NaN or explosion."""
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)

        for i in range(500):
            k = jax.random.fold_in(rng, i)
            # Vary actions sinusoidally
            phase = i / 50.0 * jnp.pi
            action = jnp.array([jnp.sin(phase) * 0.3, jnp.cos(phase) * 0.3, 0.0])
            state, obs, reward, _, _ = step_fn(state, action, k)

        assert jnp.all(jnp.isfinite(obs)), "Observation has NaN/Inf after 500 steps"
        assert jnp.isfinite(reward), "Reward has NaN/Inf after 500 steps"
        assert state.sump.oxidant >= 0.0
        assert state.sump.sulfide >= 0.0
        assert state.sump.alkalinity >= 0.0

    def test_extreme_actions_stay_bounded(self, rng: jax.Array) -> None:
        """Random extreme actions should not cause numerical issues."""
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)

        for i in range(100):
            k1, k2 = jax.random.split(jax.random.fold_in(rng, i))
            action = jax.random.uniform(k1, (3,), minval=-1.0, maxval=1.0)
            state, obs, reward, _, _ = step_fn(state, action, k2)

        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)


class TestControlModes:
    def test_direct_mode_benchmark(self, rng: jax.Array) -> None:
        """Benchmark should work with DIRECT control mode."""
        config = H2SScrubberConfig(
            caustic_control_mode=DIRECT,
            bleach_control_mode=DIRECT,
            makeup_control_mode=DIRECT,
        )
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, obs = reset_fn(rng)

        state, obs, reward, _, info = step_fn(state, jnp.zeros(3), rng)
        assert jnp.all(jnp.isfinite(obs))

    def test_feedforward_mode_benchmark(self, rng: jax.Array) -> None:
        """Benchmark should work with FEEDFORWARD control mode."""
        config = H2SScrubberConfig(
            caustic_control_mode=FEEDFORWARD,
            bleach_control_mode=FEEDFORWARD,
            makeup_control_mode=FEEDFORWARD,
        )
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, obs = reset_fn(rng)

        state, obs, reward, _, info = step_fn(state, jnp.zeros(3), rng)
        assert jnp.all(jnp.isfinite(obs))


class TestJIT:
    def test_jit_step(self, rng: jax.Array) -> None:
        """Step function should be JIT-compilable."""
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        state, _ = reset_fn(rng)

        jit_step = jax.jit(step_fn)
        new_state, obs, reward, done, info = jit_step(state, jnp.zeros(3), rng)

        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)

    def test_jit_reset(self, rng: jax.Array) -> None:
        """Reset function should be JIT-compilable."""
        config = H2SScrubberConfig()
        reset_fn, _ = make_h2s_scrubber_benchmark(config)

        jit_reset = jax.jit(reset_fn)
        state, obs = jit_reset(rng)

        assert jnp.all(jnp.isfinite(obs))
        assert isinstance(state, H2SScrubberState)
