from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.equalization_tank import (
    EqualizationTankBenchmarkConfig,
    make_equalization_tank_benchmark,
)
from process_control.benchmarks.ph_neutralization import (
    PhNeutralizationBenchmarkConfig,
    make_ph_neutralization_benchmark,
)


class TestPhNeutralizationBenchmark:
    def test_default_dose_authority_covers_nominal_acid_load(self) -> None:
        config = PhNeutralizationBenchmarkConfig()

        assert config.pump_min_dose <= config.pi_ff <= config.pump_max_dose
        assert config.pi_output_max == config.pump_max_dose

    def test_calibrated_midpoint_stays_off_pH_rails(self) -> None:
        """The calibrated nominal action avoids persistent pH rail saturation."""
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        key = jax.random.PRNGKey(42)
        key, reset_key = jax.random.split(key)
        state, _ = reset_fn(reset_key)
        midpoint = jnp.array((config.pump_min_dose + config.pump_max_dose) / 2.0)
        measured_values = []

        for _ in range(256):
            key, step_key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, midpoint, step_key)
            measured_values.append(info["measured_ph"])

        measured_ph = jnp.asarray(measured_values)[128:]
        assert jnp.all((measured_ph > 0.5) & (measured_ph < 13.5))

    def test_reset_returns_state_and_obs(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, _step_fn = make_ph_neutralization_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array(7.5)
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "true_ph" in info
        assert "measured_ph" in info
        assert "realized_dose" in info

    def test_obs_ph_channel_is_normalized(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(20):
            k_step, key = jax.random.split(key)
            state, obs, _, _, _ = step_fn(state, jnp.array(7.5), k_step)

        # obs[0] = measured_ph / 14 → should be in (0, 1)
        assert float(obs[0]) > 0.0
        assert float(obs[0]) < 1.0

    def test_reward_is_negative_mse(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        state, _ = reset_fn(k1)
        _, _, reward, _, info = step_fn(state, jnp.array(7.5), k2)

        # Reward is computed from sensor reading, not true state
        expected = -((info["measured_ph"] - config.target_ph) ** 2)
        assert jnp.allclose(reward, expected)

    def test_jit_compatible(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        state, obs = jit_reset(k1)
        new_state, obs2, reward, done, info = jit_step(state, jnp.array(5.0), k2)

        assert obs2.shape == (4,)
        assert reward.shape == ()


class TestEqualizationTankBenchmark:
    def test_reset_returns_state_and_obs(self) -> None:
        config = EqualizationTankBenchmarkConfig()
        reset_fn, _step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array(75.0)
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "true_level" in info
        assert "requested_outlet" in info
        assert "realized_outlet" in info
        assert "overflow_flow" in info
        assert "unmet_outlet_flow" in info
        assert "tank_constraint_status" in info
        assert "flow_rate_penalty" in info
        assert "flow_deviation_penalty" in info
        assert "level_band_penalty" in info
        assert "constraint_penalty" in info
        assert "inlet_flow" in info

    def test_level_stays_within_bounds(self) -> None:
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(5)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(50):
            k_step, key = jax.random.split(key)
            # zero outlet: tank fills up but should be clamped
            state, _, _, _, info = step_fn(state, jnp.array(0.0), k_step)
            level = float(info["true_level"])
            assert level <= config.tank_max_level
            assert level >= config.tank_min_level

    def test_reward_sums_named_penalty_components(self) -> None:
        """Reward is the negated sum of the flow, level-band and constraint costs."""
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        state, _ = reset_fn(k1)
        _, _, reward, _, info = step_fn(state, jnp.array(75.0), k2)

        expected = -(
            info["flow_rate_penalty"]
            + info["flow_deviation_penalty"]
            + info["level_band_penalty"]
            + info["overflow_penalty"]
            + info["unmet_outlet_penalty"]
        )
        assert jnp.allclose(reward, expected)

    def test_overflow_worsens_reward_beyond_flow_and_band_cost(self) -> None:
        """Overflow contributes a negative cost when the tank reaches capacity."""
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(13)
        key, reset_key = jax.random.split(key)
        state, _ = reset_fn(reset_key)

        for _ in range(200):
            key, step_key = jax.random.split(key)
            state, _, reward, _, info = step_fn(state, jnp.array(0.0), step_key)
            if float(info["overflow_flow"]) > 0.0:
                assert float(info["constraint_penalty"]) > 0.0
                assert float(reward) < -float(
                    info["flow_rate_penalty"]
                    + info["flow_deviation_penalty"]
                    + info["level_band_penalty"]
                )
                return

        raise AssertionError("zero outlet did not produce an overflow constraint violation")

    def test_unmet_outlet_flow_worsens_reward_beyond_flow_and_band_cost(self) -> None:
        """Unmet requested outlet flow contributes a negative cost."""
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(17)
        key, reset_key = jax.random.split(key)
        state, _ = reset_fn(reset_key)
        state = replace(
            state,
            tank_state=replace(state.tank_state, level=jnp.array(config.tank_min_level)),
        )

        for _ in range(20):
            key, step_key = jax.random.split(key)
            state, _, reward, _, info = step_fn(
                state,
                jnp.array(config.pump_max_flow),
                step_key,
            )
            if float(info["unmet_outlet_flow"]) > 0.0:
                assert float(info["constraint_penalty"]) > 0.0
                assert float(reward) < -float(
                    info["flow_rate_penalty"]
                    + info["flow_deviation_penalty"]
                    + info["level_band_penalty"]
                )
                return

        raise AssertionError("maximum outlet did not produce an unmet-flow violation")

    def test_jit_compatible(self) -> None:
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)
        state, obs = jit_reset(k1)
        new_state, obs2, reward, done, info = jit_step(state, jnp.array(75.0), k2)

        assert obs2.shape == (4,)
        assert reward.shape == ()

    def test_outlet_change_is_penalized_and_steady_outlet_is_not(self) -> None:
        """Only a change in realized outlet flow incurs the rate penalty.

        Holding the previous realized outlet is free; stepping it by 20 is charged.
        """
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(19), 3)
        state, _ = reset_fn(k1)
        state, _, _, _, first = step_fn(state, jnp.array(75.0), k2)
        _, _, _, _, steady = step_fn(state, first["realized_outlet"], k3)
        _, _, _, _, stepped = step_fn(state, first["realized_outlet"] + 20.0, k3)

        assert float(steady["flow_rate_penalty"]) == 0.0
        assert float(stepped["flow_rate_penalty"]) > 0.0

    def test_level_band_penalty_applies_only_outside_the_band(self) -> None:
        """The level-band cost is zero at the centred start and positive near the brim.

        The tank is forced to 97% of capacity to leave the configured band.
        """
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        k1, k2 = jax.random.split(jax.random.PRNGKey(23))
        state, _ = reset_fn(k1)
        _, _, _, _, centred = step_fn(state, jnp.array(75.0), k2)
        crowded_level = jnp.array(0.97 * config.tank_max_level)
        crowded = replace(state, tank_state=replace(state.tank_state, level=crowded_level))
        _, _, _, _, high = step_fn(crowded, jnp.array(75.0), k2)

        assert float(centred["level_band_penalty"]) == 0.0
        assert float(high["level_band_penalty"]) > 0.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"flow_rate_weight": -1.0},
            {"level_band_weight": -0.1},
            {"level_band_low": 0.9, "level_band_high": 0.1},
            {"level_band_high": 1.5},
            {"pump_max_ramp_rate": 0.0},
        ],
    )
    def test_invalid_reward_config_is_rejected(self, overrides: dict[str, Any]) -> None:
        """Negative weights, an inverted or out-of-range band and a zero ramp are rejected.

        Each parametrized override breaks one reward or ramp configuration constraint.
        """
        with pytest.raises(ValueError):
            make_equalization_tank_benchmark(EqualizationTankBenchmarkConfig(**overrides))
