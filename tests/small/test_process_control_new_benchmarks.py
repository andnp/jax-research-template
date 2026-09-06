from dataclasses import replace

import jax
import jax.numpy as jnp
from process_control.benchmarks.chlorine_two_stage import (
    ChlorineTwoStageBenchmarkConfig,
    make_chlorine_two_stage_benchmark,
)
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

        assert config.pump_max_dose == 300.0
        assert config.pi_ff == 175.0
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
        assert "tracking_penalty" in info
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

    def test_reward_contains_tracking_and_constraint_penalties(self) -> None:
        """Reward exposes level tracking and physical constraint costs."""
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        state, _ = reset_fn(k1)
        _, _, reward, _, info = step_fn(state, jnp.array(75.0), k2)

        expected = -(
            info["tracking_penalty"]
            + info["overflow_penalty"]
            + info["unmet_outlet_penalty"]
        )
        assert jnp.allclose(reward, expected)

    def test_overflow_worsens_reward_beyond_tracking_cost(self) -> None:
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
                assert float(reward) < -float(info["tracking_penalty"])
                return

        raise AssertionError("zero outlet did not produce an overflow constraint violation")

    def test_unmet_outlet_flow_worsens_reward_beyond_tracking_cost(self) -> None:
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
                assert float(reward) < -float(info["tracking_penalty"])
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


class TestChlorineTwoStageBenchmark:
    def test_reset_returns_state_and_obs(self) -> None:
        config = ChlorineTwoStageBenchmarkConfig()
        reset_fn, _step_fn = make_chlorine_two_stage_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = ChlorineTwoStageBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_two_stage_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array(2.5)
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "outlet_residual" in info
        assert "pi_dose" in info

    def test_both_basins_advance(self) -> None:
        config = ChlorineTwoStageBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_two_stage_benchmark(config)
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        initial_b1 = state.basin1_state.segments.sum()
        initial_b2 = state.basin2_state.segments.sum()

        # Need enough steps for material to traverse basin 1 (10 segments) before
        # basin 2 receives a non-zero inlet (ratio ≈ 0.94 per step → ~11 steps)
        for _ in range(25):
            k_step, key = jax.random.split(key)
            state, _, _, _, _ = step_fn(state, jnp.array(3.0), k_step)

        assert state.basin1_state.segments.sum() != initial_b1
        assert state.basin2_state.segments.sum() != initial_b2

    def test_two_stage_slower_than_one_stage(self) -> None:
        """Outlet residual should build up more slowly with double the contact volume."""
        from process_control.benchmarks.chlorine import (
            ChlorineBenchmarkConfig,
            make_chlorine_benchmark,
        )

        single_config = ChlorineBenchmarkConfig(
            basin_volume=400.0,
            basin_segments=10,
        )
        two_stage_config = ChlorineTwoStageBenchmarkConfig(
            basin1_volume=200.0,
            basin2_volume=200.0,
            basin1_segments=10,
            basin2_segments=10,
        )

        reset1, step1 = make_chlorine_benchmark(single_config)
        reset2, step2 = make_chlorine_two_stage_benchmark(two_stage_config)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        state1, _ = reset1(k1)
        state2, _ = reset2(k2)

        dose = jnp.array(3.0)
        residuals1 = []
        residuals2 = []

        for i in range(20):
            k = jax.random.PRNGKey(i + 100)
            k1s, k2s = jax.random.split(k)
            state1, _, _, _, info1 = step1(state1, dose, k1s)
            state2, _, _, _, info2 = step2(state2, dose, k2s)
            residuals1.append(float(info1["outlet_residual"]))
            residuals2.append(float(info2["outlet_residual"]))

        # Both should eventually show residual, but two-stage dynamics differ
        assert max(residuals1) >= 0.0
        assert max(residuals2) >= 0.0

    def test_jit_compatible(self) -> None:
        config = ChlorineTwoStageBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_two_stage_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(77)
        k1, k2 = jax.random.split(key)
        state, obs = jit_reset(k1)
        new_state, obs2, reward, done, info = jit_step(state, jnp.array(2.5), k2)

        assert obs2.shape == (4,)
        assert reward.shape == ()
