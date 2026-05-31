import jax
import jax.numpy as jnp

from process_control.benchmarks.bsm1_recycle import (
    BSM1RecycleConfig,
    make_bsm1_recycle_benchmark,
)


class TestBSM1RecycleBenchmark:
    def test_reset_returns_state_and_obs(self) -> None:
        config = BSM1RecycleConfig()
        reset_fn, _ = make_bsm1_recycle_benchmark(config)
        state, obs = reset_fn(jax.random.PRNGKey(0))

        assert obs.shape == (6,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        state, _ = reset_fn(k1)

        action = jnp.array([3.0, 1.0])  # Q_a=3×Q_in, Q_rs=1×Q_in (BSM1 defaults)
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (6,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "q_a_ratio" in info
        assert "q_rs_ratio" in info

    def test_high_recycle_reduces_nitrate(self) -> None:
        """Higher internal recycle should bring more nitrate back to anoxic zone,
        increasing denitrification and lowering effluent NO₃."""
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)

        # Run with low recycle
        key = jax.random.PRNGKey(10)
        k1, key = jax.random.split(key)
        state_low, _ = reset_fn(k1)
        for _ in range(500):
            k_step, key = jax.random.split(key)
            state_low, _, _, _, info_low = step_fn(
                state_low, jnp.array([1.0, 1.0]), k_step)

        # Run with high recycle
        key = jax.random.PRNGKey(10)
        k1, key = jax.random.split(key)
        state_high, _ = reset_fn(k1)
        for _ in range(500):
            k_step, key = jax.random.split(key)
            state_high, _, _, _, info_high = step_fn(
                state_high, jnp.array([5.0, 1.0]), k_step)

        # Higher recycle → more denitrification → lower effluent NO₃
        assert float(info_high["s_no_effluent"]) < float(info_low["s_no_effluent"])

    def test_recycle_actuator_is_ramp_limited(self) -> None:
        """Actuator should not jump instantly to requested value."""
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)
        k1, k2 = jax.random.split(jax.random.PRNGKey(20))
        state, _ = reset_fn(k1)

        # Request max recycle from BSM1 default (3.0)
        _, _, _, _, info = step_fn(state, jnp.array([6.0, 3.0]), k2)

        # With ramp_rate=2.0 h⁻¹/h and dt=0.02h, max change = 0.04 per step
        # From initial 3.0, should not reach 6.0 in one step
        assert float(info["q_a_ratio"]) < 6.0
        assert float(info["q_a_ratio"]) > 3.0

    def test_effluent_non_negative(self) -> None:
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)
        key = jax.random.PRNGKey(30)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(50):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([3.0, 1.0]), k_step)
            assert float(info["s_nh_effluent"]) >= 0.0
            assert float(info["s_no_effluent"]) >= 0.0

    def test_reward_includes_energy_penalty(self) -> None:
        """Higher recycle rates should incur greater energy penalty."""
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)

        k1, k2 = jax.random.split(jax.random.PRNGKey(40))
        state, _ = reset_fn(k1)

        # Low recycle
        _, _, reward_low, _, _ = step_fn(state, jnp.array([1.0, 0.5]), k2)
        # High recycle (same initial state, same RNG)
        _, _, reward_high, _, _ = step_fn(state, jnp.array([5.0, 2.5]), k2)

        # Reward is negative; higher recycle → larger energy penalty → more negative
        # (assuming effluent quality doesn't change drastically in one step)
        assert float(reward_high) < float(reward_low)

    def test_jit_compatible(self) -> None:
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        k1, k2 = jax.random.split(jax.random.PRNGKey(99))
        state, obs = jit_reset(k1)
        _, obs2, reward, _, _ = jit_step(state, jnp.array([3.0, 1.0]), k2)

        assert obs.shape == (6,)
        assert obs2.shape == (6,)
        assert reward.shape == ()

    def test_anoxic_zone_no3_responds_to_recycle(self) -> None:
        """R2 nitrate should increase when more nitrate-rich water is recycled."""
        config = BSM1RecycleConfig()
        reset_fn, step_fn = make_bsm1_recycle_benchmark(config)

        key = jax.random.PRNGKey(50)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        # Run with high recycle
        for _ in range(200):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([5.0, 1.0]), k_step)

        # R2 should have measurable NO₃ from the high recycle
        assert float(info["s_no_r2"]) > 0.0
