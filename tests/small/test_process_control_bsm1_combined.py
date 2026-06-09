import jax
import jax.numpy as jnp
from process_control.benchmarks.bsm1_combined import (
    BSM1CombinedConfig,
    make_bsm1_combined_benchmark,
)


class TestBSM1CombinedBenchmark:
    def test_reset_shapes(self) -> None:
        config = BSM1CombinedConfig()
        reset, _ = make_bsm1_combined_benchmark(config)
        state, obs = reset(jax.random.PRNGKey(0))
        assert obs.shape == (11,)
        assert jnp.all(jnp.isfinite(obs))

    def test_step_shapes(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([5.0, 3.5, 3.0, 1.0])
        new_state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(1))
        assert obs.shape == (11,)
        assert reward.shape == ()
        assert done.shape == ()
        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)

    def test_four_actuators_tracked(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([5.0, 3.5, 3.0, 1.0])
        _, _, _, _, info = step(state, action, jax.random.PRNGKey(1))
        assert "kla_34" in info
        assert "kla_5" in info
        assert "q_a_ratio" in info
        assert "q_rs_ratio" in info

    def test_kla_ramps_after_startup_delay(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([5.0, 3.5, 3.0, 1.0])

        # First few steps: blower in startup delay (0.05h / 0.02h ≈ 3 steps)
        _, _, _, _, info0 = step(state, action, jax.random.PRNGKey(0))
        assert float(info0["kla_34"]) == 0.0

        # After startup + ramp, kla should increase
        for i in range(15):
            state, _, _, _, info = step(state, action, jax.random.PRNGKey(i))
        assert float(info["kla_34"]) > 1.0

    def test_asymmetric_coast_down(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))

        # Ramp up for 50 steps
        high_action = jnp.array([8.0, 5.0, 3.0, 1.0])
        for i in range(50):
            state, _, _, _, _ = step(state, high_action, jax.random.PRNGKey(i))

        kla_before = float(state.loops.kla_34.pump_output)
        assert kla_before > 1.0

        # Drop to zero and check coast-down rate
        zero_action = jnp.array([0.0, 0.0, 3.0, 1.0])
        state, _, _, _, info = step(state, zero_action, jax.random.PRNGKey(99))
        kla_after = float(info["kla_34"])

        # Coast-down at 8.0 h⁻¹/h × 0.02h = 0.16 per step
        # vs ramp-up at 5.0 h⁻¹/h × 0.02h = 0.10 per step
        expected_drop = 8.0 * 0.02
        actual_drop = kla_before - kla_after
        assert abs(actual_drop - expected_drop) < 0.01

    def test_stability_500_steps(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(42))
        action = jnp.array([5.0, 3.5, 3.0, 1.0])
        for i in range(500):
            state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(i))
        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)

    def test_jit_compatible(self) -> None:
        config = BSM1CombinedConfig()
        reset, step = make_bsm1_combined_benchmark(config)
        jit_reset = jax.jit(reset)
        jit_step = jax.jit(step)
        state, obs = jit_reset(jax.random.PRNGKey(0))
        assert obs.shape == (11,)
        action = jnp.array([5.0, 3.5, 3.0, 1.0])
        state2, obs2, reward, done, info = jit_step(state, action, jax.random.PRNGKey(1))
        assert obs2.shape == (11,)
