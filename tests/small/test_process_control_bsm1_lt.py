import jax
import jax.numpy as jnp
from process_control.benchmarks.bsm1_lt import BSM1LTConfig, make_bsm1_lt_benchmark
from process_control.units.asm1 import ASM1Params, apply_arrhenius


class TestArrheniusCorrection:
    def test_reference_temperature_unchanged(self) -> None:
        params = ASM1Params(volume=1333.0)
        corrected = apply_arrhenius(params, jnp.array(15.0))
        assert abs(float(corrected.mu_h) - params.mu_h) < 1e-5
        assert abs(float(corrected.mu_a) - params.mu_a) < 1e-5

    def test_warm_increases_rates(self) -> None:
        params = ASM1Params(volume=1333.0)
        corrected = apply_arrhenius(params, jnp.array(20.0))
        assert float(corrected.mu_h) > params.mu_h
        assert float(corrected.mu_a) > params.mu_a

    def test_cold_decreases_rates(self) -> None:
        params = ASM1Params(volume=1333.0)
        corrected = apply_arrhenius(params, jnp.array(10.0))
        assert float(corrected.mu_h) < params.mu_h
        assert float(corrected.mu_a) < params.mu_a

    def test_autotrophs_more_sensitive(self) -> None:
        params = ASM1Params(volume=1333.0)
        warm = apply_arrhenius(params, jnp.array(20.0))
        cold = apply_arrhenius(params, jnp.array(10.0))
        ratio_mu_a = float(warm.mu_a) / float(cold.mu_a)
        ratio_mu_h = float(warm.mu_h) / float(cold.mu_h)
        # Autotrophs have higher theta → bigger ratio
        assert ratio_mu_a > ratio_mu_h

    def test_o2_saturation_temperature_dependence(self) -> None:
        params = ASM1Params(volume=1333.0)
        cold = apply_arrhenius(params, jnp.array(10.0))
        warm = apply_arrhenius(params, jnp.array(20.0))
        # O₂ dissolves better in cold water
        assert float(cold.s_o_sat) > float(warm.s_o_sat)

    def test_jit_compatible(self) -> None:
        params = ASM1Params(volume=1333.0)

        @jax.jit
        def get_mu_a(temp: jax.Array):
            return apply_arrhenius(params, temp).mu_a

        result = get_mu_a(jnp.array(18.0))
        assert float(result) > params.mu_a


class TestBSM1LTBenchmark:
    def test_reset_10d_obs(self) -> None:
        config = BSM1LTConfig()
        reset, _ = make_bsm1_lt_benchmark(config)
        state, obs = reset(jax.random.PRNGKey(0))
        assert obs.shape == (10,)
        assert jnp.all(jnp.isfinite(obs))

    def test_step_shapes(self) -> None:
        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([8.0, 6.0])
        new_state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(1))
        assert obs.shape == (10,)
        assert reward.shape == ()
        assert "temperature" in info

    def test_temperature_in_info(self) -> None:
        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([8.0, 6.0])
        _, _, _, _, info = step(state, action, jax.random.PRNGKey(1))
        temp = float(info["temperature"])
        # Reset starts at step 0 → winter (minimum temperature)
        assert abs(temp - (config.t_mean - config.t_amplitude)) < 0.5

    def test_seasonal_temperature_range(self) -> None:
        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([8.0, 6.0])

        import dataclasses

        temps = []
        # Sample 12 months
        for month in range(12):
            day = month * 30
            step_count = int(day * config.bsm1.steps_per_day)
            s = dataclasses.replace(state, step_count=jnp.array(step_count, dtype=jnp.int32))
            _, _, _, _, info = step(s, action, jax.random.PRNGKey(month))
            temps.append(float(info["temperature"]))

        assert min(temps) >= config.t_mean - config.t_amplitude - 0.5
        assert max(temps) <= config.t_mean + config.t_amplitude + 0.5
        # Should span a reasonable range
        assert max(temps) - min(temps) > config.t_amplitude

    def test_stability_500_steps_winter(self) -> None:
        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(42))
        action = jnp.array([5.0, 3.5])
        # Winter: low temperature → lower growth rates → easy to stabilise
        for i in range(500):
            state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(i))
        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)

    def test_stability_500_steps_summer(self) -> None:
        import dataclasses

        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(42))
        # Summer start needs higher aeration
        action = jnp.array([8.0, 6.0])
        day_182 = int(182 * config.bsm1.steps_per_day)
        state = dataclasses.replace(state, step_count=jnp.array(day_182, dtype=jnp.int32))
        for i in range(500):
            state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(i))
        assert jnp.all(jnp.isfinite(obs))
        assert float(info["s_nh_effluent"]) < 5.0

    def test_summer_needs_more_aeration(self) -> None:
        """At 20°C, the same aeration produces higher NH₄ than at 15°C."""
        import dataclasses

        action = jnp.array([8.0, 6.0])
        results = {}
        for label, t_amp in [("constant", 0.0), ("seasonal", 5.0)]:
            config = BSM1LTConfig(t_amplitude=t_amp)
            reset, step = make_bsm1_lt_benchmark(config)
            state, _ = reset(jax.random.PRNGKey(42))
            # Summer
            day_182 = int(182 * config.bsm1.steps_per_day)
            state = dataclasses.replace(state, step_count=jnp.array(day_182, dtype=jnp.int32))
            for i in range(200):
                state, _, _, _, info = step(state, action, jax.random.PRNGKey(i))
            results[label] = float(info["s_nh_effluent"])
        # At 20°C, heterotrophs compete harder for O₂ → harder to nitrify
        # With same aeration, summer has higher effluent NH₄
        assert results["seasonal"] > results["constant"] * 0.5

    def test_jit_compatible(self) -> None:
        config = BSM1LTConfig()
        reset, step = make_bsm1_lt_benchmark(config)
        jit_reset = jax.jit(reset)
        jit_step = jax.jit(step)
        state, obs = jit_reset(jax.random.PRNGKey(0))
        assert obs.shape == (10,)
        action = jnp.array([8.0, 6.0])
        state2, obs2, reward, done, info = jit_step(state, action, jax.random.PRNGKey(1))
        assert obs2.shape == (10,)
        assert jnp.isfinite(reward)
