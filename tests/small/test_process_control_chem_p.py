import jax
import jax.numpy as jnp
from process_control.benchmarks.chem_p_dosing import (
    ChemPDosingConfig,
    make_chem_p_dosing_benchmark,
)
from process_control.chemistry.precipitation import PrecipitationParams, precipitate


class TestPrecipitation:
    def test_no_dose_no_removal(self) -> None:
        params = PrecipitationParams()
        eff, consumed = precipitate(jnp.array(5.0), jnp.array(0.0), params)
        assert float(eff) == 5.0
        assert float(consumed) == 0.0

    def test_high_dose_removes_most(self) -> None:
        params = PrecipitationParams()
        eff, _ = precipitate(jnp.array(5.0), jnp.array(50.0), params)
        assert float(eff) < 0.5

    def test_floor_respected(self) -> None:
        params = PrecipitationParams(p_min=0.05)
        eff, _ = precipitate(jnp.array(5.0), jnp.array(1000.0), params)
        assert float(eff) >= 0.05

    def test_monod_saturation(self) -> None:
        params = PrecipitationParams()
        # With a moderate dose, absolute removal is limited when P is low
        # due to Monod half-saturation. At higher P, absolute removal increases.
        eff_low, _ = precipitate(jnp.array(0.2), jnp.array(5.0), params)
        eff_high, _ = precipitate(jnp.array(5.0), jnp.array(5.0), params)
        removed_low = 0.2 - float(eff_low)
        removed_high = 5.0 - float(eff_high)
        assert removed_low < removed_high


class TestChemPDosingBenchmark:
    def test_reset_shapes(self) -> None:
        config = ChemPDosingConfig()
        reset, _ = make_chem_p_dosing_benchmark(config)
        state, obs = reset(jax.random.PRNGKey(0))
        assert obs.shape == (5,)
        assert jnp.all(jnp.isfinite(obs))

    def test_step_shapes(self) -> None:
        config = ChemPDosingConfig()
        reset, step = make_chem_p_dosing_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([10.0])
        new_state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(1))
        assert obs.shape == (5,)
        assert reward.shape == ()
        assert done.shape == ()
        assert jnp.all(jnp.isfinite(obs))

    def test_info_keys(self) -> None:
        config = ChemPDosingConfig()
        reset, step = make_chem_p_dosing_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([10.0])
        _, _, _, _, info = step(state, action, jax.random.PRNGKey(1))
        assert "po4_eff" in info
        assert "po4_inf" in info
        assert "fe_dose" in info
        assert "fe_consumed" in info
        assert "flow" in info

    def test_higher_dose_lower_effluent(self) -> None:
        config = ChemPDosingConfig()
        reset, step = make_chem_p_dosing_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(42))

        action_low = jnp.array([5.0])
        action_high = jnp.array([25.0])

        # Run 50 steps with each dose from same initial state
        state_lo = state
        state_hi = state
        for i in range(50):
            state_lo, _, _, _, info_lo = step(state_lo, action_low, jax.random.PRNGKey(i))
            state_hi, _, _, _, info_hi = step(state_hi, action_high, jax.random.PRNGKey(i))

        assert float(info_hi["po4_eff"]) < float(info_lo["po4_eff"])

    def test_stability_500_steps(self) -> None:
        config = ChemPDosingConfig()
        reset, step = make_chem_p_dosing_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(42))
        action = jnp.array([10.0])
        for i in range(500):
            state, obs, reward, done, info = step(state, action, jax.random.PRNGKey(i))
        assert jnp.all(jnp.isfinite(obs))
        assert jnp.isfinite(reward)

    def test_jit_compatible(self) -> None:
        config = ChemPDosingConfig()
        reset, step = make_chem_p_dosing_benchmark(config)
        jit_reset = jax.jit(reset)
        jit_step = jax.jit(step)
        state, obs = jit_reset(jax.random.PRNGKey(0))
        assert obs.shape == (5,)
        action = jnp.array([10.0])
        state2, obs2, reward, done, info = jit_step(state, action, jax.random.PRNGKey(1))
        assert obs2.shape == (5,)
