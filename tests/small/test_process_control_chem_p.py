import jax
import jax.numpy as jnp
from process_control.benchmarks.chem_p_dosing import (
    ChemPDosingConfig,
    make_chem_p_dosing_benchmark,
)


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
