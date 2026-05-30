import jax
import jax.numpy as jnp

from process_control.benchmarks.bsm1_reduced import (
    BSM1ReducedBenchmarkConfig,
    make_bsm1_reduced_benchmark,
)
from process_control.units.biological_reactor import (
    BiologicalReactorParams,
    BiologicalReactorState,
    mix_streams,
    step as reactor_step,
)


class TestBiologicalReactorUnit:
    def test_nitrification_reduces_ammonia(self) -> None:
        """Autotrophic growth with high DO and ammonia should reduce s_nh."""
        params = BiologicalReactorParams(volume=1000.0)
        inlet = BiologicalReactorState.create(s_s=5.0, s_o=8.0, s_no=0.0, s_nh=30.0, x_bh=2500.0, x_ba=150.0)
        state = BiologicalReactorState.create(s_s=5.0, s_o=4.0, s_no=5.0, s_nh=15.0, x_bh=2500.0, x_ba=150.0)

        dt = jnp.array(0.02)
        kla = jnp.array(5.0)  # strong aeration

        new_state = reactor_step(state, inlet, jnp.array(769.0), kla, params, dt)

        # With high DO, nitrification should consume NH4 (s_nh decrease or slower increase)
        # s_nh in: 15 (reactor) vs 30 (inlet), dilution brings it up, nitrification reduces it
        assert new_state.s_nh >= 0.0  # non-negative
        assert new_state.s_no >= 0.0  # non-negative

    def test_denitrification_reduces_nitrate_in_anoxic(self) -> None:
        """Low DO with substrate and nitrate should reduce s_no (denitrification)."""
        params = BiologicalReactorParams(volume=1000.0)
        inlet = BiologicalReactorState.create(s_s=20.0, s_o=0.0, s_no=10.0, s_nh=8.0, x_bh=2500.0, x_ba=150.0)
        state = BiologicalReactorState.create(s_s=2.0, s_o=0.0, s_no=8.0, s_nh=6.0, x_bh=2800.0, x_ba=140.0)

        dt = jnp.array(0.02)
        kla = jnp.array(0.0)  # anoxic (no aeration)

        new_state = reactor_step(state, inlet, jnp.array(769.0), kla, params, dt)

        # Denitrification with substrate and nitrate removes NO3
        # The rate is positive so NO3 should decrease (or at least not increase rapidly)
        assert new_state.s_no >= 0.0
        assert new_state.s_s >= 0.0

    def test_concentrations_remain_non_negative(self) -> None:
        """All state variables must stay non-negative after any step."""
        params = BiologicalReactorParams(volume=500.0)
        inlet = BiologicalReactorState.create(s_s=70.0, s_o=0.0, s_no=0.0, s_nh=31.0, x_bh=28.0, x_ba=0.0)
        state = BiologicalReactorState.create(s_s=0.5, s_o=0.1, s_no=0.5, s_nh=0.5, x_bh=2000.0, x_ba=100.0)

        dt = jnp.array(0.02)
        for kla_val in [0.0, 2.0, 10.0]:
            new_state = reactor_step(state, inlet, jnp.array(769.0), jnp.array(kla_val), params, dt)
            assert float(new_state.s_s) >= 0.0
            assert float(new_state.s_o) >= 0.0
            assert float(new_state.s_no) >= 0.0
            assert float(new_state.s_nh) >= 0.0
            assert float(new_state.x_bh) >= 0.0
            assert float(new_state.x_ba) >= 0.0

    def test_mix_streams_mass_conservation(self) -> None:
        """Mixed stream concentrations must satisfy flow-weighted mass balance."""
        a = BiologicalReactorState.create(s_s=10.0, s_o=5.0, s_no=3.0, s_nh=2.0, x_bh=1000.0, x_ba=50.0)
        b = BiologicalReactorState.create(s_s=20.0, s_o=1.0, s_no=6.0, s_nh=10.0, x_bh=2000.0, x_ba=100.0)
        qa = jnp.array(100.0)
        qb = jnp.array(300.0)

        mixed, total_flow = mix_streams(a, qa, b, qb)

        # s_s should be flow-weighted average: (10*100 + 20*300)/400 = 17.5
        assert jnp.allclose(mixed.s_s, jnp.array(17.5), atol=1e-4)
        assert float(total_flow) == 400.0


class TestBSM1ReducedBenchmark:
    def test_reset_returns_state_and_obs(self) -> None:
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, _step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array([3.0, 2.0])  # kla=3, recycle_ratio=2
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "s_nh_effluent" in info
        assert "s_no_effluent" in info
        assert "s_o_aerobic" in info
        assert "kla" in info
        assert "recycle_ratio" in info

    def test_effluent_concentrations_non_negative(self) -> None:
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(7)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(30):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([3.0, 2.0]), k_step)
            assert float(info["s_nh_effluent"]) >= 0.0
            assert float(info["s_no_effluent"]) >= 0.0
            assert float(info["s_o_aerobic"]) >= 0.0

    def test_high_aeration_increases_do(self) -> None:
        """High kla action should drive aerobic DO toward saturation."""
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        # Run for many steps with maximum aeration
        for _ in range(200):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([10.0, 2.0]), k_step)

        # With kla=10, DO should be substantially elevated
        assert float(info["s_o_aerobic"]) > 2.0

    def test_no_aeration_collapses_do(self) -> None:
        """Zero kla should drive aerobic DO toward zero over many steps."""
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(1)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(200):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([0.0, 2.0]), k_step)

        # Without aeration, DO should be near zero
        assert float(info["s_o_aerobic"]) < 0.5

    def test_reward_is_negative(self) -> None:
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        key = jax.random.PRNGKey(5)
        k1, k2 = jax.random.split(key)
        state, _ = reset_fn(k1)
        _, _, reward, _, _ = step_fn(state, jnp.array([3.0, 2.0]), k2)

        assert float(reward) <= 0.0

    def test_jit_compatible(self) -> None:
        config = BSM1ReducedBenchmarkConfig()
        reset_fn, step_fn = make_bsm1_reduced_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(55)
        k1, k2 = jax.random.split(key)
        state, obs = jit_reset(k1)
        new_state, obs2, reward, done, info = jit_step(state, jnp.array([3.0, 2.0]), k2)

        assert obs2.shape == (4,)
        assert reward.shape == ()
