from dataclasses import fields

import jax
import jax.numpy as jnp
from process_control.benchmarks.sludge_blanket import (
    SludgeBlanketConfig,
    make_sludge_blanket_benchmark,
)
from process_control.units.takacs_settler import (
    TakacsSettlerParams,
    TakacsSettlerState,
    compute_blanket_height,
    get_effluent_tss,
    get_underflow_tss,
)
from process_control.units.takacs_settler import (
    reset as settler_reset,
)
from process_control.units.takacs_settler import (
    step as settler_step,
)


class TestTakacsSettler:
    def test_reset_creates_profile(self) -> None:
        params = TakacsSettlerParams()
        state = settler_reset(3500.0, params, jax.random.PRNGKey(0))
        assert state.layer_tss.shape == (10,)
        # Bottom layer should have higher TSS than top
        assert float(state.layer_tss[0]) > float(state.layer_tss[-1])

    def test_settling_reduces_effluent_tss(self) -> None:
        """With gravity settling and no feed, effluent TSS should decrease."""
        params = TakacsSettlerParams()
        # Start with uniform high TSS
        state = TakacsSettlerState(layer_tss=jnp.full(10, 3000.0))
        dt = jnp.array(0.02)

        initial_eff = float(get_effluent_tss(state))

        # Run with moderate underflow (no feed into settler for simplicity,
        # use very low feed flow to approximate)
        for _ in range(100):
            state = settler_step(
                state, jnp.array(3000.0), jnp.array(769.0),
                jnp.array(400.0), params, dt,
            )

        final_eff = float(get_effluent_tss(state))
        # With settling, top layer should have lower TSS than initial uniform
        assert final_eff < initial_eff

    def test_underflow_concentrates_solids(self) -> None:
        """Bottom layer should have higher TSS than feed after settling."""
        params = TakacsSettlerParams()
        state = settler_reset(3500.0, params, jax.random.PRNGKey(1))
        dt = jnp.array(0.02)

        for _ in range(500):
            state = settler_step(
                state, jnp.array(3500.0), jnp.array(769.0),
                jnp.array(400.0), params, dt,
            )

        underflow_tss = float(get_underflow_tss(state))
        # Underflow should be more concentrated than feed
        assert underflow_tss > 3500.0

    def test_all_layers_non_negative(self) -> None:
        params = TakacsSettlerParams()
        state = settler_reset(3500.0, params, jax.random.PRNGKey(2))
        dt = jnp.array(0.02)

        for _ in range(200):
            state = settler_step(
                state, jnp.array(3500.0), jnp.array(769.0),
                jnp.array(400.0), params, dt,
            )

        assert jnp.all(state.layer_tss >= 0.0)

    def test_blanket_height_responds_to_underflow(self) -> None:
        """Low underflow → blanket rises. High underflow → blanket drops."""
        params = TakacsSettlerParams()
        dt = jnp.array(0.02)

        # Run with low underflow
        state_low = settler_reset(3500.0, params, jax.random.PRNGKey(3))
        for _ in range(500):
            state_low = settler_step(
                state_low, jnp.array(3500.0), jnp.array(769.0),
                jnp.array(100.0), params, dt,
            )
        blanket_low_u = float(compute_blanket_height(state_low, params))

        # Run with high underflow
        state_high = settler_reset(3500.0, params, jax.random.PRNGKey(3))
        for _ in range(500):
            state_high = settler_step(
                state_high, jnp.array(3500.0), jnp.array(769.0),
                jnp.array(600.0), params, dt,
            )
        blanket_high_u = float(compute_blanket_height(state_high, params))

        # Low underflow should give higher blanket
        assert blanket_low_u > blanket_high_u

    def test_high_feed_flow_raises_blanket(self) -> None:
        """Higher feed flow → more upward velocity → blanket rises."""
        params = TakacsSettlerParams()
        dt = jnp.array(0.02)
        q_u = jnp.array(400.0)

        # Normal flow
        state_normal = settler_reset(3500.0, params, jax.random.PRNGKey(4))
        for _ in range(500):
            state_normal = settler_step(
                state_normal, jnp.array(3500.0), jnp.array(769.0),
                q_u, params, dt,
            )
        blanket_normal = float(compute_blanket_height(state_normal, params))

        # Storm flow (2x normal)
        state_storm = settler_reset(3500.0, params, jax.random.PRNGKey(4))
        for _ in range(500):
            state_storm = settler_step(
                state_storm, jnp.array(3500.0), jnp.array(1500.0),
                q_u, params, dt,
            )
        blanket_storm = float(compute_blanket_height(state_storm, params))

        # Storm should raise blanket
        assert blanket_storm > blanket_normal

    def test_jit_compatible(self) -> None:
        params = TakacsSettlerParams()
        state = settler_reset(3500.0, params, jax.random.PRNGKey(5))

        @jax.jit
        def do_step(s: TakacsSettlerState) -> TakacsSettlerState:
            return settler_step(
                s, jnp.array(3500.0), jnp.array(769.0),
                jnp.array(400.0), params, jnp.array(0.02),
            )

        new_state = do_step(state)
        assert new_state.layer_tss.shape == (10,)


class TestSludgeBlanketBenchmark:
    def test_does_not_expose_unsupported_transport_disturbances(self) -> None:
        """Solids scenarios must not advertise water-transport disturbances."""
        config_fields = {field.name for field in fields(SludgeBlanketConfig)}
        assert "max_disturbance_events" not in config_fields

        reset_fn, _ = make_sludge_blanket_benchmark(SludgeBlanketConfig())
        state, _ = reset_fn(jax.random.PRNGKey(0))

        state_fields = {field.name for field in fields(state)}
        assert "disturbance_schedule" not in state_fields

    def test_reset_returns_state_and_obs(self) -> None:
        config = SludgeBlanketConfig()
        reset_fn, _ = make_sludge_blanket_benchmark(config)
        state, obs = reset_fn(jax.random.PRNGKey(0))

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = SludgeBlanketConfig()
        reset_fn, step_fn = make_sludge_blanket_benchmark(config)
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        state, _ = reset_fn(k1)

        action = jnp.array([400.0])
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "effluent_tss" in info
        assert "underflow_tss" in info
        assert "blanket_height" in info
        assert "q_underflow" in info

    def test_effluent_tss_non_negative(self) -> None:
        config = SludgeBlanketConfig()
        reset_fn, step_fn = make_sludge_blanket_benchmark(config)
        key = jax.random.PRNGKey(10)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(50):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([400.0]), k_step)
            assert float(info["effluent_tss"]) >= 0.0
            assert float(info["underflow_tss"]) >= 0.0

    def test_low_underflow_raises_blanket_vs_high(self) -> None:
        """Low underflow should give a higher blanket than high underflow."""
        config = SludgeBlanketConfig()
        reset_fn, step_fn = make_sludge_blanket_benchmark(config)

        # Run with low underflow
        key = jax.random.PRNGKey(20)
        k1, key = jax.random.split(key)
        state_low, _ = reset_fn(k1)
        for _ in range(500):
            k_step, key = jax.random.split(key)
            state_low, _, _, _, info_low = step_fn(state_low, jnp.array([100.0]), k_step)

        # Run with high underflow (same seed)
        key = jax.random.PRNGKey(20)
        k1, key = jax.random.split(key)
        state_high, _ = reset_fn(k1)
        for _ in range(500):
            k_step, key = jax.random.split(key)
            state_high, _, _, _, info_high = step_fn(state_high, jnp.array([800.0]), k_step)

        assert float(info_low["blanket_height"]) > float(info_high["blanket_height"])

    def test_reward_is_negative(self) -> None:
        config = SludgeBlanketConfig()
        reset_fn, step_fn = make_sludge_blanket_benchmark(config)
        k1, k2 = jax.random.split(jax.random.PRNGKey(30))
        state, _ = reset_fn(k1)

        _, _, reward, _, _ = step_fn(state, jnp.array([400.0]), k2)
        assert float(reward) < 0.0

    def test_jit_compatible(self) -> None:
        config = SludgeBlanketConfig()
        reset_fn, step_fn = make_sludge_blanket_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        k1, k2 = jax.random.split(jax.random.PRNGKey(99))
        state, obs = jit_reset(k1)
        _, obs2, reward, _, _ = jit_step(state, jnp.array([400.0]), k2)

        assert obs.shape == (4,)
        assert obs2.shape == (4,)
        assert reward.shape == ()
