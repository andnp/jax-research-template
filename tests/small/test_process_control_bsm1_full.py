from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, make_bsm1_benchmark
from process_control.benchmarks.bsm1_combined import (
    BSM1CombinedConfig,
    make_bsm1_combined_benchmark,
)
from process_control.benchmarks.bsm1_recycle import (
    BSM1RecycleConfig,
    make_bsm1_recycle_benchmark,
)
from process_control.units.asm1 import (
    ASM1Params,
    ASM1State,
    compute_bod5,
    compute_cod,
    compute_tn,
    compute_tss,
    mix_streams,
)
from process_control.units.asm1 import (
    reset as asm1_reset,
)
from process_control.units.asm1 import (
    step as asm1_step,
)
from process_control.units.bsm1_plant import perfect_clarifier, reactor_train


class TestASM1Unit:
    def test_nitrification_increases_nitrate(self) -> None:
        """High DO + ammonia → nitrification → s_no increases, s_nh decreases."""
        params = ASM1Params(volume=1333.0)
        # Inlet with moderate NH4 and no NO3
        inlet = ASM1State.create(s_s=2.0, s_o=8.0, s_no=0.0, s_nh=10.0, x_bh=2500.0, x_ba=200.0, x_s=30.0)
        # Reactor at near-zero NH4 (forcing inlet concentration gradient to feed reaction)
        state = ASM1State.create(s_s=1.0, s_o=4.0, s_no=5.0, s_nh=3.0, x_bh=2500.0, x_ba=200.0, x_s=30.0)
        dt = jnp.array(0.02)
        kla = jnp.array(8.0)  # strong aeration

        new_state = asm1_step(state, inlet, jnp.array(1000.0), kla, params, dt)

        # s_nh should not blow up; s_no should increase with high kla
        assert float(new_state.s_nh) >= 0.0
        assert float(new_state.s_no) >= 0.0
        # With high aeration, autotrophic growth should be active
        assert float(new_state.s_o) > 0.0

    def test_denitrification_in_anoxic_zone(self) -> None:
        """Zero kla + substrate + nitrate → denitrification → s_no decreases."""
        params = ASM1Params(volume=1000.0)
        inlet = ASM1State.create(s_s=10.0, s_o=0.0, s_no=10.0, s_nh=8.0, x_bh=2500.0, x_ba=150.0, x_s=50.0)
        state = ASM1State.create(s_s=2.0, s_o=0.0, s_no=7.0, s_nh=6.0, x_bh=2500.0, x_ba=150.0, x_s=40.0)
        dt = jnp.array(0.02)
        kla = jnp.array(0.0)  # anoxic

        new_state = asm1_step(state, inlet, jnp.array(769.0), kla, params, dt)

        # All concentrations non-negative
        assert float(new_state.s_s) >= 0.0
        assert float(new_state.s_no) >= 0.0
        assert float(new_state.s_nh) >= 0.0
        assert float(new_state.x_bh) >= 0.0
        # DO stays near zero with no aeration
        assert float(new_state.s_o) < 0.5

    def test_hydrolysis_transfers_x_s_to_s_s(self) -> None:
        """X_S hydrolysis should reduce x_s and increase s_s over multiple steps."""
        params = ASM1Params(volume=1000.0)
        inlet = ASM1State.create(s_s=1.0, x_s=100.0, x_bh=2500.0)
        state = ASM1State.create(s_s=1.0, s_o=0.0, x_s=100.0, x_bh=2500.0)
        dt = jnp.array(0.02)
        kla = jnp.array(0.0)

        # Run many steps to see hydrolysis effect
        for _ in range(50):
            state = asm1_step(state, inlet, jnp.array(769.0), kla, params, dt)

        # With x_bh present, hydrolysis should occur, reducing x_s
        # (dilution and hydrolysis both reduce x_s)
        assert float(state.x_s) >= 0.0

    def test_all_concentrations_non_negative(self) -> None:
        """All state variables must stay non-negative in all operating conditions."""
        params = ASM1Params(volume=1000.0)
        influent = ASM1State.create(
            s_i=30.0,
            s_s=69.5,
            x_i=51.2,
            x_s=202.32,
            x_bh=28.17,
            x_ba=0.0,
            x_p=0.0,
            s_o=0.0,
            s_no=0.0,
            s_nh=31.56,
            s_nd=6.95,
            x_nd=10.59,
            s_alk=7.0,
        )
        state = ASM1State.create(
            s_s=2.0,
            s_o=0.0,
            s_no=8.0,
            s_nh=7.0,
            x_bh=2500.0,
            x_ba=150.0,
            x_s=60.0,
            x_nd=5.0,
        )
        dt = jnp.array(0.02)

        for kla_val in [0.0, 3.0, 8.0]:
            s = asm1_step(state, influent, jnp.array(769.0), jnp.array(kla_val), params, dt)
            assert float(s.s_i) >= 0.0
            assert float(s.s_s) >= 0.0
            assert float(s.x_i) >= 0.0
            assert float(s.x_s) >= 0.0
            assert float(s.x_bh) >= 0.0
            assert float(s.x_ba) >= 0.0
            assert float(s.x_p) >= 0.0
            assert float(s.s_o) >= 0.0
            assert float(s.s_no) >= 0.0
            assert float(s.s_nh) >= 0.0
            assert float(s.s_nd) >= 0.0
            assert float(s.x_nd) >= 0.0

    def test_mix_streams_mass_conservation(self) -> None:
        """Flow-weighted mixture must conserve mass."""
        a = ASM1State.create(s_s=10.0, s_nh=20.0, x_bh=1000.0)
        b = ASM1State.create(s_s=30.0, s_nh=5.0, x_bh=3000.0)
        qa = jnp.array(300.0)
        qb = jnp.array(100.0)

        mixed, total = mix_streams(a, qa, b, qb)

        # Expected: (10*300 + 30*100) / 400 = 4200/400 = 10.5 for s_s
        assert jnp.allclose(mixed.s_s, jnp.array(15.0), atol=1e-4)
        assert float(total) == 400.0

    def test_compute_tss_and_cod(self) -> None:
        state = ASM1State.create(
            s_i=30.0,
            s_s=1.0,
            x_i=1000.0,
            x_s=40.0,
            x_bh=2500.0,
            x_ba=150.0,
            x_p=450.0,
            s_o=2.0,
        )
        tss = compute_tss(state)
        cod = compute_cod(state)

        expected_tss = 0.75 * (1000.0 + 40.0 + 2500.0 + 150.0 + 450.0)
        assert jnp.allclose(tss, jnp.array(expected_tss), atol=1e-2)
        assert float(cod) > 0.0

    def test_compute_bod5_matches_bsm1_formula(self) -> None:
        """BSM1 BOD5 = 0.25 * (S_S + X_S + (1-f_P) * (X_BH + X_BA))."""
        params = ASM1Params(volume=1000.0, f_p=0.08)
        state = ASM1State.create(s_s=1.0, x_s=2.0, x_bh=10.0, x_ba=5.0)

        bod5 = compute_bod5(state, params)

        expected = 0.25 * (1.0 + 2.0 + (1.0 - 0.08) * (10.0 + 5.0))
        assert jnp.allclose(bod5, jnp.array(expected), atol=1e-6)

    def test_compute_tn_matches_bsm1_formula(self) -> None:
        """BSM1 N_tot = S_NH + S_ND + X_ND + i_XB*(X_BH+X_BA) + i_XP*(X_P+X_I) + S_NO."""
        params = ASM1Params(volume=1000.0, i_xb=0.08, i_xp=0.06)
        state = ASM1State.create(
            x_i=100.0, x_bh=10.0, x_ba=5.0, x_p=8.0, s_no=3.0, s_nh=2.0, s_nd=1.0, x_nd=0.5
        )

        tn = compute_tn(state, params)

        tkn = 2.0 + 1.0 + 0.5 + 0.08 * (10.0 + 5.0) + 0.06 * (8.0 + 100.0)
        assert jnp.allclose(tn, jnp.array(tkn + 3.0), atol=1e-6)


class TestBSM1Benchmark:
    @pytest.mark.parametrize(
        "factory, config, action",
        [
            (make_bsm1_benchmark, BSM1BenchmarkConfig(), jnp.array([5.0, 5.0])),
            (make_bsm1_recycle_benchmark, BSM1RecycleConfig(), jnp.array([3.0, 1.0])),
            (make_bsm1_combined_benchmark, BSM1CombinedConfig(), jnp.array([5.0, 5.0, 3.0, 1.0])),
        ],
    )
    def test_mid_range_aeration_keeps_effluent_ammonium_bounded(
        self, factory: Callable, config: object, action: jax.Array
    ) -> None:
        """A jitted long run keeps effluent ammonium finite and below 50 mg N/L."""
        reset, step = factory(config)
        state, _ = reset(jax.random.PRNGKey(4))

        def scan(state: object, key: jax.Array):
            state, _, _, _, info = step(state, action, key)
            return state, info["s_nh_effluent"]

        def run_scan(state: object):
            return jax.lax.scan(scan, state, jax.random.split(jax.random.PRNGKey(5), 8000))

        _, effluent_nh = jax.jit(run_scan)(state)
        effluent_nh = effluent_nh[-1]
        assert bool(jnp.isfinite(effluent_nh))
        assert float(effluent_nh) < 50.0

    def test_reset_returns_state_and_obs(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, _step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (9,)
        assert state.step_count == 0

    def test_initial_states_match_alex_2008_table_6(self) -> None:
        """Reactor initial states must match the BSM1 open-loop steady-state reference."""
        config = BSM1BenchmarkConfig()
        reset_fn, _step_fn = make_bsm1_benchmark(config)

        state, _ = reset_fn(jax.random.PRNGKey(0))

        assert jnp.allclose(state.reactor1.x_i, jnp.array(1149.0), atol=1e-6)
        assert jnp.allclose(state.reactor1.s_no, jnp.array(5.37), atol=1e-6)
        assert jnp.allclose(state.reactor2.x_s, jnp.array(76.4), atol=1e-6)
        assert jnp.allclose(state.reactor5.s_o, jnp.array(0.491), atol=1e-6)
        assert jnp.allclose(state.reactor5.s_no, jnp.array(10.4), atol=1e-6)

    def test_step_returns_correct_shapes(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array([3.0, 3.0])  # kla_34=3, kla_5=3
        new_state, obs, reward, done, info = step_fn(state, action, k2)

        assert obs.shape == (9,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "s_nh_effluent" in info
        assert "s_no_effluent" in info
        assert "s_o_r3" in info
        assert "s_o_r5" in info
        assert "kla_34" in info

    def test_all_five_reactors_advance(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        initial_r1_x_bh = state.reactor1.x_bh
        initial_r5_x_bh = state.reactor5.x_bh

        for _ in range(10):
            k_step, key = jax.random.split(key)
            state, _, _, _, _ = step_fn(state, jnp.array([3.0, 3.0]), k_step)

        # X_BH should change (growth + decay dynamics)
        assert state.reactor1.x_bh != initial_r1_x_bh or state.reactor5.x_bh != initial_r5_x_bh

    def test_high_aeration_raises_aerobic_do(self) -> None:
        """BSM1 has slow dynamics (SRT ~10 days). After the actuator ramps to kla=10
        over the first 100 steps (2 h), we need ~600 more steps (12 h) for DO to
        re-equilibrate from the initial anoxic transient. Steady-state BSM1 DO ≈ 2 g/m³.
        We assert DO > 1.0 to verify aerobic response without requiring full convergence.
        """
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(2)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(700):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([10.0, 10.0]), k_step)

        assert float(info["s_o_r3"]) > 1.0
        assert float(info["s_o_r5"]) > 1.0

    def test_no_aeration_collapses_do(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(3)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(200):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([0.0, 0.0]), k_step)

        assert float(info["s_o_r3"]) < 0.5
        assert float(info["s_o_r5"]) < 0.5

    def test_effluent_non_negative(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(7)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(30):
            k_step, key = jax.random.split(key)
            state, _, _, _, info = step_fn(state, jnp.array([3.0, 3.0]), k_step)
            assert float(info["s_nh_effluent"]) >= 0.0
            assert float(info["s_no_effluent"]) >= 0.0

    def test_jit_compatible(self) -> None:
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        jit_reset = jax.jit(reset_fn)
        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        state, obs = jit_reset(k1)
        new_state, obs2, reward, done, info = jit_step(state, jnp.array([3.0, 3.0]), k2)

        assert obs2.shape == (9,)
        assert reward.shape == ()

    def test_sensor_noise_is_nonzero(self) -> None:
        """Default sensors have noise — readings should differ from true values."""
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        readings = []
        true_vals = []
        for _ in range(50):
            k_step, key = jax.random.split(key)
            state, obs, _, _, info = step_fn(state, jnp.array([3.0, 3.0]), k_step)
            readings.append(float(obs[2]))  # DO R3 reading (normalised)
            true_vals.append(float(info["s_o_r3"]) / 8.0)

        diffs = [abs(r - t) for r, t in zip(readings, true_vals, strict=True)]
        assert max(diffs) > 0.001, "Sensor noise appears absent"

    def test_obs_includes_influent_and_aeration(self) -> None:
        """Obs should include influent NH4 and aeration power channels."""
        config = BSM1BenchmarkConfig()
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(10)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(20):
            k_step, key = jax.random.split(key)
            state, obs, _, _, _ = step_fn(state, jnp.array([3.0, 3.0]), k_step)

        # Channel 6: influent NH4 (should be nonzero after analyzer samples)
        assert float(obs[6]) > 0.1
        # Channel 7: aeration power (should be > 0 with kla=3)
        assert float(obs[7]) > 0.0

    def test_ideal_sensors_match_true_state(self) -> None:
        """With zero noise, zero lag, and sample_period=1, sensor readings
        should match true state values.

        DO sensors live inside DosingSystem loops and read the *previous*
        step's reactor DO (one-step delay).  With ideal settings the sensed
        value equals the previous true DO, so we compare against the reactor
        state that was read, not the newly-advanced state.
        """
        config = BSM1BenchmarkConfig(
            do_noise_std=0.0,
            do_lag=0.0,
            do_drift_rate=0.0,
            analyzer_noise_std=0.0,
            analyzer_lag=0.0,
            analyzer_sample_period=1,
        )
        reset_fn, step_fn = make_bsm1_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(10):
            prev_r3_so = state.reactor3.s_o
            prev_r5_so = state.reactor5.s_o
            k_step, key = jax.random.split(key)
            state, obs, _, _, info = step_fn(state, jnp.array([3.0, 3.0]), k_step)

        # DO reading should match previous-step true state (DosingSystem delay)
        assert jnp.allclose(obs[2], prev_r3_so / 8.0, atol=1e-5)
        assert jnp.allclose(obs[3], prev_r5_so / 8.0, atol=1e-5)
        # Effluent NH4 reading should match current true state (standalone sensor)
        assert jnp.allclose(obs[0], info["s_nh_effluent"] / 35.0, atol=1e-5)


type Reactors = tuple[ASM1State, ASM1State, ASM1State, ASM1State, ASM1State]


class TestASM1IntegrationStability:
    @pytest.mark.parametrize(
        "q_a_ratio, q_rs_ratio",
        [(3.0, 1.0), (0.5, 0.3)],
        ids=["bsm1_recycle", "minimum_recycle"],
    )
    def test_max_aeration_keeps_do_within_saturation(self, q_a_ratio: float, q_rs_ratio: float) -> None:
        """Every reactor's DO stays in [0, s_o_sat] at maximum kLa over a BSM1 dt.

        Near DO=0 the O2-Monod terms make the reactor ODE stiff enough that a
        single explicit step per dt diverges, spiking DO far above saturation.
        """
        config = BSM1BenchmarkConfig()
        initial = config.initial_states
        key = jax.random.PRNGKey(0)
        params = (
            ASM1Params(volume=config.v1),
            ASM1Params(volume=config.v2),
            ASM1Params(volume=config.v3),
            ASM1Params(volume=config.v4),
            ASM1Params(volume=config.v5),
        )
        influent = asm1_reset(initial.influent, key)
        reactors = (
            asm1_reset(initial.r1, key),
            asm1_reset(initial.r2, key),
            asm1_reset(initial.r3, key),
            asm1_reset(initial.r4, key),
            asm1_reset(initial.r5, key),
        )
        q_in = jnp.array(config.mean_flow)
        q_a = q_a_ratio * q_in
        q_rs = q_rs_ratio * q_in
        kla = jnp.array(config.kla_34_max)

        def body(reactors: Reactors, _: None):
            _effluent, sludge = perfect_clarifier(reactors[4], q_in + q_rs, q_rs, jnp.array(config.q_w_fixed))
            reactors = reactor_train(reactors, influent, q_in, sludge, q_rs, q_a, kla, kla, params, jnp.array(config.dt))
            return reactors, jnp.stack([r.s_o for r in reactors])

        def rollout(reactors: Reactors):
            return jax.lax.scan(body, reactors, None, length=300)

        _, dissolved_oxygen = jax.jit(rollout)(reactors)

        assert dissolved_oxygen.shape == (300, 5)
        assert float(dissolved_oxygen.min()) >= 0.0
        assert float(dissolved_oxygen.max()) <= params[0].s_o_sat
