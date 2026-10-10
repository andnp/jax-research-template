import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.dewatering import DewateringConfig, make_dewatering_benchmark
from process_control.units.dewatering import DewateringParams
from process_control.units.dewatering import reset as dw_reset
from process_control.units.dewatering import step as dw_step


class TestDewateringUnitCapture:
    def test_peak_capture_reaches_max_capture(self) -> None:
        """At the optimum dose ratio sqrt(k*ki) the solids captured equal max_capture of the feed."""
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(0))
        feed_tss = jnp.array(10000.0)
        optimum_ratio = float(jnp.sqrt(params.k_polymer_capture * params.ki_polymer_capture))
        dose = jnp.array(optimum_ratio * float(feed_tss) / 1000.0)

        new_state, *_ = dw_step(state, feed_tss, jnp.array(20.0), dose, jnp.array(0.5), params, jnp.array(0.05))

        solids_in = float(feed_tss) * 20.0 * 0.05 / 1e3
        assert abs(float(new_state.cake_produced) / solids_in - params.max_capture) < 1e-4

    def test_idle_belt_reports_zero_dryness(self) -> None:
        """With no throughput there is no cake, so dryness must not report a phantom value."""
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(0))

        _, dryness, filtrate_tss, *_ = dw_step(
            state, jnp.array(10000.0), jnp.array(20.0), jnp.array(50.0), jnp.array(0.0), params, jnp.array(0.05)
        )

        assert float(dryness) == 0.0
        assert float(filtrate_tss) == 0.0


def _rollout(config: DewateringConfig, action: tuple[float, float], n_steps: int, seed: int = 0):
    reset_fn, step_fn = make_dewatering_benchmark(config)
    step_fn = jax.jit(step_fn)
    state, obs = reset_fn(jax.random.PRNGKey(seed))
    initial = state
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_steps)
    steps = []
    for key in keys:
        state, obs, reward, _, info = step_fn(state, jnp.array(action), key)
        steps.append((obs, reward, info))
    return initial, state, steps


class TestDewateringBenchmarkContract:
    def test_shapes_and_finite_over_a_day(self) -> None:
        """Reset and step return the documented 6D observation and finite rewards."""
        config = DewateringConfig()
        reset_fn, _ = make_dewatering_benchmark(config)
        _, obs = reset_fn(jax.random.PRNGKey(0))
        _, _, steps = _rollout(config, (0.5, 0.5), 480)

        assert obs.shape == (6,)
        assert all(o.shape == (6,) and bool(jnp.isfinite(r)) for o, r, _ in steps)


class TestDewateringTankBalance:
    @pytest.mark.parametrize(
        ("config", "action"),
        [
            (DewateringConfig(tank_init_fraction=0.95), (0.0, 0.0)),
            (DewateringConfig(), (0.6, 0.5)),
        ],
        ids=["belt_off_overflow", "no_overflow"],
    )
    def test_volume_and_solids_reconcile(self, config: DewateringConfig, action: tuple[float, float]) -> None:
        """Initial inventory plus inflow equals processed, overflowed and remaining volume and solids."""
        initial, final, steps = _rollout(config, action, 200)
        infos = [info for _, _, info in steps]
        dt = config.dt

        def total(key: str) -> float:
            return float(sum(info[key] for info in infos))

        volume_in = float(initial.tank_volume) + dt * total("inflow")
        volume_out = dt * total("throughput") + total("overflow_volume") + float(final.tank_volume)
        solids_in = float(initial.tank_solids) + dt * sum(float(i["inflow"] * i["inflow_tss"]) for i in infos) / 1e3
        solids_out = (
            total("cake_solids") + total("filtrate_solids") + total("overflow_solids") + float(final.tank_solids)
        )

        assert abs(volume_in - volume_out) < 1e-2
        assert abs(solids_in - solids_out) < 1e-2
        assert (total("overflow_volume") > 0.0) == (action == (0.0, 0.0))

    def test_belt_off_fills_the_tank_to_its_limit(self) -> None:
        """With the belt idle the tank level saturates at 1 rather than growing without bound."""
        _, final, _ = _rollout(DewateringConfig(), (0.0, 0.0), 200)

        assert abs(float(final.tank_volume) - DewateringConfig().tank_volume_max) < 1e-3


class TestDewateringAnalyzer:
    def test_reading_is_held_while_belt_is_idle(self) -> None:
        """An idle belt produces no filtrate, so the analyzer must keep its last reading."""
        config = DewateringConfig()
        reset_fn, step_fn = make_dewatering_benchmark(config)
        state, obs = reset_fn(jax.random.PRNGKey(0))
        keys = jax.random.split(jax.random.PRNGKey(1), 40)
        for key in keys[:30]:
            state, obs, *_ = step_fn(state, jnp.array([0.6, 0.5]), key)
        last_running = float(obs[1])
        readings = []
        for key in keys[30:]:
            state, obs, *_ = step_fn(state, jnp.array([0.6, 0.0]), key)
            readings.append(float(obs[1]))

        assert last_running > 0.0
        assert readings == [last_running] * 10

    def test_reward_ignores_analyzer_noise(self) -> None:
        """Quality is scored on the true filtrate, so a noisier analyzer cannot change the reward."""
        quiet = _rollout(DewateringConfig(filtrate_noise_std=0.0), (0.7, 0.8), 60)[2]
        noisy = _rollout(DewateringConfig(filtrate_noise_std=200.0), (0.7, 0.8), 60)[2]

        assert any(abs(float(q[0][1]) - float(n[0][1])) > 1e-3 for q, n in zip(quiet, noisy, strict=True))
        assert all(float(q[1]) == float(n[1]) for q, n in zip(quiet, noisy, strict=True))
