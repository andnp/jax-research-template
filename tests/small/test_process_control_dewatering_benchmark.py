from collections.abc import Callable
from dataclasses import replace

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

    def test_reward_ignores_dryness_analyzer_noise(self) -> None:
        """Cost is computed from the true cake dryness, so a noisier dryness analyzer changes only the observation."""
        quiet = _rollout(DewateringConfig(dryness_noise_std=0.0), (0.7, 0.8), 60)[2]
        noisy = _rollout(DewateringConfig(dryness_noise_std=0.05), (0.7, 0.8), 60)[2]

        assert any(abs(float(q[0][0]) - float(n[0][0])) > 1e-3 for q, n in zip(quiet, noisy, strict=True))
        assert all(float(q[1]) == float(n[1]) for q, n in zip(quiet, noisy, strict=True))


Policy = Callable[[jax.Array], jax.Array]


def _episode_stats(
    policy: Policy, seeds: tuple[int, ...], n_steps: int = 960
) -> tuple[list[float], list[float]]:
    """Per-seed mean reward and fraction of steps whose true filtrate exceeds the limit."""
    config = DewateringConfig()
    reset_fn, step_fn = make_dewatering_benchmark(config)

    def episode(seed: int):
        state, obs = reset_fn(jax.random.PRNGKey(seed))

        def body(carry: tuple, key: jax.Array):
            state, obs = carry
            state, obs, reward, _, info = step_fn(state, policy(obs), key)
            return (state, obs), (reward, info["filtrate_tss"] > config.filtrate_limit)

        keys = jax.random.split(jax.random.PRNGKey(seed + 1000), n_steps)
        _, (rewards, violating) = jax.lax.scan(body, (state, obs), keys)
        return float(rewards.mean()), float(violating.mean())

    stats = [episode(seed) for seed in seeds]
    return [r for r, _ in stats], [v for _, v in stats]


def _total_reward(stats: tuple[list[float], list[float]]) -> float:
    return sum(stats[0])


def _mean_reward(policy: Policy, seeds: tuple[int, ...], n_steps: int = 960):
    return _episode_stats(policy, seeds, n_steps)[0]


def _constant(dose: float, speed: float) -> Policy:
    def policy(obs: jax.Array):
        return jnp.array([dose, speed])

    return policy


def _dose_per_solids(obs: jax.Array):
    """Dose proportional to tank solids (ratio feedforward) with belt speed raised as the tank fills."""
    return jnp.clip(jnp.array([8.0 * obs[4] * 15.0 / 150.0, 0.4 + 0.5 * (obs[5] - 0.5)]), 0.0, 1.0)


def _dryness_feedback(obs: jax.Array):
    """Integral action on measured cake dryness, in dose-per-tank-solids space, using only observations.

    The previous dose is read back from the observation, so the policy is stateless. The 0.26 target
    is the dryness at the capture-optimal dose ratio of the default unit; the gain is tuned for the benchmark.
    """
    running = obs[0] > 0.0
    dose_per_solids = obs[3] / jnp.maximum(obs[4], 1e-3)
    dose_per_solids = jnp.where(running, dose_per_solids + 2.0 * (0.26 - obs[0]), dose_per_solids)
    dose_per_solids = jnp.where(obs[3] == 0.0, 0.8, dose_per_solids)
    return jnp.clip(jnp.array([dose_per_solids * obs[4], 0.4 + 0.5 * (obs[5] - 0.5)]), 0.0, 1.0)


class TestDewateringPolicyRanking:
    SEEDS = (0, 1, 2)

    def test_idle_and_full_throttle_lose_to_a_moderate_policy(self) -> None:
        """Doing nothing overflows the tank and maxing everything blows the filtrate limit."""
        moderate = _mean_reward(_constant(0.6, 0.55), self.SEEDS)
        idle = _mean_reward(_constant(0.0, 0.0), self.SEEDS)
        full = _mean_reward(_constant(1.0, 1.0), self.SEEDS)

        assert all(m > i + 1.0 for m, i in zip(moderate, idle, strict=True))
        assert all(m > f + 1.0 for m, f in zip(moderate, full, strict=True))

    def test_observation_driven_dosing_beats_the_best_constant_on_every_seed(self) -> None:
        """Scaling dose with measured tank solids beats a hindsight-tuned constant dose and speed."""
        constants = [(d, s) for d in (0.5, 0.6, 0.7, 0.8) for s in (0.5, 0.55, 0.6)]
        best_constant = max(
            (_mean_reward(_constant(d, s), self.SEEDS) for d, s in constants), key=sum
        )
        adaptive = _mean_reward(_dose_per_solids, self.SEEDS)

        assert all(a > c for a, c in zip(adaptive, best_constant, strict=True))

    def test_dryness_feedback_beats_the_best_constant_on_held_out_seeds(self) -> None:
        """Feedback on the lagged, noisy dryness analyzer beats a constant tuned on other seeds and cuts violations."""
        tuning, held_out = (0, 1, 2), (100, 101, 102)
        constants = [(d, s) for d in (0.5, 0.6, 0.7) for s in (0.5, 0.55, 0.6)]
        tuning_reward = {c: _total_reward(_episode_stats(_constant(*c), tuning)) for c in constants}
        best_constant = max(tuning_reward, key=tuning_reward.__getitem__)
        best_rewards, best_violations = _episode_stats(_constant(*best_constant), held_out)
        rewards, violations = _episode_stats(_dryness_feedback, held_out)

        assert all(r > c for r, c in zip(rewards, best_rewards, strict=True))
        assert sum(violations) < 0.5 * sum(best_violations)

    def test_filtrate_limit_is_reachable_at_peak_polymer(self) -> None:
        """At the optimum dose ratio and a moderate belt speed, true filtrate stays under the limit."""
        config = DewateringConfig()
        reset_fn, step_fn = make_dewatering_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        optimum_ratio = (config.dewatering.k_polymer_capture * config.dewatering.ki_polymer_capture) ** 0.5
        dose = optimum_ratio * config.feed_tss_mean / 1000.0 / config.dose_max

        _, _, _, _, info = step_fn(state, jnp.array([dose, 0.5]), jax.random.PRNGKey(1))

        assert float(info["filtrate_tss"]) < config.filtrate_limit


class TestDewateringPolymerDemand:
    @staticmethod
    def _filtrate(log_demand: float, dose_fraction: float) -> float:
        config = DewateringConfig(polymer_demand_drift_scale=0.0)
        reset_fn, step_fn = make_dewatering_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        state = replace(state, log_polymer_demand=jnp.array(log_demand))
        _, _, _, _, info = step_fn(state, jnp.array([dose_fraction, 0.5]), jax.random.PRNGKey(1))
        return float(info["filtrate_tss"])

    def test_dose_optimum_scales_with_polymer_demand(self) -> None:
        """A sludge that needs twice the polymer is underdosed by the dose that suits nominal sludge."""
        nominal_optimum = 0.63
        nominal = self._filtrate(0.0, nominal_optimum)
        demanding_underdosed = self._filtrate(float(jnp.log(2.0)), nominal_optimum)
        demanding_matched = self._filtrate(float(jnp.log(2.0)), 2 * nominal_optimum)

        assert demanding_underdosed > 1.2 * nominal
        assert abs(demanding_matched - nominal) < 0.05 * nominal

    def test_demand_stays_within_its_bounds(self) -> None:
        """The latent demand random walk is clipped so the optimum dose stays within a bounded range."""
        config = DewateringConfig(polymer_demand_drift_scale=1.0)
        _, _, steps = _rollout(config, (0.6, 0.5), 200)
        demands = [float(info["polymer_demand"]) for _, _, info in steps]

        bound = float(jnp.exp(config.polymer_demand_clip))
        assert max(demands) <= bound + 1e-4
        assert min(demands) >= 1 / bound - 1e-4
