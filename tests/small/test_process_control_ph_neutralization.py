from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.ph_neutralization import (
    PhNeutralizationBenchmarkConfig,
    make_ph_neutralization_benchmark,
)
from process_control.scenarios.pulse_train import PulseTrainParams

NO_PULSES = PulseTrainParams(0.0, 1.0, 0.0, 0.0)
# Constant flow, acid and alkalinity: no diurnal swing, drift or slugs.
STEADY = PhNeutralizationBenchmarkConfig(
    flow_amplitude=0.0,
    min_flow=50.0,
    max_flow=50.0,
    flow_drift_scale=0.0,
    flow_slug=NO_PULSES,
    acid_amplitude=0.0,
    acid_drift_scale=0.0,
    acid_slug=NO_PULSES,
    alk_drift_scale=0.0,
    alk_drift_clip=0.0,
)


def _rollout(config: PhNeutralizationBenchmarkConfig, action: Any, n_steps: int, seed: int = 0) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    reset_fn, step_fn = make_ph_neutralization_benchmark(config)
    reset_key, run_key = jax.random.split(jax.random.PRNGKey(seed))
    state, _ = reset_fn(reset_key)
    action = jnp.asarray(action, dtype=float)

    def body(state: Any, key: jax.Array) -> tuple[Any, tuple[dict[str, jax.Array], jax.Array, jax.Array]]:
        state, _, reward, done, info = step_fn(state, action, key)
        return state, (info, reward, done)

    _, (infos, rewards, dones) = jax.lax.scan(body, state, jax.random.split(run_key, n_steps))
    return infos, rewards, dones


class TestContract:
    def test_reset_and_step_shapes(self) -> None:
        """Observation is [pH1, pH2, flow, dose1, dose2], normalised; the action is two pump commands."""
        reset_fn, step_fn = make_ph_neutralization_benchmark(PhNeutralizationBenchmarkConfig())
        state, obs = reset_fn(jax.random.PRNGKey(0))

        new_state, new_obs, reward, done, info = step_fn(state, jnp.array([0.5, 0.2]), jax.random.PRNGKey(1))

        assert obs.shape == (5,)
        assert new_obs.shape == (5,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.isfinite(new_obs))
        assert info["realized_dose"].shape == (2,)

    def test_episodes_never_terminate(self) -> None:
        _, _, dones = _rollout(PhNeutralizationBenchmarkConfig(), [0.5, 0.2], 500)

        assert not jnp.any(dones)

    def test_a_day_has_1440_one_minute_steps_and_the_tanks_hold_a_20_minute_residence_time(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        infos, _, _ = _rollout(STEADY, [0.0, 0.0], 3)

        assert config.steps_per_day * config.dt == 1440.0
        assert jnp.allclose(infos["hrt_min"], 20.0)

    def test_jit_and_vmap_over_seeds(self) -> None:
        reset_fn, step_fn = make_ph_neutralization_benchmark(PhNeutralizationBenchmarkConfig())

        def one(key: jax.Array) -> tuple[jax.Array, jax.Array]:
            state, _ = reset_fn(key)
            _, obs, reward, _, _ = step_fn(state, jnp.array([0.5, 0.2]), key)
            return obs, reward

        obs, reward = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(0), 4))

        assert obs.shape == (4, 5)
        assert reward.shape == (4,)
        assert jnp.all(jnp.isfinite(obs))

    def test_same_key_gives_the_same_trajectory(self) -> None:
        a, _, _ = _rollout(PhNeutralizationBenchmarkConfig(), [0.4, 0.1], 100, seed=3)
        b, _, _ = _rollout(PhNeutralizationBenchmarkConfig(), [0.4, 0.1], 100, seed=3)
        c, _, _ = _rollout(PhNeutralizationBenchmarkConfig(), [0.4, 0.1], 100, seed=4)

        assert jnp.array_equal(a["true_ph_2"], b["true_ph_2"])
        assert not jnp.array_equal(a["true_ph_2"], c["true_ph_2"])


class TestReward:
    def test_reward_is_the_negative_band_plus_naoh_cost_from_info(self) -> None:
        infos, rewards, _ = _rollout(PhNeutralizationBenchmarkConfig(), [1.0, 0.4], 200)

        assert jnp.allclose(rewards, -(infos["band_cost"] + infos["naoh_cost"]))
        assert jnp.all(infos["naoh_cost"] > 0.0)

    def test_reward_uses_true_state_so_sensor_noise_does_not_change_it(self) -> None:
        quiet = replace(PhNeutralizationBenchmarkConfig(), ph_noise_std=0.0, flow_noise_std=0.0)
        noisy = replace(PhNeutralizationBenchmarkConfig(), ph_noise_std=0.5, flow_noise_std=5.0)

        quiet_infos, quiet_rewards, _ = _rollout(quiet, [0.8, 0.3], 100)
        noisy_infos, noisy_rewards, _ = _rollout(noisy, [0.8, 0.3], 100)

        assert jnp.array_equal(quiet_rewards, noisy_rewards)
        assert jnp.array_equal(quiet_infos["true_ph_2"], noisy_infos["true_ph_2"])
        assert not jnp.array_equal(quiet_infos["measured_ph_2"], noisy_infos["measured_ph_2"])

    def test_no_dose_leaves_the_effluent_below_the_band_and_overdosing_above_it(self) -> None:
        config = PhNeutralizationBenchmarkConfig()
        under, under_rewards, _ = _rollout(config, [0.0, 0.0], 400)
        over, over_rewards, _ = _rollout(config, [config.pump_1_max, config.pump_2_max], 400)

        assert jnp.all(under["true_ph_2"][-100:] < config.band_low)
        assert jnp.all(over["true_ph_2"][-100:] > config.band_high)
        assert jnp.all(under["band_violation"][-100:] > 0.0)
        assert jnp.all(over["band_violation"][-100:] > 0.0)
        assert jnp.mean(under_rewards[-100:]) < 0.0 and jnp.mean(over_rewards[-100:]) < 0.0

    def test_margin_violation_is_reported_without_entering_the_reward(self) -> None:
        """A pH inside the band but outside the margin shows a margin violation and no band cost."""
        inside_band = replace(STEADY, band_low=0.0, band_high=14.0)
        infos, rewards, _ = _rollout(inside_band, [0.0, 0.0], 400)

        assert jnp.all(infos["margin_violation"][-50:] > 0.0)
        assert jnp.allclose(infos["band_cost"], 0.0)
        assert jnp.allclose(rewards, -infos["naoh_cost"])


class TestChemistry:
    @pytest.mark.parametrize("alk", [0.5, 8.0])
    def test_solver_stays_valid_across_the_alkalinity_range_for_every_dose_extreme(self, alk: float) -> None:
        for action in ([0.0, 0.0], [2.1, 0.9]):
            infos, _, _ = _rollout(replace(PhNeutralizationBenchmarkConfig(), alk_mean=alk), action, 600)

            assert jnp.all(infos["solver_valid"])
            assert jnp.all(infos["solver_converged"] & infos["solver_bracket_ok"] & infos["solver_in_domain"])

    def test_a_default_rollout_with_slugs_has_no_invalid_solver_step(self) -> None:
        infos, _, _ = _rollout(PhNeutralizationBenchmarkConfig(), [0.9, 0.3], 4320)

        assert jnp.all(infos["solver_valid"])
        assert jnp.all(jnp.isfinite(infos["true_ph_2"]))

    def test_more_bicarbonate_buffers_the_same_dose_so_it_reaches_a_higher_ph(self) -> None:
        """The dose to reach a target pH falls as alkalinity rises, so at a fixed dose pH rises with alkalinity."""
        dose = 2.5 * 0.05  # eq per m3 of influent at 50 L/min, into stage 1
        phs = []
        for alk in (0.5, 1.5, 3.0, 5.0, 8.0):
            infos, _, _ = _rollout(replace(STEADY, alk_mean=alk), [dose, 0.0], 300)
            phs.append(float(infos["true_ph_1"][-1]))

        assert all(later > earlier for earlier, later in zip(phs, phs[1:], strict=False))

    def test_influent_alkalinity_is_reported_and_hidden_from_the_observation(self) -> None:
        reset_fn, step_fn = make_ph_neutralization_benchmark(replace(STEADY, alk_mean=4.0))
        state, obs = reset_fn(jax.random.PRNGKey(0))

        _, new_obs, _, _, info = step_fn(state, jnp.array([0.0, 0.0]), jax.random.PRNGKey(1))

        assert info["alkalinity"] == pytest.approx(4.0)
        assert obs.shape == new_obs.shape == (5,)


class TestPumps:
    def test_default_turndown_spans_a_stage_2_trim_dose_of_one_milli_mole_per_minute(self) -> None:
        infos, _, _ = _rollout(STEADY, [0.0, 0.001], 5)

        assert infos["realized_dose"][-1, 1] == pytest.approx(0.001)

    def test_stress_turndown_has_a_floor_of_a_tenth_of_the_maximum(self) -> None:
        config = replace(STEADY, turndown=10.0)
        stopped, _, _ = _rollout(config, [0.0, 0.04], 5)
        snapped, _, _ = _rollout(config, [0.0, 0.05], 5)

        assert stopped["realized_dose"][-1, 1] == 0.0
        assert snapped["realized_dose"][-1, 1] == pytest.approx(config.pump_2_max / 10.0)

    def test_commands_beyond_the_pump_maximum_are_clipped(self) -> None:
        infos, _, _ = _rollout(STEADY, [50.0, 50.0], 5)

        assert jnp.allclose(infos["realized_dose"][-1], jnp.array([STEADY.pump_1_max, STEADY.pump_2_max]))
