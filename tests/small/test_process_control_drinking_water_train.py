import jax
import jax.numpy as jnp
from process_control.benchmarks.drinking_water_train import (
    DrinkingWaterTrainConfig,
    make_drinking_water_train_benchmark,
)


class TestChlorineDemandTracksDOC:
    def test_underdosing_coagulant_raises_chlorine_demand(self) -> None:
        """Less coagulant removes less DOC, so more chlorine is consumed reaching the same dose."""
        config = DrinkingWaterTrainConfig()
        reset_fn, step_fn = make_drinking_water_train_benchmark(config)
        key = jax.random.PRNGKey(0)
        state, _ = reset_fn(key)

        low_action = jnp.array([0.1, 0.0, 0.5])
        high_action = jnp.array([1.0, 0.0, 0.5])

        _, _, _, _, info_low = step_fn(state, low_action, key)
        _, _, _, _, info_high = step_fn(state, high_action, key)

        assert float(info_low["permeate_doc"]) > float(info_high["permeate_doc"])
        assert float(info_low["cl2_residual"]) < float(info_high["cl2_residual"])


class TestContactTimeTracksFlow:
    def test_contact_time_shrinks_as_permeate_flow_rises(self) -> None:
        """Contact time is basin volume / flow, so higher throughput must shorten it."""
        config = DrinkingWaterTrainConfig()
        reset_fn, step_fn = make_drinking_water_train_benchmark(config)
        key = jax.random.PRNGKey(1)
        state, _ = reset_fn(key)

        _, _, _, _, info = step_fn(state, jnp.array([0.5, 0.0, 0.5]), key)
        expected_contact_time = config.contact_basin_volume / float(info["q_permeate"])

        assert jnp.allclose(info["contact_time"], expected_contact_time, rtol=1e-5)


class TestResetPressure:
    def test_clean_membrane_reset_tmp_is_a_small_fraction_of_the_limit(self) -> None:
        """compute_tmp takes flux in m/s, so a clean membrane must not start far above max TMP."""
        config = DrinkingWaterTrainConfig()
        reset_fn, _ = make_drinking_water_train_benchmark(config)

        _, obs = reset_fn(jax.random.PRNGKey(0))

        assert 0.0 < float(obs[2]) < 0.2


class TestPlantMassBalance:
    def test_permeate_follows_raw_feed_when_not_backwashing(self) -> None:
        """Production tracks raw feed flow, so the contact basin is fed at plant throughput."""
        config = DrinkingWaterTrainConfig()
        reset_fn, step_fn = make_drinking_water_train_benchmark(config)
        step_fn = jax.jit(step_fn)
        state, _ = reset_fn(jax.random.PRNGKey(0))

        for i in range(50):
            state, _, _, _, info = step_fn(state, jnp.array([0.3, 0.0, 0.5]), jax.random.PRNGKey(i))
            assert jnp.allclose(info["q_permeate"], info["q_in"], rtol=1e-4)

    def test_contact_time_stays_in_design_range_over_a_day(self) -> None:
        """Basin HRT stays within 0.5-2 h across the diurnal feed swing."""
        config = DrinkingWaterTrainConfig()
        reset_fn, step_fn = make_drinking_water_train_benchmark(config)
        step_fn = jax.jit(step_fn)
        state, _ = reset_fn(jax.random.PRNGKey(0))

        times = []
        for i in range(config.steps_per_day):
            action = jnp.array([0.3, 0.0, 0.5])
            state, _, _, _, info = step_fn(state, action, jax.random.PRNGKey(i))
            times.append(float(info["contact_time"]))

        assert 0.5 <= min(times) and max(times) <= 2.0

    def test_backwash_lengthens_contact_time_through_lower_downstream_flow(self) -> None:
        """Backwash interrupts permeate production, so downstream HRT rises for that step."""
        config = DrinkingWaterTrainConfig()
        reset_fn, step_fn = make_drinking_water_train_benchmark(config)
        key = jax.random.PRNGKey(2)
        state, _ = reset_fn(key)

        _, _, _, _, normal = step_fn(state, jnp.array([0.3, 0.0, 0.5]), key)
        _, _, _, _, washing = step_fn(state, jnp.array([0.3, 1.0, 0.5]), key)

        assert float(washing["q_permeate"]) < float(normal["q_permeate"])
        assert float(washing["contact_time"]) > float(normal["contact_time"])
