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
