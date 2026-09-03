import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.chlorine import (
    ChlorineBenchmarkConfig,
    make_chlorine_benchmark,
)


def _step(config: ChlorineBenchmarkConfig, action: float = 2.0):
    reset, step = make_chlorine_benchmark(config)
    state, _ = reset(jax.random.PRNGKey(0))
    return step(state, jnp.array(action), jax.random.PRNGKey(1))


def test_tracking_reward_remains_the_default() -> None:
    config = ChlorineBenchmarkConfig()
    _state, observation, reward, _done, _info = _step(config)

    assert jnp.allclose(
        reward,
        -((observation[1] - config.target_residual) ** 2),
    )


def test_supervisory_floor_reward_uses_quality_and_dose_costs() -> None:
    config = ChlorineBenchmarkConfig(
        reward_profile="supervisory-floor",
        quality_floor=1.0,
        dose_cost_weight=0.02,
    )
    _state, observation, reward, _done, info = _step(config)
    quality_cost = jnp.maximum(config.quality_floor - observation[1], 0.0) ** 2
    normalized_dose = info["realized_dose"] / config.pump_max_dose

    expected = -(quality_cost + config.dose_cost_weight * normalized_dose**2)
    assert jnp.allclose(reward, expected)
    assert jnp.allclose(info["quality_cost"], quality_cost)


def test_supervisory_floor_reward_reports_movement_cost() -> None:
    config = ChlorineBenchmarkConfig(
        reward_profile="supervisory-floor",
        dose_movement_weight=0.5,
    )
    _state, _observation, reward, _done, info = _step(config, action=4.0)

    expected = -(
        info["quality_cost"]
        + info["dose_cost"]
        + info["dose_movement_cost"]
    )
    assert jnp.allclose(reward, expected)


def test_unknown_reward_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward_profile"):
        make_chlorine_benchmark(ChlorineBenchmarkConfig(reward_profile="other"))
