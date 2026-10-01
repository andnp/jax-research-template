"""Environment configs build the environment they name and compare by value."""

import pytest
from rl_envs.env_config import GymnaxConfig, ProcessControlConfig


def test_gymnax_config_builds_an_environment_and_its_params() -> None:
    env, env_params = GymnaxConfig("CartPole-v1").build()

    assert env.spec(env_params).num_actions == 2


def test_normalized_gymnax_config_scales_continuous_actions_to_unit_bounds() -> None:
    env, env_params = GymnaxConfig("Pendulum-v1", NORMALIZED=True).build()
    spec = env.spec(env_params)

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert float(spec.action_low[0]) == pytest.approx(-1.0)
    assert float(spec.action_high[0]) == pytest.approx(1.0)


def test_gymnax_configs_compare_and_hash_by_value() -> None:
    assert GymnaxConfig("CartPole-v1") == GymnaxConfig("CartPole-v1")
    assert hash(GymnaxConfig("CartPole-v1")) == hash(GymnaxConfig("CartPole-v1"))
    assert GymnaxConfig("Pendulum-v1") != GymnaxConfig("Pendulum-v1", NORMALIZED=True)


def test_process_control_config_builds_an_environment_without_params() -> None:
    env, env_params = ProcessControlConfig("chlorine").build()

    assert env_params is None
    assert env.spec().id == "process_control:chlorine"


def test_process_control_scenario_reaches_the_environment_id() -> None:
    nominal, _ = ProcessControlConfig("h2s_scrubber", SCENARIO="nominal").build()
    shifted, _ = ProcessControlConfig("h2s_scrubber", SCENARIO="load_shift").build()

    assert nominal.spec().id.endswith(":nominal")
    assert shifted.spec().id.endswith(":load_shift")
