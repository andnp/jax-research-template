"""Medium tests: TD3 gradient flow on MuJoCo Playground dm_control_suite envs."""

import jax
import jax.numpy as jnp
import pytest
from rl_agents.td3 import TD3Config, make_train
from rl_components.gymnax_bridge import make_gymnax_compat_env
from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter


def _make_env(env_name: str):
    from mujoco_playground import dm_control_suite

    raw = dm_control_suite.load(env_name, config_overrides={"impl": "jax"})
    return make_gymnax_compat_env(MujocoPlaygroundAdapter(raw))


def _short_config(env_name: str) -> TD3Config:
    return TD3Config(
        ENV_NAME=env_name,
        TOTAL_TIMESTEPS=200,
        LEARNING_STARTS=50,
        BUFFER_SIZE=64,
        BATCH_SIZE=16,
        TRAIN_FREQUENCY=1,
        POLICY_DELAY=2,
    )


class TestMujocoPlaygroundBridge:
    def test_adapter_spec_cheetah(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("CheetahRun", config_overrides={"impl": "jax"}))
        spec = adapter.spec()

        assert spec.observation_shape[0] > 0
        assert spec.action_shape[0] > 0
        assert spec.action_dtype == jnp.float32

    def test_adapter_reset_returns_obs(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("HopperHop", config_overrides={"impl": "jax"}))
        spec = adapter.spec()
        result = adapter.reset(jax.random.key(0))

        assert result.observation.shape == spec.observation_shape

    def test_adapter_step_returns_transition(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("HopperHop", config_overrides={"impl": "jax"}))
        spec = adapter.spec()
        reset = adapter.reset(jax.random.key(0))
        action = jnp.zeros(spec.action_shape)

        transition = adapter.step(jax.random.key(1), reset.state, action)

        assert transition.observation.shape == spec.observation_shape
        assert transition.reward.shape == ()
        assert transition.terminated.shape == ()


class TestTD3OnPlaygroundEnvs:
    @pytest.mark.parametrize("env_name", ["CheetahRun", "HopperHop"])
    def test_critic_params_change_after_update(self, env_name: str) -> None:
        env = _make_env(env_name)
        config = _short_config(env_name)
        train = make_train(config, env=env)

        key = jax.random.key(0)
        init_state = make_train(
            TD3Config(
                ENV_NAME=env_name,
                TOTAL_TIMESTEPS=1,
                LEARNING_STARTS=100,
                BUFFER_SIZE=64,
                BATCH_SIZE=16,
            ),
            env=env,
        )(key)["runner_state"]
        params_before = init_state.critic_state.params

        out = jax.jit(train)(key)
        params_after = out["runner_state"].critic_state.params

        flat_before = jax.tree_util.tree_leaves(params_before)
        flat_after = jax.tree_util.tree_leaves(params_after)
        any_changed = any(not jnp.allclose(b, a) for b, a in zip(flat_before, flat_after, strict=True))
        assert any_changed, f"Critic params did not change on {env_name}"

    @pytest.mark.parametrize("env_name", ["CheetahRun", "HopperHop"])
    def test_train_jit_compiles(self, env_name: str) -> None:
        env = _make_env(env_name)
        config = _short_config(env_name)
        train = jax.jit(make_train(config, env=env))

        out = train(jax.random.key(42))
        assert "runner_state" in out
        assert "metrics" in out
