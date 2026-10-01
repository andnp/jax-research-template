"""Medium tests: TD3 learning on MuJoCo Playground dm_control_suite envs."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from rl_agents.td3 import TD3Agent, TD3Config
from rl_components.loop import run
from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

STEPS = 200
GAMMA = 0.99


def _make_env(env_name: str) -> MujocoPlaygroundAdapter:
    from mujoco_playground import dm_control_suite

    return MujocoPlaygroundAdapter(dm_control_suite.load(env_name, config_overrides={"impl": "jax"}))


def _short_agent(env_name: str) -> TD3Agent:
    return TD3Agent(
        TD3Config(
            ENV_NAME=env_name,
            total_timesteps=STEPS,
            LEARNING_STARTS=50,
            BUFFER_SIZE=64,
            BATCH_SIZE=16,
            TRAIN_FREQUENCY=1,
            POLICY_DELAY=2,
        )
    )


class TestMujocoPlaygroundBridge:
    def test_adapter_spec_cheetah(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("CheetahRun", config_overrides={"impl": "jax"}))
        spec = adapter.spec()

        assert spec.observation_shape[0] > 0
        assert spec.action_shape[0] > 0
        assert spec.action_dtype == jnp.float32

    def test_adapter_reset_returns_obs(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("HopperHop", config_overrides={"impl": "jax"}))
        spec = adapter.spec()
        result = adapter.reset(jax.random.key(0))

        assert result.observation.shape == spec.observation_shape

    def test_adapter_step_returns_transition(self) -> None:
        from mujoco_playground import dm_control_suite
        from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

        adapter = MujocoPlaygroundAdapter(dm_control_suite.load("HopperHop", config_overrides={"impl": "jax"}))
        spec = adapter.spec()
        reset = adapter.reset(jax.random.key(0))
        action = jnp.zeros(spec.action_shape)

        transition = adapter.step(jax.random.key(1), reset.state, action)

        assert transition.observation.shape == spec.observation_shape
        assert transition.reward.shape == ()
        assert transition.terminated.shape == ()

    def test_episode_wrapper_time_limit_reports_truncated_not_terminated(self) -> None:
        from brax.envs.wrappers.training import EpisodeWrapper
        from mujoco_playground import dm_control_suite
        from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

        raw = dm_control_suite.load("CartpoleBalance", config_overrides={"impl": "jax"})
        wrapped = EpisodeWrapper(cast(Any, raw), episode_length=1, action_repeat=1)
        adapter = MujocoPlaygroundAdapter(cast(Any, wrapped))
        spec = adapter.spec()
        reset = adapter.reset(jax.random.key(0))
        action = jnp.zeros(spec.action_shape)

        transition = adapter.step(jax.random.key(1), reset.state, action)

        assert bool(transition.truncated) is True
        assert bool(transition.terminated) is False

        reset_state = cast(Any, reset.state)
        blown_up_data = reset_state.data.replace(qpos=jnp.full_like(reset_state.data.qpos, jnp.nan))
        genuinely_terminal_state = reset_state.replace(data=blown_up_data)

        genuine_transition = adapter.step(jax.random.key(2), genuinely_terminal_state, action)

        assert bool(genuine_transition.terminated) is True
        assert bool(genuine_transition.truncated) is False


class TestTD3OnPlaygroundEnvs:
    """``run`` is entered eagerly here rather than under an outer ``jax.jit``.

    :meth:`MujocoPlaygroundAdapter.spec` performs JAX operations, and ``run`` calls it
    before the scan, so an enclosing trace turns the adapter's action bounds into
    tracers and its ``jnp.all`` guard raises. The scan body still compiles.
    """

    @pytest.mark.parametrize("env_name", ["CheetahRun", "HopperHop"])
    def test_critic_params_change_after_update(self, env_name: str) -> None:
        env = _make_env(env_name)
        agent = _short_agent(env_name)
        key = jax.random.key(0)
        params_before = agent.init(key, env.spec()).algorithm_state.critic_state.params

        final_state, _ = run(agent, env, key, steps=STEPS, gamma=GAMMA)
        params_after = final_state.agent_state.algorithm_state.critic_state.params

        flat_before = jax.tree_util.tree_leaves(params_before)
        flat_after = jax.tree_util.tree_leaves(params_after)
        any_changed = any(not jnp.allclose(b, a) for b, a in zip(flat_before, flat_after, strict=True))
        assert any_changed, f"Critic params did not change on {env_name}"

    @pytest.mark.parametrize("env_name", ["CheetahRun", "HopperHop"])
    def test_run_reports_finite_agent_metrics_every_step(self, env_name: str) -> None:
        env = _make_env(env_name)
        agent = _short_agent(env_name)

        _, metrics = run(agent, env, jax.random.key(42), steps=STEPS, gamma=GAMMA)

        for key_name in ("critic_loss", "actor_loss", "loop/reward", "loop/episode_end"):
            assert metrics[key_name].shape == (STEPS,), key_name
        assert bool(jnp.all(jnp.isfinite(metrics["critic_loss"])))
