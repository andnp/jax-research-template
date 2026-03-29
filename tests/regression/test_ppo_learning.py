import gymnax
import gymnax.wrappers
import jax
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig

_config = PPOConfig(TOTAL_TIMESTEPS=80_000, ENV_NAME="CartPole-v1")


def test_ppo_cartpole_learns():
    # Simple PPO configuration for a quick test
    rng = jax.random.PRNGKey(_config.SEED)
    env, env_params = gymnax.make(_config.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = make_train(_config, env=env, env_params=env_params)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    # Check the last 10 updates mean return
    final_return = returns[-10:].mean()
    print(f"Final mean return: {final_return}")

    # CartPole-v1 max is 500
    assert final_return > 400, f"PPO failed to learn CartPole. Return: {final_return}"
