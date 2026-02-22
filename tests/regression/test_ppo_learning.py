import jax

from agents.ppo import make_train
from components.types import PPOConfig


def test_ppo_cartpole_learns():
    # Simple PPO configuration for a quick test
    config = PPOConfig(TOTAL_TIMESTEPS=200_000, NUM_ENVS=32, NUM_STEPS=64, UPDATE_EPOCHS=4, NUM_MINIBATCHES=4, LR=1e-3, ENV_NAME="CartPole-v1", SEED=42)

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    # Check the last 10 updates mean return
    final_return = returns[-10:].mean()
    print(f"Final mean return: {final_return}")

    # CartPole-v1 max is 500
    assert final_return > 400, f"PPO failed to learn CartPole. Return: {final_return}"
