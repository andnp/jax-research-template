import jax

from agents.dqn import DQNConfig, make_train


def test_dqn_cartpole_learns():
    config = DQNConfig(
        TOTAL_TIMESTEPS=100_000, NUM_ENVS=4, BUFFER_SIZE=50000, BATCH_SIZE=128, LEARNING_STARTS=1000, TARGET_NETWORK_FREQUENCY=500, LR=1e-3, ENV_NAME="CartPole-v1", SEED=42
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    # Check the last 100 steps of updates (which are every 4 env steps)
    final_return = returns[-100:].mean()
    print(f"Final mean return: {final_return}")

    assert final_return > 100, f"DQN failed to learn CartPole. Return: {final_return}"
