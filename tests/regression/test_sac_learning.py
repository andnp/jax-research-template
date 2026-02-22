import jax

from agents.sac import SACConfig, make_train


def test_sac_mountaincar_continuous_learns():
    # MountainCarContinuous is easier for SAC than discrete for PPO sometimes
    config = SACConfig(
        TOTAL_TIMESTEPS=50_000,
        BUFFER_SIZE=100000,
        BATCH_SIZE=256,
        LEARNING_STARTS=1000,
        LR=1e-3,
        ENV_NAME="MountainCarContinuous-v0",
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    final_return = returns[-100:].mean()
    print(f"Final mean return: {final_return}")

    # MountainCarContinuous solved is > 90
    assert final_return > 0, f"SAC failed to learn MountainCarContinuous. Return: {final_return}"
