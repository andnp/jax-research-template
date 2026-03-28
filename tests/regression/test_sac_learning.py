import jax
from rl_agents.sac import SACConfig, make_train


def test_sac_pendulum_learns():
    config = SACConfig(
        TOTAL_TIMESTEPS=50_000,
        ENV_NAME="Pendulum-v1",
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    final_return = returns[-100:].mean()
    print(f"Final mean return: {final_return}")

    assert final_return > -300, f"SAC failed to learn Pendulum. Return: {final_return}"
