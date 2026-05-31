import time

import gymnax
import gymnax.wrappers
import jax
import matplotlib.pyplot as plt
from rl_agents.sac import SACConfig, make_train


def main() -> None:
    config = SACConfig(
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=100_000,
        BATCH_SIZE=256,
        LEARNING_STARTS=1000,
        LR=1e-3,
        ENV_NAME="MountainCarContinuous-v0",
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    env, env_params = gymnax.make(config.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = make_train(config, env=env, env_params=env_params)  # type: ignore[arg-type]
    train_jit = jax.jit(train_fn)

    print(f"--- Training SAC on {config.ENV_NAME} ---")
    start = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - start

    metrics = out["metrics"]
    returns = metrics["returned_episode_returns"]
    sps = config.TOTAL_TIMESTEPS / elapsed
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.plot(returns)
    plt.xlabel("Update Step")
    plt.ylabel("Episode Return")
    plt.title(f"SAC on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("sac_results.png")
    print("Plot saved as sac_results.png")


if __name__ == "__main__":
    main()
