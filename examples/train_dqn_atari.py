import time
from typing import cast

import jax
from rl_agents.dqn import DQNConfig, make_train
from rl_components.atari import JAXAtariConfig, make_atari_adapter
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_compat_env


def main():
    config = DQNConfig(
        TOTAL_TIMESTEPS=5_000,
        BUFFER_SIZE=10_000,
        BATCH_SIZE=32,
        LEARNING_STARTS=1_000,
        TARGET_NETWORK_FREQUENCY=500,
        LR=1e-4,
        SEED=42,
        NETWORK_PRESET="nature_cnn",
    )
    env = make_gymnax_compat_env(
        cast(EnvProtocol[jax.Array, object, jax.Array, None], make_atari_adapter(JAXAtariConfig(game="pong")))
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config, env=env, env_params=None)
    train_jit = jax.jit(train_fn)

    print("--- Training DQN on JAXAtari Pong ---")
    print("Compiling & Training (1st run)...")
    compile_start = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    total_time = time.time() - compile_start

    print("Executing second run for accurate SPS...")
    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    execution_time = time.time() - start_time

    metrics = out["metrics"]
    returns = metrics["returned_episode_returns"]
    sps = config.TOTAL_TIMESTEPS / execution_time

    print(f"Compilation Time: {max(0, total_time - execution_time):.2f}s")
    print(f"Execution Time:   {execution_time:.2f}s")
    print(f"SPS:              {sps:.2f}")
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")


if __name__ == "__main__":
    main()