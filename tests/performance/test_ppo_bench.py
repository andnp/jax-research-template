import jax
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig


def test_ppo_speed(benchmark):
    config = PPOConfig(
        TOTAL_TIMESTEPS=100_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LR=3e-4,
        ENV_NAME="CartPole-v1",
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    # Warmup (compilation)
    jax.block_until_ready(train_jit(rng))

    def run_train():
        out = train_jit(rng)
        jax.block_until_ready(out)
        return out

    benchmark.pedantic(run_train, rounds=5, warmup_rounds=1)
