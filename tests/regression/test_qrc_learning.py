import gymnax
import gymnax.wrappers
import jax
import pytest
from rl_agents.qrc import QRCConfig, make_train


def test_qrc_make_train_requires_explicit_env() -> None:
    config_only_args = [QRCConfig()]

    with pytest.raises(TypeError, match="env"):
        make_train(*config_only_args)  # type: ignore[arg-type]


def test_qrc_cartpole_learns() -> None:
    config = QRCConfig(
        ENV_NAME="CartPole-v1",
        TOTAL_TIMESTEPS=50_000,
    )

    rng = jax.random.PRNGKey(config.SEED)
    env, env_params = gymnax.make(config.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = make_train(config, env=env, env_params=env_params)  # type: ignore[arg-type]
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    returns = out["metrics"]["returned_episode_returns"]

    # Check the last 100 steps of updates (which are every 4 env steps)
    final_return = returns[-100:].mean()
    print(f"Final mean return: {final_return}")

    assert final_return > 100, f"QRC failed to learn CartPole. Return: {final_return}"
