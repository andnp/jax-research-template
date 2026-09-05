"""Behavioural gates for QRC through the shared training loop.

Two gates with different jobs.

``test_qrc_cartpole_learns`` is the coarse smoke gate: does QRC learn at all. It is
deliberately loose and cannot see anything subtler than a failure to learn.

``TestQRCRegularisationFloors`` exists because that looseness hid a real regression. A
measured move of 282.0 -> 176.96 on the core QRC regression, when the h-head regulariser
was switched to decoupled AdamW, sat entirely inside the CartPole gate's slack and passed
silently. These floors catch a collapse of either regularisation path.

WHAT THESE FLOORS DO NOT DO, and it matters: they cannot distinguish the two
``H_REGULARIZATION`` values from each other. Measured on Acrobot at 5 seeds, the two
configurations overlap heavily -- a mean difference of 6.35 against a pooled standard error
of about 4.00, roughly 1.6 SE. A separate CartPole measurement could not separate them
either, there under a saturating 500 ceiling. So a passing gate here is evidence that
neither path has collapsed. It is NOT evidence that they are equivalent, and NOT evidence
that either is preferable. Do not cite it for that.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_agents.qrc import HRegularization, QRCAgent, QRCConfig
from rl_components.loop import run
from rl_envs.gymnax_bridge import make_gymnax_env

GAMMA = 0.99
LATE_FRACTION = 0.9
"""Fraction of the horizon after which an episode counts towards the smoke gate's score."""

FLOOR_SEEDS = 5
FLOOR_EPISODES = 100
"""Completed episodes averaged for the floor gates.

The smoke gate scores the final tenth of the horizon, as the sibling DQN gate does. The
floors instead average the last 100 completed episodes, which is a larger and therefore
steadier sample -- roughly 100 episodes rather than the 30-odd that fall in a tenth of an
Acrobot run. A floor is only useful if seed noise cannot trip it, so stability wins over
matching the smoke gate's reduction. The thresholds below were measured with this exact
reduction and do not transfer to another one.
"""


def _cartpole_return(config: QRCConfig) -> jax.Array:
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = QRCAgent(config)

    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(train_fn)
    _, metrics = train_jit(jax.random.PRNGKey(config.SEED))

    # ``loop/episode_return`` is a sparse impulse: it carries a completed episode's return on
    # the boundary step and zero everywhere else, so it must be masked, not averaged.
    boundaries = jnp.flatnonzero(metrics["loop/episode_end"])
    late = boundaries[boundaries >= int(LATE_FRACTION * config.TOTAL_TIMESTEPS)]
    assert late.size > 0, "no episode completed in the final tenth of the run"
    return metrics["loop/episode_return"][late].mean()


def test_qrc_cartpole_learns() -> None:
    config = QRCConfig(ENV_NAME="CartPole-v1", TOTAL_TIMESTEPS=50_000)
    final_return = _cartpole_return(config)
    print(f"Final mean return: {final_return}")
    assert final_return > 100, f"QRC failed to learn CartPole. Return: {final_return}"


def _acrobot_seed_means(style: HRegularization) -> np.ndarray:
    """Mean of each seed's last ``FLOOR_EPISODES`` completed episode returns."""
    config = QRCConfig(ENV_NAME="Acrobot-v1", TOTAL_TIMESTEPS=50_000, H_REGULARIZATION=style)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = QRCAgent(config)

    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(jax.vmap(train_fn))
    _, metrics = train_jit(jax.random.split(jax.random.key(config.SEED), FLOOR_SEEDS))

    returns = np.asarray(metrics["loop/episode_return"])
    ends = np.asarray(metrics["loop/episode_end"]).astype(bool)
    means = []
    for seed in range(FLOOR_SEEDS):
        completed = returns[seed][ends[seed]]
        assert completed.size >= FLOOR_EPISODES, (
            f"{style} seed {seed} completed only {completed.size} episodes; "
            f"the floor needs at least {FLOOR_EPISODES}"
        )
        means.append(completed[-FLOOR_EPISODES:].mean())
    return np.asarray(means)


class TestQRCRegularisationFloors:
    """Each ``H_REGULARIZATION`` path must not collapse. Acrobot, because CartPole saturates.

    Thresholds are derived from a 5-seed measurement at this exact configuration and
    reduction:

        adamw  [-121.85, -102.78, -111.41, -114.92, -119.23]  mean -114.04  std 7.46
        l2     [-102.20, -103.60, -110.81, -107.73, -114.11]  mean -107.69  std 4.95

    Each floor sits far enough below its measured mean that seed noise cannot reach it --
    about 6 standard errors for ``adamw`` and about 10 for ``l2`` -- while still lying above
    the value a regression of the size that motivated this gate would produce. A 37% drop,
    the magnitude of the 282 -> 177 move, would put ``adamw`` near -156 and ``l2`` near -147,
    and both fail their floor. Re-measure before changing the horizon, the network width, the
    epsilon schedule or the reduction; none of these numbers transfer.
    """

    @pytest.mark.parametrize(("style", "floor"), [("adamw", -135.0), ("l2", -130.0)])
    def test_regularisation_style_does_not_collapse(self, style: HRegularization, floor: float) -> None:
        seed_means = _acrobot_seed_means(style)
        observed = float(seed_means.mean())
        print(f"{style}: per-seed={[round(float(v), 2) for v in seed_means]} mean={observed:.2f} floor={floor}")
        assert observed > floor, (
            f"QRC with H_REGULARIZATION={style!r} collapsed on Acrobot: "
            f"mean of {FLOOR_SEEDS} seeds was {observed:.2f}, floor {floor}"
        )
