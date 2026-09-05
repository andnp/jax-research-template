"""Golden baseline for PPO, captured before the loop port and asserted against it after.

`test_ppo_seed_averaged_returns_match_golden_baseline` is THE STEP-8 GATE of the agent-port
migration. It asserts only quantities that survive a change of RNG stream: the across-seed
mean return at five update checkpoints, banded at `_RETURN_SE_BAND` standard errors, and the
exact final `observation_count`.

The golden numbers in `ppo_golden_baseline.json` were measured on the PRE-PORT `make_train`
and MUST NOT be re-baselined. A breach is a report-and-diagnose, never an update.

Why the gate cannot pin a single trajectory. The port moves per-step work across
`jax.random.split` boundaries: the pre-port loop consumed `2 * NUM_STEPS + UPDATE_EPOCHS`
splits per update, ordered `(act, step)` per rollout step and then one `permute` per epoch,
while `rl_components.loop.run` splits three ways per iteration and the agent splits three
more. Reshuffling those splits changes which key each sample draws from, so every sampled
action after the first diverges even when the algorithm is unchanged. Emulating that
reshuffle three different ways moved the pinned single-seed checkpoint returns by 25% to 85%
while leaving the 512-seed means within 2.3 standard errors. A single-seed return is
therefore not a fidelity measurement; the seed mean is. A companion
`test_ppo_single_seed_trajectory_is_unchanged` pinned one such trajectory on the pre-port
implementation, and step 8 deleted it rather than re-baselining it, along with the
`pre_port_determinism` section of the golden document that fed it.

Four semantic changes the port cannot avoid. Deltas attributable to these are LEGITIMATE:

1. Transition indexing. Pre-port, `Transition` slot `i` held `(o_i, a_i, V(o_i), r_i, d_i)`:
   the reward and done arrived with the action that had not yet been applied when the slot
   was written, and slot 0 was a valid sample. The port uses completion-indexed slots, so
   the rollout window shifts by one step and the run's very first slot is masked out.
2. Fused `not_done`. Pre-port, one `not_done = 1 - done` term drove both the TD residual and
   the GAE trace cut, with no distinction between termination and truncation. The port
   separates them, which changes advantages at every truncated boundary. CartPole's 500-step
   limit is reached rarely at these returns, so this configuration barely exercises it.
3. Update timing. The pre-port update ran between rollouts, so the new parameters first
   acted on the step after the window closed. In the port the update runs inside the
   iteration that closes the window, before that iteration selects its action, so the new
   parameters act one step earlier.
4. Observation count. The loop's `N` iterations close `N - 1` transitions, so the agent
   never sees the state the final iteration reached and the running statistics take one
   fewer sample than the pre-port loop did for the same budget.

Measured on this configuration after the port: the five checkpoints land at 2.25, 0.49,
0.20, 0.17 and 0.81 SE. The largest is checkpoint 0, which is where the one masked slot and
the one-step update shift carry the most weight.

Cost: the seed sweep is one `jax.jit(jax.vmap(...))` call over 512 keys and takes about 4
seconds. `ppo.py` is deliberately untouched by this file; it only observes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import gymnax
import jax
import jax.numpy as jnp
from rl_agents.ppo import PPOAgent
from rl_components.loop import run
from rl_components.types import PPOConfig
from rl_envs.gymnax_bridge import make_gymnax_env

_GOLDEN_PATH = Path(__file__).with_name("ppo_golden_baseline.json")

_CONFIG = PPOConfig(
    TOTAL_TIMESTEPS=1024,
    NUM_STEPS=32,
    ENV_NAME="CartPole-v1",
    NORMALIZE_OBSERVATIONS=True,
    SEED=42,
)

# The discount the golden run used, which now lives on `run` rather than on the config.
_GAMMA = 0.99

# The pre-port metric was `traj_batch.info["returned_episode_returns"].mean()` over the
# NUM_STEPS rollout: the rollout mean of a `LogWrapper` statistic that itself carried the
# last completed episode's return forward at every step. `LogWrapper` wraps the fused
# `step` the port removes, so the same signal is rebuilt from the loop's own metrics in
# `_smoothed_return_curve`. It is a smoothed learning-progress signal, not any single
# episode's return.
_CHECKPOINTS = (0, 7, 15, 23, 31)

# The gate runs 512 keys under one `vmap`. 24 keys is not enough: with SE ~4.7 on a mean of
# ~45, a bare RNG reshuffle moved the mean by up to 3.0 SE while a materially wrong discount
# (GAMMA 0.99 -> 0.50) moved it by 4.9 SE, so signal and stream noise were the same size. At
# 512 keys the three emulated reshuffles land at 2.24, 1.89 and 1.70 SE while GAMMA 0.50,
# GAE_LAMBDA 0.50 and GAE_LAMBDA 0.0 land at 9.1, 16.9 and 27.4 SE.
_NUM_SEEDS = 512

# k = 6 sits 2.7x above the largest reshuffle deviation measured above and 1.5x below the
# smallest genuine behavioural deviation, so a correct port passes and a wrong advantage or
# discount pathway fails. CartPole returns are skewed, so the band is deliberately wider than
# a normal-tail argument would give.
_RETURN_SE_BAND = 6.0

type _GoldenSection = dict[str, float | dict[str, float]]


def _read_golden_section(name: str) -> _GoldenSection:
    """Read one top-level section of the golden JSON document."""
    document = json.loads(_GOLDEN_PATH.read_text())
    return cast(_GoldenSection, document[name])


def _read_golden_curve(section: _GoldenSection, key: str) -> dict[str, float]:
    """Read one per-checkpoint mapping out of a golden section."""
    curve = section[key]
    assert isinstance(curve, dict)
    return curve


def _read_golden_scalar(section: _GoldenSection, key: str) -> float:
    """Read one scalar out of a golden section."""
    value = section[key]
    assert isinstance(value, float)
    return value


def _smoothed_return_curve(metrics: dict[str, jax.Array]) -> jax.Array:
    """Rebuild the pre-port smoothed-return curve from the loop's own episode metrics.

    `loop/episode_return` is a SPARSE IMPULSE: it carries a completed episode's undiscounted
    return on the boundary step and zero everywhere else. Carrying the most recent impulse
    forward reproduces `LogWrapper.returned_episode_returns` exactly, including its zero
    before any episode has finished, and the loop's metric index equals the pre-port
    environment-step index, so reshaping into `NUM_STEPS` windows recovers the same
    per-update grouping the golden numbers were averaged over.

    Args:
        metrics: The scan-stacked metrics from one vmapped `run`, shaped `(seeds, steps)`.

    Returns:
        The per-seed, per-update rollout-mean smoothed return, shaped
        `(_NUM_SEEDS, num_updates)`.
    """
    steps = _CONFIG.TOTAL_TIMESTEPS
    episode_return = metrics["loop/episode_return"]
    impulse_index = jnp.asarray(
        jnp.where(metrics["loop/episode_end"], jnp.arange(steps, dtype=jnp.int32), jnp.int32(-1))
    )
    latest = jax.lax.cummax(impulse_index, axis=1)
    carried = jnp.where(latest >= 0, jnp.take_along_axis(episode_return, jnp.maximum(latest, 0), axis=1), 0.0)
    return carried.reshape(_NUM_SEEDS, steps // _CONFIG.NUM_STEPS, _CONFIG.NUM_STEPS).mean(axis=-1)


def _run_seed_sweep() -> tuple[jax.Array, jax.Array]:
    """Train the pinned configuration on `_NUM_SEEDS` keys under a single `jax.vmap`.

    Returns:
        The per-seed rollout-mean smoothed returns, shaped `(_NUM_SEEDS, num_updates)`, and
        the per-seed final observation counts, shaped `(_NUM_SEEDS,)`.
    """
    raw_env, env_params = gymnax.make(_CONFIG.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = PPOAgent(_CONFIG)

    def train_one_key(key: jax.Array):
        return run(
            agent,
            env,
            key,
            steps=_CONFIG.TOTAL_TIMESTEPS,
            gamma=_GAMMA,
            env_params=env_params,
        )

    train_fn = jax.jit(jax.vmap(train_one_key))
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_CONFIG.SEED, _CONFIG.SEED + _NUM_SEEDS))
    final_state, metrics = train_fn(keys)
    return _smoothed_return_curve(metrics), final_state.agent_state.obs_norm_state.observation_count


def test_ppo_seed_averaged_returns_match_golden_baseline() -> None:
    """Gate the port on statistics that a change of RNG stream cannot move."""
    golden = _read_golden_section("port_gate")
    golden_mean = _read_golden_curve(golden, "rollout_mean_smoothed_return_seed_mean")
    golden_se = _read_golden_curve(golden, "rollout_mean_smoothed_return_seed_standard_error")
    returns, observation_count = _run_seed_sweep()

    assert set(golden_mean) == set(golden_se) == {str(i) for i in _CHECKPOINTS}
    for update, expected in golden_mean.items():
        actual = float(returns[:, int(update)].mean())
        band = _RETURN_SE_BAND * golden_se[update]
        assert abs(actual - expected) <= band, (
            f"update {update} seed mean {actual} left the +/-{band} band "
            f"({_RETURN_SE_BAND:.0f} SE) around {expected}"
        )

    # `observation_count` is a pure step count, so it is exact on every backend and identical
    # across seeds. The port takes exactly one sample fewer than the pre-port loop did: the
    # statistics are updated once per loop iteration from the state the agent acts from, and
    # the pre-port loop instead took one per environment step plus the reset observation,
    # which is one more because the loop's final iteration opens a transition that never
    # closes. Any offset other than exactly one means the rollout budget or the normalisation
    # wiring moved.
    assert bool(jnp.all(observation_count == _read_golden_scalar(golden, "observation_count") - 1.0))
