"""Golden baseline for `rl_agents.ppo.make_train`, captured before the loop port.

Two independent things live in this file, and they must not be confused:

1. `test_ppo_seed_averaged_returns_match_golden_baseline` is THE STEP-8 GATE of the
   agent-port migration. It asserts only quantities that survive a change of RNG stream:
   the across-seed mean return at five update checkpoints, banded at `_RETURN_SE_BAND`
   standard errors, and the exact final `observation_count`.
2. `test_ppo_single_seed_trajectory_is_unchanged` is a PRE-PORT DETERMINISM REGRESSION on
   the current implementation. It is NOT a port gate and the port WILL break it. Step 8
   deletes it; it must never be re-baselined to make the port look green.

Why the gate cannot pin a single trajectory. Step 8 rewrites PPO onto
`rl_components.loop.run` + `AgentProtocol`, which moves per-step work across
`jax.random.split` boundaries: today an update consumes exactly
`2 * NUM_STEPS + UPDATE_EPOCHS` splits, ordered `(act, step)` per rollout step and then one
`permute` per epoch. Reshuffling those splits changes which key each sample draws from, so
every sampled action after the first diverges even when the algorithm is unchanged.
Emulating that reshuffle three different ways (one 3-way split in place of the two
sequential 2-way splits, the same split with the two consumers swapped, and one extra
unused split per step) moved the pinned single-seed checkpoint returns by 25% to 85% while
leaving the 512-seed means within 2.3 standard errors. A single-seed return is therefore not
a fidelity measurement; the seed mean is.

Two semantic changes the port cannot avoid. Deltas attributable to these are LEGITIMATE:

1. Transition indexing. Today `Transition` slot `i` holds `(o_i, a_i, V(o_i), r_i, d_i)`: the
   reward and done arrive together with the action that has not yet been applied to the
   environment when the slot is written, and slot 0 is a valid sample. The port uses
   completion-indexed slots, so the rollout window shifts by one step relative to this
   baseline.
2. Fused `not_done`. Today the single `not_done = 1 - done` term drives both the TD residual
   and the GAE trace cut, with no distinction between termination and truncation. The port
   separates bootstrapping from trace cutting, which changes advantages at every truncated
   boundary.

Cost: the seed sweep is one `jax.jit(jax.vmap(...))` call over 512 keys and takes about 4
seconds; the determinism check takes about 2. `ppo.py` is deliberately untouched by this
file; it only observes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import gymnax
import gymnax.wrappers
import jax
import jax.numpy as jnp
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig

_GOLDEN_PATH = Path(__file__).with_name("ppo_golden_baseline.json")

_CONFIG = PPOConfig(
    TOTAL_TIMESTEPS=1024,
    NUM_STEPS=32,
    ENV_NAME="CartPole-v1",
    NORMALIZE_OBSERVATIONS=True,
    SEED=42,
)

# The metric is `traj_batch.info["returned_episode_returns"].mean()` over the NUM_STEPS
# rollout (`ppo.py:284`): the rollout mean of a LogWrapper statistic that itself carries the
# last completed episode's return forward at every step. It is a smoothed learning-progress
# signal, not any single episode's return.
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

# Bands for the pre-port determinism check only. Rerunning this configuration on the machine
# that captured the golden reproduces it bitwise; these absorb backend floating-point drift,
# which is the only difference this check tolerates. Measured headroom: perturbing LR by up to
# 1e-4 relative leaves all five returns bitwise identical, and 1e-3 breaches.
_RETURN_RTOL = 0.10
_VALUE_HEAD_MEAN_ATOL = 5e-3
_VALUE_HEAD_STD_RTOL = 0.02

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


def _run_seed_sweep() -> tuple[jax.Array, jax.Array]:
    """Train the pinned configuration on `_NUM_SEEDS` keys under a single `jax.vmap`.

    Returns:
        The per-seed rollout-mean smoothed returns, shaped `(_NUM_SEEDS, num_updates)`, and
        the per-seed final observation counts, shaped `(_NUM_SEEDS,)`.
    """
    env, env_params = gymnax.make(_CONFIG.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = jax.jit(jax.vmap(make_train(_CONFIG, env=env, env_params=env_params)))
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_CONFIG.SEED, _CONFIG.SEED + _NUM_SEEDS))
    out = train_fn(keys)
    return out["metrics"]["returned_episode_returns"], out["runner_state"].obs_norm_state.observation_count


def _run_single_seed() -> tuple[dict[str, float], float, float, float]:
    """Run the pinned configuration on the single golden seed and reduce it to its metrics.

    Returns:
        The per-checkpoint rollout-mean smoothed returns, the final observation count, and
        the value head's flattened mean and standard deviation.
    """
    env, env_params = gymnax.make(_CONFIG.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = jax.jit(make_train(_CONFIG, env=env, env_params=env_params))
    out = train_fn(jax.random.PRNGKey(_CONFIG.SEED))

    returns = out["metrics"]["returned_episode_returns"]
    value_head = out["runner_state"].train_state.params["params"]["Dense_5"]
    flat_value_head = jnp.concatenate([jnp.ravel(value_head["kernel"]), jnp.ravel(value_head["bias"])])
    return (
        {str(i): float(returns[i]) for i in _CHECKPOINTS},
        float(out["runner_state"].obs_norm_state.observation_count),
        float(flat_value_head.mean()),
        float(flat_value_head.std()),
    )


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

    # `observation_count` is a pure step count -- one update per rollout step plus the reset
    # observation -- so it is exact on every backend and identical across seeds. Any drift
    # means the rollout budget or the normalisation wiring moved.
    assert bool(jnp.all(observation_count == _read_golden_scalar(golden, "observation_count")))


def test_ppo_single_seed_trajectory_is_unchanged() -> None:
    """PRE-PORT DETERMINISM REGRESSION on today's PPO -- NOT the step-8 port gate.

    These values pin one RNG trajectory, so the port breaks them by construction (see the
    module docstring). Step 8 deletes this test rather than re-baselining it. Until then a
    breach means today's `ppo.py` changed behaviour.
    """
    golden = _read_golden_section("pre_port_determinism")
    golden_returns = _read_golden_curve(golden, "rollout_mean_smoothed_return")
    actual_returns, observation_count, value_head_mean, value_head_std = _run_single_seed()

    assert set(actual_returns) == set(golden_returns)
    for update, expected in golden_returns.items():
        assert abs(actual_returns[update] - expected) <= _RETURN_RTOL * abs(expected), (
            f"update {update} return {actual_returns[update]} left the {_RETURN_RTOL:.0%} band around {expected}"
        )

    golden_value_head_std = _read_golden_scalar(golden, "value_head_std")
    assert observation_count == _read_golden_scalar(golden, "observation_count")
    assert abs(value_head_mean - _read_golden_scalar(golden, "value_head_mean")) <= _VALUE_HEAD_MEAN_ATOL
    assert abs(value_head_std - golden_value_head_std) <= _VALUE_HEAD_STD_RTOL * golden_value_head_std
