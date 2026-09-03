"""Golden baseline for `rl_agents.ppo.make_train`, captured before the loop port.

THIS FILE IS THE STEP-8 GATE of the agent-port migration. Step 8 rewrites PPO onto
`rl_components.loop.run` + `AgentProtocol`. The restructuring moves per-step work across
`jax.random.split` boundaries: today an update consumes exactly
`2 * NUM_STEPS + UPDATE_EPOCHS` splits, ordered `(act, step)` per rollout step and then one
`permute` per epoch. Any per-step reshuffling changes which key each sample draws from, so the
ported agent will NOT reproduce these numbers bitwise and must NOT be gated on equality.
Compare the port against the values in `ppo_golden_baseline.json` within the tolerances
declared below, and treat a breach as a real behavioural regression rather than re-baselining.

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

`ppo.py` is deliberately untouched by this commit; it only observes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnax
import gymnax.wrappers
import jax
import jax.numpy as jnp
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig

_GOLDEN_PATH = Path(__file__).with_name("ppo_golden_baseline.json")

# Tolerances. Rerunning this configuration on the machine that captured the golden reproduces
# it bitwise, so the bands exist only to absorb cross-platform XLA differences (fused-multiply
# and reduction-order changes between CPU backends, hosts and accelerators).
#
# `returned_episode_returns` gets the loosest band by a wide margin. A one-bit difference in
# the actor logits can flip a sampled discrete action, which forks the trajectory and moves an
# episode return by whole steps; the metric is chaotic in its inputs, not smoothly sensitive.
# 10% is wide enough that a re-tuned floating-point backend does not fail the suite, and still
# far tighter than the signal the gate has to catch: the deltas a genuinely wrong port
# produces here (dropped rollout step, mis-cut GAE trace) are order-of-magnitude, and the
# noisy 80k-step threshold in `tests/regression/test_ppo_learning.py` is the only other gate.
# Erring loose on this metric is the deliberate choice.
_RETURN_RTOL = 0.10

# The value-head statistics are smooth functions of the same parameter tensor, so they carry a
# much tighter band. `value_head_mean` is near zero (~0.04) at this budget, so relative
# tolerance is meaningless for it and an absolute band is used instead; the std is comfortably
# scaled and takes a relative one.
_VALUE_HEAD_MEAN_ATOL = 5e-3
_VALUE_HEAD_STD_RTOL = 0.02

_CONFIG = PPOConfig(
    TOTAL_TIMESTEPS=1024,
    NUM_STEPS=32,
    ENV_NAME="CartPole-v1",
    NORMALIZE_OBSERVATIONS=True,
    SEED=42,
)


def _run_baseline() -> dict[str, float | dict[str, float]]:
    """Run the pinned PPO configuration and reduce it to the golden's metrics."""
    env, env_params = gymnax.make(_CONFIG.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = jax.jit(make_train(_CONFIG, env=env, env_params=env_params))
    out = train_fn(jax.random.PRNGKey(_CONFIG.SEED))

    returns = out["metrics"]["returned_episode_returns"]
    value_head = out["runner_state"].train_state.params["params"]["Dense_5"]
    flat_value_head = jnp.concatenate([jnp.ravel(value_head["kernel"]), jnp.ravel(value_head["bias"])])
    checkpoints = (0, 7, 15, 23, 31)
    return {
        "returned_episode_returns": {str(i): float(returns[i]) for i in checkpoints},
        "observation_count": float(out["runner_state"].obs_norm_state.observation_count),
        "value_head_mean": float(flat_value_head.mean()),
        "value_head_std": float(flat_value_head.std()),
    }


def test_ppo_matches_golden_baseline() -> None:
    golden = json.loads(_GOLDEN_PATH.read_text())
    actual = _run_baseline()

    golden_returns = golden["returned_episode_returns"]
    actual_returns = actual["returned_episode_returns"]
    assert isinstance(actual_returns, dict)
    assert set(actual_returns) == set(golden_returns)
    for update, expected in golden_returns.items():
        assert abs(actual_returns[update] - expected) <= _RETURN_RTOL * abs(expected), (
            f"update {update} return {actual_returns[update]} left the {_RETURN_RTOL:.0%} band around {expected}"
        )

    # `observation_count` is a pure step count -- one update per rollout step plus the reset
    # observation -- so it is exact on every backend and any drift means the rollout budget or
    # the normalisation wiring moved.
    assert actual["observation_count"] == golden["observation_count"]

    assert abs(actual["value_head_mean"] - golden["value_head_mean"]) <= _VALUE_HEAD_MEAN_ATOL
    assert abs(actual["value_head_std"] - golden["value_head_std"]) <= _VALUE_HEAD_STD_RTOL * golden["value_head_std"]
