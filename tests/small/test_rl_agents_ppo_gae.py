"""Exact-value test for PPO's GAE recurrence.

The step-8 port gate in `tests/medium/test_rl_agents_ppo_golden_baseline.py` can only band
returns across seeds, and a review measured that a bootstrap-past-termination defect passes
that band cleanly. GAE itself needs no statistics: `_calculate_gae` is a pure function of a
`Transition` batch and a bootstrap value, so a fixed synthetic batch pins it exactly with no
RNG, no seeds and no tolerance.

`_calculate_gae` is a closure nested inside `make_train`, so it cannot be imported. It is
rebound here from its own code object rather than re-implemented: these tests run the
production bytecode, and every expected number below is hand-computed.
"""

from __future__ import annotations

import types
from typing import Callable, cast

import jax
import jax.numpy as jnp
from rl_agents import ppo
from rl_components.types import PPOConfig

# Powers of two, so every intermediate product is exact in float32 and the expectations
# below can be asserted bit-for-bit. gamma * lambda = 0.375, distinct from gamma alone.
_CONFIG = PPOConfig(GAMMA=0.75, GAE_LAMBDA=0.5)

# Six steps covering all three cases: a run of continuations (0-2), a termination (3), and a
# final step that bootstraps off `_LAST_VALUE` (5). The rewards and values are distinct and
# non-round so that an off-by-one in the backward recurrence cannot coincidentally match.
_DONE = jnp.array([False, False, False, True, False, False])
_VALUE = jnp.array([0.25, 1.75, -0.5, 2.5, 1.25, -0.75], dtype=jnp.float32)
_REWARD = jnp.array([1.5, -0.5, 2.25, 3.0, 0.75, -1.25], dtype=jnp.float32)
_LAST_VALUE = jnp.array(0.5, dtype=jnp.float32)

# Hand-computed, NOT derived in this file. Worked backwards from t=5 with gamma=0.75 and
# gamma*lambda=0.375; the two steps around the termination are:
#   t=3 terminates, so not_done=0 drops both the bootstrap and the incoming trace:
#     delta_3 = 3.0 + 0.75 * 1.25 * 0 - 2.5   = 0.5
#     A_3     = 0.5 + 0.375 * 0 * A_4         = 0.5
#   t=2 continues, so it accumulates A_3 -- and nothing from beyond t=3:
#     delta_2 = 2.25 + 0.75 * 2.5 - (-0.5)    = 4.625
#     A_2     = 4.625 + 0.375 * 0.5           = 4.8125
_EXPECTED_ADVANTAGES = jnp.array([2.2548828125, -0.8203125, 4.8125, 0.5, -1.109375, -0.125], dtype=jnp.float32)
_EXPECTED_TARGETS = jnp.array([2.5048828125, 0.9296875, 4.3125, 3.0, 0.140625, -0.875], dtype=jnp.float32)


def _find_code(code: types.CodeType, name: str) -> types.CodeType:
    """Find the nested code object called `name` inside `code`."""
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            if const.co_name == name:
                return const
            try:
                return _find_code(const, name)
            except LookupError:
                continue
    raise LookupError(f"no nested code object named {name!r}")


def _bind_calculate_gae(config: PPOConfig) -> Callable[[ppo.Transition, jax.Array], tuple[jax.Array, jax.Array]]:
    """Rebind `ppo.make_train`'s nested `_calculate_gae` against `config`'s hypers.

    The real function closes over `hypers` alone (GAMMA and GAE_LAMBDA are swept, traced
    values) and reads `jax`/`jnp` from the `ppo` module globals, so rebuilding it from its
    code object runs the production recurrence unchanged.
    """
    code = _find_code(ppo.make_train.__code__, "_calculate_gae")
    assert code.co_freevars == ("hypers",)
    closure = (types.CellType(ppo.ppo_hypers(config)),)
    return cast(
        Callable[[ppo.Transition, jax.Array], tuple[jax.Array, jax.Array]],
        types.FunctionType(code, vars(ppo), "_calculate_gae", None, closure),
    )


def _batch(value: jax.Array) -> ppo.Transition:
    """Build the fixed trajectory batch, with `value` as the critic's predictions."""
    return ppo.Transition(
        done=_DONE,
        action=jnp.zeros((6,), dtype=jnp.int32),
        value=value,
        reward=_REWARD,
        log_prob=jnp.zeros((6,), dtype=jnp.float32),
        obs=jnp.zeros((6, 4), dtype=jnp.float32),
        info={},
    )


class TestGAERecurrence:
    def test_advantages_and_targets_match_hand_computed_values(self) -> None:
        """Pin every advantage and target exactly against hand-computed constants."""
        advantages, targets = _bind_calculate_gae(_CONFIG)(_batch(_VALUE), _LAST_VALUE)

        assert jnp.array_equal(advantages, _EXPECTED_ADVANTAGES), advantages
        assert jnp.array_equal(targets, _EXPECTED_TARGETS), targets

    def test_termination_blocks_every_value_after_it(self) -> None:
        """No advantage up to the termination may draw on a value from the next episode.

        Cross-episode leakage is the defect the agent-port migration exists to prevent, so it
        gets its own assertion: replacing every post-termination value, and the bootstrap
        value, with wildly different numbers must leave steps 0-3 bit-identical.
        """
        leaked = _VALUE.at[4].set(-40.0).at[5].set(90.0)
        calculate_gae = _bind_calculate_gae(_CONFIG)

        advantages, targets = calculate_gae(_batch(leaked), jnp.array(-70.0, dtype=jnp.float32))

        assert jnp.array_equal(advantages[:4], _EXPECTED_ADVANTAGES[:4]), advantages
        assert jnp.array_equal(targets[:4], _EXPECTED_TARGETS[:4]), targets
        # The perturbation has to bite somewhere, or the assertion above is vacuous.
        assert not bool(jnp.any(advantages[4:] == _EXPECTED_ADVANTAGES[4:]))
