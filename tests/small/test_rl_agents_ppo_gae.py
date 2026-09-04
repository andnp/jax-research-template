"""Exact-value test for PPO's GAE recurrence.

The step-8 port gate in `tests/medium/test_rl_agents_ppo_golden_baseline.py` can only band
returns across seeds, and a review measured that a bootstrap-past-termination defect passes
that band cleanly. GAE itself needs no statistics: `ppo_gae` is a pure function of a
`PPOTransition` window, so a fixed synthetic window pins it exactly with no RNG, no seeds
and no tolerance.

Every expected number below is hand-computed. The first two cases carry the pre-port
expectations forward unchanged: the port re-indexed the rollout and split `bootstrap_value`
out of the following slot's `value`, but on a window with no truncation those are two
spellings of the same arithmetic, so re-deriving the constants would have thrown away the
only independent check that the re-indexing preserved the recurrence.

The third case is the one the pre-port layout could not express at all. A truncation has
`discount != 0` and `episode_end` true simultaneously, which the fused `not_done` collapsed
into a single flag; the constants there separate the two implementations by construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_agents.ppo import PPOTransition, ppo_gae

# Powers of two, so every intermediate product is exact in float32 and the expectations
# below can be asserted bit-for-bit. gamma * lambda = 0.375, distinct from gamma alone.
_GAMMA = jnp.asarray(0.75, dtype=jnp.float32)
_GAE_LAMBDA = jnp.asarray(0.5, dtype=jnp.float32)

# Six steps covering all three cases: a run of continuations (0-2), a termination (3), and a
# final step that bootstraps off `_LAST_VALUE` (5). The rewards and values are distinct and
# non-round so that an off-by-one in the backward recurrence cannot coincidentally match.
_EPISODE_END = jnp.array([False, False, False, True, False, False])
_DISCOUNT = jnp.where(_EPISODE_END, jnp.zeros((), jnp.float32), _GAMMA)
_VALUE = jnp.array([0.25, 1.75, -0.5, 2.5, 1.25, -0.75], dtype=jnp.float32)
_REWARD = jnp.array([1.5, -0.5, 2.25, 3.0, 0.75, -1.25], dtype=jnp.float32)
_LAST_VALUE = jnp.array(0.5, dtype=jnp.float32)

# Under the completion-indexed layout each slot stores the value of the state its own
# transition reached, which off a boundary is the state the following slot acted from. So
# the window that reproduces the pre-port numbers is `_VALUE` shifted left, closed by the
# bootstrap value the old code carried in as `last_val`.
_BOOTSTRAP_VALUE = jnp.concatenate([_VALUE[1:], _LAST_VALUE[None]])

# Hand-computed, NOT derived in this file. Worked backwards from t=5 with gamma=0.75 and
# gamma*lambda=0.375; the two steps around the termination are:
#   t=3 terminates, so discount=0 drops the bootstrap and episode_end drops the trace:
#     delta_3 = 3.0 + 0.0 * 1.25 - 2.5      = 0.5
#     A_3     = 0.5 + 0.375 * 0 * A_4       = 0.5
#   t=2 continues, so it accumulates A_3 -- and nothing from beyond t=3:
#     delta_2 = 2.25 + 0.75 * 2.5 - (-0.5)  = 4.625
#     A_2     = 4.625 + 0.375 * 0.5         = 4.8125
_EXPECTED_ADVANTAGES = jnp.array([2.2548828125, -0.8203125, 4.8125, 0.5, -1.109375, -0.125], dtype=jnp.float32)
_EXPECTED_TARGETS = jnp.array([2.5048828125, 0.9296875, 4.3125, 3.0, 0.140625, -0.875], dtype=jnp.float32)

# A three-slot window whose middle slot is a truncation: the bootstrap survives with
# coefficient 0.5 while the trace is cut. Hand-computed with gamma=0.5, lambda=0.5:
#   A_2 = -1.0 + 0.5 * 8.0 - 0.5            = 2.5
#   A_1 = 2.0 + 0.5 * 4.0 - 1.0             = 3.0    (trace cut, so A_2 does not enter)
#   A_0 = (1.0 + 0.5 * 2.0 - 0.0) + 0.25 * 3.0       = 2.75
# A fused `not_done` would drop the middle bootstrap and give A_1 = 1.0, A_0 = 2.25.
_TRUNCATION_GAE_LAMBDA = jnp.asarray(0.5, dtype=jnp.float32)
_TRUNCATION_DISCOUNT = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
_TRUNCATION_EPISODE_END = jnp.array([False, True, False])
_TRUNCATION_VALUE = jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32)
_TRUNCATION_REWARD = jnp.array([1.0, 2.0, -1.0], dtype=jnp.float32)
_TRUNCATION_BOOTSTRAP_VALUE = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float32)
_EXPECTED_TRUNCATION_ADVANTAGES = jnp.array([2.75, 3.0, 2.5], dtype=jnp.float32)
_EXPECTED_TRUNCATION_TARGETS = jnp.array([2.75, 4.0, 3.0], dtype=jnp.float32)


def _window(
    value: jax.Array,
    bootstrap_value: jax.Array,
    *,
    reward: jax.Array = _REWARD,
    discount: jax.Array = _DISCOUNT,
    episode_end: jax.Array = _EPISODE_END,
) -> PPOTransition:
    """Build a rollout window; the acting fields are inert here and stay zero."""
    length = int(value.shape[0])
    return PPOTransition(
        obs=jnp.zeros((length, 4), dtype=jnp.float32),
        action=jnp.zeros((length,), dtype=jnp.int32),
        log_prob=jnp.zeros((length,), dtype=jnp.float32),
        value=value,
        reward=reward,
        discount=discount,
        episode_end=episode_end,
        bootstrap_value=bootstrap_value,
        valid=jnp.ones((length,), dtype=jnp.bool_),
    )


class TestGAERecurrence:
    def test_advantages_and_targets_match_hand_computed_values(self) -> None:
        """Pin every advantage and target exactly against hand-computed constants."""
        advantages, targets = ppo_gae(_window(_VALUE, _BOOTSTRAP_VALUE), _GAE_LAMBDA)

        assert jnp.array_equal(advantages, _EXPECTED_ADVANTAGES), advantages
        assert jnp.array_equal(targets, _EXPECTED_TARGETS), targets

    def test_termination_blocks_every_value_after_it(self) -> None:
        """No advantage up to the termination may draw on a value from the next episode.

        Cross-episode leakage is the defect the agent-port migration exists to prevent, so it
        gets its own assertion: replacing every post-termination value, and the terminating
        slot's own bootstrap value, with wildly different numbers must leave steps 0-3
        bit-identical.
        """
        leaked_value = _VALUE.at[4].set(-40.0).at[5].set(90.0)
        leaked_bootstrap = _BOOTSTRAP_VALUE.at[3].set(-40.0).at[4].set(90.0).at[5].set(-70.0)

        advantages, targets = ppo_gae(_window(leaked_value, leaked_bootstrap), _GAE_LAMBDA)

        assert jnp.array_equal(advantages[:4], _EXPECTED_ADVANTAGES[:4]), advantages
        assert jnp.array_equal(targets[:4], _EXPECTED_TARGETS[:4]), targets
        # The perturbation has to bite somewhere, or the assertion above is vacuous.
        assert not bool(jnp.any(advantages[4:] == _EXPECTED_ADVANTAGES[4:]))

    def test_truncation_keeps_its_bootstrap_while_cutting_the_trace(self) -> None:
        """Bootstrap AT a truncation, never ACROSS one.

        This is the case the pre-port fused `not_done` could not represent: one flag had to
        answer both "does the bootstrap survive?" and "does the trace continue?", and a
        truncation answers yes and no. The expected constants differ from what a fused flag
        produces at both surviving slots, so neither role can be dropped unnoticed.
        """
        window = _window(
            _TRUNCATION_VALUE,
            _TRUNCATION_BOOTSTRAP_VALUE,
            reward=_TRUNCATION_REWARD,
            discount=_TRUNCATION_DISCOUNT,
            episode_end=_TRUNCATION_EPISODE_END,
        )

        advantages, targets = ppo_gae(window, _TRUNCATION_GAE_LAMBDA)

        assert jnp.array_equal(advantages, _EXPECTED_TRUNCATION_ADVANTAGES), advantages
        assert jnp.array_equal(targets, _EXPECTED_TRUNCATION_TARGETS), targets
