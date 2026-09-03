"""Small (unit) tests for ``rl_components.timestep.bootstrap_terms``.

The expected ``(discount, episode_end)`` pairs are written as literals, never
recomputed from the flags: a test that re-derives the rule agrees with a broken
implementation that shares the same mistake.

``GAMMA`` is deliberately neither ``0.0`` nor ``1.0`` so a surviving bootstrap is
distinguishable from both a killed one and an undiscounted one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from rl_components.env_protocol import TruncationPolicy
from rl_components.timestep import bootstrap_terms

GAMMA = 0.9

# (terminated, truncated, cutoff_reached, policy, expected_discount, expected_episode_end)
CASES: list[tuple[bool, bool, bool, TruncationPolicy, float, bool]] = [
    # Continuation: no boundary, so the bootstrap survives under either policy.
    (False, False, False, "bootstrap", GAMMA, False),
    (False, False, False, "terminate", GAMMA, False),
    # Truncation via the loop's own cutoff. `cutoff_reached` behaves exactly like
    # `truncated`: the loop's time limit and the environment's are the same kind of event.
    (False, False, True, "bootstrap", GAMMA, True),
    (False, False, True, "terminate", 0.0, True),
    # Truncation reported by the environment. Under "bootstrap" the discount stays at
    # GAMMA while episode_end is True -- THE CASE THE WHOLE DESIGN EXISTS FOR: bootstrap
    # AT the truncation (its final state is an ordinary state worth valuing) but never
    # ACROSS it (the trajectory stops, so traces must be cut).
    (False, True, False, "bootstrap", GAMMA, True),
    (False, True, False, "terminate", 0.0, True),
    # Both truncation sources at once: still one truncation.
    (False, True, True, "bootstrap", GAMMA, True),
    (False, True, True, "terminate", 0.0, True),
    # Termination: no successor state to value, so the bootstrap dies under either policy.
    (True, False, False, "bootstrap", 0.0, True),
    (True, False, False, "terminate", 0.0, True),
    # TERMINATION DOMINANCE. A step flagged terminated alongside either truncation source
    # is a termination: discount 0.0 under BOTH policies. Treating it as a truncation would
    # bootstrap from a state nothing follows and bias every value estimate upward.
    (True, False, True, "bootstrap", 0.0, True),
    (True, False, True, "terminate", 0.0, True),
    (True, True, False, "bootstrap", 0.0, True),
    (True, True, False, "terminate", 0.0, True),
    (True, True, True, "bootstrap", 0.0, True),
    (True, True, True, "terminate", 0.0, True),
]


def _flag(value: bool) -> str:
    return "T" if value else "F"


def _case_id(case: tuple[bool, bool, bool, TruncationPolicy, float, bool]) -> str:
    terminated, truncated, cutoff_reached, policy, _, _ = case
    return f"term={_flag(terminated)},trunc={_flag(truncated)},cutoff={_flag(cutoff_reached)},policy={policy}"


class TestBootstrapTermsTable:
    @pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
    def test_matches_expected_table(self, case: tuple[bool, bool, bool, TruncationPolicy, float, bool]) -> None:
        terminated, truncated, cutoff_reached, policy, expected_discount, expected_episode_end = case
        discount, episode_end = bootstrap_terms(
            jnp.asarray(terminated, jnp.bool_),
            jnp.asarray(truncated, jnp.bool_),
            jnp.asarray(cutoff_reached, jnp.bool_),
            gamma=GAMMA,
            truncation_policy=policy,
        )
        assert float(discount) == pytest.approx(expected_discount)
        assert bool(episode_end) is expected_episode_end


class TestTerminationDominance:
    """The row an implementation is most likely to get wrong, called out on its own."""

    @pytest.mark.parametrize("policy", ["bootstrap", "terminate"])
    @pytest.mark.parametrize("truncation_source", ["truncated", "cutoff_reached"])
    def test_termination_kills_bootstrap_despite_truncation(self, policy: TruncationPolicy, truncation_source: str) -> None:
        discount, episode_end = bootstrap_terms(
            jnp.asarray(True, jnp.bool_),
            jnp.asarray(truncation_source == "truncated", jnp.bool_),
            jnp.asarray(truncation_source == "cutoff_reached", jnp.bool_),
            gamma=GAMMA,
            truncation_policy=policy,
        )
        assert float(discount) == 0.0
        assert bool(episode_end) is True


class TestBootstrapTermsDtypes:
    def test_discount_is_float32_and_episode_end_is_bool(self) -> None:
        discount, episode_end = bootstrap_terms(
            jnp.asarray(False, jnp.bool_),
            jnp.asarray(True, jnp.bool_),
            jnp.asarray(False, jnp.bool_),
            gamma=GAMMA,
            truncation_policy="bootstrap",
        )
        assert discount.dtype == jnp.float32
        assert episode_end.dtype == jnp.bool_


class TestBootstrapTermsUnderJit:
    @pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
    def test_jit_with_policy_closed_over(self, case: tuple[bool, bool, bool, TruncationPolicy, float, bool]) -> None:
        terminated, truncated, cutoff_reached, policy, expected_discount, expected_episode_end = case
        jitted = jax.jit(lambda t, tr, c: bootstrap_terms(t, tr, c, gamma=GAMMA, truncation_policy=policy))
        discount, episode_end = jitted(
            jnp.asarray(terminated, jnp.bool_),
            jnp.asarray(truncated, jnp.bool_),
            jnp.asarray(cutoff_reached, jnp.bool_),
        )
        assert float(discount) == pytest.approx(expected_discount)
        assert bool(episode_end) is expected_episode_end
        assert discount.dtype == jnp.float32
        assert episode_end.dtype == jnp.bool_
