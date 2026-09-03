"""The action-space contract every ported continuous agent shares.

``rl_agents.continuous_actions`` is the one place that contract is stated, and each of
the four continuous agents calls it from ``init``. Its rejections are the loud failures
that stand in for four silent ones: a discrete spec would crash later inside a network, a
rank-2 action would index the wrong axis, and unnormalized bounds would not fail at all --
the policy would simply be confined to the middle of the real action range and keep
training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from rl_agents.continuous_actions import continuous_action_dim, uniform_exploration_action
from rl_components.env_protocol import EnvSpec


def _spec(
    action_shape: tuple[int, ...] = (2,),
    low: float | None = -1.0,
    high: float | None = 1.0,
) -> EnvSpec:
    if low is None or high is None:
        return EnvSpec(
            id="fake-continuous",
            observation_shape=(3,),
            action_shape=action_shape,
            action_dtype=jnp.float32,
        )
    return EnvSpec(
        id="fake-continuous",
        observation_shape=(3,),
        action_shape=action_shape,
        action_dtype=jnp.float32,
        action_low=jnp.full(action_shape, low, dtype=jnp.float32),
        action_high=jnp.full(action_shape, high, dtype=jnp.float32),
    )


@pytest.mark.parametrize("action_dim", [1, 2, 7])
def test_action_dim_is_the_length_of_the_action_vector(action_dim: int) -> None:
    assert continuous_action_dim(_spec((action_dim,))) == action_dim


def test_a_spec_without_declared_bounds_is_accepted() -> None:
    """There is nothing to disagree with, so the agent is left to trust the environment."""
    assert continuous_action_dim(_spec((2,), low=None, high=None)) == 2


def test_a_discrete_spec_is_rejected() -> None:
    spec = EnvSpec(id="fake-discrete", observation_shape=(3,), action_shape=(), num_actions=4)
    with pytest.raises(ValueError, match="continuous action space"):
        continuous_action_dim(spec)


def test_a_higher_rank_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        continuous_action_dim(_spec((2, 3)))


@pytest.mark.parametrize(("low", "high"), [(-2.0, 2.0), (0.0, 1.0), (-1.0, 2.0)])
def test_unnormalized_bounds_are_rejected(low: float, high: float) -> None:
    """The failure this replaces is silent: a squashed policy simply never leaves [-1, 1]."""
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        continuous_action_dim(_spec((2,), low=low, high=high))


def test_exploration_action_stays_inside_the_box() -> None:
    action = uniform_exploration_action(jax.random.key(0), (256,), jnp.float32)
    assert action.shape == (256,)
    assert action.dtype == jnp.float32
    assert bool(jnp.all(action >= -1.0)) and bool(jnp.all(action <= 1.0))
    assert float(jnp.min(action)) < 0.0 < float(jnp.max(action))
