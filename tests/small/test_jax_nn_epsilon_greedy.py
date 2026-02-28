"""Small (unit) tests for epsilon_greedy_action."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_nn.heads import epsilon_greedy_action

SEED = 42


class TestEpsilonGreedyDeterministic:
    def test_epsilon_zero_always_greedy(self) -> None:
        q = jnp.array([1.0, 3.0, 2.0])
        for seed in range(20):
            action = epsilon_greedy_action(q, 0.0, key=jax.random.key(seed))
            assert action == 1

    def test_epsilon_one_always_random(self) -> None:
        q = jnp.array([1.0, 3.0, 2.0])
        actions = set()
        for seed in range(100):
            action = epsilon_greedy_action(q, 1.0, key=jax.random.key(seed))
            actions.add(int(action))
        assert len(actions) == 3


class TestEpsilonGreedyShape:
    def test_scalar_output(self) -> None:
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        action = epsilon_greedy_action(q, 0.1, key=jax.random.key(SEED))
        assert action.shape == ()

    def test_batched_q_values(self) -> None:
        q = jnp.ones((8, 4))
        action = epsilon_greedy_action(q, 0.1, key=jax.random.key(SEED))
        assert action.shape == (8,)


class TestEpsilonGreedyRange:
    def test_action_in_range(self) -> None:
        num_actions = 5
        q = jnp.ones((num_actions,))
        for seed in range(50):
            action = epsilon_greedy_action(q, 0.5, key=jax.random.key(seed))
            assert 0 <= int(action) < num_actions


class TestEpsilonGreedyDeterministicKey:
    def test_same_key_same_result(self) -> None:
        q = jnp.array([1.0, 2.0, 3.0])
        a1 = epsilon_greedy_action(q, 0.5, key=jax.random.key(SEED))
        a2 = epsilon_greedy_action(q, 0.5, key=jax.random.key(SEED))
        assert a1 == a2
