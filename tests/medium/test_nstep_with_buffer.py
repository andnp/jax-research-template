"""Medium (integration) test: n-step returns applied to buffer data.

Fills a buffer with a known trajectory, extracts it, applies compute_nstep_returns,
and verifies correct discounted returns.
Target duration: < 1s.
"""

from __future__ import annotations

from typing import NamedTuple

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.nstep import compute_nstep_returns
from jax_replay.uniform import add, init_buffer


class Transition(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class TestNstepWithBuffer:
    def test_nstep_on_buffer_trajectory(self) -> None:
        proto = Transition(obs=jnp.zeros((2,)), reward=jnp.zeros(()), done=jnp.zeros(()))
        state = init_buffer(proto, capacity=16)

        # Store a known trajectory: rewards [1, 2, 3, 4, 5], no dones
        for i in range(5):
            t = Transition(
                obs=jnp.full(2, float(i)),
                reward=jnp.float32(i + 1),
                done=jnp.float32(0.0),
            )
            state = add(state, t)

        # Extract rewards and dones from buffer (first 5 entries)
        rewards = state.data["1"][:5]  # reward is leaf index 1
        dones = state.data["2"][:5]  # done is leaf index 2

        nstep_r, _, _ = compute_nstep_returns(rewards, dones, gamma=0.9, n=3)

        # Manual: R_0 = 1 + 0.9*2 + 0.81*3 = 1 + 1.8 + 2.43 = 5.23
        assert jnp.allclose(nstep_r[0], 5.23, atol=1e-4)
        # R_3 = 4 + 0.9*5 = 8.5
        assert jnp.allclose(nstep_r[3], 8.5, atol=1e-4)
        # R_4 = 5 (only 1 step left)
        assert jnp.allclose(nstep_r[4], 5.0, atol=1e-4)

    def test_nstep_with_episode_boundary_in_buffer(self) -> None:
        proto = Transition(obs=jnp.zeros((2,)), reward=jnp.zeros(()), done=jnp.zeros(()))
        state = init_buffer(proto, capacity=16)

        rewards_list = [1.0, 2.0, 3.0, 4.0]
        dones_list = [0.0, 1.0, 0.0, 0.0]  # episode ends at t=1

        for r, d in zip(rewards_list, dones_list, strict=True):
            t = Transition(obs=jnp.zeros(2), reward=jnp.float32(r), done=jnp.float32(d))
            state = add(state, t)

        rewards = state.data["1"][:4]
        dones = state.data["2"][:4]

        nstep_r, nstep_dones, _ = compute_nstep_returns(rewards, dones, gamma=0.9, n=3)

        # R_0: 1 + 0.9*2 + 0 (done at t=1 blocks t=2) = 2.8
        assert jnp.allclose(nstep_r[0], 2.8, atol=1e-4)
        assert float(nstep_dones[0]) == 1.0

    def test_nstep_jit_compiled(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        dones = jnp.zeros(4)

        @jax.jit
        def _compute(rewards, dones):
            return compute_nstep_returns(rewards, dones, gamma=0.99, n=2)

        nstep_r, _, _ = _compute(rewards, dones)
        assert nstep_r.shape == (4,)
