"""Small (unit) tests for jax_replay n-step return computation.

Verifies math for known reward sequences, episode boundaries, and identity cases.
Target duration: << 1 ms per test (no JIT).
"""

from __future__ import annotations

import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.nstep import compute_nstep_returns


class TestNstepIdentities:
    def test_n1_returns_immediate_reward(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0])
        dones = jnp.array([0.0, 0.0, 0.0])
        nstep_r, _, _ = compute_nstep_returns(rewards, dones, gamma=0.99, n=1)
        assert jnp.allclose(nstep_r, rewards)

    def test_gamma_zero_returns_immediate_reward(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        dones = jnp.array([0.0, 0.0, 0.0, 0.0])
        nstep_r, _, _ = compute_nstep_returns(rewards, dones, gamma=0.0, n=3)
        assert jnp.allclose(nstep_r, rewards)


class TestNstepMath:
    def test_n2_no_dones(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        dones = jnp.array([0.0, 0.0, 0.0, 0.0])
        gamma = 0.9
        nstep_r, _, _ = compute_nstep_returns(rewards, dones, gamma=gamma, n=2)
        # R_0 = 1.0 + 0.9*2.0 = 2.8
        # R_1 = 2.0 + 0.9*3.0 = 4.7
        # R_2 = 3.0 + 0.9*4.0 = 6.6
        # R_3 = 4.0 (only 1 step available, padded with 0)
        expected = jnp.array([2.8, 4.7, 6.6, 4.0])
        assert jnp.allclose(nstep_r, expected, atol=1e-5)

    def test_n3_no_dones(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dones = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        gamma = 0.5
        nstep_r, _, _ = compute_nstep_returns(rewards, dones, gamma=gamma, n=3)
        # R_0 = 1 + 0.5*2 + 0.25*3 = 1 + 1 + 0.75 = 2.75
        # R_1 = 2 + 0.5*3 + 0.25*4 = 2 + 1.5 + 1.0 = 4.5
        # R_2 = 3 + 0.5*4 + 0.25*5 = 3 + 2 + 1.25 = 6.25
        # R_3 = 4 + 0.5*5 = 6.5
        # R_4 = 5
        expected = jnp.array([2.75, 4.5, 6.25, 6.5, 5.0])
        assert jnp.allclose(nstep_r, expected, atol=1e-5)


class TestNstepEpisodeBoundaries:
    def test_done_truncates_nstep_window(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        dones = jnp.array([0.0, 1.0, 0.0, 0.0])
        gamma = 0.9
        nstep_r, nstep_dones, _ = compute_nstep_returns(rewards, dones, gamma=gamma, n=3)
        # R_0: window is [1, 2, 3] but done at t=1, so:
        #   k=0: gamma^0 * 1.0 * 1.0 = 1.0 (not_done_before[0] = 1.0)
        #   k=1: gamma^1 * 2.0 * (1-d_0) = 0.9 * 2.0 * 1.0 = 1.8
        #   k=2: gamma^2 * 3.0 * (1-d_0)*(1-d_1) = 0.81 * 3.0 * 0.0 = 0.0
        # R_0 = 1.0 + 1.8 + 0.0 = 2.8
        assert jnp.allclose(nstep_r[0], 2.8, atol=1e-5)
        assert float(nstep_dones[0]) == 1.0  # done within window

    def test_done_at_boundary_sets_nstep_done(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0])
        dones = jnp.array([0.0, 0.0, 1.0])
        _, nstep_dones, _ = compute_nstep_returns(rewards, dones, gamma=0.99, n=3)
        # All windows that include t=2 should have nstep_done=1.0
        assert float(nstep_dones[0]) == 1.0
        assert float(nstep_dones[1]) == 1.0
        assert float(nstep_dones[2]) == 1.0


class TestNstepBootstrapIndices:
    def test_bootstrap_index_without_dones(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dones = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, bootstrap_idx = compute_nstep_returns(rewards, dones, gamma=0.99, n=2)
        # bootstrap_idx[0] = min(0+2, 4) = 2
        # bootstrap_idx[3] = min(3+2, 4) = 4
        # bootstrap_idx[4] = min(4+2, 4) = 4
        assert int(bootstrap_idx[0]) == 2
        assert int(bootstrap_idx[3]) == 4
        assert int(bootstrap_idx[4]) == 4

    def test_bootstrap_index_with_done(self) -> None:
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        dones = jnp.array([0.0, 1.0, 0.0, 0.0])
        _, _, bootstrap_idx = compute_nstep_returns(rewards, dones, gamma=0.99, n=3)
        # At t=0, first done in window is at position k=1 → bootstrap from t+1 = 1
        assert int(bootstrap_idx[0]) == 1
