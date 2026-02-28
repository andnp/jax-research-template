"""Medium (integration) tests for jax_replay PER buffer under JIT.

Verifies JIT compilation of PER add/sample/update cycle and sampling distribution.
Target duration: < 1s.
"""

from __future__ import annotations

from typing import NamedTuple

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.per import init_per_buffer, per_add, per_sample, per_update_priorities


class Transition(NamedTuple):
    obs: jax.Array
    reward: jax.Array


def _proto() -> Transition:
    return Transition(obs=jnp.zeros((2,)), reward=jnp.zeros(()))


class TestPerJit:
    def test_add_sample_update_under_jit(self) -> None:
        proto = _proto()
        state = init_per_buffer(proto, capacity=16)

        @jax.jit
        def _step(state, key):
            t = Transition(obs=jnp.ones(2), reward=jnp.float32(1.0))
            state = per_add(state, t, alpha=0.6)
            batch, weights, indices = per_sample(state, key, batch_size=4, beta=0.4, prototype=proto)
            td_errors = jnp.ones(4) * 0.5
            state = per_update_priorities(state, indices, td_errors, alpha=0.6)
            return state, weights

        key = jax.random.key(0)
        for _i in range(50):
            key, subkey = jax.random.split(key)
            state, weights = _step(state, subkey)

        assert int(state.count) == 16
        assert weights.shape == (4,)


class TestPerSamplingDistribution:
    def test_high_priority_sampled_more_often(self) -> None:
        proto = _proto()
        capacity = 4
        state = init_per_buffer(proto, capacity=capacity)

        # Add 4 transitions with equal priority
        for i in range(4):
            state = per_add(state, Transition(obs=jnp.full(2, float(i)), reward=jnp.float32(i)), alpha=1.0)

        # Set priorities: index 0 gets 100x, others stay at 1.0
        indices = jnp.array([0, 1, 2, 3], dtype=jnp.uint32)
        td_errors = jnp.array([100.0, 1.0, 1.0, 1.0])
        state = per_update_priorities(state, indices, td_errors, alpha=1.0, epsilon=0.0)

        # Sample 10000 times and count how often index 0 is picked
        num_samples = 10_000
        batch_size = 4

        @jax.jit
        def _sample(state, key):
            _, _, idx = per_sample(state, key, batch_size=batch_size, beta=1.0, prototype=proto)
            return idx

        key = jax.random.key(42)
        counts = jnp.zeros(capacity)
        for _i in range(num_samples):
            key, subkey = jax.random.split(key)
            idx = _sample(state, subkey)
            counts = counts + jnp.bincount(idx.astype(jnp.int32), length=capacity)

        total = counts.sum()
        freq_0 = counts[0] / total
        # Priority fraction for index 0: 100 / (100+1+1+1) = 0.9709
        # Allow some variance (stratified sampling reduces it)
        assert freq_0 > 0.90, f"Expected index 0 to be sampled > 90% of the time, got {freq_0:.3f}"
