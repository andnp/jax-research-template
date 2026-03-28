"""Medium (integration) tests for jax_replay uniform buffer under JIT.

Verifies JIT compilation, buffer state invariants over many steps, and vmap.
Target duration: < 1s.
"""

from __future__ import annotations

from typing import NamedTuple

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
from jax_replay.uniform import add, init_buffer, sample


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array


def _proto() -> Transition:
    return Transition(obs=jnp.zeros((4,)), action=jnp.zeros((), dtype=jnp.int32), reward=jnp.zeros(()))


def _batch_obs_shape(batch: Transition) -> tuple[int, ...]:
    return batch.obs.shape


class TestUniformJit:
    def test_add_sample_loop_under_jit(self) -> None:
        proto = _proto()
        capacity = 64
        state = init_buffer(proto, capacity)
        batch = Transition(obs=jnp.zeros((8, 4)), action=jnp.zeros((8,), dtype=jnp.int32), reward=jnp.zeros((8,)))

        @jax.jit
        def _step(state, key):
            t = Transition(obs=jnp.ones(4), action=jnp.int32(1), reward=jnp.float32(1.0))
            state = add(state, t)
            batch = sample(state, key, batch_size=8, prototype=proto)
            return state, batch

        key = jax.random.key(0)
        for _i in range(1000):
            key, subkey = jax.random.split(key)
            state, batch = _step(state, subkey)

        assert int(state.pointer) == 1000
        assert int(state.count) == capacity  # capped at capacity
        assert _batch_obs_shape(batch) == (8, 4)

    def test_pointer_wraps_correctly_after_many_steps(self) -> None:
        proto = _proto()
        capacity = 16
        state = init_buffer(proto, capacity)

        @jax.jit
        def _add_one(state):
            t = Transition(obs=jnp.ones(4), action=jnp.int32(0), reward=jnp.float32(0.0))
            return add(state, t)

        for _i in range(100):
            state = _add_one(state)

        assert int(state.pointer) == 100
        # Effective idx = 100 % 16 = 4 → next write goes to slot 4
        assert int(state.count) == capacity

    def test_vmap_over_independent_buffers(self) -> None:
        proto = _proto()
        capacity = 8

        def _run_single(seed):
            state = init_buffer(proto, capacity)
            key = jax.random.key(seed)
            t = Transition(
                obs=jnp.full(4, seed.astype(jnp.float32)),
                action=jnp.int32(seed),
                reward=seed.astype(jnp.float32),
            )
            state = add(state, t)
            batch = sample(state, key, batch_size=1, prototype=proto)
            return batch.obs[0, 0]  # first element of obs

        seeds = jnp.arange(4, dtype=jnp.uint32)
        results = jax.vmap(_run_single)(seeds)
        # Each buffer had obs filled with its seed value
        assert jnp.allclose(results, jnp.array([0.0, 1.0, 2.0, 3.0]))
