"""Medium integration tests for Collector inside jit/scan/vmap.

These tests verify that write() and eval() work correctly when compiled
with jax.jit and called from inside jax.lax.scan (the primary use-case
from the spec), and that the vmap axis is captured correctly.

Target duration: < 1 s per test (involves JIT compilation but no full runs).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from research_instrument.collector import Collector, InMemoryBackend


class TestWriteInsideScan:
    """write() must stream correct frames from inside jax.lax.scan."""

    def test_write_called_from_jit(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend)

        @jax.jit
        def step(value: jax.Array, global_step: jax.Array) -> None:
            c.write("reward", value, global_step)

        step(jnp.float32(5.0), jnp.int32(10))
        assert len(backend.records) == 1
        assert abs(backend.records[0].value - 5.0) < 1e-4
        assert backend.records[0].global_step == 10

    def test_write_called_from_scan(self) -> None:
        """write() inside scan should emit one frame per iteration."""
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend)

        def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
            c.write("reward", jnp.float32(step), step)
            return carry, None

        jax.lax.scan(body, jnp.int32(0), jnp.arange(5, dtype=jnp.int32))
        assert len(backend.records) == 5
        steps = sorted(f.global_step for f in backend.records)
        assert steps == [0, 1, 2, 3, 4]

    def test_write_not_whitelisted_no_frames_in_scan(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset(), backend)  # empty whitelist

        def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
            c.write("reward", jnp.float32(1.0), step)
            return carry, None

        jax.lax.scan(body, jnp.int32(0), jnp.arange(10, dtype=jnp.int32))
        assert backend.records == []

    def test_write_jit_compiled_scan(self) -> None:
        """Verify the full jit(scan(...)) path matches per-step expectations."""
        backend = InMemoryBackend()
        c = Collector(frozenset({"loss"}), backend)

        def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
            c.write("loss", jnp.float32(0.1) * step, step)
            return carry, None

        jit_scan = jax.jit(lambda: jax.lax.scan(body, jnp.int32(0), jnp.arange(4, dtype=jnp.int32)))
        jit_scan()
        assert len(backend.records) == 4


class TestEvalInsideScan:
    """eval() schedule must fire exactly at global_step % every == 0."""

    def test_eval_schedule_in_scan(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"weight_norm"}), backend)

        def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
            c.eval("weight_norm", lambda: jnp.float32(1.0), step, every=3)
            return carry, None

        # Steps 0..8; schedule fires at 0, 3, 6 → 3 frames
        jax.lax.scan(body, jnp.int32(0), jnp.arange(9, dtype=jnp.int32))
        assert len(backend.records) == 3
        fired = sorted(f.global_step for f in backend.records)
        assert fired == [0, 3, 6]

    def test_eval_not_whitelisted_skips_in_scan(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset(), backend)

        def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
            c.eval("weight_norm", lambda: jnp.float32(1.0), step, every=1)
            return carry, None

        jax.lax.scan(body, jnp.int32(0), jnp.arange(5, dtype=jnp.int32))
        assert backend.records == []


class TestVmapAxisCapture:
    """seed_id should reflect the vmap axis index when axis_name is set."""

    def test_seed_id_zero_outside_vmap(self) -> None:
        """Outside a named vmap, seed_id must default to 0."""
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend, vmap_axis_name="seed")

        @jax.jit
        def step(v: jax.Array) -> None:
            c.write("reward", v, jnp.int32(0))

        step(jnp.float32(1.0))
        assert backend.records[0].seed_id == 0

    def test_seed_id_captured_in_named_vmap(self) -> None:
        """Inside vmap(axis_name='seed'), seed_id must match the batch index."""
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend, vmap_axis_name="seed")

        def single(v: jax.Array) -> None:
            c.write("reward", v, jnp.int32(0))

        batch = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
        jax.vmap(single, axis_name="seed")(batch)

        assert len(backend.records) == 3
        seed_ids = sorted(f.seed_id for f in backend.records)
        assert seed_ids == [0, 1, 2]

    def test_seed_id_in_vmap_scan(self) -> None:
        """vmap over a scan: each seed emits frames tagged with its index."""
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend, vmap_axis_name="seed")

        def train_single(rng: jax.Array) -> None:
            def body(carry: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
                c.write("reward", jnp.float32(1.0), step)
                return carry, None

            jax.lax.scan(body, rng, jnp.arange(3, dtype=jnp.int32))

        keys = jnp.arange(4, dtype=jnp.int32)
        jax.vmap(train_single, axis_name="seed")(keys)

        # 4 seeds × 3 steps = 12 frames
        assert len(backend.records) == 12
        # Every seed_id 0..3 must appear exactly 3 times
        from collections import Counter

        counts = Counter(f.seed_id for f in backend.records)
        assert dict(counts) == {0: 3, 1: 3, 2: 3, 3: 3}
