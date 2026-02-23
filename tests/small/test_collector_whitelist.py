"""Small (unit) tests for Collector whitelist and no-op semantics.

These tests run in pure Python without JAX compilation, verifying the
trace-time (Python-level) whitelist gate and basic data flow.

Target duration: << 1 ms per test (no JIT, no GPU).
"""

from __future__ import annotations

import jax.numpy as jnp
from research_instrument.collector import Collector, InMemoryBackend, MetricFrame, configure


class TestWhitelistGating:
    """write() and eval() must be true no-ops when name is not whitelisted."""

    def test_write_not_whitelisted_produces_no_frames(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"loss"}), backend)
        # "reward" is not in the whitelist — no frame should be written.
        c.write("reward", jnp.float32(1.0), jnp.int32(0))
        assert backend.records == []

    def test_write_whitelisted_produces_frame(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"reward"}), backend)
        c.write("reward", jnp.float32(3.14), jnp.int32(42))
        assert len(backend.records) == 1
        frame = backend.records[0]
        assert frame.name == "reward"
        assert abs(frame.value - 3.14) < 1e-4
        assert frame.global_step == 42
        assert frame.seed_id == 0

    def test_write_empty_whitelist_produces_no_frames(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset(), backend)
        c.write("reward", jnp.float32(1.0), jnp.int32(0))
        c.write("loss", jnp.float32(0.5), jnp.int32(1))
        assert backend.records == []

    def test_multiple_whitelisted_metrics(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"a", "b"}), backend)
        c.write("a", jnp.float32(1.0), jnp.int32(0))
        c.write("b", jnp.float32(2.0), jnp.int32(1))
        c.write("c", jnp.float32(3.0), jnp.int32(2))  # not whitelisted
        assert len(backend.records) == 2
        names = {f.name for f in backend.records}
        assert names == {"a", "b"}


class TestEvalScheduling:
    """eval() must respect global_step % every == 0 deterministically."""

    def test_eval_not_whitelisted_produces_no_frames(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"loss"}), backend)
        c.eval("weight_norm", lambda: jnp.float32(0.5), jnp.int32(0), every=1)
        assert backend.records == []

    def test_eval_fires_at_step_zero(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"weight_norm"}), backend)
        c.eval("weight_norm", lambda: jnp.float32(9.9), jnp.int32(0), every=100)
        assert len(backend.records) == 1
        assert backend.records[0].name == "weight_norm"
        assert abs(backend.records[0].value - 9.9) < 1e-4

    def test_eval_skips_non_schedule_steps(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"weight_norm"}), backend)
        c.eval("weight_norm", lambda: jnp.float32(1.0), jnp.int32(1), every=100)
        c.eval("weight_norm", lambda: jnp.float32(1.0), jnp.int32(50), every=100)
        c.eval("weight_norm", lambda: jnp.float32(1.0), jnp.int32(99), every=100)
        assert backend.records == []

    def test_eval_fires_at_exact_multiples(self) -> None:
        backend = InMemoryBackend()
        c = Collector(frozenset({"metric"}), backend)
        for step in range(300):
            s = step
            c.eval("metric", lambda s=s: jnp.float32(float(s)), jnp.int32(step), every=100)
        # steps 0, 100, 200 → 3 frames
        assert len(backend.records) == 3
        fired_steps = sorted(f.global_step for f in backend.records)
        assert fired_steps == [0, 100, 200]


class TestMetricFrame:
    """MetricFrame dataclass contract."""

    def test_is_frozen(self) -> None:
        frame = MetricFrame(name="x", value=1.0, global_step=0, seed_id=0)
        try:
            frame.name = "y"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except Exception:
            pass


class TestConfigureSingleton:
    """configure() replaces the module-level singleton."""

    def test_configure_returns_new_collector(self) -> None:
        c = configure(frozenset({"reward"}))
        assert isinstance(c, Collector)
        assert "reward" in c._whitelist

    def test_configure_with_backend(self) -> None:
        backend = InMemoryBackend()
        c = configure(frozenset({"loss"}), backend=backend)
        c.write("loss", jnp.float32(0.1), jnp.int32(5))
        assert len(backend.records) == 1
