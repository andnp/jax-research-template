"""JAX-native metrics collector.

Instruments training code that runs inside jit/vmap/scan without passing
logging objects through the call stack.  Any metric not present in the
whitelist becomes a pure Python-level no-op — no JAX ops are emitted.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MetricFrame:
    """A single recorded data point produced by one callback invocation."""

    name: str
    value: float
    global_step: int
    seed_id: int


class StorageBackend(Protocol):
    """Pluggable persistence layer for collected metrics.

    Implementations must be thread-safe because write_batch is called from
    a background thread spawned by jax.debug.callback.
    """

    def write_batch(self, frames: list[MetricFrame]) -> None:
        """Persist a batch of metric frames."""
        ...

    def flush(self) -> None:
        """Block until all in-flight writes are durable."""
        ...

    def close(self) -> None:
        """Finalize the connection and release resources."""
        ...


class InMemoryBackend:
    """Thread-safe in-memory storage backend.

    Useful for unit tests and interactive experimentation.
    All frames are accessible via the ``records`` attribute after a run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[MetricFrame] = []

    def write_batch(self, frames: list[MetricFrame]) -> None:
        with self._lock:
            self.records.extend(frames)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class Collector:
    """JAX-native, whitelist-driven metrics collector.

    The collector is designed to be called from inside JIT-compiled functions,
    including ``jax.lax.scan`` loops and ``vmap``-parallelised training runs.

    Whitelist semantics
    -------------------
    The whitelist is checked at JIT trace time (Python level).  A ``write``
    or ``eval`` call for a metric that is *not* in the whitelist returns
    immediately without emitting any JAX operations — making it a true
    zero-cost no-op after compilation.

    vmap axis capture
    -----------------
    When the caller is inside a ``jax.vmap`` with ``axis_name`` matching
    ``vmap_axis_name`` (default ``"seed"``), the collector automatically
    retrieves the current axis index via ``jax.lax.axis_index`` and attaches
    it to every frame.  Outside a named vmap the seed id defaults to 0.

    Example
    -------
    >>> collector = Collector(frozenset({"reward", "loss"}))
    >>> # inside a jax.lax.scan body:
    >>> collector.write("reward", reward_scalar, global_step)
    >>> collector.eval("loss", compute_loss, global_step, every=100)
    """

    def __init__(
        self,
        whitelist: frozenset[str],
        backend: StorageBackend | None = None,
        vmap_axis_name: str = "seed",
    ) -> None:
        """Initialise the collector.

        Args:
            whitelist: Set of metric names that will actually be recorded.
                       Any name not in this set is a no-op.
            backend: Storage backend to receive frames.  Defaults to
                     ``InMemoryBackend`` when *None*.
            vmap_axis_name: The vmap axis name used to identify parallel seeds.
                            Must match the ``axis_name`` passed to ``jax.vmap``.
        """
        self._whitelist = whitelist
        self._backend: StorageBackend = backend if backend is not None else InMemoryBackend()
        self._axis_name = vmap_axis_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_seed_id(self) -> jax.Array:
        """Return the current vmap axis index, falling back to 0.

        This is called at JIT trace time.  The try/except catches the
        ``NameError`` raised by JAX when ``axis_name`` is not in scope
        (i.e. the function is not running inside a matching named vmap).
        """
        try:
            return jax.lax.axis_index(self._axis_name)
        except Exception:  # noqa: BLE001  (NameError from JAX internals)
            return jnp.int32(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, name: str, value: jax.Array, global_step: jax.Array) -> None:
        """Record a scalar metric value.

        This is a zero-cost no-op at trace time when *name* is not in the
        whitelist.  Otherwise it schedules a ``jax.debug.callback`` that
        delivers the concrete value to the storage backend on the Python
        thread after each step.

        Args:
            name: Metric name.  Must be in the whitelist to have any effect.
            value: The JAX array scalar to record.
            global_step: Monotonically increasing step counter (JAX int32/int64).
        """
        if name not in self._whitelist:
            return

        seed_id = self._get_seed_id()

        # Capture `name` in a closure; only JAX arrays cross the callback
        # boundary — Python strings cannot be passed as callback arguments.
        def _callback(v: np.ndarray, step: np.ndarray, sid: np.ndarray) -> None:
            frame = MetricFrame(name=name, value=float(v), global_step=int(step), seed_id=int(sid))
            self._backend.write_batch([frame])

        jax.debug.callback(_callback, value, global_step, seed_id)

    def eval(
        self,
        name: str,
        fn: Callable[[], jax.Array],
        global_step: jax.Array,
        every: int,
    ) -> None:
        """Evaluate an expensive metric on a deterministic schedule.

        The callable *fn* is compiled by JAX but only **executed** at runtime
        when both conditions hold:

        1. *name* is in the whitelist (checked at trace time — Python level).
        2. ``global_step % every == 0`` (checked at runtime via
           ``jax.lax.cond``).

        ``jax.lax.cond`` compiles both branches but executes only one, so the
        computation inside *fn* is skipped entirely on the accelerator when
        the schedule condition is false.

        Args:
            name: Metric name.  Must be in the whitelist to have any effect.
            fn: Zero-argument callable that returns the metric value as a JAX
                array.  It is only *executed* when the schedule fires.
            global_step: Monotonically increasing step counter.
            every: Evaluation frequency in steps.  *fn* runs when
                   ``global_step % every == 0``.
        """
        if name not in self._whitelist:
            return

        seed_id = self._get_seed_id()
        should_run = (global_step % every) == 0

        def _callback(v: np.ndarray, step: np.ndarray, sid: np.ndarray) -> None:
            frame = MetricFrame(name=name, value=float(v), global_step=int(step), seed_id=int(sid))
            self._backend.write_batch([frame])

        # Both branches must return the same pytree structure; we use a dummy
        # int32 sentinel so the return value can be discarded by the caller.
        def _true_fn(_: None) -> jax.Array:
            result = fn()
            jax.debug.callback(_callback, result, global_step, seed_id)
            return jnp.int32(0)

        def _false_fn(_: None) -> jax.Array:
            return jnp.int32(0)

        jax.lax.cond(should_run, _true_fn, _false_fn, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# A convenience singleton that experiments can import directly.
# Replace with a properly configured instance at the start of a run.
_default_whitelist: frozenset[str] = frozenset()

#: Global collector instance.  Configure by calling ``configure()``.
collector: Collector = Collector(_default_whitelist)


def configure(
    whitelist: frozenset[str],
    backend: StorageBackend | None = None,
    vmap_axis_name: str = "seed",
) -> Collector:
    """Replace the global collector with a new configured instance.

    Call this once before compiling your training function, typically in the
    experiment entry-point, after loading the metric whitelist from the DB.

    Args:
        whitelist: Set of metric names to record.
        backend: Storage backend (defaults to ``InMemoryBackend``).
        vmap_axis_name: vmap axis name used for seed identification.

    Returns:
        The newly created :class:`Collector` (also available as
        ``research_instrument.collector``).
    """
    global collector  # noqa: PLW0603
    collector = Collector(whitelist, backend, vmap_axis_name)
    return collector


__all__ = [
    "Collector",
    "InMemoryBackend",
    "MetricFrame",
    "StorageBackend",
    "collector",
    "configure",
]

