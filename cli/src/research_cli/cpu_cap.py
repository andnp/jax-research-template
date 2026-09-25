from __future__ import annotations

import os
import sys
import warnings


def apply_cpu_cap() -> None:
    if sys.platform != "linux":
        return

    raw_limit = os.environ.get("RESEARCH_MAX_CPUS")
    if raw_limit is None:
        return
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 0
    if limit <= 0:
        warnings.warn(
            f"Ignoring invalid RESEARCH_MAX_CPUS={raw_limit!r}; expected a positive integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    original_affinity = os.sched_getaffinity(0)
    if limit >= len(original_affinity):
        return

    try:
        os.sched_setaffinity(0, set(sorted(original_affinity)[:limit]))
        # Initialize the backend while its worker pool can see only the capped CPU set.
        import jax.numpy as jnp

        jnp.zeros(1).block_until_ready()
    finally:
        for thread_id in os.listdir("/proc/self/task"):
            os.sched_setaffinity(int(thread_id), original_affinity)
