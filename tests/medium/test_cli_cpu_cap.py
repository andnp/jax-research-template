from __future__ import annotations

import os
import subprocess
import sys

import pytest

_THREAD_PROBE = """
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from research_cli.cpu_cap import apply_cpu_cap
original = sorted(os.sched_getaffinity(0))
apply_cpu_cap()
import jax.numpy as jnp
value = jnp.ones((256, 256), dtype=jnp.float32)
(value @ value).block_until_ready()
print(" ".join(map(str, original)))
for thread_id in sorted(os.listdir("/proc/self/task"), key=int):
    print(" ".join(map(str, sorted(os.sched_getaffinity(int(thread_id))))))
"""

_NOOP_PROBE = """
import os
import sys
import warnings
from research_cli.cpu_cap import apply_cpu_cap
original = os.sched_getaffinity(0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    apply_cpu_cap()
assert os.sched_getaffinity(0) == original
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
expected_count = int(os.environ["EXPECTED_WARNING_COUNT"])
assert len(caught) == expected_count
if caught:
    assert "RESEARCH_MAX_CPUS" in str(caught[0].message)
    assert "positive integer" in str(caught[0].message)
"""


def _run_thread_probe(max_cpus: str | None) -> tuple[set[int], list[set[int]]]:
    environment = os.environ.copy()
    environment.pop("RESEARCH_MAX_CPUS", None)
    if max_cpus is not None:
        environment["RESEARCH_MAX_CPUS"] = max_cpus
    completed = subprocess.run(
        [sys.executable, "-c", _THREAD_PROBE],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
        timeout=120,
    )
    lines = completed.stdout.splitlines()
    original_affinity = set(map(int, lines[0].split()))
    thread_affinities = [set(map(int, line.split())) for line in lines[1:]]
    return original_affinity, thread_affinities


def _run_noop_probe(max_cpus: str | None, expected_warning_count: int) -> None:
    environment = os.environ.copy()
    environment.pop("RESEARCH_MAX_CPUS", None)
    if max_cpus is not None:
        environment["RESEARCH_MAX_CPUS"] = max_cpus
    environment["EXPECTED_WARNING_COUNT"] = str(expected_warning_count)
    subprocess.run(
        [sys.executable, "-c", _NOOP_PROBE],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
        timeout=30,
    )


def test_cpu_pool_is_capped_while_worker_threads_keep_full_affinity() -> None:
    """Cap backend thread count while leaving workers free across all allowed CPUs.

    The subprocess initializes the backend under a four-CPU mask, then checks each worker.
    """
    if sys.platform != "linux":
        pytest.skip("thread-affinity cap is Linux-only")
    if len(os.sched_getaffinity(0)) < 8:
        pytest.skip("requires at least eight available CPUs")

    uncapped_original, uncapped_threads = _run_thread_probe(None)
    capped_original, capped_threads = _run_thread_probe("4")

    assert capped_original == uncapped_original
    assert len(capped_threads) < len(uncapped_threads)
    assert capped_threads
    assert all(affinity == capped_original for affinity in capped_threads)


@pytest.mark.parametrize("max_cpus", ["four", "0", "-4"])
def test_invalid_cpu_cap_warns_without_changing_affinity(max_cpus: str) -> None:
    """Malformed and non-positive caps warn before backend initialization.

    Invalid values must not change the calling thread's affinity.
    """
    if sys.platform != "linux":
        pytest.skip("invalid-cap warnings are Linux-only")

    _run_noop_probe(max_cpus, expected_warning_count=1)


def test_unset_cpu_cap_leaves_affinity_and_backend_untouched() -> None:
    """An unset cap leaves process setup untouched.

    The no-op path must not import the CPU backend.
    """
    if sys.platform != "linux":
        pytest.skip("thread-affinity cap is Linux-only")

    _run_noop_probe(None, expected_warning_count=0)
