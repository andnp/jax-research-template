"""Pure scheduling policy for the parallel experiment runner: threads per worker and memory admission."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


def worker_threads(
    budget: int,
    running: Sequence[int],
    waiting_groups: int,
    *,
    max_threads: int = 8,
    fixed: int | None = None,
) -> int | None:
    """Threads for the next worker, spreading the unused budget over waiting groups; None if it cannot start."""
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if fixed is not None and fixed < 1:
        raise ValueError(f"fixed threads must be at least 1, got {fixed}")
    free = budget - sum(running)
    if free < 1 or waiting_groups < 1:
        return None
    if fixed is not None:
        return fixed if fixed <= free else None
    return max(1, min(math.ceil(free / waiting_groups), max_threads, free))


@dataclass(frozen=True)
class RunningWorker:
    projection_bytes: int
    rss_bytes: int


class Admission(StrEnum):
    ADMIT = "admit"
    WAIT = "wait"
    IMPOSSIBLE = "impossible"


def admit_worker(
    mem_available: int,
    reserve: int,
    projection: int,
    running: Sequence[RunningWorker],
) -> Admission:
    """Admit when available memory minus the reserve covers the new projection plus running workers' remaining growth."""
    growth = sum(max(0, worker.projection_bytes - worker.rss_bytes) for worker in running)
    if mem_available - reserve >= projection + growth:
        return Admission.ADMIT
    return Admission.WAIT if running else Admission.IMPOSSIBLE


def worker_projection(reported_peaks: Sequence[int], default: int) -> int:
    """Largest peak RSS reported by a completed worker, else the default."""
    return max(reported_peaks, default=default)


@dataclass(frozen=True)
class MemInfo:
    available_bytes: int
    total_bytes: int


def parse_meminfo(text: bytes) -> MemInfo:
    """Read MemAvailable and MemTotal (reported in kB) from /proc/meminfo contents."""
    values: dict[str, int] = {}
    for line in text.decode().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemAvailable", "MemTotal"):
            values[key] = int(rest.split()[0]) * 1024
    missing = {"MemAvailable", "MemTotal"} - values.keys()
    if missing:
        raise ValueError(f"meminfo is missing {', '.join(sorted(missing))}")
    return MemInfo(available_bytes=values["MemAvailable"], total_bytes=values["MemTotal"])


def parse_vm_rss(text: bytes) -> int:
    """Resident set size in bytes from /proc/<pid>/status contents; 0 for a process that has exited."""
    for line in text.decode().splitlines():
        key, _, rest = line.partition(":")
        if key == "VmRSS":
            return int(rest.split()[0]) * 1024
    return 0
