"""Execution owners: the process whose death means a live execution will never finish (ADR 014)."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_LOCAL_PREFIX = "local"
# Fields 1 and 2 (pid, comm) precede the closing parenthesis; start time is field 22.
_START_TICKS_FIELD = 22
_FIELDS_BEFORE_STATE = 2


class OwnerState(StrEnum):
    ALIVE = "ALIVE"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LocalOwner:
    hostname: str
    boot_id: str
    pid: int
    start_ticks: int


def format_local_owner(owner: LocalOwner) -> str:
    return f"{_LOCAL_PREFIX}:{owner.hostname}:{owner.boot_id}:{owner.pid}:{owner.start_ticks}"


def parse_local_owner(owner: str) -> LocalOwner | None:
    """Decode a ``local:`` owner, or None when ``owner`` is not one."""
    prefix, _, rest = owner.partition(":")
    parts = rest.split(":")
    if prefix != _LOCAL_PREFIX or len(parts) != 4 or not (parts[2].isdigit() and parts[3].isdigit()):
        return None
    return LocalOwner(parts[0], parts[1], int(parts[2]), int(parts[3]))


def parse_start_ticks(stat: str) -> int:
    """Read the process start time from ``/proc/<pid>/stat``; the comm field may hold spaces and parentheses."""
    fields_after_comm = stat.rpartition(")")[2].split()
    return int(fields_after_comm[_START_TICKS_FIELD - _FIELDS_BEFORE_STATE - 1])


def decide_local_owner_state(owner: LocalOwner, hostname: str, boot_id: str, observed_ticks: int | None) -> OwnerState:
    if owner.hostname != hostname:
        return OwnerState.UNKNOWN
    if owner.boot_id != boot_id:
        return OwnerState.DEAD
    return OwnerState.ALIVE if observed_ticks == owner.start_ticks else OwnerState.DEAD


def _read_start_ticks(pid: int) -> int | None:
    try:
        return parse_start_ticks(Path(f"/proc/{pid}/stat").read_text())
    except FileNotFoundError:
        return None


def local_owner(pid: int | None = None) -> str:
    """Describe a process on this host and boot; defaults to the current one."""
    pid = os.getpid() if pid is None else pid
    start_ticks = _read_start_ticks(pid)
    if start_ticks is None:
        raise ProcessLookupError(f"No process {pid} to own an execution.")
    owner = LocalOwner(socket.gethostname(), _BOOT_ID_PATH.read_text().strip(), pid, start_ticks)
    return format_local_owner(owner)


def local_owner_state(owner: str) -> OwnerState:
    """Report whether a ``local:`` owner is still running; owners of any other kind are UNKNOWN."""
    parsed = parse_local_owner(owner)
    if parsed is None:
        return OwnerState.UNKNOWN
    return decide_local_owner_state(
        parsed, socket.gethostname(), _BOOT_ID_PATH.read_text().strip(), _read_start_ticks(parsed.pid)
    )
