"""Contract tests for describing and probing a local process as an execution owner."""

import os
import socket

import pytest
from experiment_definition.ownership import (
    LocalOwner,
    OwnerState,
    decide_local_owner_state,
    format_local_owner,
    local_owner,
    local_owner_state,
    parse_local_owner,
    parse_start_ticks,
)

_OWNER = LocalOwner("node01", "3f2a9c1e-0000-4000-8000-123456789abc", 4242, 987654)


class TestOwnerEncoding:
    def test_local_owner_round_trips(self) -> None:
        """An encoded owner decodes to the same fields, so reclaim sees what planning recorded."""
        assert format_local_owner(_OWNER) == "local:node01:3f2a9c1e-0000-4000-8000-123456789abc:4242:987654"
        assert parse_local_owner(format_local_owner(_OWNER)) == _OWNER

    @pytest.mark.parametrize("owner", ["slurm:12_3", "local:node01:boot:4242", "local:node01:boot:pid:7", ""])
    def test_other_owners_are_not_local(self, owner: str) -> None:
        """Slurm and malformed owners are not decoded here."""
        assert parse_local_owner(owner) is None


class TestStartTicks:
    def test_plain_stat_line(self) -> None:
        """Start time is field 22 of the stat line."""
        stat = "42 (python) S 1 42 42 0 -1 4194304 100 0 0 0 1 2 0 0 20 0 1 0 555 1000 100 18446744073709551615"
        assert parse_start_ticks(stat) == 555

    def test_comm_with_spaces_and_parentheses(self) -> None:
        """A comm like 'a b) (c' must not shift the fields."""
        stat = "42 (a b) (c) S 1 42 42 0 -1 4194304 100 0 0 0 1 2 0 0 20 0 1 0 555 1000 100 18446744073709551615"
        assert parse_start_ticks(stat) == 555


class TestOwnerStateDecision:
    @pytest.mark.parametrize(
        ("hostname", "boot_id", "observed_ticks", "expected"),
        [
            ("node01", _OWNER.boot_id, _OWNER.start_ticks, OwnerState.ALIVE),
            ("node01", _OWNER.boot_id, _OWNER.start_ticks + 1, OwnerState.DEAD),
            ("node01", _OWNER.boot_id, None, OwnerState.DEAD),
            ("node01", "another-boot", _OWNER.start_ticks, OwnerState.DEAD),
            ("node02", _OWNER.boot_id, _OWNER.start_ticks, OwnerState.UNKNOWN),
            ("node02", "another-boot", None, OwnerState.UNKNOWN),
        ],
    )
    def test_decision(self, hostname: str, boot_id: str, observed_ticks: int | None, expected: OwnerState) -> None:
        """A reused pid, a reboot and a missing process are DEAD; another host cannot be judged."""
        assert decide_local_owner_state(_OWNER, hostname, boot_id, observed_ticks) is expected


class TestLocalProcess:
    def test_current_process_is_alive(self) -> None:
        """The owner of the running process reads back ALIVE."""
        assert local_owner_state(local_owner()) is OwnerState.ALIVE

    def test_owner_names_this_host_and_pid(self) -> None:
        """The encoded owner records the hostname and pid it was built for."""
        parsed = parse_local_owner(local_owner())
        assert parsed is not None
        assert (parsed.hostname, parsed.pid) == (socket.gethostname(), os.getpid())

    def test_wrong_start_ticks_is_dead(self) -> None:
        """A pid that now belongs to a different process is DEAD."""
        parsed = parse_local_owner(local_owner())
        assert parsed is not None
        impostor = format_local_owner(LocalOwner(parsed.hostname, parsed.boot_id, parsed.pid, parsed.start_ticks + 1))
        assert local_owner_state(impostor) is OwnerState.DEAD

    def test_slurm_owner_is_unknown(self) -> None:
        """Owners this module does not understand are UNKNOWN."""
        assert local_owner_state("slurm:12_3") is OwnerState.UNKNOWN
