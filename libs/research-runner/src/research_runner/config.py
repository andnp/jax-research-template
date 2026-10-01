"""Config fields that may vary between the runs of one batch."""

import dataclasses

_TRACED = "research_runner.traced"


def traced[T](default: T) -> T:
    """Mark a config field as vmappable: a run reads it only in JAX arithmetic."""
    return dataclasses.field(default=default, metadata={_TRACED: True})
