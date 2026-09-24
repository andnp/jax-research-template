"""Declare an experiment's primary analysis beside its spec, before any results exist."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from research_analysis.dataset.types import SweepValue


@dataclass(frozen=True)
class BakeoffAnalysis:
    """Compare ``conditions`` with ``bakeoff_report``; ``pairs`` of None plans every pair."""

    name: str
    conditions: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...] | None = None
    block_by: tuple[str, ...] = ()
    select_champion_over: tuple[str, ...] | None = None
    min_seeds: int = 5


@dataclass(frozen=True)
class BestHypersAnalysis:
    """Select the bias-corrected champion configuration, separately per ``group_by`` combination."""

    name: str
    group_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class HypersDifferenceAnalysis:
    """Paired difference between the champions of two values of ``group_axis``."""

    name: str
    group_axis: str
    minuend: SweepValue
    subtrahend: SweepValue


@dataclass(frozen=True)
class SliceHypersAnalysis:
    """Sensitivity of the champion to one swept axis."""

    name: str
    vary: str
    group_by: tuple[str, ...] = ()


type Analysis = BakeoffAnalysis | BestHypersAnalysis | HypersDifferenceAnalysis | SliceHypersAnalysis


def against_baseline(conditions: Sequence[str], baseline: str) -> tuple[tuple[str, str], ...]:
    """Plan each condition minus ``baseline``, and nothing else."""
    return tuple((condition, baseline) for condition in conditions if condition != baseline)
