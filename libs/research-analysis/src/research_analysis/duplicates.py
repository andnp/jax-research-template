"""Explain why one analysis cell holds more than one run for the same seed."""

from __future__ import annotations

from collections.abc import Iterable


def duplicate_seed_error(cell: str, seed: object, run_ids: Iterable[object]) -> ValueError:
    return ValueError(
        f"{cell} has more than one run for seed {seed} (runs {sorted(str(run_id) for run_id in run_ids)}); this "
        "usually means an undeclared swept axis, an ablation, several environments, or several version sets were "
        "pooled. Declare the axis, add it to group_by/block_by, or narrow the dataset with CohortSelection or "
        "VersionSelection."
    )
