from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from experiment_definition.experiment import Experiment
from experiment_definition.parameter import ParameterValue

from .config import _build_config, _static_names

if TYPE_CHECKING:
    import jax
    from _typeshed import DataclassInstance


@dataclass(frozen=True, slots=True)
class RunPoint:
    """One logical run's resolved hyperparameters, including its seed."""

    run_id: int
    hyperparameters: Mapping[str, ParameterValue]


@dataclass(frozen=True, slots=True)
class ExperimentSpec[C: DataclassInstance]:
    """An experiment, its base config with named choices, its JAX ``train`` and the results tree it writes.

    ``train(config, key)`` returns ``(state, metrics)``; the runner records the declared
    metrics and discards the state.
    """

    experiment: Experiment
    _: KW_ONLY
    config: C
    train: Callable[[C, jax.Array], tuple[object, Mapping[str, jax.Array]]]
    results_root: Path
    choices: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    max_runs_per_batch: int | None = None
    capture_git: bool = True

    def __post_init__(self) -> None:
        if not self.results_root.is_absolute():
            raise ValueError(
                f"results_root must be absolute, got {self.results_root}. "
                "A relative path resolves against the caller's working directory, "
                "which silently moves the results tree.",
            )

    def sync(self, db_path: Path | str):
        return self.experiment.sync(db_path, static_names=self._static_names)

    def _static_names(self, params: Mapping[str, ParameterValue]) -> frozenset[str]:
        return _static_names(_build_config(self.config, self.choices, params), params)

    @property
    def db_path(self) -> Path:
        return self.results_root / "experiments.sqlite"

    @property
    def metrics_db_path(self) -> Path:
        return self.results_root / "metrics.sqlite"

    @property
    def executions_root(self) -> Path:
        return self.results_root / "executions"
