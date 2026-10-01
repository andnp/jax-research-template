"""Contract tests for public synthetic experiment test support."""

from pathlib import Path

from experiment_definition.component import ComponentType
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ComponentVersionRef, VersionSelection
from research_analysis.testing import SyntheticComponentVersion, SyntheticExperiment, SyntheticRunSpec


def test_synthetic_experiment_writes_metrics_and_pins_component_ids(tmp_path: Path) -> None:
    """Load writer-backed curves and scalars from explicitly selected version IDs.

    The helper should preserve the identities needed for exact historical selection.
    """
    algorithm = SyntheticComponentVersion("ppo", ComponentType.ALGO, "algo-v2", version_id=101)
    environment = SyntheticComponentVersion("cartpole", ComponentType.ENV, "env-v3", version_id=202)
    experiment = SyntheticExperiment(tmp_path, "synthetic", component_versions=(algorithm, environment))
    execution = experiment.add_execution(
        [
            SyntheticRunSpec(
                seed=7,
                hyperparameters={"condition_name": "baseline"},
                curve_metrics={"reward": (1.0, 2.0)},
                run_scalars={"auc": 3.5},
                component_version_ids=(101, 202),
            )
        ],
        execution_scalars={"compile_seconds": 0.25},
    )[0]

    dataset = load_experiment_dataset(
        experiment.db_path,
        "synthetic",
        version_selection=VersionSelection.exact(
            (
                ComponentVersionRef.by_id("ppo", "ALGO", 101),
                ComponentVersionRef.by_id("cartpole", "ENV", 202),
            )
        ),
    )

    assert dataset.runs.select("run_id", "algo_version_id", "env_version_id").rows() == [
        (execution.run_id, 101, 202)
    ]
    assert dataset.metrics.select("step", "metric", "value").rows() == [
        (0, "reward", 1.0),
        (1, "reward", 2.0),
    ]
    assert dataset.scalar_metrics.select("metric", "value").rows() == [("auc", 3.5)]
    assert dataset.execution_metrics.rows() == [(execution.execution_id, "compile_seconds", 0.25)]
