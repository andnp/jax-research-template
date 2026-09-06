"""Contract tests for dataset-backed statistical reporting."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import pytest
from experiment_definition.db import DatabaseManager
from experiment_definition.parameter import ParameterValue
from research_analysis.aggregates import ReferenceRange
from research_analysis.comparison import ComparisonSpec
from research_analysis.dataset import (
    ExperimentDataset,
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    load_experiment_dataset,
)
from research_analysis.reporting import (
    ABComparisonReport,
    BenchmarkBakeoffReport,
    HyperparameterConfigReport,
    HyperparameterDifferenceReport,
    HyperparameterSliceReport,
    best_hypers,
    compare_bakeoff,
    compare_pairwise,
    slice_hypers,
)


class ExperimentSetup(TypedDict):
    db_path: Path
    metrics_path: Path
    experiment_id: int
    algo_version_id: int
    env_version_id: int
    root: Path


def _ensure_metrics_table(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )


def _add_observation(
    setup: ExperimentSetup,
    metadata: dict[str, ParameterValue],
    seed: int,
    value: float,
    *,
    ablation: str | None = None,
) -> None:
    with DatabaseManager(setup["db_path"]) as database:
        hyper_id = database.add_hyperparam_config({**metadata, "seed": seed})
        run_id = database.add_run(
            setup["experiment_id"],
            setup["algo_version_id"],
            setup["env_version_id"],
            hyper_id,
            seed,
            ablation,
        )
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id,
            str(setup["root"] / str(execution_id)),
            metadata={"metrics_db_path": str(setup["metrics_path"])},
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(setup["metrics_path"]) as connection:
        connection.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
            (run_id, execution_id, "metric", 0, value),
        )
        connection.commit()


def _load(
    setup: ExperimentSetup,
    dimensions: ExperimentDimensions | None = None,
) -> ExperimentDataset:
    return load_experiment_dataset(setup["db_path"], "test-exp", dimensions=dimensions)


@pytest.fixture
def temp_experiment_setup() -> Generator[ExperimentSetup, None, None]:
    """Create a hermetic experiment database and a shared recorded metrics database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "experiments.sqlite"
        metrics_path = root / "metrics.sqlite"
        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_id = database.add_experiment("test-exp")
            algo_component = database.add_component("ppo", "ALGO")
            algo_version_id = database.add_component_version(algo_component, "algo-hash")
            env_component = database.add_component("cartpole", "ENV")
            env_version_id = database.add_component_version(env_component, "env-hash")
        _ensure_metrics_table(metrics_path)
        yield {
            "db_path": db_path,
            "metrics_path": metrics_path,
            "experiment_id": experiment_id,
            "algo_version_id": algo_version_id,
            "env_version_id": env_version_id,
            "root": root,
        }


def test_compare_pairwise_consumes_selected_dataset(temp_experiment_setup: ExperimentSetup) -> None:
    """Compare paired conditions from materialized metrics without database arguments."""
    setup = temp_experiment_setup
    for condition in ("baseline", "candidate"):
        for seed in range(4):
            _add_observation(setup, {"condition_name": condition}, seed, 100.0 + seed + (50.0 if condition == "candidate" else 0.0))

    report = compare_pairwise(_load(setup), "baseline", "candidate", "metric", verbose=False)

    assert isinstance(report, ABComparisonReport)
    assert report.pairing == "paired"
    assert report.difference_in_means == pytest.approx(-50.0)


def test_compare_pairwise_honors_declared_pairing_keys(temp_experiment_setup: ExperimentSetup) -> None:
    """Reject partial automatic pairing unless complete-case behavior is explicit."""
    setup = temp_experiment_setup
    for condition, seeds in (("a", (0, 1, 2, 4)), ("b", (0, 1, 2, 3))):
        for seed in seeds:
            _add_observation(setup, {"condition_name": condition}, seed, float(seed))
    dataset = _load(setup)

    with pytest.raises(ValueError, match="partially overlapping"):
        compare_pairwise(dataset, "a", "b", "metric", verbose=False)

    report = compare_pairwise(
        dataset,
        "a",
        "b",
        "metric",
        verbose=False,
        spec=ComparisonSpec(missingness="complete_case", n_resamples=64),
    )
    assert report.pairing == "paired"


def test_best_hypers_consumes_declared_sweep_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """Find the best declared sweep value using final values from the dataset metrics table."""
    setup = temp_experiment_setup
    for learning_rate, score in (("low", 1.0), ("high", 3.0)):
        for seed in range(3):
            _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, score + seed)
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    report = best_hypers(dataset, "metric", verbose=False)

    assert isinstance(report, HyperparameterConfigReport)
    assert report.winning_config == {"learning_rate": "high"}
    assert report.raw_mean == pytest.approx(4.0)


def test_best_hypers_rejects_undeclared_group_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """Require grouping keys to be declared typed sweep axes in the dataset."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "learning_rate": "low", "task": "cartpole"},
            seed,
            float(seed),
        )
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    with pytest.raises(ValueError, match="declared typed sweep axes"):
        best_hypers(dataset, "metric", group_by=["task"], verbose=False)


def test_best_hypers_optimizes_jointly_over_multiple_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """A grid champion is the best joint combination, not a per-axis marginal pick."""
    setup = temp_experiment_setup
    combos = {("low", 1): 4.0, ("low", 2): 2.0, ("high", 1): 3.0, ("high", 2): 5.0}
    for (learning_rate, eta), value in combos.items():
        for seed in range(3):
            _add_observation(
                setup,
                {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta},
                seed,
                value,
            )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    report = best_hypers(dataset, "metric", verbose=False)

    assert report.winning_config == {"learning_rate": "high", "eta": 2}
    assert report.raw_mean == pytest.approx(5.0)


def test_best_hypers_higher_is_better_false_finds_the_minimizer(temp_experiment_setup: ExperimentSetup) -> None:
    """For an error-like metric, the champion is the lowest raw mean, reported in its own units."""
    setup = temp_experiment_setup
    for learning_rate, score in (("low", 1.0), ("high", 3.0)):
        for seed in range(3):
            _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, score + seed)
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    report = best_hypers(dataset, "metric", higher_is_better=False, verbose=False)

    assert report.winning_config == {"learning_rate": "low"}
    assert report.raw_mean == pytest.approx(2.0)
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.corrected_mean <= ci_high


def test_slice_hypers_pins_champion_config_while_varying_one_axis(
    temp_experiment_setup: ExperimentSetup,
) -> None:
    """The best-achievable curve re-optimizes lr per eta; the slice curve pins lr at the champion's value."""
    setup = temp_experiment_setup
    combos = {("low", 1): 4.0, ("low", 2): 2.0, ("high", 1): 3.0, ("high", 2): 5.0}
    for (learning_rate, eta), value in combos.items():
        for seed in range(3):
            _add_observation(
                setup,
                {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta},
                seed,
                value,
            )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    report = slice_hypers(dataset, "metric", vary="eta", verbose=False)

    assert isinstance(report, HyperparameterSliceReport)
    assert report.values == [1, 2]
    assert report.best_performance == pytest.approx([4.0, 5.0])
    assert report.slice_performance == pytest.approx([3.0, 5.0])


def test_slice_hypers_rejects_varying_a_grouped_axis(temp_experiment_setup: ExperimentSetup) -> None:
    """vary and group_by must name disjoint axes."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "learning_rate": "low", "eta": 1},
            seed,
            float(seed),
        )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    with pytest.raises(ValueError, match="cannot be both varied and grouped by"):
        slice_hypers(dataset, "metric", vary="eta", group_by=["eta"], verbose=False)


def _add_arm_observation(setup: ExperimentSetup, algorithm: str, learning_rate: str, seed: int, value: float) -> None:
    _add_observation(setup, {"condition_name": algorithm, "algorithm": algorithm, "learning_rate": learning_rate}, seed, value)


def test_best_hypers_groups_by_the_condition_dimension(temp_experiment_setup: ExperimentSetup) -> None:
    """group_by can name the dataset's condition dimension, not just declared sweep axes."""
    setup = temp_experiment_setup
    for algorithm, base in (("shared", 1.0), ("adversarial", 3.0)):
        for learning_rate, offset in (("low", 0.0), ("high", 1.0)):
            for seed in range(3):
                _add_observation(setup, {"algorithm": algorithm, "learning_rate": learning_rate}, seed, base + offset + seed)
    dataset = _load(
        setup,
        ExperimentDimensions(condition="algorithm", sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)),
    )

    reports = best_hypers(dataset, "metric", group_by=["algorithm"], verbose=False)

    assert reports[("shared",)].winning_config == {"learning_rate": "high"}
    assert reports[("shared",)].raw_mean == pytest.approx(3.0)
    assert reports[("adversarial",)].winning_config == {"learning_rate": "high"}
    assert reports[("adversarial",)].raw_mean == pytest.approx(5.0)


def test_best_hypers_difference_pairs_by_seed_across_two_groups(temp_experiment_setup: ExperimentSetup) -> None:
    """The paired difference compares each arm's own champion, correlated by shared seed."""
    setup = temp_experiment_setup
    arm_a = {("low",): (1.0, 2.0, 3.0), ("high",): (4.0, 5.0, 6.0)}
    arm_b = {("low",): (1.0, 1.0, 1.0), ("high",): (2.0, 2.0, 2.0)}
    for (learning_rate,), values in arm_a.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "a", learning_rate, seed, value)
    for (learning_rate,), values in arm_b.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "b", learning_rate, seed, value)
    dataset = _load(
        setup,
        ExperimentDimensions(sweep_axes=(SweepAxis("algorithm", SweepValueType.STRING), SweepAxis("learning_rate", SweepValueType.STRING))),
    )

    report = best_hypers(dataset, "metric", group_by=["algorithm"], difference_between=("a", "b"), verbose=False)

    assert isinstance(report, HyperparameterDifferenceReport)
    assert report.minuend_config == {"learning_rate": "high"}
    assert report.subtrahend_config == {"learning_rate": "high"}
    assert report.mean_diff > 0
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.mean_diff <= ci_high


def test_best_hypers_difference_higher_is_better_false_finds_minimizers(
    temp_experiment_setup: ExperimentSetup,
) -> None:
    """For an error-like metric, each side's champion is its own minimizer, in original units."""
    setup = temp_experiment_setup
    arm_a = {("low",): (1.0, 2.0, 3.0), ("high",): (4.0, 5.0, 6.0)}
    arm_b = {("low",): (1.0, 1.0, 1.0), ("high",): (2.0, 2.0, 2.0)}
    for (learning_rate,), values in arm_a.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "a", learning_rate, seed, value)
    for (learning_rate,), values in arm_b.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "b", learning_rate, seed, value)
    dataset = _load(
        setup,
        ExperimentDimensions(sweep_axes=(SweepAxis("algorithm", SweepValueType.STRING), SweepAxis("learning_rate", SweepValueType.STRING))),
    )

    report = best_hypers(
        dataset, "metric", group_by=["algorithm"], difference_between=("a", "b"), higher_is_better=False, verbose=False
    )

    assert isinstance(report, HyperparameterDifferenceReport)
    assert report.minuend_config == {"learning_rate": "low"}
    assert report.subtrahend_config == {"learning_rate": "low"}
    assert report.mean_diff == pytest.approx(1.0, abs=0.5)
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.mean_diff <= ci_high


def test_best_hypers_difference_requires_a_single_group_axis(temp_experiment_setup: ExperimentSetup) -> None:
    """difference_between only makes sense against exactly one group_by axis."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "algorithm": "a", "learning_rate": "low", "eta": 1},
            seed,
            float(seed),
        )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("algorithm", SweepValueType.STRING),
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    with pytest.raises(ValueError, match="difference_between requires group_by to name exactly one axis"):
        best_hypers(dataset, "metric", group_by=["algorithm", "learning_rate"], difference_between=("a", "b"), verbose=False)


def test_compare_bakeoff_consumes_canonical_environment_dimension(temp_experiment_setup: ExperimentSetup) -> None:
    """Compare algorithms and environments using canonical dataset dimensions."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac", "td3"]
    environments = ["cartpole", "mountaincar"]
    for algorithm in algorithms:
        for environment in environments:
            for seed in range(3):
                _add_observation(setup, {"algorithm": algorithm, "env_source": environment}, seed, float(seed))
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = compare_bakeoff(
        dataset,
        algorithms,
        environments,
        "metric",
        verbose=False,
        references={environment: ReferenceRange(0.0, 10.0) for environment in environments},
    )

    assert isinstance(report, BenchmarkBakeoffReport)
    assert report.algorithms == algorithms
    assert report.environments == environments


def test_compare_bakeoff_retains_blockwise_missingness(temp_experiment_setup: ExperimentSetup) -> None:
    """Keep incomplete environment-seed blocks for block-stratified inference."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac", "td3"]
    for seed in range(3):
        for index, algorithm in enumerate(algorithms):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, float(seed + index))
    for algorithm, seed in (("ppo", 3), ("sac", 3)):
        _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, float(seed))
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = compare_bakeoff(
        dataset,
        algorithms,
        ["cartpole"],
        "metric",
        verbose=False,
        references={"cartpole": ReferenceRange(0.0, 10.0)},
    )

    assert report.omnibus_details.assumptions["has_missing"] is True
