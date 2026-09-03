"""Small tests for research_analysis.reporting — automated reporting and analysis."""

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.reporting import (
    ABComparisonReport,
    BenchmarkBakeoffReport,
    HyperparameterSensitivityReport,
    analyze_hypers,
    compare_bakeoff,
    compare_pairwise,
)


def populate_metrics_db(db_path: Path, run_id: int, metric_name: str, values: list[float], execution_id: int | None = None) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if execution_id is not None:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )
        for step, val in enumerate(values):
            cursor.execute(
                "INSERT INTO metrics (run_id, execution_id, metric_name, global_step, value) VALUES (?, ?, ?, ?, ?)",
                (run_id, execution_id, metric_name, step, val),
            )
    else:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS metrics (run_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )
        for step, val in enumerate(values):
            cursor.execute(
                "INSERT INTO metrics (run_id, metric_name, global_step, value) VALUES (?, ?, ?, ?)",
                (run_id, metric_name, step, val),
            )
    conn.commit()
    conn.close()


@pytest.fixture
def temp_experiment_setup() -> Generator[dict[str, Any], None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "experiments.sqlite"
        metrics_db_path = tmp_path / "metrics.sqlite"

        # Initialize experiment DB
        with DatabaseManager(db_path) as db:
            db.initialize()
            exp_id = db.add_experiment("test-exp", "Test experiment description")
            
            algo_comp_id = db.add_component("ppo", "ALGO")
            algo_ver_id = db.add_component_version(algo_comp_id, "hash_algo")
            
            env_comp_id = db.add_component("CartPole", "ENV")
            env_ver_id = db.add_component_version(env_comp_id, "hash_env")

            yield {
                "db_path": db_path,
                "metrics_db_path": metrics_db_path,
                "exp_id": exp_id,
                "algo_ver_id": algo_ver_id,
                "env_ver_id": env_ver_id,
                "tmp_path": tmp_path,
            }


def test_compare_pairwise_paired(temp_experiment_setup: dict[str, Any]) -> None:
    setup = temp_experiment_setup
    db_path = setup["db_path"]

    # 4 seeds for ppo-baseline, 4 seeds for ppo-with-mask (identical seed values: 0, 1, 2, 3)
    conditions = ["ppo-baseline", "ppo-with-mask"]
    seeds = [0, 1, 2, 3]

    with DatabaseManager(db_path) as db:
        for condition in conditions:
            for seed in seeds:
                hyper_id = db.add_hyperparam_config({"condition_name": condition, "seed": seed})

                run_id = db.add_run(
                    experiment_id=setup["exp_id"],
                    algo_version_id=setup["algo_ver_id"],
                    env_version_id=setup["env_ver_id"],
                    hyper_id=hyper_id,
                    seed=seed,
                )

                exec_id = db.add_execution()
                db.update_execution_status(exec_id, "COMPLETED")
                db.link_execution_run(exec_id, run_id)
                db.record_execution_artifacts(
                    exec_id,
                    str(setup["tmp_path"]),
                    metadata={"metrics_db_path": str(setup["metrics_db_path"])},
                )

                # Populate run outcomes
                # Baseline gets lower performance, Mask gets higher performance
                val = 100.0 + seed if condition == "ppo-baseline" else 150.0 + seed
                populate_metrics_db(setup["metrics_db_path"], run_id, "returned_episode_returns", [val], execution_id=exec_id)

    report = compare_pairwise(
        db_path=setup["db_path"],
        experiment_slug="test-exp",
        condition_a="ppo-baseline",
        condition_b="ppo-with-mask",
        metric_name="returned_episode_returns",
        verbose=True,
    )

    assert isinstance(report, ABComparisonReport)
    assert report.condition_a == "ppo-baseline"
    assert report.condition_b == "ppo-with-mask"
    assert report.mean_a == pytest.approx(101.5)
    assert report.mean_b == pytest.approx(151.5)
    assert report.difference_in_means == pytest.approx(-50.0)
    assert report.test_details.assumptions["paired"] is True
    # Verify plotting artifact exists
    assert report.distribution_plot_path is not None
    assert report.distribution_plot_path.exists()


def test_compare_pairwise_honors_confidence_level(temp_experiment_setup: dict[str, Any]) -> None:
    """Use the requested confidence level for significance and interval width."""
    setup = temp_experiment_setup
    differences = [-2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

    with DatabaseManager(setup["db_path"]) as db:
        for seed, difference in enumerate(differences):
            hyper_id = db.add_hyperparam_config({"condition_name": "a", "seed": seed})
            run_id = db.add_run(setup["exp_id"], setup["algo_ver_id"], setup["env_ver_id"], hyper_id, seed)
            execution_id = db.add_execution()
            db.update_execution_status(execution_id, "COMPLETED")
            db.link_execution_run(execution_id, run_id)
            db.record_execution_artifacts(execution_id, str(setup["tmp_path"]), metadata={"metrics_db_path": str(setup["metrics_db_path"])})
            populate_metrics_db(setup["metrics_db_path"], run_id, "metric", [difference], execution_id=execution_id)

            hyper_id = db.add_hyperparam_config({"condition_name": "b", "seed": seed})
            run_id = db.add_run(setup["exp_id"], setup["algo_ver_id"], setup["env_ver_id"], hyper_id, seed)
            execution_id = db.add_execution()
            db.update_execution_status(execution_id, "COMPLETED")
            db.link_execution_run(execution_id, run_id)
            db.record_execution_artifacts(execution_id, str(setup["tmp_path"]), metadata={"metrics_db_path": str(setup["metrics_db_path"])})
            populate_metrics_db(setup["metrics_db_path"], run_id, "metric", [0.0], execution_id=execution_id)

    report_90 = compare_pairwise(setup["db_path"], "test-exp", "a", "b", "metric", confidence_level=0.90, verbose=False)
    report_95 = compare_pairwise(setup["db_path"], "test-exp", "a", "b", "metric", confidence_level=0.95, verbose=False)
    assert report_90.test_details.is_significant is True
    assert report_95.test_details.is_significant is False
    assert report_90.difference_ci[0] >= report_95.difference_ci[0]
    assert report_90.difference_ci[1] <= report_95.difference_ci[1]


def test_compare_pairwise_rejects_invalid_confidence_level() -> None:
    """Reject confidence levels outside the open unit interval."""
    with pytest.raises(ValueError, match="confidence_level must be in"):
        compare_pairwise("unused.sqlite", "test-exp", "a", "b", "metric", confidence_level=1.0, verbose=False)


def test_analyze_hypers(temp_experiment_setup: dict[str, Any]) -> None:
    setup = temp_experiment_setup
    db_path = setup["db_path"]
    
    # Tuning learning rate: 1e-3, 3e-4, 1e-4
    lrs = ["1e-3", "3e-4", "1e-4"]
    seeds = [0, 1, 2]

    with DatabaseManager(db_path) as db:
        for lr in lrs:
            for seed in seeds:
                hyper_id = db.add_hyperparam_config({"learning_rate": lr, "seed": seed})
                
                run_id = db.add_run(
                    experiment_id=setup["exp_id"],
                    algo_version_id=setup["algo_ver_id"],
                    env_version_id=setup["env_ver_id"],
                    hyper_id=hyper_id,
                    seed=seed,
                )
                
                exec_id = db.add_execution()
                db.update_execution_status(exec_id, "COMPLETED")
                db.link_execution_run(exec_id, run_id)
                db.record_execution_artifacts(exec_id, str(setup["tmp_path"]))

                # Performance: 1e-3 -> ~200, 3e-4 -> ~500 (winner), 1e-4 -> ~300
                if lr == "1e-3":
                    val = 200.0 + seed
                elif lr == "3e-4":
                    val = 500.0 + seed
                else:
                    val = 300.0 + seed
                populate_metrics_db(setup["metrics_db_path"], run_id, "returned_episode_returns", [val])

    report = analyze_hypers(
        db_path=setup["db_path"],
        experiment_slug="test-exp",
        target_hyperparameter="learning_rate",
        metric_name="returned_episode_returns",
        verbose=True,
    )

    assert isinstance(report, HyperparameterSensitivityReport)
    assert report.target_hyperparameter == "learning_rate"
    assert report.winning_value == "3e-4"
    assert report.raw_mean == pytest.approx(501.0)
    assert report.corrected_mean <= report.raw_mean
    assert report.sensitivity_plot_path is not None
    assert report.sensitivity_plot_path.exists()


def test_analyze_hypers_group_by_partitions_instead_of_pooling(temp_experiment_setup: dict[str, Any]) -> None:
    """group_by must analyze each hyperparameter-key combination separately, not pool them."""
    setup = temp_experiment_setup
    db_path = setup["db_path"]

    lrs = ["1e-3", "3e-4", "1e-4"]
    seeds = [0, 1, 2]
    tasks = ["alpha", "beta"]

    with DatabaseManager(db_path) as db:
        for task in tasks:
            for lr in lrs:
                for seed in seeds:
                    hyper_id = db.add_hyperparam_config({"learning_rate": lr, "task": task, "seed": seed})

                    run_id = db.add_run(
                        experiment_id=setup["exp_id"],
                        algo_version_id=setup["algo_ver_id"],
                        env_version_id=setup["env_ver_id"],
                        hyper_id=hyper_id,
                        seed=seed,
                    )

                    exec_id = db.add_execution()
                    db.update_execution_status(exec_id, "COMPLETED")
                    db.link_execution_run(exec_id, run_id)
                    db.record_execution_artifacts(exec_id, str(setup["tmp_path"]))

                    # task "alpha" scores hundreds of points positive; task "beta" scores strongly
                    # negative. Pooling the two tasks together (the bug) drowns beta's true winner
                    # in alpha's much larger positive values.
                    if task == "alpha":
                        val = {"1e-3": 200.0, "3e-4": 500.0, "1e-4": 300.0}[lr] + seed
                    else:
                        val = {"1e-3": -200.0, "3e-4": -180.0, "1e-4": -100.0}[lr] + seed
                    populate_metrics_db(setup["metrics_db_path"], run_id, "returned_episode_returns", [val])

    report = analyze_hypers(
        db_path=setup["db_path"],
        experiment_slug="test-exp",
        target_hyperparameter="learning_rate",
        metric_name="returned_episode_returns",
        group_by=["task"],
        verbose=False,
    )

    assert isinstance(report, dict)
    assert set(report.keys()) == {("alpha",), ("beta",)}

    alpha_report = report[("alpha",)]
    beta_report = report[("beta",)]
    assert isinstance(alpha_report, HyperparameterSensitivityReport)
    assert isinstance(beta_report, HyperparameterSensitivityReport)

    assert alpha_report.winning_value == "3e-4"
    assert alpha_report.raw_mean == pytest.approx(501.0)

    assert beta_report.winning_value == "1e-4"
    assert beta_report.raw_mean == pytest.approx(-99.0)

    assert alpha_report.sensitivity_plot_path is not None
    assert beta_report.sensitivity_plot_path is not None
    assert alpha_report.sensitivity_plot_path != beta_report.sensitivity_plot_path
    assert alpha_report.sensitivity_plot_path.exists()
    assert beta_report.sensitivity_plot_path.exists()


def test_compare_bakeoff(temp_experiment_setup: dict[str, Any]) -> None:
    setup = temp_experiment_setup
    db_path = setup["db_path"]
    
    # Bakeoff of 3 algorithms (ppo, sac, td3) on 2 environments (CartPole, MountainCar)
    algos = ["ppo", "sac", "td3"]
    envs = ["CartPole", "MountainCar"]
    seeds = [0, 1]

    with DatabaseManager(db_path) as db:
        for algo in algos:
            for env in envs:
                for seed in seeds:
                    hyper_id = db.add_hyperparam_config({"algorithm": algo, "env_name": env, "seed": seed})
                    
                    run_id = db.add_run(
                        experiment_id=setup["exp_id"],
                        algo_version_id=setup["algo_ver_id"],
                        env_version_id=setup["env_ver_id"],
                        hyper_id=hyper_id,
                        seed=seed,
                    )
                    
                    exec_id = db.add_execution()
                    db.update_execution_status(exec_id, "COMPLETED")
                    db.link_execution_run(exec_id, run_id)
                    db.record_execution_artifacts(exec_id, str(setup["tmp_path"]))

                    # Output values
                    val = 100.0 + seed
                    populate_metrics_db(setup["metrics_db_path"], run_id, "returned_episode_returns", [val])

    report = compare_bakeoff(
        db_path=setup["db_path"],
        experiment_slug="test-exp",
        algorithms=algos,
        environments=envs,
        metric_name="returned_episode_returns",
        verbose=True,
    )

    assert isinstance(report, BenchmarkBakeoffReport)
    assert sorted(report.algorithms) == sorted(algos)
    assert sorted(report.environments) == sorted(envs)
    assert report.ecdf_plot_path is not None
    assert report.ecdf_plot_path.exists()


def _populate_bakeoff_rows(setup: dict[str, Any], rows: list[tuple[str, str, int, float]]) -> None:
    with DatabaseManager(setup["db_path"]) as db:
        for algorithm, environment, seed, value in rows:
            hyper_id = db.add_hyperparam_config({"algorithm": algorithm, "env_name": environment})
            run_id = db.add_run(setup["exp_id"], setup["algo_ver_id"], setup["env_ver_id"], hyper_id, seed)
            execution_id = db.add_execution()
            db.update_execution_status(execution_id, "COMPLETED")
            db.link_execution_run(execution_id, run_id)
            db.record_execution_artifacts(execution_id, str(setup["tmp_path"]))
            populate_metrics_db(setup["metrics_db_path"], run_id, "metric", [value])


def test_compare_bakeoff_reports_listwise_missing_data(temp_experiment_setup: dict[str, Any]) -> None:
    """Run Friedman on complete rows while reporting listwise deletion."""
    algorithms = ["ppo", "sac", "td3"]
    rows = [
        (algorithm, "CartPole", seed, float(seed + index))
        for seed in range(3)
        for index, algorithm in enumerate(algorithms)
    ] + [("ppo", "CartPole", 3, 10.0), ("sac", "CartPole", 3, 11.0)]
    _populate_bakeoff_rows(temp_experiment_setup, rows)

    report = compare_bakeoff(temp_experiment_setup["db_path"], "test-exp", algorithms, ["CartPole"], "metric", verbose=False)

    assert report.omnibus_details.test_name == "Friedman Test (Listwise Deleted for Missing Data)"
    assert report.omnibus_details.assumptions["has_missing"] is True


def test_compare_bakeoff_rejects_insufficient_complete_data(temp_experiment_setup: dict[str, Any]) -> None:
    """Fail clearly when missingness leaves fewer than three complete rows."""
    algorithms = ["ppo", "sac", "td3"]
    rows = [(algorithm, "CartPole", 0, float(index)) for index, algorithm in enumerate(algorithms)]
    rows += [("ppo", "CartPole", 1, 10.0), ("sac", "CartPole", 1, 11.0)]
    _populate_bakeoff_rows(temp_experiment_setup, rows)

    with pytest.raises(ValueError, match="Insufficient complete bakeoff data"):
        compare_bakeoff(temp_experiment_setup["db_path"], "test-exp", algorithms, ["CartPole"], "metric", verbose=False)


def test_compare_bakeoff_rejects_fewer_than_three_algorithms() -> None:
    """Fail with an actionable message instead of an opaque scipy error below the Friedman minimum."""
    with pytest.raises(ValueError, match="compare_pairwise") as exc_info:
        compare_bakeoff("unused.sqlite", "test-exp", ["ppo", "sac"], ["CartPole"], "metric", verbose=False)
    assert exc_info.type is ValueError
