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
