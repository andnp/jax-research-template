"""Contract tests for the staged completed-run and versioned-candidate objects."""

from __future__ import annotations

from pathlib import Path

from experiment_definition.db import DatabaseManager
from research_analysis.dataset.candidates import RunState, fetch_run_candidates
from research_analysis.dataset.types import VersionSelection
from research_analysis.dataset.versions import resolve_versioned_candidates, select_version_sets
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator


def test_fetch_and_select_use_named_run_candidate_fields(tmp_path: Path) -> None:
    """Discovery stays separate from version identity while both expose named fields."""
    db_path = tmp_path / "experiments.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("candidate-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": 0})
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        metrics_path = tmp_path / "metrics.sqlite"
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(execution_id, "COMPLETED")

        experiment = database.get_experiment("candidate-test")
        assert experiment is not None
        run_candidates = fetch_run_candidates(database, experiment)

        assert len(run_candidates) == 1
        assert run_candidates[0].state is RunState.COMPLETED
        completed_run = run_candidates[0].completed
        assert completed_run is not None
        assert completed_run.run.id == run_id
        assert completed_run.execution.id == execution_id
        assert completed_run.artifacts.metrics_db_path == metrics_path

        candidates = resolve_versioned_candidates(database, [completed_run])
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.completed_run == completed_run
        assert candidate.version_set.algo_version_id == algo_version
        assert candidate.version_set.env_version_id == env_version
        assert len(candidate.provenance) == 2

        selected_sets, _ = select_version_sets(candidates, VersionSelection.latest_completed(), DiagnosticAccumulator())
        assert candidate.version_set in selected_sets
