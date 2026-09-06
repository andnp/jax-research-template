"""Contract tests for named-field access on RunCandidate and VersionSet."""

from __future__ import annotations

from pathlib import Path

from experiment_definition.db import DatabaseManager
from research_analysis.dataset.candidates import fetch_run_candidates
from research_analysis.dataset.types import VersionSelection
from research_analysis.dataset.versions import select_version_sets
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator


def test_fetch_and_select_use_named_run_candidate_fields(tmp_path: Path) -> None:
    """fetch_run_candidates and select_version_sets expose run/version_set/provenance by name, not tuple index."""
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
        database.record_execution_artifacts(execution_id, str(tmp_path / "execution"))
        database.update_execution_status(execution_id, "COMPLETED")

        experiment = database.get_experiment("candidate-test")
        assert experiment is not None
        diagnostics = DiagnosticAccumulator()
        candidates = fetch_run_candidates(database, experiment, diagnostics)

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.run.id == run_id
        assert candidate.execution.id == execution_id
        assert candidate.version_set.algo_version_id == algo_version
        assert candidate.version_set.env_version_id == env_version
        assert len(candidate.provenance) == 2

        selected_sets, _ = select_version_sets(candidates, VersionSelection.latest_completed(), diagnostics)
        assert candidate.version_set in selected_sets
