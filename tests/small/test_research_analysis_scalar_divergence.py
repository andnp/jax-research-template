"""Loading contracts for stored per-run scalars that diverged (NaN or infinite)."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from experiment_definition.db import DatabaseManager
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.types import DiagnosticCode
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars

_CONDITIONS = ("baseline", "candidate")


def _load(
    tmp_path: Path, *, candidate_auc: float, candidate_curve: float = 1.0, throughput: float = 2.0
) -> ExperimentDataset:
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("scalar-divergence")
        algo = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        runs: list[tuple[int, int, int]] = []
        for seed, condition in enumerate(_CONDITIONS):
            hyper_id = database.add_hyperparam_config({"condition_name": condition, "seed": seed})
            run_id = database.add_run(experiment_id, algo, env, hyper_id, seed)
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
            )
            database.update_execution_status(execution_id, "COMPLETED")
            runs.append((run_id, execution_id, seed))

    for (run_id, execution_id, seed), auc in zip(runs, (0.5, candidate_auc), strict=True):
        ids = {"experiment_id": experiment_id, "execution_id": execution_id}
        write_seed_batch_scalars(metrics_path, {"auc": (auc,)}, **ids, run_ids=(run_id,), seed_ids=(seed,))
        write_execution_scalars(metrics_path, {"throughput": throughput}, **ids)
    with sqlite3.connect(metrics_path) as connection:
        connection.executemany(
            "INSERT INTO metrics (metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id) "
            "VALUES ('auc', ?, 0, ?, ?, ?, ?)",
            [
                (value, seed, experiment_id, run_id, execution_id)
                for (run_id, execution_id, seed), value in zip(runs, (1.0, candidate_curve), strict=True)
            ],
        )
    return load_experiment_dataset(db_path, "scalar-divergence")


def _flags(dataset: ExperimentDataset) -> dict[str, list[str]]:
    return dict(zip(dataset.runs["condition_name"], dataset.runs["non_finite_metrics"].to_list(), strict=True))


def _diverged(dataset: ExperimentDataset) -> dict[str, dict[str, int]]:
    return {str(dict(cell.cell)["condition_name"]): dict(cell.diverged) for cell in dataset.accounting.cells}


class TestDivergedStoredScalar:
    def test_diverged_scalar_is_flagged_diagnosed_and_counted(self, tmp_path: Path) -> None:
        """A NaN stored value is a diverged outcome: its run is flagged, diagnosed, and counted in the accounting."""
        dataset = _load(tmp_path, candidate_auc=math.nan)

        candidate_run = dataset.runs.filter(dataset.runs["condition_name"] == "candidate")["run_id"].item()
        diagnostics = [item for item in dataset.diagnostics if item.code is DiagnosticCode.NON_FINITE_METRIC]
        assert _flags(dataset) == {"baseline": [], "candidate": ["auc"]}
        assert _diverged(dataset) == {"baseline": {}, "candidate": {"auc": 1}}
        assert [item.run_id for item in diagnostics] == [candidate_run]
        assert "stored value" in diagnostics[0].message
        assert "step" not in diagnostics[0].message

    def test_a_metric_diverged_as_curve_and_scalar_counts_once(self, tmp_path: Path) -> None:
        """One run diverging in the same metric in both tables is one diverged outcome, not two."""
        dataset = _load(tmp_path, candidate_auc=math.inf, candidate_curve=math.nan)

        diagnostics = [item for item in dataset.diagnostics if item.code is DiagnosticCode.NON_FINITE_METRIC]
        assert _flags(dataset) == {"baseline": [], "candidate": ["auc"]}
        assert _diverged(dataset) == {"baseline": {}, "candidate": {"auc": 1}}
        assert [item.message.split("metric 'auc' ")[1] for item in diagnostics] == [
            "is non-finite from step 0 (1 values) and has a non-finite stored value"
        ]

    def test_a_diverged_execution_value_is_not_a_run_outcome(self, tmp_path: Path) -> None:
        """A whole-execution value describes no single run, so it never marks a run as diverged."""
        dataset = _load(tmp_path, candidate_auc=0.7, throughput=math.nan)

        assert _flags(dataset) == {"baseline": [], "candidate": []}
        assert _diverged(dataset) == {"baseline": {}, "candidate": {}}
        assert not any(item.code is DiagnosticCode.NON_FINITE_METRIC for item in dataset.diagnostics)
