from __future__ import annotations

import textwrap
from pathlib import Path

from experiment_definition.db import DatabaseManager
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_experiment_list_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "list", str(db_path)])
    assert result.exit_code == 0
    assert "Name" in result.output
    assert "Description" in result.output


def test_experiment_list_with_experiments(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        db.add_experiment("my-experiment", "Test experiment")
        db.add_experiment("another-exp", "Another one")

    result = runner.invoke(app, ["experiment", "list", str(db_path)])
    assert result.exit_code == 0
    assert "my-experiment" in result.output
    assert "another-exp" in result.output
    assert "Test experiment" in result.output


def test_experiment_status_shows_run_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("status-exp", "For status test")
        comp_id = db.add_component("test-algo", "ALGO")
        ver_id = db.add_component_version(comp_id, "abc123")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "def456")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})
        db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=1)

    result = runner.invoke(app, ["experiment", "status", str(db_path)])
    assert result.exit_code == 0
    assert "status-exp" in result.output
    assert "2 runs" in result.output
    assert "0 completed" in result.output
    assert "0 failed" in result.output
    assert "2 pending" in result.output
    assert "algo_version" not in result.output


def test_experiment_status_breaks_down_by_version_pair(tmp_path: Path) -> None:
    """A single-pair experiment stays a one-liner; a multi-pair one gets a per-pair breakdown."""
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("multi-version-exp")
        comp_id = db.add_component("test-algo", "ALGO")
        old_ver_id = db.add_component_version(comp_id, "abc123")
        new_ver_id = db.add_component_version(comp_id, "abc456")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "def456")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})

        old_run_id = db.add_run(exp_id, old_ver_id, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, old_ver_id, env_ver_id, hyper_id, seed=1)
        db.add_run(exp_id, new_ver_id, env_ver_id, hyper_id, seed=0)

        execution = db.add_execution(hostname="n1")
        db.link_execution_run(execution, old_run_id)
        db.update_execution_status(execution, "COMPLETED")

    result = runner.invoke(app, ["experiment", "status", str(db_path)])
    assert result.exit_code == 0
    assert "multi-version-exp: 3 runs (1 completed, 0 failed, 2 pending)" in result.output
    assert f"algo_version={old_ver_id} env_version={env_ver_id}: 2 runs (1 completed, 0 failed, 1 pending)" in result.output
    assert f"algo_version={new_ver_id} env_version={env_ver_id}: 1 runs (0 completed, 0 failed, 1 pending)" in result.output


def test_status_marks_the_newest_version_that_has_results(tmp_path: Path) -> None:
    """The marker names the pair analysis will read, which is not always the newest pair.

    A sync mints a component version and its unexecuted runs for any source edit, cosmetic
    ones included, so the newest pair is routinely one nothing has run. Marking that pair
    would point the reader at an empty version.
    """
    db_path = tmp_path / "newest_with_results.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("refactored-exp")
        comp_id = db.add_component("test-algo", "ALGO")
        ran_ver_id = db.add_component_version(comp_id, "ran")
        unrun_ver_id = db.add_component_version(comp_id, "unrun")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "env")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})

        ran_run_id = db.add_run(exp_id, ran_ver_id, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, unrun_ver_id, env_ver_id, hyper_id, seed=0)
        execution = db.add_execution(hostname="n1")
        db.link_execution_run(execution, ran_run_id)
        db.update_execution_status(execution, "COMPLETED")
    assert unrun_ver_id > ran_ver_id  # the fixture premise: the empty version really is newer

    result = runner.invoke(app, ["experiment", "status", str(db_path)])

    assert result.exit_code == 0, result.output
    assert f"algo_version={ran_ver_id} env_version={env_ver_id}: 1 runs (1 completed, 0 failed, 0 pending) (newest with results)" in result.output
    assert f"algo_version={unrun_ver_id} env_version={env_ver_id}: 1 runs (0 completed, 0 failed, 1 pending)\n" in result.output


def test_experiment_status_renders_null_version_ids(tmp_path: Path) -> None:
    """Runs with a NULL algo_version_id (ambiguous provenance) render as 'unknown', not crash."""
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("ambiguous-exp")
        comp_id = db.add_component("test-algo", "ALGO")
        ver_id = db.add_component_version(comp_id, "abc123")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "def456")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})

        db.add_run(exp_id, None, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=1)

    result = runner.invoke(app, ["experiment", "status", str(db_path)])
    assert result.exit_code == 0
    assert f"algo_version=unknown env_version={env_ver_id}: 1 runs" in result.output
    assert f"algo_version={ver_id} env_version={env_ver_id}: 1 runs" in result.output


def test_experiment_status_separates_failed_from_never_run(tmp_path: Path) -> None:
    """A FAILED execution must not read as merely 'pending' alongside a never-run seed."""
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("failure-exp")
        comp_id = db.add_component("test-algo", "ALGO")
        ver_id = db.add_component_version(comp_id, "abc123")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "def456")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})

        failed_run_id = db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=1)

        execution = db.add_execution(hostname="n1")
        db.link_execution_run(execution, failed_run_id)
        db.update_execution_status(execution, "FAILED")

    result = runner.invoke(app, ["experiment", "status", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "failure-exp: 2 runs (0 completed, 1 failed, 1 pending)" in result.output


def test_experiment_status_a_retried_run_counts_as_pending_not_failed(tmp_path: Path) -> None:
    """A run with a FAILED execution and a queued retry is in-flight, not failed.

    Precedence must match research_analysis.dataset.candidates._run_state: completed >
    in-flight pending/running > failed > never run. Covers both the overall count and
    the per-version-pair breakdown.
    """
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment("retry-exp")
        comp_id = db.add_component("test-algo", "ALGO")
        ver_id = db.add_component_version(comp_id, "abc123")
        other_ver_id = db.add_component_version(comp_id, "def456")
        env_id = db.add_component("test-env", "ENV")
        env_ver_id = db.add_component_version(env_id, "ghi789")
        hyper_id = db.add_hyperparam_config({"lr": 0.01})

        retried_run_id = db.add_run(exp_id, ver_id, env_ver_id, hyper_id, seed=0)
        db.add_run(exp_id, other_ver_id, env_ver_id, hyper_id, seed=0)

        failed_execution = db.add_execution(hostname="n1")
        db.link_execution_run(failed_execution, retried_run_id)
        db.update_execution_status(failed_execution, "FAILED")
        retry_execution = db.add_execution(hostname="n1")
        db.link_execution_run(retry_execution, retried_run_id)

    result = runner.invoke(app, ["experiment", "status", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "retry-exp: 2 runs (0 completed, 0 failed, 2 pending)" in result.output
    assert f"algo_version={ver_id} env_version={env_ver_id}: 1 runs (0 completed, 0 failed, 1 pending)" in result.output
    assert f"algo_version={other_ver_id} env_version={env_ver_id}: 1 runs (0 completed, 0 failed, 1 pending)" in result.output


def _write_spec_file(tmp_path: Path, results_root: Path) -> Path:
    spec_file = tmp_path / "test_spec.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            from __future__ import annotations
            from pathlib import Path
            from research_runner.registry import spec
            from research_runner.types import ExecutionContext, ExecutionResult, ExperimentSpec
            from experiment_definition.component import Component, ComponentType
            from experiment_definition.experiment import Experiment

            @spec
            def my_spec() -> ExperimentSpec:
                experiment = Experiment("test-experiment", description="Test")
                with experiment.for_component(Component(name="test-algo", path=Path("algo.py"), type=ComponentType.ALGO)):
                    experiment.add_parameter("learning_rate", [0.01])
                with experiment.for_component(Component(name="test-env", path=Path("env.py"), type=ComponentType.ENV)):
                    pass
                experiment.add_parameter("seed", [0])

                def train(ctx: ExecutionContext) -> ExecutionResult:
                    return ExecutionResult(metadata={{"trained": True}})

                return ExperimentSpec(
                    experiment=experiment,
                    train_fn=train,
                    results_root=Path("{results_root}"),
                    capture_git=False,
                )

            def not_a_spec():
                return "I am not a spec"
        """),
    )
    return spec_file


def test_experiment_plan_with_spec(tmp_path: Path) -> None:
    results_root = tmp_path / "plan"

    spec_file = _write_spec_file(tmp_path, results_root)

    result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output
    assert "batch" in result.output.lower()


def test_experiment_run_with_spec(tmp_path: Path) -> None:
    results_root = tmp_path / "run"

    spec_file = _write_spec_file(tmp_path, results_root)

    result = runner.invoke(app, ["experiment", "run", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output
    assert "completed" in result.output.lower()


def test_experiment_run_with_spec_filter(tmp_path: Path) -> None:
    results_root = tmp_path / "run_filtered"

    spec_file = _write_spec_file(tmp_path, results_root)

    result = runner.invoke(app, ["experiment", "run", str(spec_file), "--spec", "my_spec"])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output


def test_spec_discovery_ignores_non_spec_functions(tmp_path: Path) -> None:
    results_root = tmp_path / "disc"

    spec_file = _write_spec_file(tmp_path, results_root)
    module = __import__("importlib").util
    loader_spec = module.spec_from_file_location("test_disc", spec_file)
    loaded = module.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loaded)

    from research_cli.experiment import _discover_specs

    specs = _discover_specs(loaded, None)
    assert "my_spec" in specs
    assert "not_a_spec" not in specs


def test_experiment_plan_fails_when_no_specs_registered(tmp_path: Path) -> None:
    spec_file = tmp_path / "unregistered_spec.py"
    spec_file.write_text(
        textwrap.dedent("""\
            from __future__ import annotations
            from research_runner.types import ExperimentSpec

            def my_spec() -> ExperimentSpec:
                raise NotImplementedError
        """),
    )

    result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert result.exit_code == 1
    assert "no experiment specs registered" in result.output.lower()
    assert "@spec" in result.output


# ---------------------------------------------------------------------------
# Executions listing
# ---------------------------------------------------------------------------


def test_executions_shows_execution_table(tmp_path: Path) -> None:
    db_path = tmp_path / "exec.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        eid = db.add_execution(hostname="gpu-node-01", git_commit="abc12345deadbeef")
        db.update_execution_status(eid, "COMPLETED", start_time="2026-05-01T10:00:00Z")

    result = runner.invoke(app, ["experiment", "executions", str(db_path)])
    assert result.exit_code == 0
    assert "COMPLETED" in result.output
    assert "gpu-node-01" in result.output
    assert "abc12345" in result.output


def test_executions_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "empty_exec.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "executions", str(db_path)])
    assert result.exit_code == 0
    assert "ID" in result.output
    assert "Status" in result.output


# ---------------------------------------------------------------------------
# Invalidation CLI
# ---------------------------------------------------------------------------


def test_invalidate_by_execution_id(tmp_path: Path) -> None:
    db_path = tmp_path / "inv.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        eid = db.add_execution(hostname="node01")
        db.update_execution_status(eid, "COMPLETED")

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--execution", str(eid)],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Invalidated 1" in result.output


def test_invalidate_by_git_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "inv_commit.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        e1 = db.add_execution(hostname="n1", git_commit="fff999abcdef")
        db.update_execution_status(e1, "COMPLETED")
        e2 = db.add_execution(hostname="n2", git_commit="fff999abcdef")
        db.update_execution_status(e2, "COMPLETED")

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--git-commit", "fff999abcdef"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Invalidated 2" in result.output


def test_invalidate_yes_flag_skips_confirmation(tmp_path: Path) -> None:
    db_path = tmp_path / "inv_yes.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        eid = db.add_execution(hostname="node01")
        db.update_execution_status(eid, "COMPLETED")

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--execution", str(eid), "--yes"],
    )
    assert result.exit_code == 0
    assert "Invalidated 1" in result.output


def test_invalidate_no_args_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "inv_noargs.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "invalidate", str(db_path)])
    assert result.exit_code == 1


def _seed_two_arm_experiment(db: DatabaseManager) -> tuple[int, int, int]:
    """Seed one experiment with two hyperparameter arms, each with a completed execution.

    Returns (experiment_id, changed_arm_run_id, stable_arm_run_id).
    """
    exp_id = db.add_experiment("bakeoff")
    algo_id = db.add_component("algo", "ALGO")
    algo_ver_id = db.add_component_version(algo_id, "abc")
    env_id = db.add_component("env", "ENV")
    env_ver_id = db.add_component_version(env_id, "def")

    changed_hyper_id = db.add_hyperparam_config({"algorithm": "gi_qrc"})
    stable_hyper_id = db.add_hyperparam_config({"algorithm": "baseline"})

    changed_run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, changed_hyper_id, seed=0)
    stable_run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, stable_hyper_id, seed=0)

    changed_execution = db.add_execution(hostname="n1")
    db.link_execution_run(changed_execution, changed_run_id)
    db.update_execution_status(changed_execution, "COMPLETED")

    stable_execution = db.add_execution(hostname="n2")
    db.link_execution_run(stable_execution, stable_run_id)
    db.update_execution_status(stable_execution, "COMPLETED")

    return exp_id, changed_run_id, stable_run_id


def test_invalidate_by_where_unsatisfies_only_matched_arm(tmp_path: Path) -> None:
    """
    The load-bearing case: invalidating one arm by --where must make that
    arm's runs unsatisfied while leaving the other arm's runs satisfied --
    the whole point of the feature is avoiding a full re-identify/re-run.
    """
    db_path = tmp_path / "bakeoff.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id, changed_run_id, stable_run_id = _seed_two_arm_experiment(db)

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--where", "algorithm=gi_qrc"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert "Invalidated 1" in result.output

    with DatabaseManager(db_path) as db:
        db.initialize()
        unsatisfied_ids = {r.id for r in db.list_runs_without_completed_execution(exp_id)}
    assert changed_run_id in unsatisfied_ids
    assert stable_run_id not in unsatisfied_ids


def test_invalidate_by_where_no_match_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "bakeoff_none.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        _seed_two_arm_experiment(db)

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--where", "algorithm=nonexistent"],
    )
    assert result.exit_code != 0
    assert "matched no executions" in (result.output + (result.stderr or "")).lower()


def test_invalidate_by_where_requires_experiment(tmp_path: Path) -> None:
    db_path = tmp_path / "bakeoff_noexp.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "invalidate", str(db_path), "--where", "algorithm=gi_qrc"])
    assert result.exit_code == 1
    assert "requires --experiment" in result.output.lower()


def test_invalidate_by_where_yes_flag_skips_confirmation(tmp_path: Path) -> None:
    db_path = tmp_path / "bakeoff_yes.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        _seed_two_arm_experiment(db)

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--where", "algorithm=gi_qrc", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "Invalidated 1" in result.output


def test_invalidate_nonexistent_warns(tmp_path: Path) -> None:
    db_path = tmp_path / "inv_none.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--execution", "99999"],
        input="y\n",
    )
    assert "no executions were invalidated" in (result.output + (result.stderr or "")).lower()


# ---------------------------------------------------------------------------
# Execute-batch CLI
# ---------------------------------------------------------------------------


def test_execute_batch_runs_planned_execution(tmp_path: Path) -> None:
    results_root = tmp_path / "eb"
    db_path = results_root / "experiments.sqlite"
    spec_file = _write_spec_file(tmp_path, results_root)

    # Sync experiment and plan execution batches
    plan_result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert plan_result.exit_code == 0, plan_result.output

    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment("test-experiment")
        assert exp_row is not None
        planned = database.plan_experiment_execution_batches(
            exp_row.id, results_root / "executions",
        )
        execution_id = planned[0].execution_id

    result = runner.invoke(app, [
        "experiment", "execute-batch", str(db_path),
        "--execution-id", str(execution_id),
        "--spec-file", str(spec_file),
        "--spec", "my_spec",
    ])
    assert result.exit_code == 0, result.output
    assert "completed" in result.output.lower()


def _write_metric_logging_spec_file(tmp_path: Path, results_root: Path) -> Path:
    spec_file = tmp_path / "metric_spec.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            from __future__ import annotations
            from pathlib import Path
            from research_runner.registry import spec
            from research_runner.types import ExecutionContext, ExecutionResult, ExperimentSpec
            from experiment_definition.component import Component, ComponentType
            from experiment_definition.experiment import Experiment

            @spec
            def my_spec() -> ExperimentSpec:
                experiment = Experiment("test-experiment", description="Test")
                with experiment.for_component(Component(name="test-algo", path=Path("algo.py"), type=ComponentType.ALGO)):
                    experiment.add_parameter("learning_rate", [0.01])
                with experiment.for_component(Component(name="test-env", path=Path("env.py"), type=ComponentType.ENV)):
                    pass
                experiment.add_parameter("seed", [0])

                def train(ctx: ExecutionContext) -> ExecutionResult:
                    return ExecutionResult(metric_curves={{"reward": [[1.0, 2.0]]}})

                return ExperimentSpec(
                    experiment=experiment,
                    train_fn=train,
                    results_root=Path("{results_root}"),
                    capture_git=False,
                )
        """),
    )
    return spec_file


def test_execute_batch_honors_results_root_override(tmp_path: Path) -> None:
    """--results-root must redirect metrics writes, not just the spec's own db_path."""
    results_root = tmp_path / "eb_original"
    override_root = tmp_path / "eb_override"
    db_path = override_root / "experiments.sqlite"
    spec_file = _write_metric_logging_spec_file(tmp_path, results_root)

    plan_result = runner.invoke(app, ["experiment", "plan", str(spec_file), "--results-root", str(override_root)])
    assert plan_result.exit_code == 0, plan_result.output

    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment("test-experiment")
        assert exp_row is not None
        planned = database.plan_experiment_execution_batches(
            exp_row.id, override_root / "executions",
        )
        execution_id = planned[0].execution_id

    result = runner.invoke(app, [
        "experiment", "execute-batch", str(db_path),
        "--execution-id", str(execution_id),
        "--spec-file", str(spec_file),
        "--spec", "my_spec",
        "--results-root", str(override_root),
    ])
    assert result.exit_code == 0, result.output
    assert (override_root / "metrics.sqlite").exists()
    assert not (results_root / "metrics.sqlite").exists()


def test_execute_batch_nonexistent_execution_fails(tmp_path: Path) -> None:
    results_root = tmp_path / "eb_fail"
    db_path = results_root / "experiments.sqlite"
    spec_file = _write_spec_file(tmp_path, results_root)

    # Sync experiment so the DB exists
    plan_result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert plan_result.exit_code == 0, plan_result.output

    result = runner.invoke(app, [
        "experiment", "execute-batch", str(db_path),
        "--execution-id", "99999",
        "--spec-file", str(spec_file),
        "--spec", "my_spec",
    ])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# Submit CLI (dry-run)
# ---------------------------------------------------------------------------


def test_submit_dry_run_before_any_sync_creates_no_database(tmp_path: Path) -> None:
    """A first-use dry run has nothing synced yet; it must not create experiments.sqlite."""
    results_root = tmp_path / "submit"
    spec_file = _write_spec_file(tmp_path, results_root)
    db_path = results_root / "experiments.sqlite"
    script_path = tmp_path / "slurm_submit.sh"

    result = runner.invoke(app, [
        "experiment", "submit", str(spec_file),
        "--dry-run",
        "--script-path", str(script_path),
    ])
    assert result.exit_code == 0, result.output
    assert "nothing planned yet" in result.output.lower()
    assert not db_path.exists()
    assert not script_path.exists()


def test_submit_dry_run_does_not_mutate_the_database(tmp_path: Path) -> None:
    """A dry run must not claim runs PENDING; planning them is a real DB commit."""
    results_root = tmp_path / "submit_dry_no_mutate"
    spec_file = _write_spec_file(tmp_path, results_root)
    db_path = results_root / "experiments.sqlite"

    plan_result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert plan_result.exit_code == 0, plan_result.output
    before_bytes = db_path.read_bytes()

    result = runner.invoke(app, [
        "experiment", "submit", str(spec_file),
        "--dry-run",
        "--script-path", str(tmp_path / "slurm_submit.sh"),
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()

    assert db_path.read_bytes() == before_bytes


def test_submit_dry_run_forwards_results_root_to_job_script(tmp_path: Path) -> None:
    """A results-root override is included in the generated Slurm command."""
    results_root = tmp_path / "submit_override"
    spec_file = _write_spec_file(tmp_path, results_root)
    script_path = tmp_path / "slurm_submit.sh"

    plan_result = runner.invoke(app, [
        "experiment", "plan", str(spec_file), "--results-root", str(results_root),
    ])
    assert plan_result.exit_code == 0, plan_result.output

    result = runner.invoke(app, [
        "experiment", "submit", str(spec_file),
        "--dry-run", "--results-root", str(results_root),
        "--script-path", str(script_path),
    ])

    assert result.exit_code == 0, result.output
    assert f'--results-root "{results_root.resolve()}"' in script_path.read_text()



def test_submit_reads_slurm_settings_from_a_config_file_with_flags_overriding(tmp_path: Path) -> None:
    """--slurm-config supplies cluster settings; an explicit flag still wins over the file."""
    results_root = tmp_path / "submit_config"
    spec_file = _write_spec_file(tmp_path, results_root)
    config_path = tmp_path / "slurm.json"
    config_path.write_text('{"account": "acct-from-file", "time": "1:00:00", "mem_per_cpu": "8G"}')
    script_path = tmp_path / "slurm_submit.sh"
    assert runner.invoke(app, ["experiment", "plan", str(spec_file)]).exit_code == 0

    result = runner.invoke(app, [
        "experiment", "submit", str(spec_file),
        "--dry-run",
        "--slurm-config", str(config_path),
        "--time", "0:30:00",
        "--script-path", str(script_path),
    ])

    assert result.exit_code == 0, result.output
    script = script_path.read_text()
    assert "#SBATCH --account=acct-from-file" in script
    assert "#SBATCH --mem-per-cpu=8G" in script
    assert "#SBATCH --time=0:30:00" in script

# ── prune ────────────────────────────────────────────────────────────────────────


def _orphan_run(db_path: Path) -> int:
    """Add a run for a hyperparameter value the current definition does not generate.

    Keeps the definition's own component versions so that adding this row cannot change
    which version counts as latest, which would make the definition's own run the orphan.
    """
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_row = db.get_experiment("test-experiment")
        assert exp_row is not None
        algo = db.get_component("test-algo")
        env = db.get_component("test-env")
        assert algo is not None and env is not None
        algo_ver = db.conn.execute(
            "SELECT id FROM ComponentVersions WHERE component_id = ? ORDER BY version_number DESC LIMIT 1",
            (algo.id,),
        ).fetchone()
        env_ver = db.conn.execute(
            "SELECT id FROM ComponentVersions WHERE component_id = ? ORDER BY version_number DESC LIMIT 1",
            (env.id,),
        ).fetchone()
        assert algo_ver is not None and env_ver is not None
        hyper_id = db.add_hyperparam_config({"learning_rate": 0.99})
        return db.add_run(exp_row.id, algo_ver[0], env_ver[0], hyper_id, seed=0)


def _run_count(db_path: Path) -> int:
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_row = db.get_experiment("test-experiment")
        assert exp_row is not None
        return len(db.list_runs(exp_row.id))


def test_prune_reports_nothing_for_a_clean_definition(tmp_path: Path) -> None:
    """A study whose rows all come from the current definition has no orphans."""
    results_root = tmp_path / "prune_clean"
    spec_file = _write_spec_file(tmp_path, results_root)
    runner.invoke(app, ["experiment", "plan", str(spec_file)])

    result = runner.invoke(app, ["experiment", "prune", str(spec_file), "--yes"])

    assert result.exit_code == 0, result.output
    assert "no orphan runs" in result.output
    assert _run_count(results_root / "experiments.sqlite") == 1


def test_prune_deletes_runs_a_superseded_definition_left_behind(tmp_path: Path) -> None:
    """A row the current definition does not generate is deleted.

    This is the shape a component-file edit or a dropped hyperparameter produces: the new
    identity mints new rows and the old ones stay plannable, so ``run`` would execute them
    and record results the analysis then filters out.
    """
    results_root = tmp_path / "prune_orphan"
    spec_file = _write_spec_file(tmp_path, results_root)
    runner.invoke(app, ["experiment", "plan", str(spec_file)])
    db_path = results_root / "experiments.sqlite"
    _orphan_run(db_path)
    assert _run_count(db_path) == 2

    result = runner.invoke(app, ["experiment", "prune", str(spec_file), "--yes"])

    assert result.exit_code == 0, result.output
    assert "deleted 1 orphan run(s)" in result.output
    assert _run_count(db_path) == 1


def test_prune_keeps_an_orphan_that_an_execution_touched(tmp_path: Path) -> None:
    """Results and live claims are not garbage, so a run with any execution survives."""
    results_root = tmp_path / "prune_executed"
    spec_file = _write_spec_file(tmp_path, results_root)
    runner.invoke(app, ["experiment", "plan", str(spec_file)])
    db_path = results_root / "experiments.sqlite"
    run_id = _orphan_run(db_path)
    with DatabaseManager(db_path) as db:
        db.initialize()
        db.link_execution_run(db.add_execution(hostname="somewhere"), run_id)

    result = runner.invoke(app, ["experiment", "prune", str(spec_file), "--yes"])

    assert result.exit_code == 0, result.output
    assert "no orphan runs" in result.output
    assert _run_count(db_path) == 2


def test_invalidate_all_unsatisfies_every_arm(tmp_path: Path) -> None:
    """``--all`` is the clean-re-run reset: every arm becomes unsatisfied, not just one.

    The complement of the ``--where`` case above, which deliberately leaves the unmatched
    arm satisfied.
    """
    db_path = tmp_path / "inv_all.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id, changed_run_id, stable_run_id = _seed_two_arm_experiment(db)

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--all"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert "Invalidated 2" in result.output

    with DatabaseManager(db_path) as db:
        db.initialize()
        unsatisfied = {run.id for run in db.list_runs_without_completed_execution(exp_id)}
    assert unsatisfied == {changed_run_id, stable_run_id}


def test_invalidate_all_leaves_other_experiments_alone(tmp_path: Path) -> None:
    """Scoped to ``--experiment``: a shared database must not lose an unrelated study."""
    db_path = tmp_path / "inv_all_scope.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        _seed_two_arm_experiment(db)
        other_exp = db.add_experiment("other-study")
        algo = db.get_component("algo")
        env = db.get_component("env")
        assert algo is not None and env is not None
        algo_ver = db.conn.execute(
            "SELECT id FROM ComponentVersions WHERE component_id = ? ORDER BY version_number DESC LIMIT 1",
            (algo.id,),
        ).fetchone()
        env_ver = db.conn.execute(
            "SELECT id FROM ComponentVersions WHERE component_id = ? ORDER BY version_number DESC LIMIT 1",
            (env.id,),
        ).fetchone()
        assert algo_ver is not None and env_ver is not None
        other_run = db.add_run(other_exp, algo_ver[0], env_ver[0], db.add_hyperparam_config({"algorithm": "other"}), seed=0)
        other_execution = db.add_execution(hostname="n3")
        db.link_execution_run(other_execution, other_run)
        db.update_execution_status(other_execution, "COMPLETED")

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--all", "--yes"],
    )
    assert result.exit_code == 0, result.output

    with DatabaseManager(db_path) as db:
        db.initialize()
        assert db.list_runs_without_completed_execution(other_exp) == []


def test_invalidate_all_requires_an_experiment(tmp_path: Path) -> None:
    """Without a slug ``--all`` would mean every experiment in the database, so it refuses."""
    db_path = tmp_path / "inv_all_noexp.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "invalidate", str(db_path), "--all"])

    assert result.exit_code == 1
    assert "--all requires --experiment" in result.output


def test_invalidate_all_rejects_a_where_clause(tmp_path: Path) -> None:
    """The two selectors express opposite intents, so combining them is an error, not a merge."""
    db_path = tmp_path / "inv_all_where.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        _seed_two_arm_experiment(db)

    result = runner.invoke(
        app,
        ["experiment", "invalidate", str(db_path), "--experiment", "bakeoff", "--all", "--where", "algorithm=gi_qrc"],
    )

    assert result.exit_code == 1
    assert "contradictory" in result.output
