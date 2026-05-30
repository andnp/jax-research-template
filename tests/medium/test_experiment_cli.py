from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from experiment_definition.db import DatabaseManager
from research_cli.main import app

runner = CliRunner()


def test_experiment_list_empty_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "list", str(db_path)])
    assert result.exit_code == 0
    assert "Name" in result.output
    assert "Description" in result.output


def test_experiment_list_with_experiments(tmp_path: Path):
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


def test_experiment_status_shows_run_counts(tmp_path: Path):
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
    assert "2 pending" in result.output


def _write_spec_file(tmp_path: Path, db_path: Path, executions_root: Path) -> Path:
    spec_file = tmp_path / "test_spec.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            from __future__ import annotations
            from pathlib import Path
            from research_runner.types import ExperimentSpec, ExecutionContext, ExecutionResult
            from experiment_definition.experiment import Experiment
            from experiment_definition.component import Component, ComponentType

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
                    db_path=Path("{db_path}"),
                    executions_root=Path("{executions_root}"),
                    capture_git=False,
                )

            def not_a_spec():
                return "I am not a spec"
        """),
    )
    return spec_file


def test_experiment_plan_with_spec(tmp_path: Path):
    db_path = tmp_path / "plan.sqlite"
    executions_root = tmp_path / "executions"

    spec_file = _write_spec_file(tmp_path, db_path, executions_root)

    result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output
    assert "batch" in result.output.lower()


def test_experiment_run_with_spec(tmp_path: Path):
    db_path = tmp_path / "run.sqlite"
    executions_root = tmp_path / "executions"

    spec_file = _write_spec_file(tmp_path, db_path, executions_root)

    result = runner.invoke(app, ["experiment", "run", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output
    assert "completed" in result.output.lower()


def test_experiment_run_with_spec_filter(tmp_path: Path):
    db_path = tmp_path / "run_filtered.sqlite"
    executions_root = tmp_path / "executions"

    spec_file = _write_spec_file(tmp_path, db_path, executions_root)

    result = runner.invoke(app, ["experiment", "run", str(spec_file), "--spec", "my_spec"])
    assert result.exit_code == 0, result.output
    assert "my_spec" in result.output


def test_spec_discovery_ignores_non_spec_functions(tmp_path: Path):
    db_path = tmp_path / "disc.sqlite"
    executions_root = tmp_path / "executions"

    spec_file = _write_spec_file(tmp_path, db_path, executions_root)
    module = __import__("importlib").util
    loader_spec = module.spec_from_file_location("test_disc", spec_file)
    loaded = module.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loaded)

    from research_cli.experiment import _discover_specs

    specs = _discover_specs(loaded, None)
    assert "my_spec" in specs
    assert "not_a_spec" not in specs


# ---------------------------------------------------------------------------
# Executions listing
# ---------------------------------------------------------------------------


def test_executions_shows_execution_table(tmp_path: Path):
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


def test_executions_empty_db(tmp_path: Path):
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


def test_invalidate_by_execution_id(tmp_path: Path):
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


def test_invalidate_by_git_commit(tmp_path: Path):
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


def test_invalidate_no_args_fails(tmp_path: Path):
    db_path = tmp_path / "inv_noargs.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()

    result = runner.invoke(app, ["experiment", "invalidate", str(db_path)])
    assert result.exit_code == 1


def test_invalidate_nonexistent_warns(tmp_path: Path):
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


def test_execute_batch_runs_planned_execution(tmp_path: Path):
    db_path = tmp_path / "eb.sqlite"
    executions_root = tmp_path / "executions"
    spec_file = _write_spec_file(tmp_path, db_path, executions_root)

    # Sync experiment and plan execution batches
    plan_result = runner.invoke(app, ["experiment", "plan", str(spec_file)])
    assert plan_result.exit_code == 0, plan_result.output

    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment("test-experiment")
        assert exp_row is not None
        planned = database.plan_experiment_execution_batches(
            exp_row.id, executions_root,
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


def test_execute_batch_nonexistent_execution_fails(tmp_path: Path):
    db_path = tmp_path / "eb_fail.sqlite"
    executions_root = tmp_path / "executions"
    spec_file = _write_spec_file(tmp_path, db_path, executions_root)

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


def test_submit_dry_run(tmp_path: Path):
    db_path = tmp_path / "submit.sqlite"
    executions_root = tmp_path / "executions"
    spec_file = _write_spec_file(tmp_path, db_path, executions_root)
    script_path = tmp_path / "slurm_submit.sh"

    result = runner.invoke(app, [
        "experiment", "submit", str(spec_file),
        "--dry-run",
        "--script-path", str(script_path),
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()
    assert script_path.exists()
