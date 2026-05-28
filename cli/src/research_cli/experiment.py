from __future__ import annotations

import importlib.util
import sys
import typing
from collections.abc import Callable
from pathlib import Path

import typer

from experiment_definition.db import DatabaseManager, ExperimentRow
from research_runner.types import ExperimentSpec

experiment_app = typer.Typer(help="Manage experiments.")


def _discover_specs(module: object, spec_name: str | None) -> dict[str, Callable[[], ExperimentSpec]]:
    specs: dict[str, Callable[[], ExperimentSpec]] = {}
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if not callable(obj):
            continue
        try:
            hints = typing.get_type_hints(obj)
        except Exception:
            continue
        ret = hints.get("return")
        if ret is ExperimentSpec:
            specs[attr_name] = obj
    if spec_name is not None:
        if spec_name not in specs:
            typer.echo(f"Error: spec '{spec_name}' not found. Available: {', '.join(sorted(specs))}", err=True)
            raise typer.Exit(code=1)
        specs = {spec_name: specs[spec_name]}
    return specs


def _find_project_root(start: Path):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _load_spec_module(spec_file: Path):
    resolved = spec_file.resolve()
    if not resolved.is_file():
        typer.echo(f"Error: spec file '{resolved}' not found.", err=True)
        raise typer.Exit(code=1)

    project_root = _find_project_root(resolved.parent)
    if project_root is not None and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    module_name = resolved.stem
    loader_spec = importlib.util.spec_from_file_location(module_name, resolved)
    if loader_spec is None or loader_spec.loader is None:
        typer.echo(f"Error: cannot load '{resolved}'.", err=True)
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(loader_spec)
    sys.modules[module_name] = module
    loader_spec.loader.exec_module(module)
    return module


@experiment_app.command("list")
def list_experiments(
    db_path: Path = typer.Argument(..., help="Path to the experiment database."),
):
    with DatabaseManager(db_path) as db:
        db.initialize()
        rows = db.conn.execute(
            "SELECT name, description FROM Experiments ORDER BY name",
        ).fetchall()

    typer.echo(f"{'Name':<30} Description")
    typer.echo("-" * 60)
    for row in rows:
        name = row[0]
        desc = row[1] or ""
        typer.echo(f"{name:<30} {desc}")


@experiment_app.command("status")
def status(
    db_path: Path = typer.Argument(..., help="Path to the experiment database."),
    experiment: str | None = typer.Option(None, "--experiment", help="Filter by experiment slug."),
):
    with DatabaseManager(db_path) as db:
        db.initialize()
        if experiment:
            exp_row = db.get_experiment(experiment)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment}' not found.", err=True)
                raise typer.Exit(code=1)
            exp_rows = [exp_row]
        else:
            raw = db.conn.execute(
                "SELECT id, name, description, created_at FROM Experiments ORDER BY name",
            ).fetchall()
            exp_rows = [ExperimentRow(*r) for r in raw]

        for exp_row in exp_rows:
            all_runs = db.list_runs(exp_row.id)
            unsatisfied = db.list_unsatisfied_runs(exp_row.id)
            total = len(all_runs)
            pending = len(unsatisfied)
            completed = total - pending
            typer.echo(f"{exp_row.name}: {total} runs ({completed} completed, {pending} pending)")


@experiment_app.command("plan")
def plan(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    db: str | None = typer.Option(None, "--db", help="Override database path."),
):
    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = factory()
        db_path = Path(db) if db else experiment_spec.db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_spec.experiment.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            exp_row = database.get_experiment(experiment_spec.experiment.name)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
                raise typer.Exit(code=1)
            batches = database.list_unsatisfied_run_batches(
                exp_row.id,
                max_runs_per_batch=experiment_spec.max_runs_per_batch,
            )

        typer.echo(f"Spec '{name}': {len(batches)} batch(es) planned")
        for i, batch in enumerate(batches):
            typer.echo(f"  Batch {i + 1}: {len(batch.run_ids)} run(s)")


@experiment_app.command("run")
def run(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    db: str | None = typer.Option(None, "--db", help="Override database path."),
    executions_root: str | None = typer.Option(None, "--executions-root", help="Override executions root."),
    max_runs: int | None = typer.Option(None, "--max-runs", help="Override max runs per batch."),
):
    from research_runner.runner import run_experiment

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = factory()
        db_path = Path(db) if db else experiment_spec.db_path
        exec_root = Path(executions_root) if executions_root else experiment_spec.executions_root
        max_runs_per_batch = max_runs if max_runs is not None else experiment_spec.max_runs_per_batch

        typer.echo(f"Running spec '{name}'...")
        roots = run_experiment(
            db_path,
            experiment_spec.experiment,
            experiment_spec.train_fn,
            executions_root=exec_root,
            metrics_db_path=experiment_spec.metrics_db_path,
            max_runs_per_batch=max_runs_per_batch,
            capture_git=experiment_spec.capture_git,
        )
        typer.echo(f"Spec '{name}': {len(roots)} execution(s) completed")
        for root in roots:
            typer.echo(f"  {root}")
