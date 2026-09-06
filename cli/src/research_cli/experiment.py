from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import typer
from experiment_definition.db import DatabaseManager
from research_runner import ExperimentSpec, registered_specs

experiment_app = typer.Typer(help="Manage experiments.")


def _discover_specs(module: object, spec_name: str | None) -> dict[str, Callable[[], ExperimentSpec]]:
    specs = registered_specs(module)
    if not specs:
        typer.echo(
            "Error: no experiment specs registered. Decorate a factory with @spec.",
            err=True,
        )
        raise typer.Exit(code=1)
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


def _with_results_root(experiment_spec: ExperimentSpec, results_root: str | None) -> ExperimentSpec:
    """Apply a results-root override, keeping the spec's derived layout authoritative."""
    if results_root is None:
        return experiment_spec
    return replace(experiment_spec, results_root=Path(results_root).resolve())


def _render_version(version_id: int | None) -> str:
    """Render a component-version id, treating NULL as an honest 'unknown' rather than 'None'."""
    return "unknown" if version_id is None else str(version_id)


def _version_pair_sort_key(pair: tuple[int | None, int | None]) -> tuple[bool, int, bool, int]:
    """Order version pairs deterministically, grouping NULL-bearing pairs after known ones."""
    algo, env = pair
    return (algo is None, algo or 0, env is None, env or 0)


def _version_pair_newest_key(pair: tuple[int | None, int | None]) -> tuple[int, int]:
    """Rank version pairs by recency; NULL never outranks a known version id."""
    algo, env = pair
    return (-1 if algo is None else algo, -1 if env is None else env)


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
            exp_rows = db.list_experiments()

        for exp_row in exp_rows:
            all_runs = db.list_runs(exp_row.id)
            unsatisfied = db.list_unsatisfied_runs(exp_row.id)
            total = len(all_runs)
            pending = len(unsatisfied)
            completed = total - pending
            typer.echo(f"{exp_row.name}: {total} runs ({completed} completed, {pending} pending)")

            pairs = {(r.algo_version_id, r.env_version_id) for r in all_runs}
            if len(pairs) > 1:
                unsatisfied_ids = {r.id for r in unsatisfied}
                counts = {
                    pair: [r for r in all_runs if (r.algo_version_id, r.env_version_id) == pair]
                    for pair in pairs
                }
                # The pair analysis will actually read. Not simply the newest pair: a sync
                # mints a version and its unexecuted runs for any source edit, cosmetic
                # ones included, so the newest pair is routinely one nothing has run.
                with_results = [
                    pair for pair, runs in counts.items() if any(r.id not in unsatisfied_ids for r in runs)
                ]
                newest_pair = max(with_results, key=_version_pair_newest_key) if with_results else None
                for pair in sorted(pairs, key=_version_pair_sort_key):
                    pair_runs = counts[pair]
                    pair_total = len(pair_runs)
                    pair_pending = sum(1 for r in pair_runs if r.id in unsatisfied_ids)
                    pair_completed = pair_total - pair_pending
                    marker = " (newest with results)" if pair == newest_pair else ""
                    algo_display = _render_version(pair[0])
                    env_display = _render_version(pair[1])
                    typer.echo(
                        f"  algo_version={algo_display} env_version={env_display}: "
                        f"{pair_total} runs ({pair_completed} completed, {pair_pending} pending){marker}"
                    )


_ONLY_HELP = (
    "Restrict to runs matching key=value. Repeatable and AND-ed; comma-separated values OR "
    "within one clause, so seed=0,1,2 selects a range. Matches hyperparameters plus the "
    "seed and ablation columns."
)


@experiment_app.command("plan")
def plan(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    only: list[str] | None = typer.Option(None, "--only", help=_ONLY_HELP),
):
    only_clauses = _parse_key_value_options(only, "--only") if only else None

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        db_path = experiment_spec.db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_spec.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            exp_row = database.get_experiment(experiment_spec.experiment.name)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
                raise typer.Exit(code=1)
            batches = database.list_unsatisfied_run_batches(
                exp_row.id,
                max_runs_per_batch=experiment_spec.max_runs_per_batch,
                only=only_clauses,
            )

        typer.echo(f"Spec '{name}': {len(batches)} batch(es) planned")
        for i, batch in enumerate(batches):
            typer.echo(f"  Batch {i + 1}: {len(batch.run_ids)} run(s)")


@experiment_app.command("prune")
def prune(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Prune only the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Delete run rows an earlier definition left behind, so ``run`` does not execute them.

    A study planned but never run accumulates orphans whenever a component file or a
    hyperparameter changes: the new identity mints new rows and the old ones stay
    plannable, so ``run`` would execute them and record results under a stale component
    version that analysis then filters out. ``invalidate`` cannot reach them because they
    have no executions. Runs with any execution are never touched.
    """
    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        db_path = experiment_spec.db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        wanted = experiment_spec.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            exp_row = database.get_experiment(experiment_spec.experiment.name)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
                raise typer.Exit(code=1)
            orphans = database.list_orphan_runs(exp_row.id, wanted)

        if not orphans:
            typer.echo(f"Spec '{name}': no orphan runs.")
            continue

        def sort_key(pair: tuple[int | None, int | None]) -> tuple[int, int]:
            return (pair[0] or 0, pair[1] or 0)

        typer.echo(f"Spec '{name}': {len(orphans)} orphan run(s) in '{exp_row.name}'")
        for algo_vid, env_vid in sorted({(o.algo_version_id, o.env_version_id) for o in orphans}, key=sort_key):
            count = sum(1 for o in orphans if (o.algo_version_id, o.env_version_id) == (algo_vid, env_vid))
            typer.echo(f"  algo_version={algo_vid} env_version={env_vid}: {count} run(s)")
        if not yes:
            typer.confirm(f"Delete {len(orphans)} orphan run(s) from '{exp_row.name}'?", abort=True)

        with DatabaseManager(db_path) as database:
            database.initialize()
            deleted = database.delete_runs([o.id for o in orphans])
        typer.echo(f"Spec '{name}': deleted {deleted} orphan run(s)")


@experiment_app.command("run")
def run(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    max_runs: int | None = typer.Option(None, "--max-runs", help="Override max runs per batch."),
    reclaim_stale_after: float | None = typer.Option(
        None,
        "--reclaim-stale-after",
        help="Fail RUNNING executions older than this many seconds so their runs replan. Use after a crash; keep it longer than a real batch takes.",
    ),
    only: list[str] | None = typer.Option(None, "--only", help=_ONLY_HELP),
    # Unused here: the real JAX_PLATFORMS decision must happen in main.py before jax is
    # ever imported, so this is declared only so typer accepts/documents the flag.
    gpu: bool = typer.Option(False, "--gpu", help="Use the GPU backend for JAX instead of CPU."),
):
    from research_runner import run_experiment

    only_clauses = _parse_key_value_options(only, "--only") if only else None

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        max_runs_per_batch = max_runs if max_runs is not None else experiment_spec.max_runs_per_batch

        typer.echo(f"Running spec '{name}'...")
        roots = run_experiment(
            experiment_spec.db_path,
            experiment_spec.experiment,
            experiment_spec.train_fn,
            executions_root=experiment_spec.executions_root,
            metrics_db_path=experiment_spec.metrics_db_path,
            max_runs_per_batch=max_runs_per_batch,
            capture_git=experiment_spec.capture_git,
            reclaim_stale_after_seconds=reclaim_stale_after,
            static_parameter_names=experiment_spec.static_parameter_names,
            metric_subsample_factor=experiment_spec.metric_subsample_factor,
            only=only_clauses,
        )
        typer.echo(f"Spec '{name}': {len(roots)} execution(s) completed")
        for root in roots:
            typer.echo(f"  {root}")


@experiment_app.command("executions")
def executions(
    db_path: Path = typer.Argument(..., help="Path to the experiment database."),
    experiment: str | None = typer.Option(None, "--experiment", help="Filter by experiment name."),
    status: str | None = typer.Option(None, "--status", help="Filter by execution status."),
    git_commit: str | None = typer.Option(None, "--git-commit", help="Filter by git commit SHA."),
):
    with DatabaseManager(db_path) as db:
        db.initialize()
        experiment_id = None
        if experiment is not None:
            exp_row = db.get_experiment(experiment)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment}' not found.", err=True)
                raise typer.Exit(code=1)
            experiment_id = exp_row.id
        rows = db.list_executions(experiment_id, status=status, git_commit=git_commit)

    typer.echo(f"{'ID':<6} {'Status':<12} {'Hostname':<20} {'Start Time':<22} {'Git Commit':<10}")
    typer.echo("-" * 70)
    for row in rows:
        commit_display = row.git_commit[:8] if row.git_commit else "\u2014"
        hostname_display = row.hostname or "\u2014"
        start_display = row.start_time or "\u2014"
        typer.echo(f"{row.id:<6} {row.status:<12} {hostname_display:<20} {start_display:<22} {commit_display:<10}")


def _parse_key_value_options(clauses: list[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for clause in clauses:
        if "=" not in clause:
            typer.echo(f"Error: {flag} clause '{clause}' must be key=value.", err=True)
            raise typer.Exit(code=1)
        key, _, value = clause.partition("=")
        parsed[key] = value
    return parsed


@experiment_app.command("invalidate")
def invalidate(
    db_path: Path = typer.Argument(..., help="Path to the experiment database."),
    execution: list[int] | None = typer.Option(None, "--execution", help="Execution ID(s) to invalidate."),
    git_commit: str | None = typer.Option(None, "--git-commit", help="Invalidate all executions for a git commit."),
    experiment: str | None = typer.Option(None, "--experiment", help="Experiment slug. Required by --where and --all."),
    where: list[str] | None = typer.Option(
        None,
        "--where",
        help="Invalidate executions whose covered runs all match hyperparameter key=value (repeatable, requires --experiment).",
    ),
    invalidate_all: bool = typer.Option(
        False,
        "--all",
        help="Invalidate every execution of --experiment, for a deliberate clean re-run.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    if not execution and not git_commit and not where and not invalidate_all:
        typer.echo("Error: at least one of --execution, --git-commit, --where, or --all must be provided.", err=True)
        raise typer.Exit(code=1)
    if where and not experiment:
        typer.echo("Error: --where requires --experiment to scope the match.", err=True)
        raise typer.Exit(code=1)
    if invalidate_all and not experiment:
        typer.echo("Error: --all requires --experiment; it will not invalidate every experiment at once.", err=True)
        raise typer.Exit(code=1)
    if invalidate_all and where:
        typer.echo("Error: --all and --where are contradictory; pass one or the other.", err=True)
        raise typer.Exit(code=1)

    where_clauses = _parse_key_value_options(where, "--where") if where else {}
    where_display = ", ".join(f"{k}={v}" for k, v in where_clauses.items())

    matched_where_ids: list[int] = []
    matched_all_ids: list[int] = []
    with DatabaseManager(db_path) as db:
        db.initialize()
        if invalidate_all:
            assert experiment is not None
            exp_row = db.get_experiment(experiment)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment}' not found.", err=True)
                raise typer.Exit(code=1)
            matched_all_ids = [row.id for row in db.find_executions_for_experiment(exp_row.id)]
            if not matched_all_ids:
                typer.echo(f"Error: experiment '{experiment}' has no executions to invalidate.", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"Matched every execution of '{experiment}': {len(matched_all_ids)} execution(s)")
        if where_clauses:
            assert experiment is not None
            exp_row = db.get_experiment(experiment)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment}' not found.", err=True)
                raise typer.Exit(code=1)
            matched = db.find_executions_by_hyperparams(exp_row.id, where_clauses)
            if not matched:
                typer.echo(f"Error: --where {where_display} matched no executions in experiment '{experiment}'.", err=True)
                raise typer.Exit(code=1)
            matched_where_ids = [row.id for row in matched]
            typer.echo(f"Matched {len(matched_where_ids)} execution(s) for --where {where_display}: {', '.join(str(i) for i in matched_where_ids)}")

    parts = []
    if execution:
        parts.append(f"execution ID(s): {', '.join(str(e) for e in execution)}")
    if git_commit:
        parts.append(f"git commit: {git_commit[:8]}")
    if matched_where_ids:
        parts.append(f"hyperparameters ({where_display}): {len(matched_where_ids)} execution(s)")
    if matched_all_ids:
        parts.append(f"ALL of experiment '{experiment}': {len(matched_all_ids)} execution(s)")
    if not yes:
        typer.confirm(f"Invalidate executions matching {'; '.join(parts)}?", abort=True)

    invalidated: list[int] = []
    with DatabaseManager(db_path) as db:
        db.initialize()
        if execution:
            for eid in execution:
                if db.invalidate_execution(eid):
                    invalidated.append(eid)
        if git_commit:
            already = set(invalidated)
            invalidated.extend(eid for eid in db.invalidate_executions_by_commit(git_commit) if eid not in already)
        if matched_where_ids:
            already = set(invalidated)
            invalidated.extend(eid for eid in matched_where_ids if eid not in already and db.invalidate_execution(eid))
        if matched_all_ids:
            already = set(invalidated)
            invalidated.extend(eid for eid in matched_all_ids if eid not in already and db.invalidate_execution(eid))

    if not invalidated:
        typer.echo("Warning: no executions were invalidated.", err=True)
    else:
        typer.echo(f"Invalidated {len(invalidated)} execution(s): {', '.join(str(i) for i in invalidated)}")


@experiment_app.command("execute-batch")
def execute_batch_cmd(
    db_path: Path = typer.Argument(..., help="Path to the experiment database."),
    execution_id: int = typer.Option(..., "--execution-id", help="Planned execution ID to run."),
    spec_file: Path = typer.Option(..., "--spec-file", help="Path to the Python spec file."),
    spec: str = typer.Option(..., "--spec", help="Name of the spec factory."),
):
    from research_runner import execute_batch

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)
    if len(specs) != 1:
        typer.echo(f"Error: expected exactly one spec, found {len(specs)}.", err=True)
        raise typer.Exit(code=1)

    experiment_spec = next(iter(specs.values()))()

    with DatabaseManager(db_path) as database:
        database.initialize()
        planned = database.get_planned_execution(execution_id)

    if planned is None:
        typer.echo(f"Error: planned execution {execution_id} not found.", err=True)
        raise typer.Exit(code=1)

    root = execute_batch(
        db_path,
        planned,
        experiment_spec.train_fn,
        metrics_db_path=experiment_spec.metrics_db_path,
        capture_git=experiment_spec.capture_git,
        metric_subsample_factor=experiment_spec.metric_subsample_factor,
    )
    typer.echo(f"Execution {execution_id} completed: {root}")


@experiment_app.command("submit")
def submit(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    account: str | None = typer.Option(None, "--account", help="Slurm account."),
    partition: str | None = typer.Option(None, "--partition", help="Slurm partition."),
    time: str = typer.Option("2:59:00", "--time", help="Slurm time limit."),
    cpus_per_task: int = typer.Option(1, "--cpus-per-task", help="CPUs per task."),
    mem_per_cpu: str = typer.Option("4G", "--mem-per-cpu", help="Memory per CPU."),
    gpus: int | None = typer.Option(None, "--gpus", help="GPUs per task."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write script without submitting."),
    script_path: Path | None = typer.Option(None, "--script-path", help="Override output script path."),
):
    from research_cluster.config import SlurmConfig
    from research_cluster.submit import submit_experiment

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        db_path = experiment_spec.db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_spec.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            exp_row = database.get_experiment(experiment_spec.experiment.name)
            if exp_row is None:
                typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
                raise typer.Exit(code=1)
            planned = database.plan_experiment_execution_batches(
                exp_row.id,
                experiment_spec.executions_root,
                max_runs_per_batch=experiment_spec.max_runs_per_batch,
            )

        if not planned:
            typer.echo(f"Spec '{name}': no unsatisfied batches, skipping.")
            continue

        execution_ids = [p.execution_id for p in planned]
        config = SlurmConfig(
            account=account,
            partition=partition,
            time=time,
            cpus_per_task=cpus_per_task,
            mem_per_cpu=mem_per_cpu,
            gpus_per_task=gpus,
        )
        job_result = submit_experiment(
            config,
            execution_ids,
            db_path,
            spec_file,
            name,
            script_path=script_path,
            dry_run=dry_run,
        )

        if dry_run:
            typer.echo(f"Spec '{name}': {len(planned)} batch(es) planned (dry-run, not submitted)")
        else:
            typer.echo(f"Spec '{name}': submitted {len(planned)} batch(es) — {job_result}")
