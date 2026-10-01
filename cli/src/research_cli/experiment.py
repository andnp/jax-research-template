from __future__ import annotations

import importlib.util
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import typer
from experiment_definition.db import DatabaseManager
from experiment_definition.ownership import local_owner
from experiment_definition.status import RunState, run_state
from research_analysis.plan import registered_plans
from research_analysis.plan_report import run_plan
from research_analysis.plan_snapshots import read_plan_snapshots, record_plan_snapshot
from research_analysis.report import ReportStatus
from research_analysis.report_files import report_directory, write_report_document
from research_analysis.report_json import report_to_json
from research_analysis.report_markdown import render_summary
from research_runner.registry import registered_specs
from research_runner.types import ExperimentSpec
from rich.console import Console
from rich.markdown import Markdown

from research_cli.reclaim import reclaim_dead
from research_cli.supervisor import GIB, SupervisorSettings, supervise

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


def _sync(experiment_spec: ExperimentSpec, db_path: Path) -> frozenset[int]:
    """Sync the spec, turning a config error in its parameters into a CLI error."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return experiment_spec.sync(db_path)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _reclaim_dead(experiment_spec: ExperimentSpec) -> str:
    with DatabaseManager(experiment_spec.db_path) as database:
        database.initialize()
        exp_row = database.get_experiment(experiment_spec.experiment.name)
        if exp_row is None:
            typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
            raise typer.Exit(code=1)
        return reclaim_dead(database, exp_row.id)


def _record_declared_plan(module: ModuleType, name: str, experiment_spec: ExperimentSpec) -> None:
    """Snapshot the spec's declared plan before its runs start, so reports can show it predates the results."""
    plan_factory = registered_plans(module).get(name)
    if plan_factory is not None:
        record_plan_snapshot(plan_factory(), experiment_spec.results_root)


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


def _preview_dry_run_batch_sizes(experiment_spec: ExperimentSpec, db_path: Path) -> list[int] | None:
    """Preview submit's unsatisfied batches read-only, without syncing or planning.

    Returns ``None`` when there is nothing synced yet to preview (no DB file, or the
    experiment has never been synced into it).
    """
    if not db_path.exists():
        return None
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        database = DatabaseManager(db_path)
        database._conn = connection
        try:
            exp_row = database.get_experiment(experiment_spec.experiment.name)
        except sqlite3.OperationalError:
            return None
        if exp_row is None:
            return None
        batches = database.list_unsatisfied_run_batches(
            exp_row.id,
            max_runs_per_batch=experiment_spec.max_runs_per_batch,
        )
        return [len(batch.run_ids) for batch in batches]
    finally:
        connection.close()


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
            statuses = db.list_run_execution_statuses(exp_row.id)
            state_of = {r.id: run_state(statuses.get(r.id, frozenset())) for r in all_runs}
            typer.echo(f"{exp_row.name}: {_describe_states(list(state_of.values()))}")

            pairs = {(r.algo_version_id, r.env_version_id) for r in all_runs}
            if len(pairs) > 1:
                counts = {
                    pair: [r for r in all_runs if (r.algo_version_id, r.env_version_id) == pair]
                    for pair in pairs
                }
                # The pair analysis will actually read. Not simply the newest pair: a sync
                # mints a version and its unexecuted runs for any source edit, cosmetic
                # ones included, so the newest pair is routinely one nothing has run.
                with_results = [
                    pair for pair, runs in counts.items() if any(state_of[r.id] is RunState.COMPLETED for r in runs)
                ]
                newest_pair = max(with_results, key=_version_pair_newest_key) if with_results else None
                for pair in sorted(pairs, key=_version_pair_sort_key):
                    marker = " (newest with results)" if pair == newest_pair else ""
                    algo_display = _render_version(pair[0])
                    env_display = _render_version(pair[1])
                    typer.echo(
                        f"  algo_version={algo_display} env_version={env_display}: "
                        f"{_describe_states([state_of[r.id] for r in counts[pair]])}{marker}"
                    )


def _describe_states(states: list[RunState]):
    counts = Counter(states)
    return (
        f"{len(states)} runs ({counts[RunState.COMPLETED]} completed, "
        f"{counts[RunState.FAILED]} failed, {counts[RunState.PENDING]} pending)"
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

        _sync(experiment_spec, db_path)

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

        wanted = _sync(experiment_spec, db_path)

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
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Thread budget: run pending groups in parallel worker processes whose NPROC sums to at most this.",
    ),
    cores_per_worker: int | None = typer.Option(
        None, "--cores-per-worker", min=1, help="With --workers, give every worker this NPROC instead of spreading the budget."
    ),
    worker_memory: float = typer.Option(
        4.0, "--worker-memory", help="With --workers, GiB a worker is projected to need until one has reported its peak."
    ),
    memory_reserve: float | None = typer.Option(
        None, "--memory-reserve", help="With --workers, GiB of memory kept free (default 15% of MemTotal)."
    ),
    group: str | None = typer.Option(None, "--group", hidden=True, help="Run only the batches with this group key."),
    # Unused here: the real JAX_PLATFORMS decision must happen in main.py before jax is
    # ever imported, so this is declared only so typer accepts/documents the flag.
    gpu: bool = typer.Option(False, "--gpu", help="Use the GPU backend for JAX instead of CPU."),
):
    from research_runner.runner import run_experiment

    only_clauses = _parse_key_value_options(only, "--only") if only else None

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)

    if workers is not None:
        if gpu:
            typer.echo("Error: --workers runs CPU workers; drop --gpu.", err=True)
            raise typer.Exit(code=1)
        if cores_per_worker is not None and cores_per_worker > workers:
            typer.echo(f"Error: --cores-per-worker {cores_per_worker} exceeds --workers {workers}.", err=True)
            raise typer.Exit(code=1)
        if len(specs) != 1:
            typer.echo(f"Error: --workers runs one spec; name it with --spec. Available: {', '.join(sorted(specs))}", err=True)
            raise typer.Exit(code=1)
        name, factory = next(iter(specs.items()))
        experiment_spec = _with_results_root(factory(), results_root)
        _sync(experiment_spec, experiment_spec.db_path)
        _record_declared_plan(module, name, experiment_spec)
        settings = SupervisorSettings(
            spec_file=spec_file,
            spec_name=name,
            results_root=results_root,
            budget=workers,
            cores_per_worker=cores_per_worker,
            max_runs=max_runs,
            worker_memory_bytes=int(worker_memory * GIB),
            memory_reserve_bytes=None if memory_reserve is None else int(memory_reserve * GIB),
            only=only or [],
            only_clauses=only_clauses,
            reclaim_stale_after=reclaim_stale_after,
        )
        raise typer.Exit(code=supervise(experiment_spec, settings))

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        _sync(experiment_spec, experiment_spec.db_path)
        _record_declared_plan(module, name, experiment_spec)
        if group is None:
            typer.echo(_reclaim_dead(experiment_spec))
        typer.echo(f"Running spec '{name}'...")
        roots = run_experiment(
            experiment_spec,
            max_runs_per_batch=max_runs,
            reclaim_stale_after_seconds=reclaim_stale_after,
            only=only_clauses,
            group=group,
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
        help="Invalidate executions whose covered runs all match key=value over hyperparameters, seed or ablation; a,b lists alternatives (repeatable, requires --experiment).",
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

    with DatabaseManager(db_path) as db:
        db.initialize()
        commit_ids = db.find_finished_execution_ids_by_commit(git_commit) if git_commit else []
        selected = dict.fromkeys([*(execution or []), *commit_ids, *matched_where_ids, *matched_all_ids])
        invalidated = db.invalidate_executions(selected)

    if not invalidated:
        typer.echo("Warning: no executions were invalidated.", err=True)
    else:
        typer.echo(f"Invalidated {len(invalidated)} execution(s): {', '.join(str(i) for i in invalidated)}")


@experiment_app.command("execute-batch")
def execute_batch_cmd(
    execution_id: int = typer.Option(..., "--execution-id", help="Planned execution ID to run."),
    spec_file: Path = typer.Option(..., "--spec-file", help="Path to the Python spec file."),
    spec: str = typer.Option(..., "--spec", help="Name of the spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
):
    from research_runner.runner import execute_batch

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)
    if len(specs) != 1:
        typer.echo(f"Error: expected exactly one spec, found {len(specs)}.", err=True)
        raise typer.Exit(code=1)

    experiment_spec = _with_results_root(next(iter(specs.values()))(), results_root)

    with DatabaseManager(experiment_spec.db_path) as database:
        database.initialize()
        planned = database.get_planned_execution(execution_id)

    if planned is None:
        typer.echo(f"Error: planned execution {execution_id} not found.", err=True)
        raise typer.Exit(code=1)

    root = execute_batch(
        experiment_spec.db_path,
        planned,
        experiment_spec,
        capture_git=experiment_spec.capture_git,
    )
    typer.echo(f"Execution {execution_id} completed: {root}")


@experiment_app.command("submit")
def submit(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Run only the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    account: str | None = typer.Option(None, "--account", help="Slurm account."),
    partition: str | None = typer.Option(None, "--partition", help="Slurm partition."),
    time: str | None = typer.Option(None, "--time", help="Slurm time limit (default 2:59:00)."),
    cpus_per_task: int | None = typer.Option(None, "--cpus-per-task", help="CPUs per task (default 1)."),
    mem_per_cpu: str | None = typer.Option(None, "--mem-per-cpu", help="Memory per CPU (default 4G)."),
    gpus: int | None = typer.Option(None, "--gpus", help="GPUs per task."),
    slurm_config: Path | None = typer.Option(
        None, "--slurm-config", help="JSON SlurmConfig file; the flags above override its values."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write script without submitting."),
    script_path: Path | None = typer.Option(None, "--script-path", help="Override output script path."),
):
    from research_cluster.config import SlurmConfig
    from research_cluster.submit import submit_experiment

    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)
    planner_owner = local_owner()

    for name, factory in specs.items():
        experiment_spec = _with_results_root(factory(), results_root)
        db_path = experiment_spec.db_path

        # A real plan claims runs PENDING and commits; a dry run must leave the DB
        # (and, when nothing is synced yet, the DB file itself) untouched. Branch
        # before any mutating setup rather than syncing/initializing unconditionally.
        if dry_run:
            batch_sizes = _preview_dry_run_batch_sizes(experiment_spec, db_path)
            if batch_sizes is None:
                typer.echo(
                    f"Spec '{name}': nothing planned yet ({db_path} not synced). "
                    "A real submit would sync the spec, plan its unsatisfied run "
                    "batches, and write a Slurm script for them."
                )
                continue
            if not batch_sizes:
                typer.echo(f"Spec '{name}': no unsatisfied batches, skipping.")
                continue
            execution_ids = list(range(1, len(batch_sizes) + 1))
        else:
            _sync(experiment_spec, db_path)
            _record_declared_plan(module, name, experiment_spec)

            with DatabaseManager(db_path) as database:
                database.initialize()
                exp_row = database.get_experiment(experiment_spec.experiment.name)
                if exp_row is None:
                    typer.echo(f"Error: experiment '{experiment_spec.experiment.name}' not synced.", err=True)
                    raise typer.Exit(code=1)
                typer.echo(reclaim_dead(database, exp_row.id))
                planned = database.plan_experiment_execution_batches(
                    exp_row.id,
                    experiment_spec.executions_root,
                    max_runs_per_batch=experiment_spec.max_runs_per_batch,
                    owner=planner_owner,
                )
                if not planned:
                    typer.echo(f"Spec '{name}': no unsatisfied batches, skipping.")
                    continue
                execution_ids = [p.execution_id for p in planned]

        base = SlurmConfig() if slurm_config is None else SlurmConfig.from_json(str(slurm_config))
        config = replace(
            base,
            account=base.account if account is None else account,
            partition=base.partition if partition is None else partition,
            time=base.time if time is None else time,
            cpus_per_task=base.cpus_per_task if cpus_per_task is None else cpus_per_task,
            mem_per_cpu=base.mem_per_cpu if mem_per_cpu is None else mem_per_cpu,
            gpus_per_task=base.gpus_per_task if gpus is None else gpus,
        )
        job_result = submit_experiment(
            config,
            execution_ids,
            db_path,
            spec_file,
            name,
            planner_owner=planner_owner,
            script_path=script_path,
            results_root=experiment_spec.results_root,
            dry_run=dry_run,
        )

        if dry_run:
            typer.echo(
                f"Spec '{name}': {len(execution_ids)} batch(es) would be planned "
                "(dry-run, not submitted; execution IDs in the script are provisional)"
            )
        else:
            typer.echo(f"Spec '{name}': submitted {len(execution_ids)} batch(es) as Slurm job {job_result}")


_REPORT_EXIT_CODES = {ReportStatus.OK: 0, ReportStatus.LOAD_FAILED: 2, ReportStatus.ANALYSIS_FAILED: 3}


@experiment_app.command("report")
def report(
    spec_file: Path = typer.Argument(..., help="Path to the Python spec file."),
    spec: str | None = typer.Option(None, "--spec", help="Report the named spec factory."),
    results_root: str | None = typer.Option(None, "--results-root", help="Override the results root."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Where under the results root to write the report; defaults to <results_root>/reports/<experiment>/<UTC time>/."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report JSON to stdout instead of a summary."),
):
    """Run the spec's declared analysis plan; exit 2 on fatal load diagnostics, 3 on a failed analysis."""
    module = _load_spec_module(spec_file)
    specs = _discover_specs(module, spec)
    if len(specs) != 1:
        typer.echo(f"Error: name one spec with --spec. Available: {', '.join(sorted(specs))}", err=True)
        raise typer.Exit(code=1)
    name, factory = next(iter(specs.items()))
    plans = registered_plans(module)
    if name not in plans:
        typer.echo(f"Error: spec '{name}' has no declared analysis plan; declare one with @analysis_plan({name}).", err=True)
        raise typer.Exit(code=1)

    experiment_spec = _with_results_root(factory(), results_root)
    if not experiment_spec.db_path.is_file():
        typer.echo(f"Error: no experiment database at {experiment_spec.db_path}.", err=True)
        raise typer.Exit(code=1)
    if output_dir is not None and not output_dir.resolve().is_relative_to(experiment_spec.results_root.resolve()):
        typer.echo(f"Error: --output-dir {output_dir} is outside the results root {experiment_spec.results_root}.", err=True)
        raise typer.Exit(code=1)
    slug = experiment_spec.experiment.name
    document = run_plan(experiment_spec.db_path, slug, plans[name](), read_plan_snapshots(experiment_spec.results_root))

    directory = output_dir or report_directory(experiment_spec.results_root, slug)
    document = write_report_document(document, directory)

    if as_json:
        typer.echo(report_to_json(document), nl=False)
    else:
        Console().print(Markdown(render_summary(document) + f"\n- Report: `{directory}`"))
    raise typer.Exit(code=_REPORT_EXIT_CODES[document.status])
