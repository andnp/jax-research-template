"""Workspace management commands for the research CLI."""

import importlib.resources
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import typer

from research_cli.config import ResearchConfigError, load_research_config

workspace_app = typer.Typer(help="Manage research shell workspaces.")

type CommandRunner = Callable[[list[str], Path, bool], None]


class WorkspaceRepairError(ValueError):
    """Raised when ``workspace repair`` cannot resolve a workspace target."""


@dataclass(frozen=True, slots=True)
class ResolvedRepairTarget:
    workspace_root: Path
    core_path: Path
    submodule_path: Path


def _load_template(filename: str) -> str:
    """Load a template file from the templates package.

    Args:
        filename: Template filename within the templates package.
    """
    return importlib.resources.files("research_cli.templates").joinpath(filename).read_text(encoding="utf-8")


def _run(args: list[str], cwd: Path, dry_run: bool) -> None:
    """Run a subprocess command, or print it when dry_run is True.

    Args:
        args: Command and arguments.
        cwd: Working directory for the command.
        dry_run: When True, print the command instead of executing.
    """
    cmd = " ".join(args)
    if dry_run:
        typer.echo(f"  [dry-run] {cmd}")
        return
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"Error running `{cmd}`:\n{result.stderr}", err=True)
        raise typer.Exit(code=1)


def _write(path: Path, content: str, dry_run: bool) -> None:
    """Write a file, or describe the action when dry_run is True.

    Args:
        path: Absolute path to write.
        content: File content.
        dry_run: When True, print description instead of writing.
    """
    if dry_run:
        typer.echo(f"  [dry-run] write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _find_workspace_root(start_path: Path) -> Path | None:
    resolved_start_path = start_path.resolve()
    for candidate in (resolved_start_path, *resolved_start_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "projects").is_dir():
            return candidate

    for candidate in (resolved_start_path, *resolved_start_path.parents):
        if (candidate / "research.yaml").exists():
            return candidate

    return None


def _resolve_repair_target(start_path: Path) -> ResolvedRepairTarget:
    workspace_root = _find_workspace_root(start_path)
    if workspace_root is None:
        raise WorkspaceRepairError(
            f"Could not find a research workspace from '{start_path.resolve()}'. "
            "Run this command from a workspace root or a directory inside a workspace containing research.yaml.",
        )

    resolved_workspace_root = workspace_root.resolve()
    config_path = resolved_workspace_root / "research.yaml"
    config = load_research_config(config_path)
    resolved_core_path = (config.core_path if config.core_path.is_absolute() else resolved_workspace_root / config.core_path).resolve(strict=False)
    if not resolved_core_path.exists():
        raise WorkspaceRepairError(
            f"Configured core_path '{config.core_path}' resolves to '{resolved_core_path}', which does not exist. "
            "Update research.yaml or create the Core checkout at that location.",
        )

    try:
        submodule_path = resolved_core_path.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise WorkspaceRepairError(
            f"Configured core_path '{config.core_path}' resolves to '{resolved_core_path}', which is outside workspace root '{resolved_workspace_root}'. "
            "Update research.yaml to point at the Core submodule inside this workspace.",
        ) from exc

    return ResolvedRepairTarget(
        workspace_root=resolved_workspace_root,
        core_path=resolved_core_path,
        submodule_path=submodule_path,
    )


def _repair_core_checkout(
    repair_target: ResolvedRepairTarget,
    dry_run: bool,
    run_command: CommandRunner = _run,
) -> None:
    commands = (
        ["git", "-C", str(repair_target.core_path), "clean", "-ffd"],
        ["git", "submodule", "update", "--force", "--checkout", "--", str(repair_target.submodule_path)],
    )
    for args in commands:
        run_command(args, repair_target.workspace_root, dry_run)


@workspace_app.command()
def init(
    name: str = typer.Argument(..., help="Name of the new research shell workspace."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Parent directory in which to create the workspace."),  # noqa: B008
    core_url: str | None = typer.Option(None, "--core-url", help="Git URL of research-core to add as a submodule. Omit to skip submodule setup."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview all actions without modifying the filesystem."),
) -> None:
    """Initialise a new Hub-and-Spoke research shell workspace.

    Creates a new directory <name> containing:
      - An initialised git repository.
      - A root pyproject.toml configured as a uv workspace.
      - A projects/ directory for per-experiment git repos.
      - (Optional) research-core added as a git submodule at core/.
      - A research.yaml configuration stub.

    Pass --dry-run to preview all steps without making any changes.

    Example:
        research workspace init my-phd-research --core-url https://github.com/andnpatterson/research-core
    """
    workspace_dir = (path / name).resolve()

    if workspace_dir.exists() and not dry_run:
        typer.echo(f"Error: '{workspace_dir}' already exists.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'[dry-run] ' if dry_run else ''}Initialising workspace '{name}' at {workspace_dir}")

    if not dry_run:
        workspace_dir.mkdir(parents=True)

    # Initialise git repository.
    _run(["git", "init"], cwd=workspace_dir, dry_run=dry_run)

    # Create the projects/ directory so uv workspace glob resolves cleanly.
    projects_dir = workspace_dir / "projects"
    if not dry_run:
        projects_dir.mkdir(exist_ok=True)
        (projects_dir / ".gitkeep").touch()
    else:
        typer.echo(f"  [dry-run] mkdir {projects_dir}")

    # Add research-core as a submodule when a URL is provided.
    if core_url:
        _run(["git", "submodule", "add", core_url, "core"], cwd=workspace_dir, dry_run=dry_run)
    else:
        typer.echo("  Skipping submodule setup (no --core-url provided). Run `git submodule add <url> core` later.")

    # Write the root pyproject.toml.
    _write(workspace_dir / "pyproject.toml", _load_template("pyproject.toml.tpl").format(name=name), dry_run=dry_run)

    # Write the research.yaml configuration stub.
    _write(workspace_dir / "research.yaml", _load_template("research.yaml.tpl"), dry_run=dry_run)

    # Write .gitignore.
    _write(workspace_dir / ".gitignore", _load_template("gitignore.tpl"), dry_run=dry_run)

    typer.echo(f"{'[dry-run] ' if dry_run else ''}✓ Workspace '{name}' ready.")
    if not core_url:
        typer.echo("  Tip: add research-core with `git submodule add <url> core` then run `uv sync`.")


@workspace_app.command()
def repair(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview the Core repair action plan without mutating the workspace."),
) -> None:
    """Repair the configured Core checkout to the workspace-recorded submodule revision."""
    try:
        repair_target = _resolve_repair_target(Path.cwd())
    except (ResearchConfigError, WorkspaceRepairError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _repair_core_checkout(repair_target, dry_run=dry_run)

    if not dry_run:
        typer.echo(f"✓ Repaired configured Core checkout at '{repair_target.core_path}'.")
