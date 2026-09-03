import subprocess
from pathlib import Path

import typer
from copier import run_copy

from research_cli.config import ResearchConfigError, load_research_config
from research_cli.workspace import WorkspaceResolutionError, resolve_workspace_root

project_app = typer.Typer(help="Manage projects within a research workspace.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _template_root() -> Path:
    return (_repo_root() / "templates").resolve()


def _workspace_root(cwd: Path | None = None) -> Path:
    try:
        workspace_root = resolve_workspace_root(cwd or Path.cwd())
    except WorkspaceResolutionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not (workspace_root / "projects").is_dir():
        typer.echo(f"Error: resolved workspace root '{workspace_root}' does not contain 'projects/'.", err=True)
        raise typer.Exit(code=1)
    return workspace_root


def _run(args: list[str], cwd: Path) -> None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        typer.echo(f"Error: failed to execute '{args[0]}': {exc.strerror}.", err=True)
        raise typer.Exit(code=1) from exc

    if result.returncode != 0:
        cmd = " ".join(args)
        typer.echo(f"Error running `{cmd}`:\n{result.stderr}", err=True)
        raise typer.Exit(code=1)


def _github_repo_create_command(project_root: Path, github_repo: str) -> list[str]:
    return [
        "gh",
        "repo",
        "create",
        github_repo,
        "--private",
        "--source",
        str(project_root),
        "--remote",
        "origin",
    ]


def _github_repo_url(github_repo: str) -> str:
    return f"https://github.com/{github_repo}.git"


def _submodule_add_command(workspace_root: Path, project_root: Path, github_repo: str) -> list[str]:
    project_path = project_root.relative_to(workspace_root)
    return ["git", "submodule", "add", "--force", _github_repo_url(github_repo), str(project_path)]


def _project_github_repo(workspace_root: Path, name: str, explicit_repo: str | None) -> str | None:
    if explicit_repo is not None:
        return explicit_repo

    config_path = workspace_root / "research.yaml"
    if not config_path.is_file():
        return None

    try:
        config = load_research_config(config_path)
    except ResearchConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not config.auto_create_private_project_repos:
        return None
    if config.default_github_org is None:
        typer.echo(
            "Error: auto_create_private_project_repos is enabled, but default_github_org is not configured in research.yaml.",
            err=True,
        )
        raise typer.Exit(code=1)
    return f"{config.default_github_org}/{name}"


@project_app.command()
def create(
    name: str = typer.Argument(..., help="Name of the project to create under projects/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview resolved paths without creating anything."),
    github_repo: str | None = typer.Option(None, "--github-repo", help="GitHub repo slug to create after rendering."),
) -> None:
    workspace_root = _workspace_root()
    projects_root = (workspace_root / "projects").resolve()
    template_root = _template_root()
    project_root = (projects_root / name).resolve()

    if not template_root.is_dir():
        typer.echo(f"Error: template root '{template_root}' does not exist.", err=True)
        raise typer.Exit(code=1)

    if project_root.exists():
        typer.echo(f"Error: '{project_root}' already exists.", err=True)
        raise typer.Exit(code=1)

    selected_github_repo = _project_github_repo(workspace_root, name, github_repo)

    typer.echo(f"{'[dry-run] ' if dry_run else ''}Create project '{name}'")
    typer.echo(f"  Workspace root: {workspace_root}")
    typer.echo(f"  Template root: {template_root}")
    typer.echo(f"  Target path: {project_root}")
    typer.echo(
        "  Git ownership: the shell repo keeps shared workspace files such as research.yaml and uv.lock; "
        "this command still runs 'git init' inside the new child project."
    )
    if selected_github_repo is not None:
        github_command = _github_repo_create_command(project_root, selected_github_repo)
        typer.echo(f"  GitHub repo: {' '.join(github_command)}")
        typer.echo(f"  Submodule: {' '.join(_submodule_add_command(workspace_root, project_root, selected_github_repo))}")

    if dry_run:
        return

    run_copy(str(template_root), str(projects_root), data={"project_name": name}, defaults=True)

    if not project_root.is_dir():
        typer.echo(f"Error: project render did not create '{project_root}'.", err=True)
        raise typer.Exit(code=1)

    _run(["git", "init"], cwd=project_root)

    if selected_github_repo is not None:
        _run(["git", "add", "."], cwd=project_root)
        _run(["git", "commit", "-m", "chore: initialize project"], cwd=project_root)
        _run(_github_repo_create_command(project_root, selected_github_repo), cwd=project_root)
        _run(_submodule_add_command(workspace_root, project_root, selected_github_repo), cwd=workspace_root)
