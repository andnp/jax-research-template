"""Small tests for the project create command surface."""

from pathlib import Path

import pytest
from research_cli import project as project_module
from research_cli.main import app
from research_cli.project import _template_root
from typer.testing import CliRunner

runner = CliRunner()


def test_research_help_lists_project_command() -> None:
    """The root CLI help must expose the project command group."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "project" in result.output


def test_project_create_dry_run_reports_normalized_target_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must resolve the target under projects/<name> from the workspace root."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"Template root: {_template_root()}" in result.output
    assert f"Target path: {workspace_root / 'projects' / 'demo'}" in result.output
    assert not (workspace_root / "projects" / "demo").exists()


def test_project_create_dry_run_does_not_render_or_invoke_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must not call Copier or spawn subprocesses."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    monkeypatch.chdir(workspace_root)

    calls: list[str] = []

    def fake_run_copy(*args, **kwargs):
        calls.append("copier")

    def fake_run(args: list[str], cwd: Path):
        calls.append("subprocess")

    monkeypatch.setattr(project_module, "run_copy", fake_run_copy)
    monkeypatch.setattr(project_module, "_run", fake_run)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run", "--github-repo", "acme/demo"])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert not (workspace_root / "projects" / "demo").exists()
    assert "GitHub repo: gh repo create acme/demo --private" in result.output


def test_project_create_fails_without_projects_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must fail fast outside a workspace root."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code != 0
    assert "workspace root containing 'projects/'" in result.output


def test_project_create_fails_if_template_root_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must fail fast when the repository templates root is missing."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    monkeypatch.chdir(workspace_root)
    monkeypatch.setattr(project_module, "_template_root", lambda: workspace_root / "missing-templates")

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code != 0
    assert "missing-templates" in result.output


def test_project_create_fails_if_project_already_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must fail fast when the target project already exists."""
    workspace_root = tmp_path.resolve()
    project_root = workspace_root / "projects" / "demo"
    project_root.mkdir(parents=True)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code != 0
    assert str(project_root) in result.output


def test_project_create_renders_then_initializes_git_without_github_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run without --github-repo must not invoke gh."""
    workspace_root = tmp_path.resolve()
    projects_root = workspace_root / "projects"
    project_root = projects_root / "demo"
    projects_root.mkdir()
    monkeypatch.chdir(workspace_root)

    calls: list[tuple[str, object, object]] = []

    def fake_run_copy(src_path: str, dst_path: str, data: dict[str, str] | None = None, **kwargs):
        calls.append(("copier", Path(src_path), Path(dst_path)))
        assert data == {"project_name": "demo"}
        assert kwargs == {"defaults": True}
        project_root.mkdir()
        (project_root / "README.md").write_text("# demo\n", encoding="utf-8")

    def fake_run(args: list[str], cwd: Path):
        calls.append((args[0], args, cwd))

    monkeypatch.setattr(project_module, "run_copy", fake_run_copy)
    monkeypatch.setattr(project_module, "_run", fake_run)

    result = runner.invoke(app, ["project", "create", "demo"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("copier", _template_root(), projects_root),
        ("git", ["git", "init"], project_root),
    ]


def test_project_create_renders_then_initializes_git_then_creates_github_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run with --github-repo must create the remote after git init."""
    workspace_root = tmp_path.resolve()
    projects_root = workspace_root / "projects"
    project_root = projects_root / "demo"
    projects_root.mkdir()
    monkeypatch.chdir(workspace_root)

    calls: list[tuple[str, object, object]] = []

    def fake_run_copy(src_path: str, dst_path: str, data: dict[str, str] | None = None, **kwargs):
        calls.append(("copier", Path(src_path), Path(dst_path)))
        assert data == {"project_name": "demo"}
        assert kwargs == {"defaults": True}
        project_root.mkdir()
        (project_root / "README.md").write_text("# demo\n", encoding="utf-8")

    def fake_run(args: list[str], cwd: Path):
        calls.append((args[0], args, cwd))

    monkeypatch.setattr(project_module, "run_copy", fake_run_copy)
    monkeypatch.setattr(project_module, "_run", fake_run)

    result = runner.invoke(app, ["project", "create", "demo", "--github-repo", "acme/demo"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("copier", _template_root(), projects_root),
        ("git", ["git", "init"], project_root),
        (
            "gh",
            [
                "gh",
                "repo",
                "create",
                "acme/demo",
                "--private",
                "--source",
                str(project_root),
                "--remote",
                "origin",
            ],
            project_root,
        ),
    ]
