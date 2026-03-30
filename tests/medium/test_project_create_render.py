"""Medium integration tests for project creation."""

from pathlib import Path

import pytest
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_workspace_config(workspace_root: Path) -> None:
    (workspace_root / "research.yaml").write_text("core_path: core\nstorage_backend: local\n", encoding="utf-8")


def test_project_create_renders_template_and_initializes_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must render the template into projects/<name> and initialize git."""
    workspace_root = tmp_path.resolve()
    project_root = workspace_root / "projects" / "demo"
    (workspace_root / "projects").mkdir()
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo"])

    assert result.exit_code == 0, result.output
    assert (project_root / "README.md").is_file()
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "train.py").is_file()
    assert (project_root / ".git").is_dir()
    assert not (project_root / "demo").exists()

    readme = (project_root / "README.md").read_text(encoding="utf-8")
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert readme.startswith("# demo\n")
    assert 'name = "demo"' in pyproject


def test_project_create_from_child_project_repo_renders_under_shell_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must still create a sibling child project when invoked inside another child repo."""
    workspace_root = tmp_path.resolve()
    child_project_root = workspace_root / "projects" / "existing"
    project_root = workspace_root / "projects" / "demo"
    child_project_root.mkdir(parents=True)
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(child_project_root)

    result = runner.invoke(app, ["project", "create", "demo"])

    assert result.exit_code == 0, result.output
    assert (project_root / "README.md").is_file()
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "train.py").is_file()
    assert (project_root / ".git").is_dir()