"""Small (unit) tests for the workspace init command.

All tests must execute in < 1ms and must not touch the real filesystem
or invoke git. Subprocess calls are mocked throughout.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from research_cli.main import app
from research_cli.workspace import (
    _load_template,
    _run,
    _write,
)
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------


def test_run_dry_run_prints_without_executing(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """_run must print the command and skip execution when dry_run=True."""
    with patch("subprocess.run") as mock_sub:
        _run(["git", "init"], cwd=tmp_path, dry_run=True)
        mock_sub.assert_not_called()
    captured = capsys.readouterr()
    assert "git init" in captured.out


def test_run_executes_command(tmp_path: Path) -> None:
    """_run must delegate to subprocess.run when dry_run=False."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_sub:
        _run(["git", "init"], cwd=tmp_path, dry_run=False)
        mock_sub.assert_called_once_with(["git", "init"], cwd=tmp_path, capture_output=True, text=True)


def test_run_raises_on_nonzero_exit(tmp_path: Path) -> None:
    """_run must raise typer.Exit when the subprocess exits with an error."""
    import typer

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "fatal: not a git repo"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(typer.Exit):
            _run(["git", "status"], cwd=tmp_path, dry_run=False)


# ---------------------------------------------------------------------------
# _write helper
# ---------------------------------------------------------------------------


def test_write_dry_run_prints_without_writing(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """_write must describe the write action and skip disk IO when dry_run=True."""
    target = tmp_path / "pyproject.toml"
    _write(target, "content", dry_run=True)
    assert not target.exists()
    captured = capsys.readouterr()
    assert "pyproject.toml" in captured.out


def test_write_creates_file(tmp_path: Path) -> None:
    """_write must create the file with the given content when dry_run=False."""
    target = tmp_path / "sub" / "file.txt"
    _write(target, "hello", dry_run=False)
    assert target.read_text() == "hello"


# ---------------------------------------------------------------------------
# pyproject.toml template
# ---------------------------------------------------------------------------


def test_pyproject_template_contains_workspace_members() -> None:
    """The pyproject.toml template must reference core/cli, core/libs/*, and projects/*."""
    rendered = _load_template("pyproject.toml.tpl").format(name="my-workspace")
    assert "core/cli" in rendered
    assert "core/libs/*" in rendered
    assert "projects/*" in rendered
    assert 'name = "my-workspace"' in rendered


def test_pyproject_template_sets_uv_package_false() -> None:
    """The root pyproject.toml must set package = false (workspace root, not a package)."""
    rendered = _load_template("pyproject.toml.tpl").format(name="test")
    assert "package = false" in rendered


# ---------------------------------------------------------------------------
# research.yaml template
# ---------------------------------------------------------------------------


def test_research_yaml_template_has_core_path() -> None:
    """research.yaml must contain the core_path key."""
    assert "core_path: core" in _load_template("research.yaml.tpl")


def test_research_yaml_template_has_storage_backend() -> None:
    """research.yaml must default to local storage (Low Floor principle)."""
    assert "storage_backend: local" in _load_template("research.yaml.tpl")


# ---------------------------------------------------------------------------
# CLI integration: workspace init (mocked git)
# ---------------------------------------------------------------------------


def _mock_subprocess_ok() -> MagicMock:
    """Return a mock subprocess result with returncode 0."""
    m = MagicMock()
    m.returncode = 0
    m.stderr = ""
    return m


def test_workspace_init_creates_directory(tmp_path: Path) -> None:
    """workspace init must create the named workspace directory."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        result = runner.invoke(app, ["workspace", "init", "my-shell", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "my-shell").is_dir()


def test_workspace_init_creates_pyproject_toml(tmp_path: Path) -> None:
    """workspace init must write a root pyproject.toml with uv workspace members."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    pyproject = (tmp_path / "shell" / "pyproject.toml").read_text()
    assert "core/libs/*" in pyproject
    assert "projects/*" in pyproject


def test_workspace_init_creates_research_yaml(tmp_path: Path) -> None:
    """workspace init must create a research.yaml configuration stub."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    assert (tmp_path / "shell" / "research.yaml").exists()


def test_workspace_init_creates_projects_dir(tmp_path: Path) -> None:
    """workspace init must create the projects/ directory."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    assert (tmp_path / "shell" / "projects").is_dir()


def test_workspace_init_adds_submodule_when_core_url_given(tmp_path: Path) -> None:
    """workspace init must invoke `git submodule add` when --core-url is provided."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()) as mock_sub:
        result = runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path), "--core-url", "https://github.com/org/research-core"])
    submodule_calls = [c for c in mock_sub.call_args_list if "submodule" in c.args[0]]
    assert len(submodule_calls) == 1
    assert "https://github.com/org/research-core" in submodule_calls[0].args[0]
    assert "uv sync --all-packages" in result.output
    assert "uv run research doctor" in result.output


def test_workspace_init_without_core_url_reports_truthful_next_steps(tmp_path: Path) -> None:
    """workspace init without --core-url must tell the user to add Core before syncing."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        result = runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "git submodule add <url> core" in result.output
    assert "uv sync --all-packages" in result.output
    assert "uv run research doctor" in result.output


def test_workspace_init_skips_submodule_without_core_url(tmp_path: Path) -> None:
    """workspace init must NOT call `git submodule add` when --core-url is omitted."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()) as mock_sub:
        runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    submodule_calls = [c for c in mock_sub.call_args_list if "submodule" in c.args[0]]
    assert len(submodule_calls) == 0


def test_workspace_init_dry_run_does_not_create_directory(tmp_path: Path) -> None:
    """workspace init --dry-run must not create any files or directories."""
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        result = runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert not (tmp_path / "shell").exists()


def test_workspace_init_fails_if_directory_exists(tmp_path: Path) -> None:
    """workspace init must exit non-zero when the target directory already exists."""
    (tmp_path / "shell").mkdir()
    with patch("subprocess.run", return_value=_mock_subprocess_ok()):
        result = runner.invoke(app, ["workspace", "init", "shell", "--path", str(tmp_path)])
    assert result.exit_code != 0
