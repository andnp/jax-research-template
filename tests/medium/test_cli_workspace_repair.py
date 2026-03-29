"""Medium tests for real git-backed workspace repair behavior."""

import subprocess
from pathlib import Path

from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _git(cwd: Path, *args: str):
    result = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def _init_repo(repo_root: Path):
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "user.email", "test@example.com")


def _create_shell_workspace(workspace_root: Path):
    _init_repo(workspace_root)
    (workspace_root / "projects").mkdir()
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")
    (workspace_root / "research.yaml").write_text("core_path: core\nstorage_backend: local\n", encoding="utf-8")


def test_workspace_repair_restores_recorded_submodule_revision_and_cleans_submodule(tmp_path: Path, monkeypatch) -> None:
    core_origin = tmp_path / "core-origin"
    _init_repo(core_origin)
    tracked_file = core_origin / "tracked.txt"
    tracked_file.write_text("recorded\n", encoding="utf-8")
    _git(core_origin, "add", "tracked.txt")
    _git(core_origin, "commit", "-m", "recorded revision")
    recorded_revision = _git(core_origin, "rev-parse", "HEAD").stdout.strip()

    tracked_file.write_text("advanced\n", encoding="utf-8")
    _git(core_origin, "commit", "-am", "advanced revision")
    advanced_revision = _git(core_origin, "rev-parse", "HEAD").stdout.strip()
    assert advanced_revision != recorded_revision

    workspace_root = tmp_path / "shell"
    _create_shell_workspace(workspace_root)
    _git(workspace_root, "-c", "protocol.file.allow=always", "submodule", "add", str(core_origin), "core")
    _git(workspace_root / "core", "checkout", recorded_revision)
    _git(workspace_root, "add", "pyproject.toml", "research.yaml", ".gitmodules", "core")
    _git(workspace_root, "commit", "-m", "record core submodule revision")

    dirty_tracked_file = workspace_root / "core" / "tracked.txt"
    _git(workspace_root / "core", "checkout", advanced_revision)
    dirty_tracked_file.write_text("dirty\n", encoding="utf-8")
    blocking_path = workspace_root / "core" / "scratch" / "blocker.txt"
    blocking_path.parent.mkdir(parents=True)
    blocking_path.write_text("blocker\n", encoding="utf-8")

    assert _git(workspace_root / "core", "rev-parse", "HEAD").stdout.strip() == advanced_revision
    assert _git(workspace_root / "core", "status", "--porcelain").stdout != ""

    monkeypatch.chdir(workspace_root)
    result = runner.invoke(app, ["workspace", "repair"])

    assert result.exit_code == 0, result.output
    assert result.output == f"✓ Repaired configured Core checkout at '{(workspace_root / 'core').resolve()}'.\n"
    assert _git(workspace_root / "core", "rev-parse", "HEAD").stdout.strip() == recorded_revision
    assert _git(workspace_root / "core", "status", "--porcelain").stdout == ""
    assert dirty_tracked_file.read_text(encoding="utf-8") == "recorded\n"
    assert not blocking_path.exists()