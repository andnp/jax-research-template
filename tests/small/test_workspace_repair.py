from pathlib import Path

import pytest
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_workspace_markers(workspace_root: Path) -> None:
    (workspace_root / "projects").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")


def _write_research_config(workspace_root: Path, content: str) -> None:
    (workspace_root / "research.yaml").write_text(content, encoding="utf-8")


def test_workspace_repair_help_is_exposed() -> None:
    result = runner.invoke(app, ["workspace", "repair", "--help"])

    assert result.exit_code == 0, result.output
    assert "repair" in result.output
    assert "--dry-run" in result.output


def test_workspace_repair_dry_run_reports_resolved_core_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    nested_dir = workspace_root / "projects" / "demo"
    core_dir = workspace_root / "vendor" / "core"
    nested_dir.mkdir(parents=True)
    core_dir.mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")
    _write_research_config(workspace_root, "core_path: vendor/core\nstorage_backend: local\n")
    monkeypatch.chdir(nested_dir)

    result = runner.invoke(app, ["workspace", "repair", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.output == f"[dry-run] Would repair configured Core checkout at '{core_dir.resolve()}'.\n"


def test_workspace_repair_fails_when_research_yaml_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_workspace_markers(workspace_root)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["workspace", "repair", "--dry-run"])

    assert result.exit_code == 1
    assert "research.yaml not found" in result.output


def test_workspace_repair_fails_when_research_yaml_is_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_workspace_markers(workspace_root)
    _write_research_config(workspace_root, "core_path: [\n")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["workspace", "repair", "--dry-run"])

    assert result.exit_code == 1
    assert "Malformed research.yaml" in result.output


def test_workspace_repair_fails_when_configured_core_path_does_not_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_workspace_markers(workspace_root)
    _write_research_config(workspace_root, "core_path: vendor/core\nstorage_backend: local\n")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["workspace", "repair", "--dry-run"])

    assert result.exit_code == 1
    assert "Configured core_path 'vendor/core' resolves to" in result.output
    assert "does not exist" in result.output


def test_workspace_repair_fails_outside_a_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["workspace", "repair", "--dry-run"])

    assert result.exit_code == 1
    assert "Could not find a research workspace" in result.output