import re
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from research_cli.main import app
from research_cli.workspace import ResolvedRepairTarget, _repair_core_checkout
from typer.testing import CliRunner

runner = CliRunner()


def _normalized_output(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output).replace("\r", "")


def _write_workspace_markers(workspace_root: Path) -> None:
    (workspace_root / "projects").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")


def _write_research_config(workspace_root: Path, content: str) -> None:
    (workspace_root / "research.yaml").write_text(content, encoding="utf-8")


def test_workspace_repair_help_is_exposed() -> None:
    result = runner.invoke(app, ["workspace", "repair", "--help"])

    assert result.exit_code == 0, result.output
    assert "repair" in result.output
    assert "--dry-run" in _normalized_output(result.output)


def test_workspace_repair_dry_run_reports_resolved_core_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    nested_dir = workspace_root / "projects" / "demo"
    core_dir = workspace_root / "vendor" / "core"
    nested_dir.mkdir(parents=True)
    core_dir.mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")
    _write_research_config(workspace_root, "core_path: vendor/core\nstorage_backend: local\n")
    monkeypatch.chdir(nested_dir)

    with patch("subprocess.run") as mock_sub:
        result = runner.invoke(app, ["workspace", "repair", "--dry-run"])
        mock_sub.assert_not_called()

    assert result.exit_code == 0, result.output
    assert result.output == (
        f"  [dry-run] git -C {core_dir.resolve()} clean -ffd\n"
        "  [dry-run] git submodule update --force --checkout -- vendor/core\n"
    )


def test_repair_core_checkout_uses_workspace_root_and_deterministic_command_order(tmp_path: Path) -> None:
    workspace_root = (tmp_path / "workspace").resolve()
    core_dir = workspace_root / "vendor" / "core"
    core_dir.mkdir(parents=True)
    repair_target = ResolvedRepairTarget(
        workspace_root=workspace_root,
        core_path=core_dir,
        submodule_path=Path("vendor/core"),
    )
    recorded_calls: list[tuple[list[str], Path, bool]] = []

    def record_run(args: list[str], cwd: Path, dry_run: bool) -> None:
        recorded_calls.append((args, cwd, dry_run))

    _repair_core_checkout(repair_target, dry_run=False, run_command=record_run)

    assert recorded_calls == [
        (["git", "-C", str(core_dir), "clean", "-ffd"], workspace_root, False),
        (["git", "submodule", "update", "--force", "--checkout", "--", "vendor/core"], workspace_root, False),
    ]


def test_workspace_repair_runs_real_repair_sequence_from_workspace_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    nested_dir = workspace_root / "projects" / "demo"
    core_dir = workspace_root / "vendor" / "core"
    nested_dir.mkdir(parents=True)
    core_dir.mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text("[tool.uv]\npackage = false\n", encoding="utf-8")
    _write_research_config(workspace_root, "core_path: vendor/core\nstorage_backend: local\n")
    monkeypatch.chdir(nested_dir)

    clean_result = MagicMock(returncode=0, stderr="")
    update_result = MagicMock(returncode=0, stderr="")
    with patch("subprocess.run", side_effect=[clean_result, update_result]) as mock_sub:
        result = runner.invoke(app, ["workspace", "repair"])

    assert result.exit_code == 0, result.output
    assert result.output == f"✓ Repaired configured Core checkout at '{core_dir.resolve()}'.\n"
    assert mock_sub.call_args_list == [
        call(["git", "-C", str(core_dir.resolve()), "clean", "-ffd"], cwd=workspace_root.resolve(), capture_output=True, text=True),
        call(["git", "submodule", "update", "--force", "--checkout", "--", "vendor/core"], cwd=workspace_root.resolve(), capture_output=True, text=True),
    ]


def test_workspace_repair_surfaces_failing_subprocess_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    core_dir = workspace_root / "vendor" / "core"
    core_dir.mkdir(parents=True)
    _write_workspace_markers(workspace_root)
    _write_research_config(workspace_root, "core_path: vendor/core\nstorage_backend: local\n")
    monkeypatch.chdir(workspace_root)

    failing_result = MagicMock(returncode=1, stderr="fatal: unable to remove blocker")
    with patch("subprocess.run", return_value=failing_result) as mock_sub:
        result = runner.invoke(app, ["workspace", "repair"])

    assert result.exit_code == 1
    assert result.output == (
        f"Error running `git -C {core_dir.resolve()} clean -ffd`:\n"
        "fatal: unable to remove blocker\n"
    )
    assert mock_sub.call_args_list == [
        call(["git", "-C", str(core_dir.resolve()), "clean", "-ffd"], cwd=workspace_root.resolve(), capture_output=True, text=True),
    ]


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