"""Small tests for the lifecycle preview command seam."""

import re
from pathlib import Path

import pytest
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _normalized_output(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output).replace("\r", "")


def _create_library(workspace_root: Path, library_name: str) -> Path:
    import_package = library_name.replace("-", "_")
    lib_root = workspace_root / "libs" / library_name
    module_root = lib_root / "src" / import_package
    module_root.mkdir(parents=True)
    (lib_root / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{library_name}"\n'
        'version = "0.1.0"\n'
        'description = "Test library"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
    (module_root / "__init__.py").write_text("", encoding="utf-8")
    return module_root


def _create_library_manifest_only(workspace_root: Path, library_name: str) -> Path:
    lib_root = workspace_root / "libs" / library_name
    (lib_root / "src").mkdir(parents=True)
    (lib_root / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{library_name}"\n'
        'version = "0.1.0"\n'
        'description = "Test library"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
    return lib_root


def _create_project(workspace_root: Path, project_name: str) -> Path:
    project_root = workspace_root / "projects" / project_name
    project_root.mkdir(parents=True)
    return project_root


def _create_component(project_root: Path, import_package: str) -> Path:
    component_root = project_root / "components" / import_package
    component_root.mkdir(parents=True)
    (component_root / "__init__.py").write_text("", encoding="utf-8")
    return component_root


def test_research_help_lists_lifecycle_commands() -> None:
    """The root CLI help must expose the lifecycle commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "eject" in result.output
    assert "harvest" in result.output


@pytest.mark.parametrize("command", ["eject", "harvest"])
def test_lifecycle_command_help_lists_dry_run(command: str) -> None:
    """Each lifecycle command must expose the dry-run option."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in _normalized_output(result.output)


def test_eject_dry_run_reports_resolved_paths_without_mutating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eject dry-run must preview the current lib-to-project resolution only."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    source_path = _create_library(workspace_root, "jax-utils")
    train_file = project_root / "train.py"
    train_file.write_text("from jax_utils.tools import step\n", encoding="utf-8")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["eject", "demo", "jax-utils", "--dry-run"])

    target_path = project_root / "components" / "jax_utils"
    assert result.exit_code == 0, result.output
    assert "[dry-run] Eject library 'jax-utils' into project 'demo'" in result.output
    assert f"Workspace root: {workspace_root}" in result.output
    assert f"Source path: {source_path}" in result.output
    assert f"Target path: {target_path}" in result.output
    assert f"Rewrite scope: {project_root}" in result.output
    assert "Copy plan:" in result.output
    assert "libs/jax-utils/src/jax_utils/__init__.py -> projects/demo/components/jax_utils/__init__.py" in result.output
    assert "Rewrite scope (Python files):" in result.output
    assert "projects/demo/train.py" in result.output
    assert "projects/demo/components/jax_utils/__init__.py" in result.output
    assert not target_path.exists()


def test_eject_fails_when_destination_already_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eject must fail fast when the project-local component already exists."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    _create_library(workspace_root, "jax-utils")
    _create_component(project_root, "jax_utils")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["eject", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code != 0
    assert f"eject destination '{project_root / 'components' / 'jax_utils'}' already exists" in result.output


def test_harvest_dry_run_reports_resolved_paths_without_mutating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Harvest dry-run must preview the current project-to-lib resolution only."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    source_path = _create_component(project_root, "jax_utils")
    monkeypatch.chdir(workspace_root)
    target_path = _create_library_manifest_only(workspace_root, "jax-utils") / "src" / "jax_utils"
    source_init = source_path / "__init__.py"

    result = runner.invoke(app, ["harvest", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "[dry-run] Harvest library 'jax-utils' from project 'demo'" in result.output
    assert f"Workspace root: {workspace_root}" in result.output
    assert f"Source path: {source_path}" in result.output
    assert f"Target path: {target_path}" in result.output
    assert "Create library manifest: no" in result.output
    assert f"Rewrite scope: {project_root}" in result.output
    assert "Rewrite scope (Python files):" in result.output
    assert "projects/demo/components/jax_utils/__init__.py" in result.output
    assert "libs/jax-utils/src/jax_utils/__init__.py" in result.output
    assert source_init.read_text(encoding="utf-8") == ""
    assert not target_path.exists()


def test_eject_fails_when_workspace_resolution_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eject must fail fast when the cwd is not a supported workspace root."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eject", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code != 0
    assert "workspace root containing 'libs/' or 'core/libs/'" in result.output


def test_eject_fails_when_project_resolution_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eject must fail fast when the project root cannot be resolved."""
    workspace_root = tmp_path.resolve()
    _create_library(workspace_root, "jax-utils")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["eject", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code != 0
    assert f"expected project root '{workspace_root / 'projects'}'" in result.output


def test_harvest_dry_run_allows_new_library_and_reports_manifest_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Harvest dry-run must support creating a brand-new shared library without mutating the workspace."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "libs").mkdir()
    project_root = _create_project(workspace_root, "demo")
    source_path = _create_component(project_root, "jax_utils")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["harvest", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"Source path: {source_path}" in result.output
    assert f"Target path: {workspace_root / 'libs' / 'jax-utils' / 'src' / 'jax_utils'}" in result.output
    assert "Create library manifest: yes" in result.output
    assert not (workspace_root / "libs" / "jax-utils").exists()


def test_harvest_fails_when_destination_package_already_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Harvest must fail fast when the destination package already exists in libs/."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    _create_component(project_root, "jax_utils")
    target_path = _create_library(workspace_root, "jax-utils")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["harvest", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code != 0
    assert f"harvest destination '{target_path}' already exists" in result.output


def test_harvest_fails_when_destination_library_layout_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Harvest must fail fast when the destination library exists without a real manifest."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    _create_component(project_root, "jax_utils")
    incomplete_lib_root = workspace_root / "libs" / "jax-utils"
    incomplete_lib_root.mkdir(parents=True)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["harvest", "demo", "jax-utils", "--dry-run"])

    assert result.exit_code != 0
    assert f"harvest destination library '{incomplete_lib_root}' is missing 'pyproject.toml'" in result.output


def test_harvest_fails_when_module_resolution_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Harvest must fail fast when the project component cannot be resolved."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    _create_library(workspace_root, "jax-utils")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["harvest", "demo", "jax-utils", "--dry-run"])

    expected_path = project_root / "components" / "jax_utils"
    assert result.exit_code != 0
    assert f"module 'jax_utils' was not found at '{expected_path}'" in result.output