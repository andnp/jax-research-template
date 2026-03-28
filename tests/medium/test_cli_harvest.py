"""Medium tests for real lifecycle harvest behavior."""

from pathlib import Path

from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _create_workspace_pyproject(workspace_root: Path) -> None:
    (workspace_root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "research-core"\n'
        'version = "0.1.0"\n'
        'description = "Test workspace"\n'
        "dependencies = [\n"
        "]\n\n"
        "[tool.uv.sources]\n\n"
        "[tool.ty.environment]\n"
        "extra-paths = [\n"
        "]\n",
        encoding="utf-8",
    )


def _create_project(workspace_root: Path, project_name: str) -> Path:
    project_root = workspace_root / "projects" / project_name
    project_root.mkdir(parents=True)
    return project_root


def _create_component(project_root: Path, import_package: str) -> Path:
    component_root = project_root / "components" / import_package
    component_root.mkdir(parents=True)
    (component_root / "__init__.py").write_text("from components.jax_utils.helpers import helper\n", encoding="utf-8")
    (component_root / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    return component_root


def test_harvest_moves_package_creates_manifest_and_rewrites_imports(tmp_path: Path, monkeypatch) -> None:
    """Harvest must move the project component into libs/, create a real manifest, and rewrite imports back to the shared path."""
    workspace_root = tmp_path.resolve()
    _create_workspace_pyproject(workspace_root)
    (workspace_root / "libs").mkdir()
    project_root = _create_project(workspace_root, "demo")
    source_root = _create_component(project_root, "jax_utils")
    train_file = project_root / "train.py"
    train_file.write_text(
        "import components.jax_utils\nfrom components.jax_utils.helpers import helper\n",
        encoding="utf-8",
    )
    nested_dir = project_root / "pkg"
    nested_dir.mkdir()
    nested_file = nested_dir / "runner.py"
    nested_file.write_text("import components.jax_utils.helpers as helpers\n", encoding="utf-8")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["harvest", "demo", "jax-utils"])

    target_root = workspace_root / "libs" / "jax-utils" / "src" / "jax_utils"
    pyproject_path = workspace_root / "libs" / "jax-utils" / "pyproject.toml"
    workspace_pyproject = workspace_root / "pyproject.toml"

    assert result.exit_code == 0, result.output
    assert not source_root.exists()
    assert target_root.is_dir()
    assert pyproject_path.read_text(encoding="utf-8") == (
        "[project]\n"
        'name = "jax-utils"\n'
        'version = "0.1.0"\n'
        'description = "Harvested shared library"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
    )
    assert (target_root / "__init__.py").read_text(encoding="utf-8") == "from jax_utils.helpers import helper\n"
    assert (target_root / "helpers.py").read_text(encoding="utf-8") == "def helper() -> int:\n    return 1\n"
    assert train_file.read_text(encoding="utf-8") == "import jax_utils\nfrom jax_utils.helpers import helper\n"
    assert nested_file.read_text(encoding="utf-8") == "import jax_utils.helpers as helpers\n"
    workspace_text = workspace_pyproject.read_text(encoding="utf-8")
    assert '"jax-utils",' in workspace_text
    assert 'jax-utils = { workspace = true }' in workspace_text
    assert '"libs/jax-utils/src",' in workspace_text
    assert f"Target path: {target_root}" in result.output
    assert "Create library manifest: yes" in result.output
    assert "Rewritten files: 6" in result.output