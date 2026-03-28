"""Medium tests for real lifecycle eject behavior."""

from pathlib import Path

from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _create_library(workspace_root: Path, library_name: str) -> Path:
    import_package = library_name.replace("-", "_")
    module_root = workspace_root / "libs" / library_name / "src" / import_package
    module_root.mkdir(parents=True)
    (module_root / "__init__.py").write_text("from jax_utils.helpers import helper\n", encoding="utf-8")
    (module_root / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    return module_root


def _create_project(workspace_root: Path, project_name: str) -> Path:
    project_root = workspace_root / "projects" / project_name
    project_root.mkdir(parents=True)
    return project_root


def test_eject_copies_package_and_rewrites_project_imports(tmp_path: Path, monkeypatch) -> None:
    """Eject must copy the package tree and rewrite shared-lib imports inside the target project."""
    workspace_root = tmp_path.resolve()
    project_root = _create_project(workspace_root, "demo")
    source_root = _create_library(workspace_root, "jax-utils")
    train_file = project_root / "train.py"
    train_file.write_text(
        "import jax_utils\nfrom jax_utils.helpers import helper\n",
        encoding="utf-8",
    )
    nested_dir = project_root / "pkg"
    nested_dir.mkdir()
    nested_file = nested_dir / "runner.py"
    nested_file.write_text("import jax_utils.helpers as helpers\n", encoding="utf-8")
    outside_file = workspace_root / "scratch.py"
    outside_file.write_text("import jax_utils\n", encoding="utf-8")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["eject", "demo", "jax-utils"])

    target_root = project_root / "components" / "jax_utils"
    assert result.exit_code == 0, result.output
    assert source_root.is_dir()
    assert target_root.is_dir()
    assert (target_root / "__init__.py").read_text(encoding="utf-8") == "from components.jax_utils.helpers import helper\n"
    assert (target_root / "helpers.py").read_text(encoding="utf-8") == "def helper() -> int:\n    return 1\n"
    assert train_file.read_text(encoding="utf-8") == "import components.jax_utils\nfrom components.jax_utils.helpers import helper\n"
    assert nested_file.read_text(encoding="utf-8") == "import components.jax_utils.helpers as helpers\n"
    assert outside_file.read_text(encoding="utf-8") == "import jax_utils\n"
    assert f"Target path: {target_root}" in result.output
    assert "Rewritten files: 3" in result.output