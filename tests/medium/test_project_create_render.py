"""Medium integration tests for project creation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from research_cli.main import app
from research_runner.registry import registered_specs
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
    assert (project_root / "experiments" / "001_demo.py").is_file()
    assert not (project_root / "train.py").exists()
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
    assert (project_root / "experiments" / "001_demo.py").is_file()
    assert not (project_root / "train.py").exists()
    assert (project_root / ".git").is_dir()


def test_project_create_renders_importable_linted_declarative_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render a declarative starter under the project's normalized slug.

    Its registered spec must import and the generated source must pass Ruff.
    """
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = tmp_path.resolve()
    project_root = workspace_root / "projects" / "demo-project"
    (workspace_root / "projects").mkdir(parents=True)
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo-project"])

    assert result.exit_code == 0, result.output
    experiment_path = project_root / "experiments" / "001_demo_project.py"
    assert experiment_path.is_file()
    assert not (project_root / "train.py").exists()

    ruff_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(repo_root / "pyproject.toml"),
            str(experiment_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ruff_result.returncode == 0, f"stdout:\n{ruff_result.stdout}\nstderr:\n{ruff_result.stderr}"

    module_spec = importlib.util.spec_from_file_location("generated_experiment", experiment_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    spec_factories = registered_specs(module)
    assert tuple(spec_factories) == ("demo_project_spec",)
    generated_spec = spec_factories["demo_project_spec"]()
    assert generated_spec.experiment.name == "demo-project"
    assert generated_spec.results_root == project_root / "results"
    assert generated_spec.parameter_schema is not None
    assert "learning_rate" in generated_spec.parameter_schema.dynamic_keys

    plan_result = runner.invoke(app, ["experiment", "plan", str(experiment_path), "--spec", "demo_project_spec"])
    assert plan_result.exit_code == 0, plan_result.output
    assert "Spec 'demo_project_spec': 1 batch(es) planned" in plan_result.output
