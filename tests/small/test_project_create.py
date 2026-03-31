"""Small tests for the project create command surface."""

from pathlib import Path

import pytest
from jinja2 import BaseLoader, Environment
from research_cli import project as project_module
from research_cli.main import app
from research_cli.project import _template_root
from typer.testing import CliRunner

runner = CliRunner()


def _write_workspace_config(workspace_root: Path) -> None:
    (workspace_root / "research.yaml").write_text("core_path: core\nstorage_backend: local\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("algorithm", "env_name"),
    [
        ("ppo", "CartPole-v1"),
        ("dqn", "CartPole-v1"),
        ("double_dqn", "CartPole-v1"),
        ("dueling_dqn", "CartPole-v1"),
        ("sac", "MountainCarContinuous-v0"),
    ],
)
def test_project_template_train_uses_explicit_env_api(algorithm: str, env_name: str) -> None:
    """The generated smoke starter must construct and pass the environment explicitly."""
    rendered = _render_project_template(
        "train.py.jinja",
        project_name="demo",
        description="A demo experiment",
        env_name=env_name,
        algorithm=algorithm,
    )

    assert "import gymnax" in rendered
    assert "import gymnax.wrappers" in rendered
    assert "env, env_params = gymnax.make(config.ENV_NAME)" in rendered
    assert "env = gymnax.wrappers.LogWrapper(env)" in rendered
    assert "train_fn = make_train(config, env=env, env_params=env_params)" in rendered
    assert "make_train(config)" not in rendered


def _render_project_template(template_name: str, **context: str) -> str:
    template_path = _template_root() / "{{project_name}}" / template_name
    template = template_path.read_text(encoding="utf-8")
    environment = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    return environment.from_string(template).render(**context)


def test_research_help_lists_project_command() -> None:
    """The root CLI help must expose the project command group."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "project" in result.output


def test_project_create_dry_run_reports_normalized_target_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must resolve the target under projects/<name> from the workspace root."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"Template root: {_template_root()}" in result.output
    assert f"Target path: {workspace_root / 'projects' / 'demo'}" in result.output
    assert "Git ownership: the shell repo keeps shared workspace files" in result.output
    assert not (workspace_root / "projects" / "demo").exists()


def test_project_create_dry_run_resolves_workspace_root_from_child_project_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must resolve the shell workspace root even when invoked inside a child project repo."""
    workspace_root = tmp_path.resolve()
    child_project_root = workspace_root / "projects" / "existing"
    child_project_root.mkdir(parents=True)
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(child_project_root)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"Workspace root: {workspace_root}" in result.output
    assert f"Target path: {workspace_root / 'projects' / 'demo'}" in result.output
    assert not (workspace_root / "projects" / "demo").exists()


def test_project_create_dry_run_does_not_render_or_invoke_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must not call Copier or spawn subprocesses."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    _write_workspace_config(workspace_root)
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
    assert "Could not find a research workspace" in result.output


def test_project_create_fails_if_template_root_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must fail fast when the repository templates root is missing."""
    workspace_root = tmp_path.resolve()
    (workspace_root / "projects").mkdir()
    _write_workspace_config(workspace_root)
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
    _write_workspace_config(workspace_root)
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["project", "create", "demo", "--dry-run"])

    assert result.exit_code != 0
    assert str(project_root) in result.output


@pytest.mark.parametrize("algorithm", ["ppo", "dqn", "double_dqn", "dueling_dqn", "sac"])
def test_project_template_pyproject_declares_truthful_runtime_dependencies(algorithm: str) -> None:
    """The generated project must declare the runtime packages its starter imports require."""
    rendered = _render_project_template(
        "pyproject.toml.jinja",
        project_name="demo",
        description="A demo experiment",
        python_version="3.13",
        algorithm=algorithm,
    )

    assert 'dependencies = [' in rendered
    assert 'dependencies = []' not in rendered
    assert '"gymnax>=0.0.9"' in rendered
    assert '"jax~=0.9"' in rendered
    assert '"rl-agents"' in rendered
    assert "[tool.uv.sources]" in rendered
    assert "rl-agents = { workspace = true }" in rendered

    if algorithm == "ppo":
        assert '"matplotlib>=3.9"' in rendered
        assert '"rl-components"' in rendered
        assert "rl-components = { workspace = true }" in rendered
    else:
        assert '"matplotlib>=3.9"' not in rendered
        assert '"rl-components"' not in rendered
        assert "rl-components = { workspace = true }" not in rendered


def test_project_template_readme_documents_workspace_aware_bootstrap() -> None:
    """The generated README must describe the supported workspace bootstrap and run flow."""
    rendered = _render_project_template(
        "README.md.jinja",
        project_name="demo",
        description="A demo experiment",
    )

    assert "From the workspace root:" in rendered
    assert "uv sync --all-packages" in rendered
    assert "uv run --directory projects/demo python train.py" in rendered


def test_project_create_renders_then_initializes_git_without_github_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run without --github-repo must not invoke gh."""
    workspace_root = tmp_path.resolve()
    projects_root = workspace_root / "projects"
    project_root = projects_root / "demo"
    projects_root.mkdir()
    _write_workspace_config(workspace_root)
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
    _write_workspace_config(workspace_root)
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
