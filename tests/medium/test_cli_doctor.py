"""Medium tests for the top-level ``research doctor`` command."""

from pathlib import Path

import research_cli.doctor as doctor_module
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_research_doctor_aggregates_config_git_and_environment_failures(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path.resolve()
    core_root = workspace_root / "core"
    core_root.mkdir(parents=True)
    (workspace_root / "research.yaml").write_text(
        "core_path: core\n"
        "storage_backend: local\n"
        "doctor:\n"
        "  expected_accelerators:\n"
        "    - gpu\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace_root)

    def run_git(args: tuple[str, ...], cwd: Path):
        assert cwd == core_root.resolve()
        if args == ("git", "rev-parse", "--is-inside-work-tree"):
            return doctor_module.GitCommandResult(returncode=128, stdout="", stderr="fatal: not a git repository")
        raise AssertionError(f"Unexpected git command: {args!r}")

    def run_environment_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return doctor_module.EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return doctor_module.JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    monkeypatch.setattr(doctor_module, "_run_git", run_git)
    monkeypatch.setattr(doctor_module, "_run_environment_command", run_environment_command)
    monkeypatch.setattr(doctor_module, "_probe_jax", probe_jax)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code != 0
    assert "[x] Config validation" in result.output
    assert "[x] research_yaml: Loaded research.yaml" in result.output
    assert "[ ] Git health" in result.output
    assert "[x] path_exists: Configured core_path resolves" in result.output
    assert "[ ] working_tree: Configured core_path 'core' resolves" in result.output
    assert "[ ] head_attached: Cannot evaluate 'head_attached'" in result.output
    assert "[ ] working_tree_clean: Cannot evaluate 'working_tree_clean'" in result.output
    assert "[ ] Environment health" in result.output
    assert "[x] uv_available: uv responded to '--version':" in result.output
    assert "[x] jax_import: JAX imported successfully." in result.output
    assert "[ ] accelerator_expectations: Configured accelerators did not match JAX detection." in result.output
    assert result.output.rstrip().endswith("overall: FAIL")