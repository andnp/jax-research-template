from pathlib import Path

from research_cli.config import DoctorConfig, ResearchConfig, ResearchConfigError
from research_cli.doctor import (
    EnvironmentCommandResult,
    GitCommandResult,
    JaxProbeResult,
    check_environment_health,
    check_git_health,
    render_doctor_report,
    run_doctor,
)


def test_check_git_health_resolves_relative_core_path_and_uses_only_read_only_git_commands(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    core_dir = workspace_root / "vendor" / "core"
    core_dir.mkdir(parents=True)

    calls: list[tuple[tuple[str, ...], Path]] = []

    def run_git(args: tuple[str, ...], cwd: Path):
        calls.append((args, cwd))
        if args == ("git", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n", stderr="")
        if args == ("git", "symbolic-ref", "--quiet", "HEAD"):
            return GitCommandResult(returncode=0, stdout="refs/heads/main\n", stderr="")
        if args == ("git", "status", "--porcelain", "--untracked-files=normal"):
            return GitCommandResult(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected git command: {args!r}")

    report = check_git_health(workspace_root=workspace_root, core_path=Path("vendor/core"), run_git=run_git)

    assert report.ok is True
    assert report.resolved_path == core_dir.resolve()
    assert [diagnostic.name for diagnostic in report.diagnostics] == ["path_exists", "working_tree", "head_attached", "working_tree_clean"]
    assert all(diagnostic.ok for diagnostic in report.diagnostics)
    assert calls == [
        (("git", "rev-parse", "--is-inside-work-tree"), core_dir.resolve()),
        (("git", "symbolic-ref", "--quiet", "HEAD"), core_dir.resolve()),
        (("git", "status", "--porcelain", "--untracked-files=normal"), core_dir.resolve()),
    ]


def test_check_git_health_reports_missing_core_path_as_actionable_failures(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    report = check_git_health(workspace_root=workspace_root, core_path=Path("missing-core"))

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [False, False, False, False]
    assert "does not exist" in report.diagnostics[0].message
    assert "Update research.yaml or create the Core checkout" in report.diagnostics[0].message
    assert "because the configured path does not exist" in report.diagnostics[1].message
    assert "because the configured path does not exist" in report.diagnostics[2].message
    assert "because the configured path does not exist" in report.diagnostics[3].message


def test_check_git_health_blocks_downstream_checks_when_path_is_not_a_git_working_tree(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    core_dir = workspace_root / "core"
    core_dir.mkdir(parents=True)

    def run_git(args: tuple[str, ...], cwd: Path):
        assert cwd == core_dir.resolve()
        assert args == ("git", "rev-parse", "--is-inside-work-tree")
        return GitCommandResult(returncode=128, stdout="", stderr="fatal: not a git repository")

    report = check_git_health(workspace_root=workspace_root, core_path=Path("core"), run_git=run_git)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, False, False, False]
    assert "not a Git working tree" in report.diagnostics[1].message
    assert "fatal: not a git repository" in report.diagnostics[1].message
    assert "because the configured path is not a Git working tree" in report.diagnostics[2].message
    assert "because the configured path is not a Git working tree" in report.diagnostics[3].message


def test_check_git_health_fails_when_head_is_detached(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    core_dir = workspace_root / "core"
    core_dir.mkdir(parents=True)

    def run_git(args: tuple[str, ...], cwd: Path):
        assert cwd == core_dir.resolve()
        if args == ("git", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n", stderr="")
        if args == ("git", "symbolic-ref", "--quiet", "HEAD"):
            return GitCommandResult(returncode=1, stdout="", stderr="")
        if args == ("git", "status", "--porcelain", "--untracked-files=normal"):
            return GitCommandResult(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected git command: {args!r}")

    report = check_git_health(workspace_root=workspace_root, core_path=Path("core"), run_git=run_git)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, True, False, True]
    assert "HEAD is detached or unreadable" in report.diagnostics[2].message
    assert "Check out a branch" in report.diagnostics[2].message


def test_check_git_health_treats_untracked_files_as_dirty(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    core_dir = workspace_root / "core"
    core_dir.mkdir(parents=True)

    def run_git(args: tuple[str, ...], cwd: Path):
        assert cwd == core_dir.resolve()
        if args == ("git", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n", stderr="")
        if args == ("git", "symbolic-ref", "--quiet", "HEAD"):
            return GitCommandResult(returncode=0, stdout="refs/heads/main\n", stderr="")
        if args == ("git", "status", "--porcelain", "--untracked-files=normal"):
            return GitCommandResult(returncode=0, stdout="?? scratch.txt\n", stderr="")
        raise AssertionError(f"Unexpected git command: {args!r}")

    report = check_git_health(workspace_root=workspace_root, core_path=Path("core"), run_git=run_git)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, True, True, False]
    assert "including untracked files, as a failure" in report.diagnostics[3].message
    assert "Commit, stash, or remove changes" in report.diagnostics[3].message


def test_check_environment_health_uses_non_mutating_uv_query_and_reports_jax_backend(tmp_path: Path) -> None:
    del tmp_path

    calls: list[tuple[str, ...]] = []

    def run_command(args: tuple[str, ...]):
        calls.append(args)
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    report = check_environment_health(expected_accelerators=None, run_command=run_command, probe_jax=probe_jax)

    assert report.ok is True
    assert [diagnostic.name for diagnostic in report.diagnostics] == ["uv_available", "jax_import", "accelerator_expectations"]
    assert calls == [("uv", "--version")]
    assert "uv responded to '--version': uv 0.7.2." in report.diagnostics[0].message
    assert "Default backend: cpu." in report.diagnostics[1].message
    assert "Detected device platforms: cpu." in report.diagnostics[1].message
    assert "No doctor.expected_accelerators were configured. Observed accelerators: cpu." in report.diagnostics[2].message


def test_check_environment_health_reports_uv_failure_without_skipping_jax_probe(tmp_path: Path) -> None:
    del tmp_path

    def run_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=127, stdout="", stderr="uv: command not found")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    report = check_environment_health(expected_accelerators=None, run_command=run_command, probe_jax=probe_jax)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [False, True, True]
    assert "uv is not discoverable or did not respond" in report.diagnostics[0].message
    assert "uv: command not found" in report.diagnostics[0].message
    assert "JAX imported successfully" in report.diagnostics[1].message


def test_check_environment_health_fails_when_expected_accelerator_is_missing(tmp_path: Path) -> None:
    del tmp_path

    def run_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    report = check_environment_health(expected_accelerators=("gpu",), run_command=run_command, probe_jax=probe_jax)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, True, False]
    assert "Missing expected accelerator(s): gpu." in report.diagnostics[2].message
    assert "Expected: gpu. Observed: cpu." in report.diagnostics[2].message


def test_check_environment_health_maps_vendor_specific_gpu_platforms_to_gpu_expectations(tmp_path: Path) -> None:
    del tmp_path

    def run_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cuda", device_platforms=("cuda", "cpu"))

    report = check_environment_health(expected_accelerators=("gpu",), run_command=run_command, probe_jax=probe_jax)

    assert report.ok is True
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, True, True]
    assert "Expected: gpu. Observed: gpu, cpu." in report.diagnostics[2].message


def test_check_environment_health_blocks_expectation_check_when_jax_probe_fails(tmp_path: Path) -> None:
    del tmp_path

    def run_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=False, backend=None, device_platforms=(), error="No module named 'jax'")

    report = check_environment_health(expected_accelerators=("cpu",), run_command=run_command, probe_jax=probe_jax)

    assert report.ok is False
    assert [diagnostic.ok for diagnostic in report.diagnostics] == [True, False, False]
    assert "No module named 'jax'" in report.diagnostics[1].message
    assert "Cannot evaluate accelerator expectations because JAX import/probe failed." in report.diagnostics[2].message


def test_run_doctor_aggregates_all_groups_when_config_is_valid(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()
    (workspace_root / "core").mkdir()

    def load_config(config_path: Path):
        assert config_path == workspace_root / "research.yaml"
        return ResearchConfig(
            core_path=Path("core"),
            storage_backend="local",
            doctor=DoctorConfig(expected_accelerators=("cpu",)),
        )

    git_calls: list[tuple[tuple[str, ...], Path]] = []
    environment_calls: list[tuple[str, ...]] = []

    def run_git(args: tuple[str, ...], cwd: Path):
        git_calls.append((args, cwd))
        if args == ("git", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n", stderr="")
        if args == ("git", "symbolic-ref", "--quiet", "HEAD"):
            return GitCommandResult(returncode=0, stdout="refs/heads/main\n", stderr="")
        if args == ("git", "status", "--porcelain", "--untracked-files=normal"):
            return GitCommandResult(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected git command: {args!r}")

    def run_environment_command(args: tuple[str, ...]):
        environment_calls.append(args)
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    report = run_doctor(
        workspace_root=workspace_root,
        load_config=load_config,
        run_git=run_git,
        run_environment_command=run_environment_command,
        probe_jax=probe_jax,
    )

    assert report.ok is True
    assert report.config.ok is True
    assert report.git.ok is True
    assert report.environment.ok is True
    assert [diagnostic.name for diagnostic in report.git.diagnostics] == ["path_exists", "working_tree", "head_attached", "working_tree_clean"]
    assert [diagnostic.name for diagnostic in report.environment.diagnostics] == ["uv_available", "jax_import", "accelerator_expectations"]
    assert git_calls == [
        (("git", "rev-parse", "--is-inside-work-tree"), workspace_root / "core"),
        (("git", "symbolic-ref", "--quiet", "HEAD"), workspace_root / "core"),
        (("git", "status", "--porcelain", "--untracked-files=normal"), workspace_root / "core"),
    ]
    assert environment_calls == [("uv", "--version")]


def test_run_doctor_reports_blocked_git_and_environment_expectations_when_config_is_invalid(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()

    def load_config(config_path: Path):
        assert config_path == workspace_root / "research.yaml"
        raise ResearchConfigError("Invalid research.yaml at '/tmp/research.yaml': missing required key(s): core_path.")

    environment_calls: list[tuple[str, ...]] = []

    def run_environment_command(args: tuple[str, ...]):
        environment_calls.append(args)
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=0, stdout="uv 0.7.2\n", stderr="")

    def probe_jax():
        return JaxProbeResult(ok=True, backend="cpu", device_platforms=("cpu",))

    report = run_doctor(
        workspace_root=workspace_root,
        load_config=load_config,
        run_environment_command=run_environment_command,
        probe_jax=probe_jax,
    )

    assert report.ok is False
    assert report.config.ok is False
    assert report.git.ok is False
    assert report.environment.ok is False
    assert [diagnostic.ok for diagnostic in report.git.diagnostics] == [False, False, False, False]
    assert [diagnostic.ok for diagnostic in report.environment.diagnostics] == [True, True, False]
    assert "research.yaml is invalid" in report.git.diagnostics[0].message
    assert "Cannot evaluate accelerator expectations because research.yaml is invalid" in report.environment.diagnostics[2].message
    assert environment_calls == [("uv", "--version")]


def test_render_doctor_report_is_deterministic_and_readable(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()

    def load_config(config_path: Path):
        assert config_path == workspace_root / "research.yaml"
        raise ResearchConfigError("research.yaml not found at '/tmp/workspace/research.yaml'.")

    def run_environment_command(args: tuple[str, ...]):
        assert args == ("uv", "--version")
        return EnvironmentCommandResult(returncode=127, stdout="", stderr="uv: command not found")

    def probe_jax():
        return JaxProbeResult(ok=False, backend=None, device_platforms=(), error="No module named 'jax'")

    report = run_doctor(
        workspace_root=workspace_root,
        load_config=load_config,
        run_environment_command=run_environment_command,
        probe_jax=probe_jax,
    )

    rendered = render_doctor_report(report)

    assert rendered == (
        "[ ] Config validation\n"
        "  [ ] research_yaml: research.yaml not found at '/tmp/workspace/research.yaml'.\n\n"
        "[ ] Git health\n"
        "  [ ] path_exists: Cannot evaluate 'path_exists' because research.yaml is invalid. research.yaml not found at '/tmp/workspace/research.yaml'.\n"
        "  [ ] working_tree: Cannot evaluate 'working_tree' because research.yaml is invalid. research.yaml not found at '/tmp/workspace/research.yaml'.\n"
        "  [ ] head_attached: Cannot evaluate 'head_attached' because research.yaml is invalid. research.yaml not found at '/tmp/workspace/research.yaml'.\n"
        "  [ ] working_tree_clean: Cannot evaluate 'working_tree_clean' because research.yaml is invalid. research.yaml not found at '/tmp/workspace/research.yaml'.\n\n"
        "[ ] Environment health\n"
        "  [ ] uv_available: uv is not discoverable or did not respond to '--version'. uv: command not found\n"
        "  [ ] jax_import: JAX could not be imported or probed without mutating the environment. No module named 'jax'\n"
        "  [ ] accelerator_expectations: Cannot evaluate accelerator expectations because research.yaml is invalid. research.yaml not found at '/tmp/workspace/research.yaml'.\n\n"
        "overall: FAIL"
    )
