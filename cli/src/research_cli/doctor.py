"""Read-only diagnostics for ``research doctor``."""

import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import typer

from research_cli.config import AcceleratorLabel, ResearchConfig, ResearchConfigError, load_research_config
from research_cli.workspace import WorkspaceResolutionError, resolve_workspace_root

ConfigCheckName = Literal["research_yaml"]
GitCheckName = Literal["path_exists", "working_tree", "head_attached", "working_tree_clean"]
EnvironmentCheckName = Literal["uv_available", "jax_import", "accelerator_expectations"]


@dataclass(frozen=True, slots=True)
class ConfigDiagnostic:
    name: ConfigCheckName
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class ConfigHealthReport:
    config_path: Path
    diagnostics: tuple[ConfigDiagnostic, ...]
    config: ResearchConfig | None = None

    @property
    def ok(self):
        return all(diagnostic.ok for diagnostic in self.diagnostics)


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    returncode: int
    stdout: str
    stderr: str


class GitCommandRunner(Protocol):
    def __call__(self, args: tuple[str, ...], cwd: Path) -> GitCommandResult:
        ...


@dataclass(frozen=True, slots=True)
class GitDiagnostic:
    name: GitCheckName
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class EnvironmentCommandResult:
    returncode: int
    stdout: str
    stderr: str


class EnvironmentCommandRunner(Protocol):
    def __call__(self, args: tuple[str, ...]) -> EnvironmentCommandResult:
        ...


@dataclass(frozen=True, slots=True)
class JaxProbeResult:
    ok: bool
    backend: str | None
    device_platforms: tuple[str, ...]
    error: str | None = None


class JaxProbeRunner(Protocol):
    def __call__(self) -> JaxProbeResult:
        ...


@dataclass(frozen=True, slots=True)
class EnvironmentDiagnostic:
    name: EnvironmentCheckName
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class EnvironmentHealthReport:
    diagnostics: tuple[EnvironmentDiagnostic, ...]

    @property
    def ok(self):
        return all(diagnostic.ok for diagnostic in self.diagnostics)


@dataclass(frozen=True, slots=True)
class GitHealthReport:
    configured_path: Path
    resolved_path: Path
    diagnostics: tuple[GitDiagnostic, ...]

    @property
    def ok(self):
        return all(diagnostic.ok for diagnostic in self.diagnostics)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    config: ConfigHealthReport
    git: GitHealthReport
    environment: EnvironmentHealthReport

    @property
    def ok(self):
        return self.config.ok and self.git.ok and self.environment.ok


class ConfigLoader(Protocol):
    def __call__(self, config_path: Path) -> ResearchConfig:
        ...


def doctor_command():
    try:
        workspace_root = resolve_workspace_root(Path.cwd())
    except WorkspaceResolutionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = run_doctor(workspace_root=workspace_root)
    typer.echo(render_doctor_report(report))
    if not report.ok:
        raise typer.Exit(code=1)


def run_doctor(
    workspace_root: Path,
    load_config: ConfigLoader | None = None,
    run_git: GitCommandRunner | None = None,
    run_environment_command: EnvironmentCommandRunner | None = None,
    probe_jax: JaxProbeRunner | None = None,
):
    config_report = check_config_health(workspace_root=workspace_root, load_config=load_config)

    if config_report.config is None:
        config_error_message = config_report.diagnostics[0].message
        git_report = blocked_git_health(workspace_root=workspace_root, reason=config_error_message)
        environment_report = check_environment_health(
            expected_accelerators=None,
            run_command=run_environment_command,
            probe_jax=probe_jax,
            accelerator_config_error=config_error_message,
        )
    else:
        git_report = check_git_health(
            workspace_root=workspace_root,
            core_path=config_report.config.core_path,
            run_git=run_git,
        )
        expected_accelerators = None
        if config_report.config.doctor is not None:
            expected_accelerators = config_report.config.doctor.expected_accelerators
        environment_report = check_environment_health(
            expected_accelerators=expected_accelerators,
            run_command=run_environment_command,
            probe_jax=probe_jax,
        )

    return DoctorReport(config=config_report, git=git_report, environment=environment_report)


def check_config_health(workspace_root: Path, load_config: ConfigLoader | None = None):
    config_loader = load_research_config if load_config is None else load_config
    config_path = workspace_root / "research.yaml"

    try:
        config = config_loader(config_path)
    except ResearchConfigError as exc:
        return ConfigHealthReport(
            config_path=config_path,
            diagnostics=(ConfigDiagnostic(name="research_yaml", ok=False, message=str(exc)),),
        )

    return ConfigHealthReport(
        config_path=config_path,
        diagnostics=(ConfigDiagnostic(name="research_yaml", ok=True, message=f"Loaded research.yaml from '{config_path}'."),),
        config=config,
    )


def blocked_git_health(workspace_root: Path, reason: str):
    blocked_path = (workspace_root / "<unavailable-core-path>").resolve(strict=False)
    diagnostics = tuple(
        GitDiagnostic(
            name=name,
            ok=False,
            message=f"Cannot evaluate '{name}' because research.yaml is invalid. {reason}",
        )
        for name in ("path_exists", "working_tree", "head_attached", "working_tree_clean")
    )
    return GitHealthReport(
        configured_path=Path("<unavailable-core-path>"),
        resolved_path=blocked_path,
        diagnostics=diagnostics,
    )


def render_doctor_report(report: DoctorReport):
    rendered_lines = [
        _render_group("Config validation", report.config.ok, report.config.diagnostics),
        _render_group("Git health", report.git.ok, report.git.diagnostics),
        _render_group("Environment health", report.environment.ok, report.environment.diagnostics),
        f"overall: {_status_label(report.ok)}",
    ]
    return "\n\n".join(rendered_lines)


def _render_group(title: str, ok: bool, diagnostics: tuple[ConfigDiagnostic, ...] | tuple[GitDiagnostic, ...] | tuple[EnvironmentDiagnostic, ...]):
    lines = [f"[{_status_marker(ok)}] {title}"]
    for diagnostic in diagnostics:
        lines.append(f"  [{_status_marker(diagnostic.ok)}] {diagnostic.name}: {diagnostic.message}")
    return "\n".join(lines)


def _status_marker(ok: bool):
    return "x" if ok else " "


def _status_label(ok: bool):
    return "PASS" if ok else "FAIL"


def check_git_health(workspace_root: Path, core_path: Path, run_git: GitCommandRunner | None = None):
    git_runner = _run_git if run_git is None else run_git
    resolved_path = _resolve_core_path(workspace_root=workspace_root, core_path=core_path)

    if not resolved_path.exists():
        missing_message = (
            f"Configured core_path '{core_path}' resolves to '{resolved_path}', which does not exist. "
            "Update research.yaml or create the Core checkout at that location."
        )
        return GitHealthReport(
            configured_path=core_path,
            resolved_path=resolved_path,
            diagnostics=(
                GitDiagnostic(name="path_exists", ok=False, message=missing_message),
                _blocked_diagnostic("working_tree", resolved_path, "the configured path does not exist"),
                _blocked_diagnostic("head_attached", resolved_path, "the configured path does not exist"),
                _blocked_diagnostic("working_tree_clean", resolved_path, "the configured path does not exist"),
            ),
        )

    work_tree_result = git_runner(("git", "rev-parse", "--is-inside-work-tree"), resolved_path)
    is_working_tree = work_tree_result.returncode == 0 and work_tree_result.stdout.strip() == "true"

    diagnostics: list[GitDiagnostic] = [
        GitDiagnostic(name="path_exists", ok=True, message=f"Configured core_path resolves to '{resolved_path}'."),
        _working_tree_diagnostic(core_path=core_path, resolved_path=resolved_path, result=work_tree_result, ok=is_working_tree),
    ]

    if not is_working_tree:
        diagnostics.extend(
            [
                _blocked_diagnostic("head_attached", resolved_path, "the configured path is not a Git working tree"),
                _blocked_diagnostic("working_tree_clean", resolved_path, "the configured path is not a Git working tree"),
            ],
        )
        return GitHealthReport(configured_path=core_path, resolved_path=resolved_path, diagnostics=tuple(diagnostics))

    head_result = git_runner(("git", "symbolic-ref", "--quiet", "HEAD"), resolved_path)
    diagnostics.append(_head_diagnostic(resolved_path=resolved_path, result=head_result))

    status_result = git_runner(("git", "status", "--porcelain", "--untracked-files=normal"), resolved_path)
    diagnostics.append(_working_tree_clean_diagnostic(resolved_path=resolved_path, result=status_result))

    return GitHealthReport(configured_path=core_path, resolved_path=resolved_path, diagnostics=tuple(diagnostics))


def check_environment_health(
    expected_accelerators: tuple[AcceleratorLabel, ...] | None,
    run_command: EnvironmentCommandRunner | None = None,
    probe_jax: JaxProbeRunner | None = None,
    accelerator_config_error: str | None = None,
):
    command_runner = _run_environment_command if run_command is None else run_command
    jax_probe_runner = _probe_jax if probe_jax is None else probe_jax

    uv_result = command_runner(("uv", "--version"))
    jax_result = jax_probe_runner()

    diagnostics: list[EnvironmentDiagnostic] = [
        _uv_diagnostic(uv_result),
        _jax_import_diagnostic(jax_result),
        _accelerator_expectation_diagnostic(
            expected_accelerators=expected_accelerators,
            jax_result=jax_result,
            accelerator_config_error=accelerator_config_error,
        ),
    ]
    return EnvironmentHealthReport(diagnostics=tuple(diagnostics))


def _resolve_core_path(workspace_root: Path, core_path: Path):
    raw_path = core_path if core_path.is_absolute() else workspace_root / core_path
    return raw_path.resolve(strict=False)


def _run_git(args: tuple[str, ...], cwd: Path):
    completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return GitCommandResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _run_environment_command(args: tuple[str, ...]):
    completed = subprocess.run(args, capture_output=True, text=True)
    return EnvironmentCommandResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _probe_jax():
    try:
        jax = importlib.import_module("jax")
        backend = jax.default_backend()
        device_platforms = tuple(sorted({device.platform for device in jax.devices()}))
    except (ImportError, RuntimeError) as exc:
        return JaxProbeResult(ok=False, backend=None, device_platforms=(), error=str(exc))

    return JaxProbeResult(ok=True, backend=backend, device_platforms=device_platforms)


def _blocked_diagnostic(name: GitCheckName, resolved_path: Path, reason: str):
    return GitDiagnostic(
        name=name,
        ok=False,
        message=f"Cannot evaluate '{name}' for '{resolved_path}' because {reason}.",
    )


def _working_tree_diagnostic(core_path: Path, resolved_path: Path, result: GitCommandResult, ok: bool):
    if ok:
        return GitDiagnostic(name="working_tree", ok=True, message=f"'{resolved_path}' is a Git working tree.")

    detail = _command_detail(result, "git")
    return GitDiagnostic(
        name="working_tree",
        ok=False,
        message=(
            f"Configured core_path '{core_path}' resolves to '{resolved_path}', but it is not a Git working tree. "
            f"Point research.yaml at a checked-out repository. {detail}"
        ),
    )


def _head_diagnostic(resolved_path: Path, result: GitCommandResult):
    if result.returncode == 0:
        return GitDiagnostic(name="head_attached", ok=True, message=f"HEAD is attached to a branch in '{resolved_path}'.")

    return GitDiagnostic(
        name="head_attached",
        ok=False,
        message=(
            f"HEAD is detached or unreadable in '{resolved_path}'. Check out a branch before running research workflows. "
            f"{_command_detail(result, 'git')}"
        ),
    )


def _working_tree_clean_diagnostic(resolved_path: Path, result: GitCommandResult):
    if result.returncode != 0:
        return GitDiagnostic(
            name="working_tree_clean",
            ok=False,
            message=f"Could not read Git status for '{resolved_path}'. {_command_detail(result, 'git')}",
        )

    if not result.stdout.strip():
        return GitDiagnostic(name="working_tree_clean", ok=True, message=f"Working tree is clean in '{resolved_path}'.")

    return GitDiagnostic(
        name="working_tree_clean",
        ok=False,
        message=(
            f"Working tree is not clean in '{resolved_path}'. Cleanliness policy treats any porcelain output, including untracked files, as a failure. "
            "Commit, stash, or remove changes before continuing."
        ),
    )


def _uv_diagnostic(result: EnvironmentCommandResult):
    if result.returncode == 0:
        return EnvironmentDiagnostic(name="uv_available", ok=True, message=f"uv responded to '--version': {result.stdout.strip()}.")

    return EnvironmentDiagnostic(
        name="uv_available",
        ok=False,
        message=f"uv is not discoverable or did not respond to '--version'. {_command_detail(result, 'uv')}",
    )


def _jax_import_diagnostic(result: JaxProbeResult):
    if result.ok:
        platform_text = ", ".join(result.device_platforms) if result.device_platforms else "none"
        return EnvironmentDiagnostic(
            name="jax_import",
            ok=True,
            message=f"JAX imported successfully. Default backend: {result.backend}. Detected device platforms: {platform_text}.",
        )

    detail = result.error if result.error else "JAX import failed."
    return EnvironmentDiagnostic(
        name="jax_import",
        ok=False,
        message=f"JAX could not be imported or probed without mutating the environment. {detail}",
    )


def _accelerator_expectation_diagnostic(
    expected_accelerators: tuple[AcceleratorLabel, ...] | None,
    jax_result: JaxProbeResult,
    accelerator_config_error: str | None = None,
):
    if accelerator_config_error is not None:
        return EnvironmentDiagnostic(
            name="accelerator_expectations",
            ok=False,
            message=(
                "Cannot evaluate accelerator expectations because research.yaml is invalid. "
                f"{accelerator_config_error}"
            ),
        )

    if not jax_result.ok:
        return EnvironmentDiagnostic(
            name="accelerator_expectations",
            ok=False,
            message="Cannot evaluate accelerator expectations because JAX import/probe failed.",
        )

    detected_accelerators = _normalize_accelerator_labels(jax_result.device_platforms)
    detected_text = ", ".join(detected_accelerators) if detected_accelerators else "none"

    if expected_accelerators is None:
        return EnvironmentDiagnostic(
            name="accelerator_expectations",
            ok=True,
            message=f"No doctor.expected_accelerators were configured. Observed accelerators: {detected_text}.",
        )

    missing_accelerators = tuple(label for label in expected_accelerators if label not in detected_accelerators)
    if not missing_accelerators:
        expected_text = ", ".join(expected_accelerators)
        return EnvironmentDiagnostic(
            name="accelerator_expectations",
            ok=True,
            message=f"Configured accelerators matched JAX detection. Expected: {expected_text}. Observed: {detected_text}.",
        )

    missing_text = ", ".join(missing_accelerators)
    expected_text = ", ".join(expected_accelerators)
    return EnvironmentDiagnostic(
        name="accelerator_expectations",
        ok=False,
        message=(
            f"Configured accelerators did not match JAX detection. Missing expected accelerator(s): {missing_text}. "
            f"Expected: {expected_text}. Observed: {detected_text}."
        ),
    )


def _normalize_accelerator_labels(device_platforms: tuple[str, ...]):
    normalized_labels: list[AcceleratorLabel] = []
    for platform in device_platforms:
        normalized_label = _normalize_accelerator_label(platform)
        if normalized_label is None or normalized_label in normalized_labels:
            continue
        normalized_labels.append(normalized_label)
    return tuple(normalized_labels)


def _normalize_accelerator_label(platform: str):
    normalized_platform = platform.casefold()
    if normalized_platform == "cpu":
        return "cpu"
    if normalized_platform == "tpu":
        return "tpu"
    if normalized_platform in {"gpu", "cuda", "rocm", "metal"}:
        return "gpu"
    return None


def _command_detail(result: GitCommandResult | EnvironmentCommandResult, command_name: str):
    stderr = result.stderr.strip()
    if stderr:
        return stderr

    stdout = result.stdout.strip()
    if stdout:
        return stdout

    return f"{command_name} exited with status {result.returncode}."