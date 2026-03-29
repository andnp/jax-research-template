"""Read-only diagnostics for ``research doctor``."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

GitCheckName = Literal["path_exists", "working_tree", "head_attached", "working_tree_clean"]


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
class GitHealthReport:
    configured_path: Path
    resolved_path: Path
    diagnostics: tuple[GitDiagnostic, ...]

    @property
    def ok(self):
        return all(diagnostic.ok for diagnostic in self.diagnostics)


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


def _resolve_core_path(workspace_root: Path, core_path: Path):
    raw_path = core_path if core_path.is_absolute() else workspace_root / core_path
    return raw_path.resolve(strict=False)


def _run_git(args: tuple[str, ...], cwd: Path):
    completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return GitCommandResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _blocked_diagnostic(name: GitCheckName, resolved_path: Path, reason: str):
    return GitDiagnostic(
        name=name,
        ok=False,
        message=f"Cannot evaluate '{name}' for '{resolved_path}' because {reason}.",
    )


def _working_tree_diagnostic(core_path: Path, resolved_path: Path, result: GitCommandResult, ok: bool):
    if ok:
        return GitDiagnostic(name="working_tree", ok=True, message=f"'{resolved_path}' is a Git working tree.")

    detail = _command_detail(result)
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
            f"{_command_detail(result)}"
        ),
    )


def _working_tree_clean_diagnostic(resolved_path: Path, result: GitCommandResult):
    if result.returncode != 0:
        return GitDiagnostic(
            name="working_tree_clean",
            ok=False,
            message=f"Could not read Git status for '{resolved_path}'. {_command_detail(result)}",
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


def _command_detail(result: GitCommandResult):
    stderr = result.stderr.strip()
    if stderr:
        return stderr

    stdout = result.stdout.strip()
    if stdout:
        return stdout

    return f"git exited with status {result.returncode}."