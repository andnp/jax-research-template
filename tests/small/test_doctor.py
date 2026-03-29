from pathlib import Path

from research_cli.doctor import GitCommandResult, check_git_health


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