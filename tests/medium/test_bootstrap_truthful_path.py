"""Medium integration test for the truthful shell bootstrap path."""

import os
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Command failed: {' '.join(command)}\n"
        f"cwd: {cwd}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _git(cwd: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, env=env)


def _workspace_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_ALLOW_PROTOCOL"] = "file"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("UV_CACHE_DIR", str(tmp_path / ".uv-cache"))
    return env


def _init_repo(repo_root: Path, env: dict[str, str]) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, env, "init", "--initial-branch=main")
    _git(repo_root, env, "config", "user.name", "Test User")
    _git(repo_root, env, "config", "user.email", "test@example.com")


def _write_fake_jax_package(core_root: Path) -> None:
    package_root = core_root / "libs" / "jax"
    (package_root / "src" / "jax").mkdir(parents=True, exist_ok=True)
    (package_root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "jax"\n'
        'version = "0.0.0"\n'
        'description = "Test-only fake JAX package"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
    (package_root / "src" / "jax" / "__init__.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class Device:\n"
        "    platform: str\n\n"
        "def default_backend() -> str:\n"
        '    return "cpu"\n\n'
        "def devices() -> list[Device]:\n"
        '    return [Device(platform="cpu")]\n',
        encoding="utf-8",
    )


def _create_core_fixture(repo_root: Path, core_root: Path, env: dict[str, str]) -> None:
    _init_repo(core_root, env)
    (core_root / "libs").mkdir(parents=True, exist_ok=True)
    (core_root / ".gitignore").write_text("__pycache__/\n*.py[cod]\n*.egg-info/\n", encoding="utf-8")
    shutil.copytree(
        repo_root / "cli" / "src" / "research_cli",
        core_root / "cli" / "src" / "research_cli",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(repo_root / "cli" / "pyproject.toml", core_root / "cli" / "pyproject.toml")
    _write_fake_jax_package(core_root)
    _git(core_root, env, "add", ".")
    _git(core_root, env, "commit", "-m", "fixture core")


def test_bootstrap_truthful_path_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = _workspace_env(tmp_path)
    core_origin = tmp_path / "core-origin"
    _create_core_fixture(repo_root, core_origin, env)

    shells_root = tmp_path / "shells"
    init_result = _run(
        [
            "uv",
            "run",
            "research",
            "workspace",
            "init",
            "truthful-shell",
            "--path",
            str(shells_root),
            "--core-url",
            str(core_origin),
        ],
        cwd=repo_root,
        env=env,
    )

    workspace_root = shells_root / "truthful-shell"
    assert workspace_root.is_dir()
    assert "uv sync --all-packages" in init_result.stdout
    assert "uv run research doctor" in init_result.stdout

    sync_result = _run(["uv", "sync", "--all-packages"], cwd=workspace_root, env=env)
    assert "Resolved" in sync_result.stderr or "Prepared" in sync_result.stderr or sync_result.stderr == ""

    doctor_result = subprocess.run(
        ["uv", "run", "research", "doctor"],
        cwd=workspace_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert doctor_result.returncode == 0, doctor_result.stdout + doctor_result.stderr
    assert "[x] Config validation" in doctor_result.stdout
    assert "[x] Git health" in doctor_result.stdout
    assert "[x] Environment health" in doctor_result.stdout
    assert "[x] working_tree: " in doctor_result.stdout
    assert "[x] head_attached: " in doctor_result.stdout
    assert "[x] working_tree_clean: " in doctor_result.stdout
    assert "[x] jax_import: JAX imported successfully." in doctor_result.stdout
    assert doctor_result.stdout.rstrip().endswith("overall: PASS")
