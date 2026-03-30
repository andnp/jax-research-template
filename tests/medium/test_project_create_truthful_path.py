"""Medium integration test for the generated project truthful smoke path."""

import os
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


def _workspace_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(tmp_path / ".uv-cache"))
    return env


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _write_workspace_root(workspace_root: Path) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "projects").mkdir()
    (workspace_root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "truthful-workspace"\n'
        'version = "0.0.0"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[tool.uv]\n"
        "package = false\n\n"
        "[tool.uv.workspace]\n"
        'members = ["libs/*", "projects/*"]\n\n'
        "[tool.uv.sources]\n"
        "jax = { workspace = true }\n"
        "gymnax = { workspace = true }\n"
        "matplotlib = { workspace = true }\n"
        'rl-agents = { workspace = true }\n'
        'rl-components = { workspace = true }\n',
        encoding="utf-8",
    )


def _write_package(package_root: Path, project_name: str, version: str, modules: dict[str, str]) -> None:
    (package_root / "src").mkdir(parents=True, exist_ok=True)
    (package_root / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{project_name}"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
    for relative_path, content in modules.items():
        module_path = package_root / "src" / relative_path
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(content, encoding="utf-8")


def _write_fake_workspace_packages(workspace_root: Path) -> None:
    libs_root = workspace_root / "libs"
    _write_package(
        libs_root / "jax",
        project_name="jax",
        version="0.9.0",
        modules={
            "jax/__init__.py": "from . import random\n\n\ndef jit(function):\n    return function\n",
            "jax/random.py": "def PRNGKey(seed: int) -> int:\n    return seed\n",
        },
    )
    _write_package(
        libs_root / "gymnax",
        project_name="gymnax",
        version="0.0.9",
        modules={
            "gymnax/__init__.py": (
                "from . import wrappers\n\n\n"
                "def make(name: str) -> tuple[dict[str, str], dict[str, int]]:\n"
                "    return {\"name\": name}, {\"max_steps_in_episode\": 1}\n"
            ),
            "gymnax/wrappers.py": "class LogWrapper:\n    def __init__(self, env: object) -> None:\n        self.env = env\n",
        },
    )
    _write_package(
        libs_root / "matplotlib",
        project_name="matplotlib",
        version="3.9.0",
        modules={
            "matplotlib/__init__.py": "from . import pyplot\n",
            "matplotlib/pyplot.py": (
                "from pathlib import Path\n\n\n"
                "def plot(values: list[float]) -> None:\n"
                "    _ = values\n\n\n"
                "def xlabel(label: str) -> None:\n"
                "    _ = label\n\n\n"
                "def ylabel(label: str) -> None:\n"
                "    _ = label\n\n\n"
                "def title(label: str) -> None:\n"
                "    _ = label\n\n\n"
                "def savefig(path: str) -> None:\n"
                "    Path(path).write_text('fake plot', encoding='utf-8')\n"
            ),
        },
    )
    _write_package(
        libs_root / "rl-agents",
        project_name="rl-agents",
        version="0.0.0",
        modules={
            "rl_agents/__init__.py": "",
            "rl_agents/ppo.py": (
                "def make_train(config, env: object, env_params: object):\n"
                "    _ = config\n"
                "    _ = env\n"
                "    _ = env_params\n\n"
                "    def train(rng: int) -> dict[str, dict[str, list[float]]]:\n"
                "        _ = rng\n"
                "        return {\"metrics\": {\"returned_episode_returns\": [0.0, 1.0]}}\n\n"
                "    return train\n"
            ),
        },
    )
    _write_package(
        libs_root / "rl-components",
        project_name="rl-components",
        version="0.0.0",
        modules={
            "rl_components/__init__.py": "",
            "rl_components/types.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True, slots=True)\n"
                "class PPOConfig:\n"
                "    ENV_NAME: str\n"
                "    TOTAL_TIMESTEPS: int\n"
                "    SEED: int\n"
            ),
        },
    )


def test_project_create_truthful_smoke_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = tmp_path / "workspace"
    env = _workspace_env(tmp_path)
    _write_workspace_root(workspace_root)
    _write_fake_workspace_packages(workspace_root)

    _run(
        ["uv", "run", "--project", str(repo_root / "cli"), "research", "project", "create", "demo"],
        cwd=workspace_root,
        env=env,
    )

    project_root = workspace_root / "projects" / "demo"
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    setup_command = ["uv", "sync", "--all-packages"]
    smoke_command = ["uv", "run", "--directory", "projects/demo", "python", "train.py"]

    assert _command_text(setup_command) in readme
    assert _command_text(smoke_command) in readme

    _run(setup_command, cwd=workspace_root, env=env)
    smoke_result = _run(smoke_command, cwd=workspace_root, env=env)

    assert smoke_result.stdout.strip() == "Final mean return: 1.00"
    assert (project_root / "results.png").is_file()