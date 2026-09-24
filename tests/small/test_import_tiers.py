"""Tests for the repository's library-to-domain import boundary."""

import ast
from pathlib import Path


def _domain_packages(root: Path) -> set[str]:
    packages: set[str] = set()
    for source_root in root.glob("domains/**/src"):
        packages.update(
            path.name
            for path in source_root.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        )
    return packages


def test_library_sources_do_not_import_domain_packages() -> None:
    """Library modules must not import packages owned by the domains tier."""
    root = Path(__file__).parents[2]
    domain_packages = _domain_packages(root)
    assert {"rl_agents", "process_control"} <= domain_packages
    violations: list[str] = []

    for path in sorted(root.glob("libs/*/src/**/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module.split(".")[0]]
            else:
                continue
            for package in imported:
                if package in domain_packages:
                    violations.append(f"{path}:{node.lineno}: {package}")

    assert not violations, "Library-to-domain imports found:\n" + "\n".join(violations)
