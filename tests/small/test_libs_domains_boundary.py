"""Enforce that libs/* packages never import from domains/* packages."""

from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]


def _domain_module_names() -> set[str]:
    domains_root = CORE_ROOT / "domains"
    names = set()
    for src_dir in domains_root.glob("*/*/src"):
        names.update(p.name for p in src_dir.iterdir() if (p / "__init__.py").is_file())
    return names


def _imported_top_level_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


class TestLibsDomainsBoundary:
    def test_libs_do_not_import_domains(self) -> None:
        domain_modules = _domain_module_names()
        violations = []
        for src_file in CORE_ROOT.glob("libs/*/src/**/*.py"):
            imported = _imported_top_level_modules(src_file)
            hit = imported & domain_modules
            if hit:
                violations.append(f"{src_file.relative_to(CORE_ROOT)} imports {sorted(hit)}")

        assert not violations, "libs/* packages may not import domains/* packages:\n" + "\n".join(violations)
