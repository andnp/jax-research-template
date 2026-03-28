import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer


@dataclass(frozen=True)
class LifecyclePreview:
    action: str
    workspace_root: Path
    source_path: Path
    target_path: Path
    rewrite_scope: Path
    copy_plan: tuple[str, ...]
    rewrite_plan: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _import_package_name(library_name: str) -> str:
    import_package = library_name.replace("-", "_").strip()
    if not import_package:
        _fail("Error: library name must not be empty.")

    segments = import_package.split(".")
    if any(not segment.isidentifier() for segment in segments):
        _fail(
            f"Error: inferred import package '{import_package}' from library '{library_name}' is invalid."
        )

    return import_package


def _module_path(root: Path, import_package: str) -> Path:
    return (root / Path(*import_package.split("."))).resolve()


def _resolve_workspace_root(cwd: Path | None = None) -> tuple[Path, Path]:
    workspace_root = (cwd or Path.cwd()).resolve()
    direct_libs_root = (workspace_root / "libs").resolve()
    nested_libs_root = (workspace_root / "core" / "libs").resolve()

    if direct_libs_root.is_dir():
        return workspace_root, direct_libs_root

    if nested_libs_root.is_dir():
        return workspace_root, nested_libs_root

    _fail(
        "Error: expected a workspace root containing 'libs/' or 'core/libs/' for lifecycle resolution."
    )


def _resolve_project_root(workspace_root: Path, project_name: str) -> Path:
    projects_root = (workspace_root / "projects").resolve()
    if not projects_root.is_dir():
        _fail(f"Error: expected project root '{projects_root}' to exist.")

    project_root = (projects_root / project_name).resolve()
    if not project_root.is_dir():
        _fail(f"Error: project '{project_name}' was not found under '{projects_root}'.")

    return project_root


def _resolve_lib_root(libs_root: Path, library_name: str) -> Path:
    lib_root = (libs_root / library_name).resolve()
    if not lib_root.is_dir():
        _fail(f"Error: library '{library_name}' was not found under '{libs_root}'.")

    return lib_root


def _resolve_existing_module(path: Path, import_package: str) -> Path:
    if not path.is_dir():
        _fail(f"Error: module '{import_package}' was not found at '{path}'.")

    return path


def _ensure_target_absent(path: Path, action: str) -> None:
    if path.exists():
        _fail(f"Error: {action} destination '{path}' already exists.")


def _relative_to(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sorted_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sorted_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _build_copy_plan(workspace_root: Path, source_path: Path, target_path: Path) -> tuple[str, ...]:
    return tuple(
        f"{_relative_to(workspace_root, source_file)} -> {_relative_to(workspace_root, target_path / source_file.relative_to(source_path))}"
        for source_file in _sorted_files(source_path)
    )


def _build_rewrite_plan(workspace_root: Path, project_root: Path, source_path: Path, target_path: Path) -> tuple[str, ...]:
    existing_paths = {_relative_to(workspace_root, path) for path in _sorted_python_files(project_root)}
    copied_paths = {
        _relative_to(workspace_root, target_path / source_file.relative_to(source_path))
        for source_file in _sorted_python_files(source_path)
    }
    return tuple(sorted(existing_paths | copied_paths))


def _split_inline_comment(line: str) -> tuple[str, str]:
    comment_index = line.find("#")
    if comment_index == -1:
        return line, ""

    return line[:comment_index], line[comment_index:]


def _rewrite_import_clause(clause: str, import_package: str) -> tuple[str, bool]:
    rewritten = re.sub(
        rf"(^\s*){re.escape(import_package)}(?=\.|\s|$)",
        rf"\1components.{import_package}",
        clause,
        count=1,
    )
    return rewritten, rewritten != clause


def _rewrite_imports(text: str, import_package: str) -> str:
    rewritten_text = re.sub(
        rf"(^\s*from\s+){re.escape(import_package)}(?=\.|\s+import\b)",
        rf"\1components.{import_package}",
        text,
        flags=re.MULTILINE,
    )

    rewritten_lines: list[str] = []
    for line in rewritten_text.splitlines(keepends=True):
        stripped_line = line.lstrip()
        if not stripped_line.startswith("import "):
            rewritten_lines.append(line)
            continue

        newline = ""
        if line.endswith("\r\n"):
            newline = "\r\n"
            line_without_newline = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            line_without_newline = line[:-1]
        else:
            line_without_newline = line

        code, comment = _split_inline_comment(line_without_newline)
        prefix, body = code.split("import ", maxsplit=1)
        rewritten_clauses: list[str] = []
        clause_changed = False
        for clause in body.split(","):
            rewritten_clause, changed = _rewrite_import_clause(clause, import_package)
            rewritten_clauses.append(rewritten_clause)
            clause_changed = clause_changed or changed

        if clause_changed:
            rewritten_lines.append(f"{prefix}import {','.join(rewritten_clauses)}{comment}{newline}")
            continue

        rewritten_lines.append(line)

    return "".join(rewritten_lines)


def _rewrite_project_imports(project_root: Path, import_package: str) -> tuple[str, ...]:
    rewritten_paths: list[str] = []
    for path in _sorted_python_files(project_root):
        original_text = path.read_text(encoding="utf-8")
        rewritten_text = _rewrite_imports(original_text, import_package)
        if rewritten_text == original_text:
            continue

        path.write_text(rewritten_text, encoding="utf-8")
        rewritten_paths.append(path.relative_to(project_root).as_posix())

    return tuple(rewritten_paths)


def _copy_package_tree(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path)


def _resolve_eject_preview(project_name: str, library_name: str, cwd: Path | None = None) -> LifecyclePreview:
    workspace_root, libs_root = _resolve_workspace_root(cwd)
    project_root = _resolve_project_root(workspace_root, project_name)
    lib_root = _resolve_lib_root(libs_root, library_name)
    import_package = _import_package_name(library_name)
    source_path = _resolve_existing_module(_module_path(lib_root / "src", import_package), import_package)
    target_path = _module_path(project_root / "components", import_package)
    _ensure_target_absent(target_path, "eject")
    return LifecyclePreview(
        action="eject",
        workspace_root=workspace_root,
        source_path=source_path,
        target_path=target_path,
        rewrite_scope=project_root,
        copy_plan=_build_copy_plan(workspace_root, source_path, target_path),
        rewrite_plan=_build_rewrite_plan(workspace_root, project_root, source_path, target_path),
    )


def _resolve_harvest_preview(project_name: str, library_name: str, cwd: Path | None = None) -> LifecyclePreview:
    workspace_root, libs_root = _resolve_workspace_root(cwd)
    project_root = _resolve_project_root(workspace_root, project_name)
    lib_root = _resolve_lib_root(libs_root, library_name)
    import_package = _import_package_name(library_name)
    source_path = _resolve_existing_module(_module_path(project_root / "components", import_package), import_package)
    target_path = _module_path(lib_root / "src", import_package)
    return LifecyclePreview(
        action="harvest",
        workspace_root=workspace_root,
        source_path=source_path,
        target_path=target_path,
        rewrite_scope=project_root,
        copy_plan=(),
        rewrite_plan=(),
    )


def _echo_preview(summary: str, preview: LifecyclePreview, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    typer.echo(f"{prefix}{summary}")
    typer.echo(f"  Workspace root: {preview.workspace_root}")
    typer.echo(f"  Source path: {preview.source_path}")
    typer.echo(f"  Target path: {preview.target_path}")
    typer.echo(f"  Rewrite scope: {preview.rewrite_scope}")
    if preview.copy_plan:
        typer.echo("  Copy plan:")
        for entry in preview.copy_plan:
            typer.echo(f"    - {entry}")

    if preview.rewrite_plan:
        typer.echo("  Rewrite scope (Python files):")
        for entry in preview.rewrite_plan:
            typer.echo(f"    - {entry}")


def _ensure_preview_only(action: str, dry_run: bool) -> None:
    if not dry_run:
        _fail(f"Error: {action} preview is implemented, but filesystem mutation is not available yet. Re-run with --dry-run.")


def _execute_eject(preview: LifecyclePreview, import_package: str) -> tuple[str, ...]:
    _copy_package_tree(preview.source_path, preview.target_path)
    return _rewrite_project_imports(preview.rewrite_scope, import_package)


def eject(
    project: str = typer.Argument(..., help="Project name under projects/."),
    library: str = typer.Argument(..., help="Library name under libs/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview resolved paths without copying anything."),
) -> None:
    preview = _resolve_eject_preview(project, library)
    if dry_run:
        _echo_preview(f"Eject library '{library}' into project '{project}'", preview, dry_run)
        return

    rewritten_paths = _execute_eject(preview, _import_package_name(library))
    _echo_preview(f"Ejected library '{library}' into project '{project}'", preview, dry_run)
    typer.echo(f"  Rewritten files: {len(rewritten_paths)}")


def harvest(
    project: str = typer.Argument(..., help="Project name under projects/."),
    library: str = typer.Argument(..., help="Library name under libs/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview resolved paths without copying anything."),
) -> None:
    preview = _resolve_harvest_preview(project, library)
    _echo_preview(f"Harvest library '{library}' from project '{project}'", preview, dry_run)
    _ensure_preview_only("harvest", dry_run)