import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer

from research_cli.workspace import WorkspaceResolutionError, resolve_workspace_root


@dataclass(frozen=True)
class LifecyclePreview:
    action: str
    workspace_root: Path
    source_path: Path
    target_path: Path
    rewrite_scope: Path
    create_library_manifest: bool
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
    try:
        workspace_root = resolve_workspace_root(cwd or Path.cwd())
    except WorkspaceResolutionError as exc:
        _fail(f"Error: {exc}")

    direct_libs_root = (workspace_root / "libs").resolve()
    nested_libs_root = (workspace_root / "core" / "libs").resolve()

    if direct_libs_root.is_dir():
        return workspace_root, direct_libs_root

    if nested_libs_root.is_dir():
        return workspace_root, nested_libs_root

    _fail(
        f"Error: resolved workspace root '{workspace_root}' does not contain 'libs/' or 'core/libs/' required for lifecycle commands."
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


def _resolve_harvest_library_root(libs_root: Path, library_name: str) -> tuple[Path, bool]:
    lib_root = (libs_root / library_name).resolve()
    if not lib_root.exists():
        return lib_root, True

    if not lib_root.is_dir():
        _fail(f"Error: harvest destination library '{lib_root}' is not a directory.")

    pyproject_path = lib_root / "pyproject.toml"
    if not pyproject_path.is_file():
        _fail(f"Error: harvest destination library '{lib_root}' is missing 'pyproject.toml'.")

    src_root = lib_root / "src"
    if src_root.exists() and not src_root.is_dir():
        _fail(f"Error: harvest destination source root '{src_root}' is not a directory.")

    return lib_root, False


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


def _rewrite_import_clause(clause: str, from_import: str, to_import: str) -> tuple[str, bool]:
    rewritten = re.sub(
        rf"(^\s*){re.escape(from_import)}(?=\.|\s|$)",
        rf"\1{to_import}",
        clause,
        count=1,
    )
    return rewritten, rewritten != clause


def _rewrite_imports(text: str, from_import: str, to_import: str) -> str:
    rewritten_text = re.sub(
        rf"(^\s*from\s+){re.escape(from_import)}(?=\.|\s+import\b)",
        rf"\1{to_import}",
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
            rewritten_clause, changed = _rewrite_import_clause(clause, from_import, to_import)
            rewritten_clauses.append(rewritten_clause)
            clause_changed = clause_changed or changed

        if clause_changed:
            rewritten_lines.append(f"{prefix}import {','.join(rewritten_clauses)}{comment}{newline}")
            continue

        rewritten_lines.append(line)

    return "".join(rewritten_lines)


def _rewrite_tree_imports(root: Path, from_import: str, to_import: str) -> tuple[str, ...]:
    rewritten_paths: list[str] = []
    for path in _sorted_python_files(root):
        original_text = path.read_text(encoding="utf-8")
        rewritten_text = _rewrite_imports(original_text, from_import, to_import)
        if rewritten_text == original_text:
            continue

        path.write_text(rewritten_text, encoding="utf-8")
        rewritten_paths.append(path.relative_to(root).as_posix())

    return tuple(rewritten_paths)


def _copy_package_tree(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path)


def _move_package_tree(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source_path, target_path)


def _library_root_from_target(target_path: Path, import_package: str) -> Path:
    return target_path.parents[len(import_package.split("."))]


def _render_library_manifest(library_name: str) -> str:
    return (
        "[project]\n"
        f'name = "{library_name}"\n'
        'version = "0.1.0"\n'
        'description = "Harvested shared library"\n'
        'requires-python = ">=3.13"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=61.0"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
    )


def _initialize_library_manifest(lib_root: Path, library_name: str) -> None:
    lib_root.mkdir(parents=True, exist_ok=False)
    (lib_root / "pyproject.toml").write_text(_render_library_manifest(library_name), encoding="utf-8")


def _find_section_bounds(lines: list[str], header: str) -> tuple[int, int]:
    start = -1
    for index, line in enumerate(lines):
        if line.strip() == header:
            start = index + 1
            break

    if start == -1:
        _fail(f"Error: expected section '{header}' in workspace pyproject.toml.")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("["):
            end = index
            break

    return start, end


def _quoted_array_entry(entry: str) -> str:
    return f'"{entry}"'


def _ensure_toml_array_entry(lines: list[str], section_header: str, key: str, entry: str) -> bool:
    start, end = _find_section_bounds(lines, section_header)
    array_start = -1
    array_end = -1
    for index in range(start, end):
        if lines[index].strip() == f"{key} = [":
            array_start = index + 1
            break

    if array_start == -1:
        _fail(f"Error: expected key '{key}' in section '{section_header}' of workspace pyproject.toml.")

    for index in range(array_start, end):
        if lines[index].strip() == "]":
            array_end = index
            break

    if array_end == -1:
        _fail(f"Error: expected closing array for key '{key}' in section '{section_header}'.")

    quoted_entry = _quoted_array_entry(entry)
    if any(line.strip().rstrip(",") == quoted_entry for line in lines[array_start:array_end]):
        return False

    lines.insert(array_end, f"    {quoted_entry},\n")
    return True


def _ensure_toml_table_entry(lines: list[str], section_header: str, key: str, value: str) -> bool:
    start, end = _find_section_bounds(lines, section_header)
    if any(line.startswith(f"{key} = ") for line in lines[start:end]):
        return False

    lines.insert(end, f"{key} = {value}\n")
    return True


def _ensure_root_workspace_library_registration(workspace_root: Path, library_name: str) -> tuple[str, ...]:
    pyproject_path = workspace_root / "pyproject.toml"
    if not pyproject_path.is_file():
        _fail(f"Error: expected workspace pyproject '{pyproject_path}' to exist.")

    lines = pyproject_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_sections: list[str] = []
    if _ensure_toml_array_entry(lines, "[project]", "dependencies", library_name):
        updated_sections.append("project.dependencies")
    if _ensure_toml_table_entry(lines, "[tool.uv.sources]", library_name, "{ workspace = true }"):
        updated_sections.append("tool.uv.sources")
    if _ensure_toml_array_entry(lines, "[tool.ty.environment]", "extra-paths", f"libs/{library_name}/src"):
        updated_sections.append("tool.ty.environment.extra-paths")

    if updated_sections:
        pyproject_path.write_text("".join(lines), encoding="utf-8")

    return tuple(updated_sections)


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
        create_library_manifest=False,
        copy_plan=_build_copy_plan(workspace_root, source_path, target_path),
        rewrite_plan=_build_rewrite_plan(workspace_root, project_root, source_path, target_path),
    )


def _resolve_harvest_preview(project_name: str, library_name: str, cwd: Path | None = None) -> LifecyclePreview:
    workspace_root, libs_root = _resolve_workspace_root(cwd)
    project_root = _resolve_project_root(workspace_root, project_name)
    import_package = _import_package_name(library_name)
    source_path = _resolve_existing_module(_module_path(project_root / "components", import_package), import_package)
    lib_root, create_library_manifest = _resolve_harvest_library_root(libs_root, library_name)
    target_path = _module_path(lib_root / "src", import_package)
    _ensure_target_absent(target_path, "harvest")
    return LifecyclePreview(
        action="harvest",
        workspace_root=workspace_root,
        source_path=source_path,
        target_path=target_path,
        rewrite_scope=project_root,
        create_library_manifest=create_library_manifest,
        copy_plan=(),
        rewrite_plan=_build_rewrite_plan(workspace_root, project_root, source_path, target_path),
    )


def _echo_preview(summary: str, preview: LifecyclePreview, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    typer.echo(f"{prefix}{summary}")
    typer.echo(f"  Workspace root: {preview.workspace_root}")
    typer.echo(f"  Source path: {preview.source_path}")
    typer.echo(f"  Target path: {preview.target_path}")
    typer.echo(f"  Create library manifest: {'yes' if preview.create_library_manifest else 'no'}")
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
    return _rewrite_tree_imports(preview.rewrite_scope, import_package, f"components.{import_package}")


def _execute_harvest(preview: LifecyclePreview, library_name: str, import_package: str) -> tuple[str, ...]:
    lib_root = _library_root_from_target(preview.target_path, import_package)
    if preview.create_library_manifest:
        _initialize_library_manifest(lib_root, library_name)

    _move_package_tree(preview.source_path, preview.target_path)
    workspace_updates = _ensure_root_workspace_library_registration(preview.workspace_root, library_name)
    project_rewrites = _rewrite_tree_imports(preview.rewrite_scope, f"components.{import_package}", import_package)
    target_rewrites = _rewrite_tree_imports(preview.target_path, f"components.{import_package}", import_package)
    return tuple(sorted({*workspace_updates, *project_rewrites, *target_rewrites}))


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
    if dry_run:
        _echo_preview(f"Harvest library '{library}' from project '{project}'", preview, dry_run)
        return

    rewritten_paths = _execute_harvest(preview, library, _import_package_name(library))
    _echo_preview(f"Harvested library '{library}' from project '{project}'", preview, dry_run)
    typer.echo(f"  Rewritten files: {len(rewritten_paths)}")