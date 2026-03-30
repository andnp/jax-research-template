# Workspace Bootstrap Reference

## Command Surface
Current command:
- `research workspace init <name> --path <parent> [--core-url <git-url>] [--dry-run]`

Primary implementation:
- `cli/src/research_cli/workspace.py`
- `tests/small/test_workspace_init.py`
- `docs/specs/research-cli.md`

## What the Command Does Today
`research workspace init` currently:
- resolves the target workspace directory as `<path>/<name>`,
- initializes a Git repository in that directory,
- creates `projects/` and a `.gitkeep` inside it,
- optionally runs `git submodule add <core-url> core` when `--core-url` is provided,
- writes `pyproject.toml`, `research.yaml`, and `.gitignore`,
- configures the generated root `pyproject.toml` with `uv` workspace members for `core/cli`, `core/libs/*`, and `projects/*`,
- supports `--dry-run` previews,
- exits with an error when the target directory already exists and the command is not a dry run.

## What It Does Not Do
The current implementation does not:
- run `uv sync` or `uv sync --all-packages`,
- run `research doctor`,
- install Git hooks,
- validate the `core_url` beyond delegating to Git,
- create any projects inside `projects/`,
- repair or upgrade the `core/` checkout.

## Post-Bootstrap Follow-Up
Truthful follow-up depends on whether the workspace already contains `core/`:
- With `--core-url`: the workspace can proceed directly to `uv sync --all-packages`, then `uv run research doctor`.
- Without `--core-url`: the command prints the exact manual sequence: add `core/` with `git submodule add <url> core`, then run `uv sync --all-packages`, then `uv run research doctor`.

Repo docs currently state these separate manual steps:
- use `uv sync --all-packages` for dependency setup once `core/` exists so the `core/cli` workspace member is installed alongside the shared libraries,
- install shared hooks explicitly with `./scripts/install-git-hooks.sh` from a core checkout when hook installation is desired.

## Expected Workspace Shape
A successful non-dry-run bootstrap creates:
- `<workspace>/pyproject.toml`
- `<workspace>/research.yaml`
- `<workspace>/.gitignore`
- `<workspace>/projects/.gitkeep`
- `<workspace>/.git/`
- a root `uv` workspace configuration that already references `core/cli`, `core/libs/*`, and `projects/*`
- optionally `<workspace>/core/` when `--core-url` is supplied
