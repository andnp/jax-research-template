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
- supports `--dry-run` previews,
- exits with an error when the target directory already exists and the command is not a dry run.

## What It Does Not Do
The current implementation does not:
- run `uv sync`,
- install Git hooks,
- validate the `core_url` beyond delegating to Git,
- create any projects inside `projects/`,
- repair or upgrade the `core/` checkout.

## Post-Bootstrap Follow-Up
Truthful follow-up depends on whether the workspace already contains `core/`:
- With `--core-url`: the workspace can proceed to environment setup after the submodule is present.
- Without `--core-url`: the command prints a tip to add `core/` later with `git submodule add <url> core`, and environment setup should wait until that checkout exists.

Repo docs currently state these separate manual steps:
- use `uv sync` for dependency setup,
- install shared hooks explicitly with `./scripts/install-git-hooks.sh` from a core checkout when hook installation is desired.

## Expected Workspace Shape
A successful non-dry-run bootstrap creates:
- `<workspace>/pyproject.toml`
- `<workspace>/research.yaml`
- `<workspace>/.gitignore`
- `<workspace>/projects/.gitkeep`
- `<workspace>/.git/`
- optionally `<workspace>/core/` when `--core-url` is supplied
