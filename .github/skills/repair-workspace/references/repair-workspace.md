# Workspace Repair Reference

## Command Surface
Current command:
- `research workspace repair [--dry-run]`

Primary implementation:
- `cli/src/research_cli/workspace.py`
- `tests/medium/test_cli_workspace_repair.py`
- `docs/specs/research-cli.md`

## Preconditions
The current implementation:
- starts from the current working directory,
- searches upward to find a workspace root,
- loads `research.yaml` from that resolved workspace root,
- resolves `core_path` relative to the workspace root when needed,
- requires the resolved `core_path` to exist,
- requires the resolved `core_path` to stay inside the workspace root.

If those conditions are not met, the command exits with an error instead of guessing another target.

## What the Command Does Today
`research workspace repair` currently:
- resolves the workspace root from the current working directory,
- loads `research.yaml` from that workspace,
- resolves the configured Core checkout path,
- runs `git -C <core_path> clean -ffd`,
- runs `git submodule update --force --checkout -- <submodule_path>` from the workspace root,
- scopes all mutation to the configured Core checkout,
- restores the Core checkout to the superproject-recorded submodule revision,
- prints a success message with the resolved Core path when the mutating run succeeds.

## Dry-Run Behavior
With `--dry-run`, the current implementation prints the ordered commands it would run and exits without mutating:
- the filesystem,
- Git state,
- `research.yaml`,
- dependencies,
- any path outside the configured Core checkout.

## What It Does Not Do
The current implementation does not:
- edit `research.yaml`,
- run `uv sync`,
- upgrade Core to a newer upstream revision,
- repair other workspace members,
- infer alternate Core targets,
- apply broader cleanup outside the configured `core_path`.

## Mutation Scope
The current cleanup scope is intentionally narrow:
- tracked-file changes inside `core_path` may be discarded if needed to restore the recorded revision,
- untracked files or directories inside `core_path` may be removed when they block or conflict with that reset,
- files outside `core_path` are out of scope for this command.
