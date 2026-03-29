---
name: bootstrap-shell
description: 'Bootstrap a new research shell workspace with the current CLI. Use when creating a workspace, previewing `research workspace init`, deciding whether to pass `--core-url`, or checking the required manual follow-up steps after workspace creation.'
---

# Bootstrap Shell

## When to Use
- Create a new shell workspace around the research core.
- Preview workspace creation before mutating the filesystem.
- Explain what `research workspace init` does today.
- Document the manual follow-up after workspace creation.

## Procedure
1. Read the current contract in [workspace bootstrap reference](./references/workspace-bootstrap.md).
2. Confirm the target parent directory, workspace name, and whether the user wants a dry run or a real mutation.
3. Prefer a preview first with `research workspace init <name> --path <parent> --dry-run`.
4. Add `--core-url <git-url>` only when the user explicitly wants the new workspace to add a `core/` submodule during creation.
5. If the user wants the actual workspace, run the same command without `--dry-run`.
6. After creation, verify the generated workspace shape against the reference before suggesting next steps.
7. Call out the manual follow-up truthfully:
   - the command does not run `uv sync`,
   - the command does not install hooks,
   - if no `--core-url` was supplied, the workspace still needs a `core/` checkout before environment sync can succeed.

## Notes
- Keep the guidance tied to the current CLI behavior in `cli/src/research_cli/workspace.py`.
- Do not claim automatic dependency installation, hook installation, or submodule repair during bootstrap.
