---
name: repair-workspace
description: 'Repair the configured Core checkout in the current workspace. Use when previewing or running `research workspace repair`, resetting the Core submodule back to the workspace-recorded revision, or removing blocking changes under `core_path`. This command mutates Git state, so prefer `--dry-run` first.'
---

# Repair Workspace

## When to Use
- Preview or run the current workspace repair flow for the configured Core checkout.
- Restore the Core submodule to the workspace-recorded revision.
- Remove tracked or untracked blockers inside the configured `core_path`.
- Explain the current repair scope and limits truthfully before mutation.

## Procedure
1. Read the current contract in [workspace repair reference](./references/repair-workspace.md).
2. Confirm the command will run from the workspace root or a directory inside the workspace so the current implementation can locate `research.yaml`.
3. Prefer `research workspace repair --dry-run` first.
4. Review the ordered action plan and make the mutation risk explicit: the command may discard tracked-file changes and remove untracked files or directories inside the configured Core checkout.
5. If the dry run matches the intended narrow repair, run `research workspace repair` without `--dry-run`.
6. Verify the outcome against the current contract, including the success message or any config/path errors.
7. Keep the remediation story narrow and current:
   - it only targets the configured `core_path`,
   - it restores the workspace-recorded submodule revision,
   - it does not edit `research.yaml`,
   - it does not run `uv sync` or broader environment repair,
   - it does not repair unrelated workspace members.

## Notes
- Keep the guidance tied to `cli/src/research_cli/workspace.py`.
- Do not claim broader cleanup, upgrade, or auto-diagnosis behavior than the current CLI supports.
