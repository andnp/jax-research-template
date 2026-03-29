---
name: create-project
description: 'Create a research project inside an existing workspace. Use when running or previewing `research project create`, checking workspace-root requirements, or deciding whether to add the optional `--github-repo` remote creation step.'
---

# Create Project

## When to Use
- Create a new project under `projects/` in an existing workspace.
- Preview project creation before writing files.
- Explain what the current project template flow creates.
- Decide whether to attach a private GitHub remote during project creation.

## Procedure
1. Read the current contract in [project creation reference](./references/project-create.md).
2. Confirm the command will run from a workspace root that already contains `projects/`.
3. Prefer a preview first with `research project create <name> --dry-run`.
4. Add `--github-repo <owner/name>` only when the user explicitly wants `gh repo create` to run after rendering.
5. For the mutating path, run `research project create <name>` with any approved optional flags.
6. Verify the rendered project shape and Git initialization against the reference.
7. Call out the current limits truthfully:
   - the command errors when `projects/` is missing,
   - the command errors when `projects/<name>` already exists,
   - the command currently passes only `project_name` to Copier and relies on template defaults for the other answers.

## Notes
- Keep the guidance tied to the current CLI behavior in `cli/src/research_cli/project.py`.
- Do not claim extra project scaffolding options beyond the existing CLI flags.
