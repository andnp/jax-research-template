# Repo-Shared Skills

This directory holds repo-shared workflow skills that agents can load on demand.

## Available Skills
- `bootstrap-shell` — create or preview a new research shell workspace with the current `research workspace init` flow.
- `create-project` — create or preview a new project inside an existing workspace with the current `research project create` flow.

## Conventions
- Use stable product-oriented names.
- Keep each `SKILL.md` lean and procedural.
- Put repo-specific contracts and caveats in `references/`.
- Add `assets/` only when a reusable artifact is needed.
- Keep the content aligned with the current CLI, tests, and docs.

## Source Material
The current skills are grounded in:
- `docs/specs/research-cli.md`
- `cli/src/research_cli/workspace.py`
- `cli/src/research_cli/project.py`
- `tests/small/test_workspace_init.py`
- `tests/medium/test_project_create_render.py`
- `CONTRIBUTING.md`
