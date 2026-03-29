# Repo-Shared Skills

This directory holds repo-shared workflow skills that agents can load on demand.

## Available Skills
- `bootstrap-shell` — create or preview a new research shell workspace with the current `research workspace init` flow.
- `contribute-to-upstream` — prepare a reusable shared change for this repo and follow the current manual upstream PR path.
- `create-project` — create or preview a new project inside an existing workspace with the current `research project create` flow.
- `decide-change-location` — choose whether work belongs in a downstream project, an ejected copy, shared `libs/`, or another shared core surface.
- `diagnose-workspace` — run or interpret the current read-only `research doctor` workflow.
- `repair-workspace` — preview or run the current mutating `research workspace repair` workflow for the configured Core checkout.

## Conventions
- Use stable product-oriented names.
- Keep each `SKILL.md` lean and procedural.
- Put repo-specific contracts and caveats in `references/`.
- Add `assets/` only when a reusable artifact is needed.
- Keep the content aligned with the current CLI, tests, and docs.

## Source Material
The current skills are grounded in:
- `docs/adrs/001-monorepo-structure.md`
- `docs/adrs/006-harvesting-lifecycle.md`
- `docs/specs/research-cli.md`
- `cli/src/research_cli/main.py`
- `cli/src/research_cli/lifecycle.py`
- `cli/src/research_cli/doctor.py`
- `cli/src/research_cli/workspace.py`
- `cli/src/research_cli/project.py`
- `tests/medium/test_cli_eject.py`
- `tests/medium/test_cli_harvest.py`
- `tests/medium/test_cli_doctor.py`
- `tests/medium/test_cli_workspace_repair.py`
- `tests/small/test_workspace_init.py`
- `tests/medium/test_project_create_render.py`
- `CONTRIBUTING.md`
