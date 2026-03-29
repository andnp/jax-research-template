---
name: decide-change-location
description: 'Decide where a change belongs in the research core ecosystem. Use when choosing between a project-local prototype, an ejected project copy, a harvested shared library in `libs/`, or a core workflow/docs update grounded in the hub-and-spoke architecture.'
---

# Decide Change Location

## When to Use
- Decide whether a change belongs in a project, an ejected copy, or shared `libs/`.
- Sort out whether a request should change the core repo, a downstream shell workspace, or both.
- Apply the repo’s eject/harvest lifecycle before editing code.
- Explain why a change should stay local versus move upstream.

## Procedure
1. Read the current contract in [change-location reference](./references/decide-change-location.md).
2. Classify the requested change before editing anything:
   - new idea or high-churn experiment,
   - project-specific divergence from a shared library,
   - third-use shared capability,
   - shared workflow, template, CLI, or documentation improvement.
3. If the work is exploratory or only serves one project, keep it project-local in a shell workspace instead of changing shared `libs/`.
4. If the work needs to break or heavily diverge from a shared library, use the eject path first so the project can iterate locally without destabilizing the core.
5. If the same component has proven reusable across projects, move it into `libs/` through the harvest path and then raise the shared-library quality bar.
6. If the request changes the shared platform itself, keep it in this repo only when the benefit is genuinely cross-project: shared libraries, CLI behavior, templates, hooks, or repo-wide docs/skills.
7. State the chosen location and why it matches the current hub-and-spoke lifecycle before implementing follow-up work.

## Notes
- Keep the guidance tied to `docs/adrs/001-monorepo-structure.md` and `docs/adrs/006-harvesting-lifecycle.md`.
- Do not skip straight to shared `libs/` for single-project experimentation.
- Do not claim this repo is the place for downstream project-local prototypes or ejected copies.