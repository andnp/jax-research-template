# Upstream Contribution Reference

## What “Upstream” Means Here
In the hub-and-spoke model, upstream means the shared Core repository: shared `libs/`, the `research` CLI, templates, hooks, and shared documentation.

A change is upstream-ready when it is no longer just a single-project experiment and is ready for shared ownership.

## Lifecycle Before the PR
Use the harvesting lifecycle from `docs/adrs/006-harvesting-lifecycle.md`:
- **Prototype in projects** for new or unstable ideas.
- **Eject** when one project must diverge from a shared library.
- **Harvest** when the capability has proven reusable and should move into shared `libs/`.
- **Then contribute upstream** once the shared version is ready for review.

Current implemented lifecycle commands:
- `research eject <project> <library>`
- `research harvest <project> <library>`

These commands are registered in `cli/src/research_cli/main.py` and implemented in `cli/src/research_cli/lifecycle.py`.

## Current Reality: `research propose` Is Not Implemented
`docs/specs/research-cli.md` and `CONTRIBUTING.md` describe a future `research propose <lib_name>` workflow.

That automation does **not** exist in the current CLI surface:
- `cli/src/research_cli/main.py` registers `project`, `workspace`, `doctor`, `eject`, `harvest`, and `info`.
- There is no `propose` command implementation or registration in the current repo.

So the current upstream path is manual after the reusable change is in this repo.

## Practical Manual Path Today
1. Make sure the reusable code or shared workflow change is represented in this repository.
   - For a project-local component, that usually means harvesting it first in the relevant workspace and then carrying the resulting shared diff here.
   - For shared CLI/docs/templates/hooks work, edit this repo directly.
2. Keep the patch focused.
   - `CONTRIBUTING.md` says to keep PRs focused.
   - For harvested code, prefer one harvested library per PR.
3. Run the relevant repo checks.
   - `CONTRIBUTING.md` currently calls out `uv sync`, `uv run ty check .`, and `uv run ruff check .`.
   - This repository’s active agent instructions and CI now treat `pyright` as the authoritative type-check gate, so use the repo’s current verification expectations for the touched surface.
4. Commit with a conventional commit message.
5. Push the branch.
6. Open the PR manually, typically with GitHub CLI or the GitHub UI.
   - Example manual shape: push the branch, then run `gh pr create --base main --head <branch> --body "Fixes #<n>"`.
7. In the PR description, explain why the change is shared and, for harvested code, why it has crossed the Rule-of-Three threshold.

## What to Say Truthfully
When an agent explains the workflow, it should say:
- harvest/eject are real current commands,
- `research propose` is planned/spec'd but not implemented,
- opening the upstream PR is currently a manual Git/GitHub step.

## Anti-Patterns
Avoid these misleading claims:
- saying the CLI can fork, branch, push, and open the PR automatically,
- treating first-draft experimental code as upstream-ready,
- skipping shared-library hardening just because code already works in one project,
- bundling multiple unrelated reusable changes into one upstream PR.