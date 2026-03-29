---
name: contribute-to-upstream
description: 'Contribute a reusable change back to the shared research core. Use when a project-local improvement is ready for harvest, when preparing a focused upstream PR against this repo, or when you need the current manual path because `research propose` is not implemented.'
---

# Contribute to Upstream

## When to Use
- Prepare a reusable project-local change for the shared core.
- Turn a harvested library or shared workflow fix into a focused PR for this repo.
- Explain the current upstream contribution path truthfully.
- Decide what is still manual because `research propose` is not available today.

## Procedure
1. Read the current contract in [upstream contribution reference](./references/contribute-to-upstream.md).
2. Confirm the candidate change is genuinely shared work, not just a project-local prototype.
3. If the change still lives in a downstream project or an ejected copy, harvest or port the reusable diff into the shared core first.
4. Keep the upstream patch focused: one reusable library or one shared workflow/docs change per PR when practical.
5. Update the shared implementation, tests, and docs in this repo to match the current behavior.
6. Run the relevant verification for the changed shared surface before opening a PR.
7. Use the current manual Git and GitHub flow to commit, push, and open the PR.
8. When describing the workflow, say plainly that `research propose` is specified in docs but not implemented in the current CLI.

## Notes
- Keep the guidance tied to `CONTRIBUTING.md`, `docs/specs/research-cli.md`, `cli/src/research_cli/main.py`, and `cli/src/research_cli/lifecycle.py`.
- Do not claim automatic fork/branch/PR automation from the CLI.
- Do not skip the harvest/eject lifecycle when the code still belongs in a downstream project first.