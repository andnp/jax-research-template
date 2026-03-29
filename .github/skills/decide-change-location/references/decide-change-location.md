# Change Location Reference

## Architecture Anchor
This repository is the shared hub/core side of the architecture described in `docs/adrs/001-monorepo-structure.md`.

That ADR defines three relevant locations:
- **Core / hub**: shared libraries, CLI, templates, and shared workflow docs.
- **Shell workspace**: a private user root that includes the Core plus one or more projects.
- **Projects**: high-churn experimental repos that consume the Core and may temporarily diverge from it.

In the downstream shell model, project-local experiments and ejected copies live in `projects/`, not in this repository.

## Decision Rules
Use these rules in order.

### 1. Start in a project when the work is new or unstable
ADR 006 says new algorithms, buffers, and layers begin in `projects/` during the prototype stage.

Choose a project-local location when:
- the code serves one experiment,
- the API is still moving quickly,
- duplication is acceptable for speed,
- the change has not yet proven reusable.

For this repo, that usually means **do not edit `libs/` yet**. The right implementation home is a downstream project workspace.

### 2. Eject when a shared library must diverge for one project
ADR 001 and ADR 006 both define eject as the path for project-specific hacking on shared code.

Choose eject when:
- a project needs to modify a shared library in a way that could break other projects,
- the project needs a forked API or different semantics,
- the shared library is close, but not stable enough for the required experiment.

Current CLI support:
- `research eject <project> <library>` is implemented in `cli/src/research_cli/lifecycle.py`.
- The current implementation copies the package into `projects/<project>/components/<import_package>` and rewrites project imports to `components.<import_package>`.
- Real behavior is covered by `tests/medium/test_cli_eject.py`.

### 3. Harvest only after reuse is proven
ADR 006 defines the harvest trigger as the Rule of Three: move code into shared `libs/` when it is needed by a third project.

Choose harvest when:
- the same capability is now shared across projects,
- the API has stabilized enough for shared ownership,
- you are ready to meet the Core’s stricter typing, docs, and testing standards.

Current CLI support:
- `research harvest <project> <library>` is implemented in `cli/src/research_cli/lifecycle.py`.
- The current implementation moves `projects/<project>/components/<import_package>` into `libs/<library>/src/<import_package>`, creates a `pyproject.toml` when needed, rewrites imports back to the shared package path, and updates the root workspace registration.
- Real behavior is covered by `tests/medium/test_cli_harvest.py`.

### 4. Change this repo directly only for shared platform concerns
Direct edits in this repo are appropriate when the change belongs to the shared platform itself, for example:
- a reusable library under `libs/`,
- the `research` CLI,
- templates under `templates/`,
- tracked hooks, agent customizations, or shared docs/ADRs/specs.

Use this repo when the change should benefit many downstream workspaces, not just one experiment.

## Practical Mapping
- **One project, one experiment, unclear reuse** → downstream project.
- **One project needs a custom fork of a shared lib** → eject in downstream project.
- **Three projects need the same capability** → harvest into `libs/`.
- **Shared workflow or platform contract change** → edit this repo.

## Anti-Patterns
Avoid these location mistakes:
- putting first-draft experimental code straight into `libs/`,
- editing the shared library when the real need is a project-only fork,
- harvesting before the API has stabilized,
- documenting `projects/` changes in this repo as if they already lived in the shared core.