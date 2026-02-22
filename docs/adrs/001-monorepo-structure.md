# ADR 001: Monorepo Hub-and-Spoke Architecture

## Status
Proposed

## Context
We need a structure that allows for:
1.  **Upstream Updates:** Users should receive improvements to shared libraries (`libs/`) without manual copy-pasting.
2.  **Access Control:** Collaborative research requires per-project permissions (not everyone should see every project).
3.  **Reproducibility:** A specific version of a project must be tied to a specific version of the shared libraries it used.

## Decision
We will adopt a **Hub-and-Spoke** architecture using Git Submodules and `uv` Workspaces.

### 1. The Components
-   **The Core (`research-core`):** A public template repository containing shared libraries, the `research` CLI, and project templates.
-   **The Shell (User's Root):** A private repository created by the user from the template. It contains:
    -   `core/`: A Git submodule pointing to `research-core`.
    -   `projects/`: A directory where each project is an independent Git repository (often added as submodules to the Shell for remote syncing).
    -   `pyproject.toml`: A `uv` workspace root that includes `core/libs/*` and `projects/*`.

### 2. The Eject/Harvest Workflow
-   **Eject:** If a user needs to modify a core library for a specific project, they "eject" it. The CLI copies the library into the project's local source. The project then uses the local version.
-   **Harvest:** If a local project-specific improvement is generalizable, the CLI provides tools to "harvest" it back into `core/libs` and submit a PR to the upstream `research-core`.

## Consequences
-   **Pros:** Clear separation of concerns; per-project permissions; easy upgrades via `git submodule update`.
-   **Cons:** Git submodules have a learning curve; `uv.lock` management at the root can be complex if projects have conflicting requirements.
