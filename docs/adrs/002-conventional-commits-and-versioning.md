# ADR 002: Conventional Commits and Versioning

## Status
Proposed

## Context
As a monorepo Core that will be used as a template and submodule by many users, we need a consistent way to:
1.  **Communicate Changes:** Users need to know if an update to `libs/` contains a bug fix, a new feature, or a breaking change.
2.  **Automate Changelogs:** Researchers shouldn't spend time manually writing "What's New" docs.
3.  **Manage Releases:** Individual libraries within the Core may need independent versioning.

## Decision
We will adopt the **Conventional Commits** specification and **Semantic Versioning (SemVer)**.

### 1. Commit Format
Every commit to the Core must follow the pattern: `<type>(<scope>): <description>`
-   `feat`: A new feature (corresponds to SemVer `minor`).
-   `fix`: A bug fix (corresponds to SemVer `patch`).
-   `refactor`: Code change that neither fixes a bug nor adds a feature.
-   `docs`: Documentation changes.
-   `breaking`: Commits that introduce breaking changes (corresponds to SemVer `major`) must include `BREAKING CHANGE:` in the footer.

### 2. Versioning Strategy
-   The **Core** as a whole will have a global version (e.g., `v0.1.0`).
-   Individual **Libraries** in `libs/*` will maintain their own versions in their respective `pyproject.toml` files.
-   We will use `~=X.Y` requirements in our templates to allow for non-breaking updates to propagate automatically.

## Consequences
-   **Pros:** Highly predictable upgrade path for users; enables automated tools like `python-semantic-release`; clear historical record.
-   **Cons:** Requires discipline from contributors; slightly higher overhead for small changes.
