# ADR 005: Workspace Dependency Alignment

## Status
Proposed

## Context
A `uv` workspace is most effective when it maintains a single `uv.lock` file. Multiple conflicting versions of core libraries (like `jax`) within the same workspace lead to non-deterministic behavior and broken JIT caches.

## Decision
We enforce **Global Dependency Alignment** for all active members of the workspace.

### 1. The Core as the Anchor
- The versions of `jax`, `jaxlib`, `flax`, and `optax` defined in the Core's `pyproject.toml` are the "Source of Truth."
- All active projects must be compatible with these versions.

### 2. Tilde Versioning
- Use `~=X.Y` (e.g., `jax~=0.4`) to allow automatic minor and patch upgrades.
- Major version upgrades must be performed centrally in the Core.

### 3. Divergence Policy
- If a project *requires* a different version of a core dependency (e.g., studying a bug in an old JAX version), it must be **archived** (moved out of the `projects/` workspace root) or treated as a standalone repository separate from the monorepo shell.

## Consequences
-   **Pros:** Guaranteed compatibility between shared libraries and projects; faster `uv sync` times; reliable reproducibility.
-   **Cons:** Limits the ability to run "legacy" projects alongside active ones without archiving.
