# ADR 008: Relational Experiment Schema & Researcher-in-the-Loop Versioning

## Status
Proposed

## Context
We need a robust way to track the relationship between code, hyperparameters, and results. We must handle "scientific invalidation" (bug fixes) without deleting historical data, while avoiding unnecessary compute waste from non-functional code changes (refactors).

## Decision
We will use a relational SQLite schema that distinguishes between the **Component** (the code), the **Version** (the scientific state), and the **Execution** (the physical run).

### 1. The Schema
- **`Components`**: `id`, `name` (e.g., "DQN"), `type` (ALGO, ENV, WRAPPER).
- **`ComponentVersions`**: `id`, `component_id`, `version_number` (Integer, auto-incrementing), `created_at`, `code_snapshot_hash`, `notes`.
- **`HyperparamConfigs`**: `id`, `hash`, `json_blob`.
- **`Experiments`**: `id`, `name`, `description`.
- **`Runs` (The Logical Unit)**: 
    - `id`, `experiment_id`, `algo_version_id`, `env_version_id`, `hyper_id`, `seed`.
    - *Note:* Represents the "Intent" to have a result for this specific combination.
- **`Executions` (The Physical Event)**:
    - `id`, `status` (PENDING, RUNNING, COMPLETED, FAILED, INVALID).
    - `hostname`, `start_time`, `end_time`, `git_commit`, `git_diff_blob`.
    - `jax_config_json` (versions, flags, platform).
- **`ExecutionRuns` (Many-to-One)**:
    - `execution_id`, `run_id`.
    - *Note:* In JAX, one physical execution (a single process) often satisfies many logical runs via `vmap`.

### 2. Automatic Versioning & Researcher Intent
Instead of forced hashing or manual labeling, the CLI manages versions automatically:
1. When `research exp run` is called, it hashes the current source of the requested components.
2. If `current_hash != last_version.code_snapshot_hash`, the CLI prompts:
   > "Source code for DQN has changed. Create new version (v3)? [y/N]"
3. **If Yes:** The system automatically increments the `version_number`. All subsequent analysis and runs default to this new version as `latest`.
4. **If No:** The system continues using the current version, acknowledging that the change was non-functional (refactor/documentation).

### 3. The "Latest" Pointer
Analysis scripts and the CLI runner will support a virtual `latest` version identifier. This resolves to the highest `version_number` for a component that hasn't been marked as `INVALID`. This allows researchers to write stable plotting scripts that automatically update as bugs are fixed and new versions are generated.


### 3. Retroactive Invalidation
The CLI will provide tools to mark entire slices of the database as `INVALID` (e.g., `research exp invalidate --algo DQN --version v1 --hypers "lr > 0.1"`). Invalidated runs are treated as "missing" by the runner and will be re-scheduled.

## Consequences
- **Pros:** Full scientific traceability; no accidental compute waste; researcher maintains control over the "meaning" of code changes.
- **Cons:** Slightly more interactive overhead during the start of a run.
