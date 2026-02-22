# Research Backlog

This backlog contains actionable tickets derived from the [ROADMAP.md](ROADMAP.md).

## Phase 1: Core Orchestration (The Brain)

### [CORE-001] Initialize `research-cli` Bootstrap
- **Description:** Implement the base `typer` application and the `workspace init` command.
- **Acceptance Criteria:**
    - `research workspace init` creates a new monorepo shell.
    - Adds `research-core` as a submodule.
    - Configures root `pyproject.toml` with `uv` workspace members.
    - Verified with a dry-run flag.

### [CORE-002] Implement Relational Experiment Schema
- **Description:** Create the SQLite initialization logic based on ADR 008.
- **Acceptance Criteria:**
    - Tables created: `Components`, `ComponentVersions`, `HyperparamConfigs`, `Experiments`, `Runs`, `Executions`, `ExecutionRuns`.
    - Support for automatic version incrementing.
    - Support for "latest" version pointer logic.

### [CORE-003] `experiment-definition` Fluent API
- **Description:** Implement the builder pattern for defining experiments.
- **Acceptance Criteria:**
    - Support `add_parameter` with `is_static` flag.
    - Support `with exp.for_component(...)` and `exp.when(...)` triggers.
    - `exp.sync()` successfully populates the SQLite DB.

### [CORE-004] `research-store` Local Mode
- **Description:** Implement the unified artifact storage API with local filesystem backend.
- **Acceptance Criteria:**
    - `store.put()` saves JAX pytrees (via Orbax) or pickles.
    - Implements atomic writes (temp-and-rename).
    - Generates `research://` URIs correctly.

## Phase 2: Runtime & Data (The Heart)

### [RUN-001] `research-instrument` Basic Collector
- **Description:** Implement the JAX-native `write()` and `eval()` API using `jax.debug.callback`.
- **Acceptance Criteria:**
    - Works inside a JIT-compiled `lax.scan` loop.
    - Correctly captures the `vmap` index.
    - No-op if metric is not in the whitelist.

### [RUN-002] Double-Buffered Storage Spooler
- **Description:** Implement the background thread and double-buffer for the instrumentation library.
- **Acceptance Criteria:**
    - Buffer A and B swap when full.
    - Background thread flushes to SQLite without blocking the JAX thread.
    - Supports a `flush()` call to ensure data integrity before job exit.

### [RUN-003] `jax-utils` Typed Wrappers
- **Description:** Implement `typed_jit` and `typed_vmap`.
- **Acceptance Criteria:**
    - Wrappers enforce `jaxtyping` annotations.
    - Provide clear error messages for shape mismatches.
    - Zero runtime overhead verified by benchmarks.

### [RUN-004] `jax-replay` Circular Buffer
- **Description:** Move the current replay buffer logic into a dedicated library and add trajectory support.
- **Acceptance Criteria:**
    - Supports both step-based and trajectory-based sampling.
    - Fully JIT-able and VMAP-ready.
    - Includes unit tests in `tests/small/`.
