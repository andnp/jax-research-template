# ADR 007: Declarative Experiment Management

## Status
Proposed

## Context
Large-scale RL research (sweeps over seeds and hypers) is prone to failures: Slurm jobs time out, nodes crash, or specific seeds fail. We need a system that:
1.  **Resumes for Free:** Never rerun work that is already in the results database.
2.  **Tracks Intent:** Separate the *definition* of an experiment from its *execution*.
3.  **Scales with JAX:** Intelligently batch work to use `jax.vmap` where possible.

## Decision
We will use a **Database-Centric Orchestration** model.

### 1. SQLite as the Source of Truth
Instead of log files or JSON configs, the state of an experiment is stored in a SQLite database.
- Each row in the `runs` table represents a unique combination of `(experiment_id, algorithm, hyperparams, seed)`.
- Each row tracks its `status` (PENDING, RUNNING, COMPLETED, FAILED) and its `result_path`.

### 2. Definition via Python
Experiment configurations are defined in pure Python scripts using a builder pattern (e.g., `sweeper.sweep("lr", [1e-3, 3e-4])`). This allows for complex logic (e.g., "only sweep this hyperparam for this algorithm") while outputting a flat schema to the DB.

### 3. VMAP-Zone Batching
The runner will query the DB for PENDING work and group it into batches that share the same **static parameters**. These batches are then executed as a single JIT-compiled function that `vmap`s over the **dynamic parameters** (seeds and simple scalar hypers).

### 4. Analysis Hooks
Analysis scripts will interface directly with the SQLite database, ensuring that plots are always based on the most up-to-date and complete set of results.

## Consequences
-   **Pros:** Robust against failures; transparent progress tracking; extremely high hardware utilization via intelligent vmapping.
-   **Cons:** Higher initial setup complexity for the CLI; requirement for a shared filesystem if running across multiple Slurm nodes (to share the SQLite file).
