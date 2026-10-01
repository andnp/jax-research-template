# PRD: `libs/research-instrument`

## 1. Overview
`research-instrument` persists experiment metrics to `metrics.sqlite`. Trainers may run
their loop under JIT and return metric arrays; the runner persists the returned metrics
afterwards through this library.
It does no in-training collection and emits no callbacks.

### Goals
- **One writer:** `seed_batch` and `scalar` are the only paths that insert metric rows.
- **One schema owner:** `sqlite_backend` holds the DDL, the insert column lists, schema
  migration, and connection handling.
- **Concurrent writers:** Slurm array tasks may write the same file; contention is handled
  by a busy timeout on a rollback journal, which assumes the shared filesystem's file
  locking works.

## 2. API

### 2.1 `seed_batch.write_seed_batch_curve`
Writes a seed-major curve array of shape `(n_seeds, n_steps)`, one row per run, each under
its own `run_id` and `seed_id`. Curves arrive already thinned to one value in every
`every` steps, and each value is stored at the step it was taken from. Non-finite values are stored (NaN as NULL) so diverged runs are recorded.

### 2.2 `scalar.write_seed_batch_scalars` and `scalar.write_execution_scalars`
Write one scalar per run (scope `run`) or one scalar shared by an execution (scope
`execution`) into `scalar_metrics`.

### 2.3 `sqlite_backend.open_metrics_connection`
Opens `metrics.sqlite` with a busy timeout, ensures the rollback journal, and creates or
upgrades the schema in place.

## 3. Storage
- `metrics`: curve rows keyed by experiment, run, execution, seed, metric name and step.
- `scalar_metrics`: scalar rows with an explicit `run` or `execution` scope.
- The default rollback journal is kept because WAL is unsafe over NFS/Lustre; a database
  left in WAL by an earlier version is converted back on open.
