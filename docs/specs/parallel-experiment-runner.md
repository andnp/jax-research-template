# Technical Specification: Parallel local experiment runs

- **Status:** Accepted for implementation (revised after external review, 2026-09-29)
- **Owner:** Andy
- **Created:** 2026-09-29
- **Touches:** `cli/` (`research experiment run`), `libs/research-runner`, `libs/experiment-definition`
- **Replaces:** `projects/actor-free-qrc/scripts/run_parallel.py`, and the affinity-based `RESEARCH_MAX_CPUS`

## 1. Problem

`research experiment run` is one process that executes its batches one after another. Using a many-core machine
today means hand-slicing a sweep into several processes, and that has failed in practice:

- **No core control that holds.** `RESEARCH_MAX_CPUS` narrows the process's CPU affinity while JAX starts up
  (`cli/src/research_cli/cpu_cap.py`), to the *lowest-numbered* N CPUs, then restores full affinity. It sizes XLA's
  thread pool but assigns no cores. The only thing that stops parallel processes oversubscribing the machine is a
  project-local launcher that pins each slice with `taskset`.
- **Concurrent reclaims crash each other.** `reclaim_stale_executions` selects stale executions inside a transaction
  but updates them after it closes, so two runners can both fail the same execution, and the second crashes with
  `Illegal execution transition: FAILED -> FAILED` (`_execution_repository.py`).
- **Nothing knows about memory.** Two sweeps in one day were stopped by the host for low memory, with no way to
  size the number of concurrent processes to fit.
- **A kill loses whole slices.** A static group is one vmapped batch by default, and a tracked command is stopped at
  two hours, so a long group is lost whole unless `--max-runs` is set by hand.
- **Cancelling leaks processes.** Killing the launcher does not reliably stop its children.
- **A hard-killed batch blocks its runs.** A worker killed by SIGKILL or the OOM killer never reaches the handler
  that marks its execution FAILED, so the execution stays `RUNNING` and its runs are not replanned until a
  stale-reclaim threshold passes.

## 2. Goal

One command:

```bash
uv run research experiment run SPEC --spec NAME --workers W
```

`W` is the sweep's **thread budget**: the sum of the XLA CPU threads (`NPROC`) of all running workers never exceeds
`W`. The runner decides how many processes run at once and how many threads each gets, and admits a process only
when the available memory fits it. Example: with `W = 8` and two pending groups, it runs 2 processes of 4 threads.

`W` is a soft budget, not a core reservation. `NPROC` caps a worker's thread pool but does not choose which cores its
threads run on; the OS scheduler places them. To leave room for other jobs on a shared machine, pick `W` below the
free physical cores.

Non-goals: Slurm and multi-node runs (`research-cluster`), GPU scheduling, checkpointing inside a batch, choosing
how many seeds an experiment should have, and more than one supervisor per experiment at a time.

## 3. Verified facts the design rests on

- **`NPROC` caps XLA's CPU pool without affinity** (JAX 0.11.1, measured 2026-09-29: a jitted 512×512 matmul scan
  allowed 8 CPUs used 6.45 cores unset, 1.00 with `NPROC=1` and 3.83 with `NPROC=4`). `XLA_FLAGS` settings,
  `intra_op_parallelism_threads` and `OMP`/`MKL`/`OPENBLAS_NUM_THREADS` do not cap it.
- **Planning already supports many workers.** `plan_next_execution_batch` claims the next unsatisfied batch inside
  `BEGIN IMMEDIATE`, so concurrent runners on one `experiments.sqlite` never claim the same runs.
- **A group is more than its static values.** `list_unsatisfied_run_batches` groups pending runs by algorithm
  version, environment version, ablation, static config and vmap zone. `--only` matches hyperparameters, seed and
  ablation, so it cannot pin a worker to exactly one group.
- **A worker's batch cap is fixed for its life.** `run_experiment` passes one `max_runs_per_batch` to every
  planning call, and every distinct vmap width is a separate XLA compile. The persistent compile cache does not make
  different shapes equivalent.
- **Loading a spec imports JAX.** Project spec modules import code that imports `jax`. Importing JAX does not
  initialise a backend; only the first array operation does.

## 4. Design

### 4.1 Supervisor and workers

`--workers W` turns `run` into a **supervisor**. The supervisor loads the spec and syncs the experiment, but never
initialises a JAX backend (no array operation, no device query). It:

1. syncs the experiment, then reclaims stale executions **once** (`--reclaim-stale-after`), before any worker starts;
2. lists the pending groups, each identified by a stable **group key** (4.5);
3. starts one **worker** per group, as a subprocess running today's single-process loop restricted to that group
   (`--group KEY`, `--max-runs b`), with the environment `NPROC=c` and `JAX_PLATFORMS=cpu`;
4. keeps the sum of running workers' `c` at or below `W` (4.2), and admits a worker only when its projected memory
   fits (4.3);
5. starts each worker in its own process group with the parent-death signal set to SIGTERM (`PR_SET_PDEATHSIG`), so
   the kernel stops the workers if the supervisor dies for any reason, including SIGKILL;
6. on SIGINT or SIGTERM, terminates every worker's process group and waits for it;
7. when a worker exits abnormally (nonzero, or killed by a signal), marks FAILED every `RUNNING` execution of that
   worker's group whose `start_time` is at or after the worker's start. Only one worker per group runs at a time, so
   those executions are that worker's. Its runs are then retried by the next launch without a reclaim threshold;
8. exits nonzero if any worker failed.

A worker runs its group's batches in turn, so the in-process jit cache is reused across them. Workers never reclaim.

Without `--workers`, `run` behaves as it does today.

### 4.2 Threads per worker (c) and seeds per batch (b)

**Threads.** A worker's `c` is set when it starts, from the budget not yet in use:
`c = clamp(ceil(W_free / G_waiting), 1, min(C_MAX, W_free))`, where `W_free` is `W` minus the running workers' `c`
and `G_waiting` is the number of groups not yet started. A worker starts only when `W_free ≥ 1`. This spreads the
whole budget: `W = 8` over three groups gives 3, 3 and 2. With many groups every worker gets one thread, which
profiling found gives the best sweep throughput. `C_MAX` defaults to 8, a tuning value measured on this machine (one
CCD's physical cores), not a topology guarantee. Running workers keep their `c`, because the pool size is fixed when
JAX starts. `--cores-per-worker C` sets `c` for every worker instead.

**Seeds per batch.** `b` is fixed for the sweep: `--max-runs` if given, else the spec's `max_runs_per_batch`, else the
whole group (today's default). Sizing `b` automatically is deferred (8.1).

**Order.** Groups start in descending order of pending runs. This is a heuristic; run count is not cost.

**Telemetry.** Every execution records, in its metadata, `runtime.nproc`, `runtime.batch_runs`,
`runtime.wall_seconds` and `runtime.peak_rss_bytes` (the worker process's `ru_maxrss`). This is the data a later
automatic sizing policy needs.

### 4.3 Memory admission

Before starting each worker the supervisor re-reads `MemAvailable` from `/proc/meminfo`, so memory used by other
jobs since the sweep started is seen. A worker is admitted if

`MemAvailable − reserve ≥ projection(new) + Σ_running max(0, projection(w) − current RSS(w))`,

the second term being the growth still to come from running workers. The reserve is 15% of `MemTotal`;
`--memory-reserve GB` overrides it. A worker's **projection** is the largest peak RSS any completed worker of this
sweep has reported (from telemetry), else `--worker-memory GB` (default 4). This is a heuristic, not a bound.

If a worker does not fit and others are running, the supervisor waits for one to finish. If none is running and it
still does not fit, the supervisor exits with an error naming the projection and the available memory, rather than
waiting forever.

### 4.4 Compile cache

Workers get JAX's persistent compilation cache: `JAX_COMPILATION_CACHE_DIR` set to the absolute path of
`~/.cache/research-jax` (expanded by the supervisor), and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0`.
Relaunches and groups with identical programs then skip recompiling. It does not help a new batch shape. No size
limit or toggle in v1; add them if the cache grows or causes trouble.

### 4.5 Changes to planning and the database

- **Group key.** `RunBatch` carries `group_key`: a short stable hash of its grouping tuple (algorithm version,
  environment version, ablation, static config JSON, vmap zone JSON). `list_unsatisfied_run_batches`,
  `plan_next_execution_batch` and `run_experiment` take `group: str | None`, restricting planning to that group. The
  CLI's `run` takes `--group KEY` (hidden; the supervisor passes it).
- **Atomic reclaim.** `reclaim_stale_executions` selects and updates inside one `BEGIN IMMEDIATE` transaction, with
  the update conditioned on `status = 'RUNNING'`, so a concurrent reclaim is a no-op rather than a crash.
- **Busy timeout.** `DatabaseManager.connect` sets `PRAGMA busy_timeout` to 30 s, matching `metrics.sqlite`, so many
  workers finishing at once wait for the write lock instead of failing after Python's 5 s default.
- **No affinity cap.** `RESEARCH_MAX_CPUS` and `cpu_cap.py` are removed. A single-process run is capped with `NPROC`,
  and the docs say so.

## 5. CLI

```
research experiment run SPEC --spec NAME
    [--workers W]                  # thread budget; enables the supervisor
    [--cores-per-worker C]         # fixed c for every worker
    [--max-runs B]                 # seeds per batch (existing flag)
    [--worker-memory GB]           # projection before any worker has reported (default 4)
    [--memory-reserve GB]          # memory kept free (default 15% of MemTotal)
    [--reclaim-stale-after S]      # done once by the supervisor when --workers is set
    [--only key=value ...]         # restricts which groups the supervisor schedules
```

## 6. Testing

- **small:**
  - thread allocation: whole budget spread with remainders, `W_free` respected with workers already running,
    `C_MAX` and `--cores-per-worker`, invalid `W`/`C`;
  - memory admission: fits, waits while others run, errors when nothing runs and nothing fits, running workers'
    remaining growth counted;
  - group keys: stable across calls, distinct for groups that differ only in component version or ablation,
    `group` filter plans only that group;
  - reclaim with two connections to one database;
  - telemetry keys present in a completed execution's metadata.
- **medium:** the supervisor end to end, on a toy `BatchTrainer` spec with three groups:
  - every run completes, each worker saw its `NPROC` (the trainer records `os.environ`), and the sum of running
    workers' `NPROC` never exceeds `W`;
  - SIGTERM to the supervisor leaves no child process;
  - SIGKILL to the supervisor leaves no child process (parent-death signal);
  - a worker that SIGKILLs itself mid-batch: the supervisor exits nonzero, that execution is FAILED, and a relaunch
    without `--reclaim-stale-after` completes its runs.
- **performance (targeted, not a PR gate):** `NPROC` caps a real critic update under concurrent load, measured as
  CPU time over wall time per worker.

## 7. Migration

- Delete `projects/actor-free-qrc/scripts/run_parallel.py` and its tests, and point its users at `--workers`.
- Update `docs/workflow.md`, and the monorepo's `run-experiments` skill: one tracked command,
  `research experiment run … --workers W`.
- Record the decision as an ADR (local parallel execution).

## 8. Deferred

1. **Automatic batch size.** Measuring one batch and resizing the next doesn't work inside one worker: `b` is fixed
   for the worker's life, every new width recompiles, and the first batch's wall time includes compilation. v2 can
   pick `b` at launch from the telemetry recorded by earlier executions of the same group (per-seed time and memory,
   with compile time separated), which needs no protocol between supervisor and worker.
2. **Growing running workers.** When groups finish, the last workers keep their small `c`. Restarting a worker
   between batches with a larger `NPROC` costs a process start and a (cached) compile. Revisit if telemetry shows
   the tail underusing the budget.
3. **Liveness beyond one supervisor.** Fail-on-death (4.1) covers workers the supervisor started. A killed
   supervisor's in-flight executions still need `--reclaim-stale-after` or `invalidate` on the next launch. A
   heartbeat or lease column would remove that; it needs a schema change.
