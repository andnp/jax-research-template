# Technical Specification: Parallel local experiment runs

- **Status:** Draft, for review
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

## 2. Goal

One command:

```bash
uv run research experiment run SPEC --spec NAME --workers W
```

`W` is the number of CPU cores the sweep may use. The runner decides how many seeds to vmap per batch, how many cores
each process gets, and how many processes run at once, so that no more than `W` cores and the available memory are
in use. Example: with `W = 8` and two pending groups of 30 seeds, it runs 2 processes of 4 cores each.

Non-goals: Slurm and multi-node runs (`research-cluster`), GPU scheduling, checkpointing inside a batch, and choosing
how many seeds an experiment should have.

## 3. Verified facts the design rests on

- **`NPROC` caps XLA's CPU pool without affinity** (JAX 0.11.1, measured 2026-09-29: a jitted 512×512 matmul scan
  allowed 8 CPUs used 6.45 cores unset, 1.00 with `NPROC=1` and 3.83 with `NPROC=4`). `XLA_FLAGS` settings,
  `intra_op_parallelism_threads` and `OMP`/`MKL`/`OPENBLAS_NUM_THREADS` do not cap it. The OS scheduler then places
  the threads, and busy cores (another user's pinned jobs) are naturally avoided.
- **Planning already supports many workers.** `plan_next_execution_batch` claims the next unsatisfied batch inside
  `BEGIN IMMEDIATE`, so concurrent runners on one `experiments.sqlite` never claim the same runs.
  `list_unsatisfied_run_batches` groups pending runs by static vmap zone, and `only` restricts a runner to one zone.
- **Batch size is already a knob.** `max_runs_per_batch` (`--max-runs`) splits a zone into several vmapped batches
  run in turn.

## 4. Design

### 4.1 Supervisor and workers

`--workers W` turns `run` into a **supervisor**. The supervisor never imports JAX. It:

1. syncs the experiment, then reclaims stale executions **once** (`--reclaim-stale-after`), before any worker starts;
2. lists the pending static groups (`list_unsatisfied_run_batches`);
3. starts one **worker** per group, as a subprocess running today's single-process loop, with
   `--only <group's static values> --max-runs b` and the environment `NPROC=c`, `JAX_PLATFORMS=cpu`;
4. keeps the total cores of running workers at or below `W`, and admits a worker only when its projected memory fits
   (4.3);
5. on SIGINT or SIGTERM, terminates every worker's process group and waits for it, so cancelling the one tracked
   command stops everything;
6. exits nonzero if any worker failed. As today, failed runs are retried by the next launch.

A worker runs its group's batches in turn, so the in-process jit cache is reused across them. Workers never reclaim.

Without `--workers`, `run` behaves as it does today.

### 4.2 Choosing cores per worker (c) and seeds per batch (b)

**Cores.** Spread the budget over the groups that can run now: `c = clamp(floor(W / G), 1, C_MAX)`, where `G` is the
number of pending groups and `C_MAX` is one CCD's physical cores (8 here; profiling found one CCD beats all 32
threads). With many groups every worker gets one core, which profiling found gives the best throughput. With few
groups each gets several. As groups finish, newly started workers get more cores; running workers keep theirs, because
the pool size is fixed when JAX starts.

**Seeds per batch.** A group's first batch is small (`b0 = min(pending, 8)`) and **measured**: wall time and peak RSS
(`VmHWM` from `/proc/<pid>/status`). Later batches of that group are sized from those measurements. The largest `b`
is chosen such that the projected batch wall time stays under `T_BATCH` (default 20 minutes, well inside the two-hour
cap) and the projected peak memory fits the worker's share (4.3). A kill then loses at most one batch of at most
`T_BATCH`.

**Order.** Groups start in descending order of pending runs, a stand-in for cost until it has been measured, so the
longest groups don't start last.

Every execution records `NPROC`, `b`, wall time and peak RSS in its metadata. That makes the policy auditable, and
tunable from data later.

### 4.3 Memory admission

The budget is `MemAvailable` when the supervisor starts, minus a reserve (default 15%, overridable with
`--memory-budget`). A worker is admitted only if the projected peak RSS of all running workers plus its own fits the
budget. Before a group has been measured, its projection is the largest per-seed peak seen so far times `b0`, or a
conservative default (1 GB per seed) if nothing has been measured. If even a single `b0` worker doesn't fit, the
supervisor waits for a running worker to finish rather than overcommit.

### 4.4 Compile cache

Workers get JAX's persistent compilation cache: `JAX_COMPILATION_CACHE_DIR=~/.cache/research-jax`,
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0`, and `JAX_COMPILATION_CACHE_MAX_SIZE` of 20 GB with LRU eviction.
Groups with identical programs, and relaunches, then skip recompiling. It can be turned off with `--no-compile-cache`.

### 4.5 Fixes that ship with this

- `reclaim_stale_executions` selects and updates inside one `BEGIN IMMEDIATE` transaction, with the update conditioned
  on `status = 'RUNNING'`, so a concurrent reclaim is a no-op rather than a crash.
- `RESEARCH_MAX_CPUS` and `cpu_cap.py` are removed. A single-process run is capped with `NPROC`, and the docs say so.

## 5. CLI

```
research experiment run SPEC --spec NAME
    [--workers W]                  # CPU core budget; enables the supervisor
    [--cores-per-worker C]         # override 4.2's c
    [--max-runs B]                 # override 4.2's b (existing flag)
    [--memory-budget GB]           # override 4.3's budget
    [--reclaim-stale-after S]      # supervisor-only when --workers is set
    [--no-compile-cache]
    [--only key=value ...]         # restricts which groups the supervisor schedules
```

## 6. Testing

- **small:** the sizing policy (`c` from `W` and `G`; `b` from measured time and memory; admission) as pure functions;
  the reclaim fix with two connections to one database.
- **medium:** the supervisor end to end, on a toy `BatchTrainer` spec with three static groups. Check that every run
  completes, that each worker saw its `NPROC` (the trainer records `os.environ`), that running workers never exceed `W`
  cores, and that SIGTERM leaves no child process.
- **performance (targeted, not a PR gate):** `NPROC` caps a real critic update under concurrent load, measured as
  CPU time over wall time per worker.

## 7. Migration

- Delete `projects/actor-free-qrc/scripts/run_parallel.py` and its tests, and point its users at `--workers`.
- Update `docs/workflow.md`, and the monorepo's `run-experiments` skill: one tracked command,
  `research experiment run … --workers W`.
- Record the decision as an ADR (local parallel execution) once this spec is accepted.

## 8. Open questions

1. **Sizing by rule or by probe.** 4.2 is a rule plus first-batch measurement, and needs nothing from the trainer. The
   alternative is a timed probe of several `(b, c)` pairs per group before running it, which would need a
   trainer-level "run k updates" hook. Recommendation: the rule, with the metadata telemetry, and add a probe only if
   the data shows throughput left on the table.
2. **Growing running workers.** A worker's cores are fixed at start. When groups finish, the last few workers keep
   their small `c`. The alternative is to restart a worker between batches with a larger `NPROC`, costing a process
   start (and a cache-hit compile). Recommendation: restart between batches once the compile cache makes it cheap,
   but not in v1.
3. **What `--workers` counts.** This spec counts physical cores. With SMT, 16 cores show 32 logical CPUs; `NPROC`
   counts threads.
