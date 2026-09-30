# ADR 013: Local Parallel Execution

## Status
Accepted

## Context
`research experiment run` executed batches one after another in one process. Using a many-core machine meant
hand-slicing sweeps into processes pinned with `taskset`, and that failed repeatedly: slices oversubscribed or
overlapped cores, concurrent reclaims crashed each other, nothing sized the process count to free memory, and
killing the launcher left workers running. The affinity-based `RESEARCH_MAX_CPUS` sized XLA's pool but assigned no
cores.

## Decision
`research experiment run --workers W` is a supervisor over worker processes, one per pending group
(`RunBatch.group_key`), specified in `docs/specs/parallel-experiment-runner.md`.
- `W` is a thread budget. Each worker's XLA pool is capped with `NPROC`, which XLA honours without affinity; the
  OS scheduler places the threads.
- A worker starts only when the budget and freshly read `MemAvailable` fit it, using peak memory reported by earlier
  executions.
- Workers die with the supervisor (`PR_SET_PDEATHSIG`), and a worker that dies has its group's RUNNING executions
  failed so the next launch retries them.
- The batch size is fixed per sweep (`--max-runs`). Sizing it automatically is deferred to a policy that reads the
  per-execution runtime telemetry.

## Consequences
- **Pros:** one command and one process tree per sweep; no core pinning to coordinate; memory-aware admission;
  cancellation and worker crashes clean up after themselves.
- **Cons:** `W` is a soft budget, not a core reservation. A SIGKILLed supervisor still leaves its workers'
  executions RUNNING until `--reclaim-stale-after` or `invalidate`. Only one supervisor per experiment at a time.
