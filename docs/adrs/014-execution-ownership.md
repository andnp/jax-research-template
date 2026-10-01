# ADR 014: Execution Ownership

## Status
Accepted. Supersedes ADR 013's start-time group sweep and `--reclaim-stale-after`.

## Context
A planned execution claims its runs while it is PENDING or RUNNING. When the process that should finish it dies
without unwinding (SIGKILL, OOM, host reaping, a cancelled Slurm array), the claim stays. Core has four ways to clear
it, and each guesses:

- `--reclaim-stale-after S` fails RUNNING executions whose `start_time` is older than `S`. The caller has to pick an
  `S` longer than any real batch. It can't tell slow from dead, and it ignores PENDING.
- `fail_group_executions_started_since` (supervisor) fails a dead worker's RUNNING executions by recomputing group
  keys from metadata and comparing start times.
- `submit` fails its planned executions only when `sbatch` itself fails. A cancelled or never-started array leaves
  PENDING claims forever.
- `invalidate` is manual.

PENDING can't be reclaimed by age. `submit` deliberately leaves executions PENDING for as long as Slurm queues them,
and that time has no upper bound. This is also why planning up front is unsafe in the `--workers` supervisor and in
serial `run`: a double SIGKILL would leave PENDING claims that no automatic path recovers. The reclaim docstring
rejected recording a PID because PIDs only make sense on the recording host and get reused. Both objections have
answers: a boot id scopes a PID to one boot of one host, the process start time rules out reuse, and Slurm job ids
are valid across hosts.

## Decision
Every live execution records its **owner**: the thing whose death means the execution will never finish. Reclaim
checks whether the owner is alive instead of looking at timestamps.

### Owner
One nullable column, `Executions.owner TEXT`, holding one of:

- `local:<hostname>:<boot_id>:<pid>:<start_ticks>`, a process on one boot of one host (`/proc/sys/kernel/random/boot_id`,
  plus field 22 of `/proc/<pid>/stat`).
- `slurm:<array_job_id>_<task_index>`, one Slurm array task.

Owners are assigned when work is planned and handed over when the work moves:

| Mode | Owner at plan time | Handover |
|---|---|---|
| serial `run` | the `run` process | none |
| `run --workers` | the supervisor process | none; workers die with it (`PR_SET_PDEATHSIG`), and while it is alive it fails a dead worker's ids itself |
| `submit` | the submitting process | after `sbatch` returns job id `J`, execution `ids[i]` gets owner `slurm:J_i` in one transaction |

Execution never changes the owner. `execute-batch` running under Slurm already belongs to its task.

### Liveness
`owner_state(owner) -> ALIVE | DEAD | UNKNOWN`:

- `local`, same host and boot: ALIVE if `/proc/<pid>/stat` exists with the same start ticks, otherwise DEAD.
- `local`, same host, different boot: DEAD, because no process survives a reboot.
- `local`, different host: UNKNOWN.
- `slurm`: `squeue -h -j J_i` lists it, so ALIVE. Otherwise `sacct` reports a terminal state (or the job is unknown to
  Slurm), so DEAD. If neither command is available, UNKNOWN.

### Reclaim
`reclaim_dead_executions(experiment_id) -> ReclaimReport` fails every PENDING or RUNNING execution whose owner is DEAD,
and reports the UNKNOWN ones. Ownerless rows are left alone. It probes outside the transaction, then transitions inside
`BEGIN IMMEDIATE` only the rows whose status and owner are unchanged, so concurrent reclaimers can't race into
FAILED→FAILED (the crash seen when parallel slices each reclaimed).

Because reclaim is now safe by construction, `run`, `run --workers` and `submit` call it automatically before
planning. `--reclaim-stale-after` is deleted outright. The owner is stored as one encoded TEXT column, because nothing needs to query its parts in SQL. A row that ownership can't resolve (UNKNOWN, or a legacy row with no
owner) is listed in the output and goes through `invalidate`.

### What this removes
`--reclaim-stale-after` and `reclaim_stale_executions`. `fail_group_executions_started_since`, `_planned_group_key`
and the group/start-time sweep, once the supervisor fails a dead worker's ids directly (it planned them). The rule
that PENDING claims from a cancelled array need `invalidate`.

### What this enables
The supervisor plans each group when it is admitted and starts `execute-batch` with that group's ids. Workers stop
re-syncing and re-planning in every child, and the hidden `--group` CLI path goes away. Planning up front is now
recoverable in every mode.

## Consequences
- **Pros:** reclaim needs no threshold and runs on every launch. PENDING and RUNNING orphans recover the same way.
  Concurrent reclaim is safe. Two heuristic sweeps and a CLI flag are deleted.
- **Cons:** a schema change on durable result DBs (an idempotent `ALTER TABLE ... ADD COLUMN owner`, same pattern as
  `_migrate_run_vmap_zone`). Liveness depends on `/proc` (Linux only) and on `squeue`/`sacct` for Slurm. A local owner on
  another host is UNKNOWN, so recovery after a launch from a different machine needs `invalidate`. That is also true
  today.
- **Legacy rows:** executions planned before this change have no owner, and only `invalidate` recovers them.
