# Running Experiments Without Wasting Compute

Companion to Workflow D in `workflow.md`, which covers the mechanics of defining and
running a sweep. This page covers the part that costs real money when you get it
wrong: what the runner considers already done, and how to re-run only what changed.

## Run identity is content-addressed

A `Run` is the intent `(experiment, algo_version, env_version, hyper_id, seed, ablation)`,
uniquely constrained. `hyper_id` addresses the *content* of a hyperparameter config, so
two studies that share a config share the row. A Run is **satisfied** by having a linked
`COMPLETED` execution; `plan` and `run` only pick up unsatisfied Runs.

Everything below follows from that one sentence.

## Change only what actually differs

If one arm of a study changes, only that arm's configs change, so only its Runs become
unsatisfied and only they re-run. This works on its own — as long as you do not destroy
the identity that makes it work.

The trap is coarse re-identification. Renaming an experiment slug creates a new
experiment id, which makes *every* Run a new row and re-runs the whole study. That
mistake once cost half a 300-run sweep: one arm had changed, both re-ran.

If your reason for a new slug is keeping old and new results apart in analysis, filter
the analysis by experiment slug instead. Separation in the query is free; separation by
new identity costs a full re-run.

## Preview before launching

`plan` is a true dry run: it syncs the definition, prints the batches it would run, and
creates no executions.

```bash
uv run research experiment plan <spec_file.py> --spec <factory>
```

Check the batch count against the arm you changed. If you changed one of two arms and
`plan` reports the whole grid, stop and find out why. That check takes seconds; the
wasted run takes as long as the sweep.

## Code changes invalidate through component hashes

`Runs` is unique on `(experiment, algo_version_id, env_version_id, hyper_id, seed,
ablation)`, and a `ComponentVersion` is the SHA-256 of its source file. So editing a
file a declared `Component` points at mints a new version id, which makes new Run rows
and re-plans them. That is usually what you want.

The catch is granularity. `_insert_runs` stamps one algo component per run, so when an
experiment declares exactly one `ALGO` component, every arm carries it — and editing one
arm's source invalidates the other arms too. Adding a third arm to a two-arm study by
editing the iterated agent's file re-planned all 450 runs, not the 150 that were new.

Declaring a `Component` per arm does not fix it: with more than one `ALGO` component the
stamp becomes NULL, which is honest about attribution but drops hash tracking entirely,
so code edits stop invalidating at all. Pick deliberately: one component gives coarse
invalidation with misattributed provenance, several give accurate provenance with no
invalidation.

Either way, check `plan`'s count against the arm you changed before launching, and use
`--where` when you need to re-run less than the hash change implies:

```bash
uv run research experiment invalidate <db> \
    --experiment <slug> --where algorithm=<arm>
```

`--where` is repeatable and all clauses must match. It requires `--experiment`, because
two studies can share a key/value. It prints what it matched before prompting, and
`--yes` skips the prompt for scripting. A batch is invalidated only if *every* run it
covers matches, so a partial match is skipped rather than silently widening.

Values are compared as strings against `str()` of the stored JSON value:
`learning_rate=0.001` matches a stored `0.001`, but not `1e-3`.

## Invalidation is deliberately optimistic

Tracking is not, and is not meant to be, complete. Component hashes cover the source files
a study declares; they do not cover the installed environment, the glue that maps
hyperparameters onto an agent's config, or anything else outside a declared `Component`.
That is a choice, not an oversight. Results cost GPU-hours, so the system errs toward
keeping them and leaves the judgment call to the experimenter, who has the manual tools
for it.

Two consequences worth internalising:

- **Do not add hashing to make invalidation exhaustive.** The glue module is the file most
  often edited for reasons that change nothing — adding an arm's branch, a refactor, a
  docstring — and hashing it would re-run an entire study on every such edit. Prefer a
  loud check at analysis time over a conservative key: a check costs nothing, a key is
  paid for in compute.
- **Invalidate by hand when you know you have invalidated something.** Changed a
  dependency, edited the glue, fixed a bug the hash cannot see? Say so with `--where` for
  one arm, or `--all` for the study.

The safety net for the optimism is a deliberate clean re-run of the critical experiments
before results are published, so the numbers that ship are uncontaminated:

```bash
uv run research experiment invalidate <db> --experiment <slug> --all
uv run research experiment run <spec_file.py> --spec <factory>
```

`--all` requires `--experiment` — it will not reset every study in a shared database — and
it refuses to combine with `--where`, since the two express opposite intents. Everything
else in this page is about avoiding waste day to day; this is the one place where
re-running everything is the point.

## Orphan runs, and pruning them

`plan` picks up every unsatisfied Run row, and a change to the definition does not remove
the rows it stops wanting. Editing a component file or dropping a hyperparameter mints a
*new* Run identity and leaves the old rows behind, still unsatisfied and still plannable.
For a study that has already run this is harmless — the old rows have COMPLETED
executions, so they are satisfied. For a study **planned but never run** it is not: those
rows have no executions, so `plan` counts them and `run` executes them, recording results
under a stale component version that analysis then filters straight back out.

`invalidate` cannot reach them. It operates on executions, and orphans have none.

```bash
uv run research experiment prune <spec_file.py> --spec <factory>
```

`prune` syncs the definition, asks it which Run rows it wants, and deletes the rest. A run
that any execution has touched is never deleted, whatever its status: COMPLETED is a
result and PENDING or RUNNING is a live claim. It prints what it matched, grouped by
component version, before prompting; `--yes` skips the prompt.

Check `plan`'s count first. A study of 60 configs reporting 90 batches is this leak, not a
grid you forgot about.

## Recovering a crashed sweep

A runner killed without unwinding never writes a terminal status, so its execution stays
`RUNNING` and keeps claiming runs that are then neither satisfied nor re-plannable. The
sweep stalls short of completion. Reclaim them:

```bash
uv run research experiment run <spec_file.py> --spec <factory> --reclaim-stale-after 1800
```

Keep the threshold comfortably longer than a real batch takes, or you will reclaim work
from a slow but healthy process.

## Two definition rules that cost whole sweeps

**Let the config classify axes.** A run's batch is decided by the fields its own config
marks `traced(...)`: runs that differ only in traced values share one compiled
`jit(vmap(train))`, and every other parameter splits batches. Declare parameter values
plainly and mark vmappability on the config field, not in the spec. Every parameter must
be a config field or a choice, so a label such as `condition_name` needs a field (or a
`choices` entry) on the config; a name the config does not have fails at sync.

**Name the arm axis `condition_name`, or declare the name you used.** Reports read arms
from `ExperimentDimensions.condition`, which defaults to a hyperparameter called
`condition_name`. The common trap: a spec that labels its arms `algorithm` fails to load
with an unresolved condition dimension, unless the analysis plan declares
`ExperimentDimensions(condition="algorithm")`. In the same way, point `environment=` at
the key that names the environment (it becomes the `env_name` column), and declare every
swept hyperparameter as a typed `SweepAxis` so `best_hypers`, `slice_hypers`, and
`select_champion_over` can use it. The seed axis is always `seed`. Put these dimensions in
the spec's `@analysis_plan`, and `research experiment report` gives you the statistics
without any hand-written analysis.
