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

**Mark every axis except `seed` as `is_static=True`.** The runner resolves one
hyperparameter dict per execution from its batch, and batching groups by static config.
A non-static axis means a batch trains at one value while each run records its own — no
error, plausible numbers, wrong results.

**Name axes `algorithm`, `env_name`, `learning_rate`, `seed`.** The reporting helpers in
`research_analysis.reporting` key off those names, so you get `analyze_hypers` and
`compare_bakeoff` for free instead of writing statistics by hand.
