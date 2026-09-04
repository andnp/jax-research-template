# Reading Results Without Re-Deriving the Library

Companion to [Running Experiments Without Wasting Compute](experiment-efficiency.md), which
covers not wasting compute when you *produce* results. This page covers not wasting time when
you *read* them: `research_analysis` already has a loader and a reporting layer, and both are
named for research concepts (an estimand, a bakeoff, a sensitivity slice) rather than for the
mechanism underneath (a SQL join, a bootstrap resample). That means grepping for the SQL you are
about to hand-write finds nothing, even though the function exists. One project hand-wrote a
run/hyperparameter loader that `research_analysis.experiment.resolve_run_artifacts` already
provided, then separately hand-wrote a paired bootstrap comparison that
`research_analysis.reporting.compare_pairwise` already provided — and the hand-written loader
happened to dodge a component-version pooling bug that the shared loader had, so duplicating the
library also hid that bug from everyone else using it.

Check the table in [The entry points, by task](#the-entry-points-by-task) before you write SQL
by hand.

## Where results live

A project's `results/` directory holds two SQLite databases with different jobs:

- **`experiments.sqlite`** — definitions and bookkeeping: `Experiments`, `Runs` (the
  `(experiment, algo_version, env_version, hyper_id, seed, ablation)` intent, unique per that
  tuple), `Executions` and `ExecutionArtifacts` (the physical attempts and their output paths),
  `HyperparamConfigs` (the hyperparameter JSON blob a run points at), and `ComponentVersions`
  (the content hash of an `ALGO`/`ENV` component's source).
- **`metrics.sqlite`** — one `metrics` table: `(run_id, execution_id, global_step, metric_name,
  value)`, the actual training curves.

`ExperimentSpec.db_path` and `.metrics_db_path` derive both paths from the same
`results_root` (`libs/research-runner/src/research_runner/types.py`), so the split is
structural, not incidental — a metrics value is meaningless without the run/hyperparameter row
in the other database that explains what produced it, which is exactly what
`resolve_run_artifacts` joins for you.

## The entry points, by task

| You want | Use |
|---|---|
| Run metadata (condition, seed, which execution/metrics DB) for a completed experiment | `research_analysis.experiment.resolve_run_artifacts(experiments_db, slug, *, schema=None, latest_component_versions=True)` |
| Long-form training curves (`condition_name, seed, step, metric, value`) for a completed experiment | `research_analysis.experiment.load_experiment_metrics(*, experiments_db, slug, metrics=None, run_artifacts=None, schema=None, latest_component_versions=True)` |
| A genuinely custom query neither of the above covers | `research_analysis.loader.load_sqlite_query(database_path, query) -> pl.DataFrame` — the escape hatch, not the default |
| A/B comparison between exactly two conditions, with declared estimand/pairing/missingness | `research_analysis.reporting.compare_pairwise(db_path, experiment_slug, condition_a, condition_b, metric_name, ..., spec=None)` |
| Hyperparameter sweep sensitivity, with cross-fitted (leave-one-seed-out) selection to avoid overstating the winner | `research_analysis.reporting.analyze_hypers(db_path, experiment_slug, target_hyperparameter, metric_name, group_by=None, ...)` |
| Multi-algorithm, multi-environment bakeoff (needs ≥3 algorithms and fixed per-env references) | `research_analysis.reporting.compare_bakeoff(db_path, experiment_slug, algorithms, environments, metric_name, references, spec=None)` |
| A confidence interval on a scalar or a learning curve, or on the difference between two samples | `research_analysis.bootstrap.bootstrap_ci` / `bootstrap_scalar_ci` / `bootstrap_difference_ci` |
| Declaring the statistical design once so `compare_pairwise` / `compare_bakeoff` share it | `research_analysis.comparison.ComparisonSpec(estimand="mean", pairing="auto", alpha=0.05, missingness="error", pairing_keys=("seed",), ...)` |

`resolve_run_artifacts` and `load_experiment_metrics` both take a `schema: AnalysisSchema | None`
— pass one when a project's hyperparameter JSON doesn't use the default `condition_name`/`env_name`
keys, and both functions read `env_name` off it, not just `condition_name`. `analyze_hypers` and
`compare_bakeoff` key off the literal hyperparameter names `condition_name`/`algorithm`/`env_name`
inside stored JSON instead, which is why `experiment-efficiency.md` tells you to name sweep axes
`algorithm`, `env_name`, `learning_rate`, `seed`.

## The component-version trap

This is the one that actually cost a project real time, so read it even if you skip the rest.

`resolve_run_artifacts` defaults to `latest_component_versions=True`, which — after finding every
run with a completed execution — restricts the result to runs whose `algo_version_id` and
`env_version_id` both equal the *maximum values among runs that have completed executions*. That
is deliberately not "the newest version that exists": a `sync` mints a new `ComponentVersion` and
inserts unexecuted `Run` rows for it the instant a declared component's source hash changes,
including for a purely cosmetic edit. If the restriction were computed over all runs instead of
only completed ones, the newest version would routinely have zero completed runs, and the
function would silently hand back an empty frame for a study that is in fact complete at the
prior version.

Why the pooling side matters as much as the empty-frame side: `experiment-003-discrete-lr-sweep`
in `projects/iterated-gtd/results/experiments.sqlite` has 4200 completed runs across five
`(algo_version_id, env_version_id)` pairs:

```
experiment-003-discrete-lr-sweep: 4200 runs (4200 completed, 0 pending)
  algo_version=4 env_version=1: 300 runs (300 completed, 0 pending)
  algo_version=9 env_version=8: 450 runs (450 completed, 0 pending)
  algo_version=10 env_version=8: 450 runs (450 completed, 0 pending)
  algo_version=12 env_version=11: 1500 runs (1500 completed, 0 pending)
  algo_version=13 env_version=11: 1500 runs (1500 completed, 0 pending) (newest with results)
```

Version 12 is a known-buggy agent and 13 is its fix. Call `resolve_run_artifacts` with the
default and you get only the 1500 runs at version 13 — the fixed behavior. Call it with
`latest_component_versions=False` and you get all 4200 pooled together, averaging broken and
fixed runs into one number that describes neither. The opt-out is there for a real use, not just
as an escape hatch: comparing version 12 against version 13 *on purpose* is how the bug was
diagnosed in the first place — you resolve twice, once per version, and compare the two frames
rather than pooling them.

`algo_version_id` is `None` whenever an experiment declares more than one `ALGO` component (the
runner can't attribute one hash to per-arm rows), and `None` is matched against `None` rather than
coerced to "oldest" or dropped — an experiment with no versioned runs still gets its rows back.

## Diagnosing which versions a study has

`research experiment status <db>` prints a per-`(algo_version_id, env_version_id)` breakdown
whenever an experiment has more than one such pair, marking the newest pair *with results* — the
same rule `resolve_run_artifacts` uses. Real output, from the messy history above:

```console
$ uv run research experiment status projects/iterated-gtd/results/experiments.sqlite \
    --experiment experiment-003-discrete-lr-sweep
experiment-003-discrete-lr-sweep: 4200 runs (4200 completed, 0 pending)
  algo_version=4 env_version=1: 300 runs (300 completed, 0 pending)
  algo_version=9 env_version=8: 450 runs (450 completed, 0 pending)
  algo_version=10 env_version=8: 450 runs (450 completed, 0 pending)
  algo_version=12 env_version=11: 1500 runs (1500 completed, 0 pending)
  algo_version=13 env_version=11: 1500 runs (1500 completed, 0 pending) (newest with results)
```

Run this before trusting any number out of `resolve_run_artifacts` for a study whose history you
don't already know cold — it costs seconds and tells you whether "latest" is one version or five.

## Comparing arms across two experiments

Sometimes the control you need was already computed under a different experiment slug. Because
`Runs` is `UNIQUE(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation)`,
re-declaring that same arm inside a new experiment slug does not reuse the row — it recomputes it
bit-for-bit, at full compute cost, purely because `experiment_id` differs. The cheaper path is to
declare only the new arm in the new experiment, and pair the two slugs together in the analysis
query instead of in the experiment definition — separation by query is free, separation by new
identity is a full re-run (the same point `experiment-efficiency.md` makes about renaming a slug).

That is only sound after three checks, and if any of them fails, don't do it:

1. **Same component versions on both sides.** Both slugs' runs must carry the same
   `algo_version_id` and `env_version_id` — check with `research experiment status` on each
   database/slug. A cross-slug comparison with mismatched component versions compares code
   versions, not conditions, which is the same trap as the component-version section above,
   just spread across two experiments instead of one.
2. **Same config coverage.** The two sides must have the same number of hyperparameter configs
   over the same key set, so every config on one side has a partner on the other. An unpaired
   leftover config silently changes what's being averaged into "the arm."
3. **Field-by-field diff of the paired hyperparameter configs.** Diffing the JSON blobs pairwise
   must show exactly the intended variable differing and nothing else. Anything else that differs
   (a seed range, an unrelated hyperparameter default) is confounded with the comparison you think
   you're running.

If any check fails, the honest move is to declare the control arm in the new experiment and pay
for the re-run — a comparison you can't verify is clean is worth less than the compute it would
cost to make it clean.

## Selection bias analysis code cannot fix

`analyze_hypers` cross-fits its winner selection (leave-one-seed-out) specifically because
picking the hyperparameter value with the best raw mean and then reporting that same mean
overstates performance — the reported "maximization bias" in its output is that gap, measured.
The same failure mode reappears one level up, where no library function catches it: reporting an
arm's difference at its own best hyperparameter value, selected on the same data used to report
the difference, overstates the effect for the identical reason. Before believing a reported gap,
check that it survives at hyperparameter values neighboring the reported best, not only at the
best value itself.
