# Reading Results Through the Dataset Boundary

`research_analysis.dataset.load_experiment_dataset` is the single entry point
for turning an experiment definition and its recorded artifacts into analysis
data. It joins run metadata to metric rows, selects a coherent completed
version set, preserves run identity and provenance, and reports incomplete data
explicitly. Statistical reporting consumes the resulting `ExperimentDataset`.

This boundary prevents each project from independently reimplementing run
selection, component-version filtering, artifact discovery, and metric joins.
For custom SQL that is genuinely outside the dataset contract, use
`research_analysis.loader.load_sqlite_query` as an escape hatch—not as the
normal way to load experiment results.

## The entry points by task

| You want | Use |
|---|---|
| Typed run metadata and long-form metrics | `load_experiment_dataset(experiments_db, slug, *, dimensions=..., version_selection=..., cohort=..., metrics=..., partial=...)` |
| A comparison between two or more conditions | `bakeoff_report(dataset, metric_name, conditions, spec=...)` |
| Bias-corrected hyperparameter selection | `best_hypers(dataset, metric_name, group_by=...)` |
| Hyperparameter sensitivity for one axis | `slice_hypers(dataset, metric_name, vary, group_by=...)` |
| A direct array comparison | `run_permutation_test(...)` and `bootstrap_difference_ci(...)` |
| A genuinely custom SQLite query | `load_sqlite_query(database_path, query)` |

The reporting functions do not accept database paths, experiment slugs,
`AnalysisSchema`, or execution roots. Build the dataset once, inspect its
diagnostics, then pass it to one or more analyses:

```python
from research_analysis.dataset import (
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    VersionSelection,
    load_experiment_dataset,
)
from research_analysis.reporting import bakeoff_report

dataset = load_experiment_dataset(
    experiments_db,
    "study-slug",
    dimensions=ExperimentDimensions(
        condition="algorithm",
        environment="environment",
        sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT),),
    ),
    version_selection=VersionSelection.latest_completed(),
)
report = bakeoff_report(dataset, "reward", ["baseline", "candidate"], verbose=False)
```

`runs` contains canonical `condition_name`, `seed`, `run_id`, `execution_id`,
`ablation`, selected dimensions, declared sweep axes, and component provenance.
`metrics` is long-form data keyed by run and execution with `step`, `metric`,
and `value`; it also carries the canonical dimensions needed by analysis.

## Version selection is part of the analysis

The default `VersionSelection.latest_completed()` chooses one coherent
component-version set represented by completed runs. It does not independently
take the maximum algorithm and environment version, and it does not include
unexecuted runs created by a later source change.

Use `VersionSelection.exact(...)` to reproduce a historical cohort. Build each
`ComponentVersionRef` with `ComponentVersionRef.by_id(...)` or
`ComponentVersionRef.by_code_snapshot_hash(...)`; an unqualified version
number is not a valid selector. Use `VersionSelection.all()` only when
intentionally analyzing every completed version set; inspect the resulting
excluded-version diagnostics before interpreting pooled data.

The loader preserves `algo_version_id=None` when an experiment contains
multiple algorithm components and the schema cannot attribute a particular run.
It does not guess from a condition name.

## Cohorts and incomplete data

A cohort is defined by its selected component versions, dimensions, ablation,
sweep values, and seed coverage. If the requested selection is ambiguous (a
partial re-run under `latest_completed`, or runs of one cell differing in an
undeclared hyperparameter), strict loading fails with structured diagnostics.
Failed and pending runs are reported as non-fatal diagnostics; malformed
artifacts, missing hyperparameter configs, unresolved dimensions, and missing
metric artifacts fail strict loading. The loader never searches parent
directories or falls back to `ExecutionArtifacts.root_path`.

If loading fails with an unresolved condition dimension, check which key labels
your arms: `ExperimentDimensions.condition` defaults to `condition_name`, so an
experiment that labels arms with `algorithm` needs
`ExperimentDimensions(condition="algorithm")`.

Use `partial=True` only when the analysis explicitly permits the reported
problems. `dataset.diagnostics` keeps every diagnostic, and
`dataset.accounting` counts each cell's planned, completed, failed, pending,
diverged, and excluded runs; read both before interpreting a comparison.

Diverged runs (NaN or infinite metrics) are outcomes, not missing data. They
stay in the dataset, are listed in `runs["non_finite_metrics"]`, and make
statistical functions raise `NonFiniteValuesError` unless you apply
`research_analysis.non_finite.substitute_non_finite(dataset, metric,
worst_case)` with a declared worst-case value. Statistical functions apply
their own pairing and missingness rules. Bakeoff comparisons use `BakeoffSpec`;
low-level array functions expose their own pairing argument.

## Curves and step grids

Dataset loading preserves metric observations; it does not invent or silently
align a training step grid. Curve operations validate or select the step-grid
policy required by that operation. Scalar reporting functions use the final
materialized observation per run where their contract calls for a scalar.

## Selection-bias reminder

`best_hypers` cross-fits winner selection because reporting the best raw mean
on the same seeds used to select it overstates performance. The hierarchical
bootstrap functions model a separate question: the performance of a champion
selected under a stated tuning budget, including selection uncertainty. A
comparison of tuned arms should make the selection procedure and its seed
coverage explicit rather than silently reusing a best configuration.

For experiment definition and rerun mechanics, see
[Running Experiments Without Wasting Compute](experiment-efficiency.md) and
[the experiment workflow](workflow.md). For the full dataset contract, see
[Canonical Experiment Dataset API](specs/research-dataset-api.md).
