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
| An A/B comparison between exactly two conditions | `compare_pairwise(dataset, condition_a, condition_b, metric_name, spec=...)` |
| Hyperparameter sensitivity with leave-one-seed-out correction | `analyze_hypers(dataset, target_hyperparameter, metric_name, group_by=...)` |
| A confidence interval for one tuned arm | `hierarchical_bootstrap_ci_from_experiment(dataset, hyperparameter_keys, metric_name, h_samples, e_samples, ...)` |
| A paired difference between independently tuned arms | `hierarchical_bootstrap_difference_ci_from_experiment(dataset, hyperparameter_keys_a, hyperparameter_keys_b, metric_name, ..., condition_a=..., condition_b=...)` |
| A multi-algorithm, multi-environment bakeoff | `compare_bakeoff(dataset, algorithms, environments, metric_name, references=..., spec=...)` |
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
from research_analysis.reporting import compare_pairwise

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
report = compare_pairwise(dataset, "baseline", "candidate", "reward", verbose=False)
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
sweep values, and seed coverage. If the requested selection is ambiguous, strict
loading fails with structured diagnostics. Incomplete executions and malformed
artifact metadata are filtered before materialization. Missing recorded metric
artifacts also fail strict loading; the loader never searches parent directories
or falls back to `ExecutionArtifacts.root_path`.

Use `partial=True` only when the analysis explicitly permits missing metric
artifacts. The returned dataset retains `DatasetDiagnostic` values for missing
metric artifacts and version-selection outcomes. Statistical functions still
apply their own pairing and missingness rules through `ComparisonSpec`.

## Curves and step grids

Dataset loading preserves metric observations; it does not invent or silently
align a training step grid. Curve operations validate or select the step-grid
policy required by that operation. Scalar reporting functions use the final
materialized observation per run where their contract calls for a scalar.

## Selection-bias reminder

`analyze_hypers` cross-fits winner selection because reporting the best raw mean
on the same seeds used to select it overstates performance. The hierarchical
bootstrap functions model a separate question: the performance of a champion
selected under a stated tuning budget, including selection uncertainty. A
comparison of tuned arms should make the selection procedure and its seed
coverage explicit rather than silently reusing a best configuration.

For experiment definition and rerun mechanics, see
[Running Experiments Without Wasting Compute](experiment-efficiency.md) and
[the experiment workflow](workflow.md). For the full dataset contract, see
[Canonical Experiment Dataset API](specs/research-dataset-api.md).
