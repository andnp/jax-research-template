# Technical Specification: Canonical Experiment Dataset API

## 1. Purpose

`research_analysis.dataset` is the only boundary between the relational experiment
database, recorded metric artifacts, and statistical reporting. Reporters consume
an `ExperimentDataset`; they do not open experiment databases or resolve run
artifacts themselves.

The package `__init__.py` is empty. Import the loader from
`research_analysis.dataset.loading` and the types (`ExperimentDimensions`,
`SweepAxis`, `VersionSelection`, `CohortSelection`, and the rest) from
`research_analysis.dataset.types`.

The API is intentionally final-state only. It does not retain wrappers for the
former `resolve_run_artifacts` or `load_experiment_metrics` entry points.

## 2. Dataset contract

`ExperimentDataset` contains:

- `runs`: one typed Polars table with logical run identity, selected execution,
  dimensions, ablation, declared sweep values, and component provenance.
- `metrics`: one typed long-form Polars table keyed by run and execution, with
  `condition_name`, optional `env_name`, ablation, step, metric, and value.
- `dimensions`: an `ExperimentDimensions` descriptor for the condition,
  optional environment, and declared typed sweep axes.
- `diagnostics`: structured non-row outcomes produced while selecting and
  materializing the dataset.
- `accounting`: per-cell run outcome counts (section 5).
- `provenance`: every selected component version.
- `executions`: one `ExecutionProvenance` per execution behind a loaded run, with
  its git commit, whether uncommitted changes were recorded (`dirty`), host, and
  start and end times.

The canonical dimension column names are `condition_name` and `env_name`.
`env_name` is present only when the experiment declares an environment
dimension. Ablation is retained separately from condition and is part of run
identity even when two runs have identical hyperparameter values.

A sweep axis that exists on only some arms declares where it applies:
`SweepAxis("num_heads", SweepValueType.INTEGER, when={"algorithm": ("gi_qrc", "gi_qrc_anchored")})`
applies to a run whose value for every `when` key is one of the listed values,
and is null on every other run. One `ExperimentDimensions` therefore covers every
arm, so an experiment loads as one dataset.

## 3. Provenance

Every selected run carries component-version provenance when the database can
attribute it:

- component identity and name;
- component type;
- component-version row ID;
- component version number only as component-scoped display metadata;
- recorded code snapshot hash.

An unset `Runs.algo_version_id` in a multi-ALGO experiment is preserved as
unknown provenance. The dataset loader does not infer an algorithm component
from condition names or hyperparameters.

## 4. Version selection

The loader requires an explicit version policy. It supports:

- `latest_completed`: one coherent set of component versions represented by
  completed runs, never independent maxima selected from different cohorts;
- `exact`: component identity plus either `ComponentVersionRef.by_id(...)` or
  `ComponentVersionRef.by_code_snapshot_hash(...)`; an unqualified version
  number is invalid;
- `all`: retain every completed version set and expose excluded or ambiguous
  cohorts through diagnostics.

Version selection is applied before metric materialization so reports cannot
silently pool incompatible code versions.

## 5. Cohorts and diagnostics

A cohort is identified by its selected component-version set, condition,
optional environment, declared sweep-axis values, ablation, and seed coverage.
The loader fails strict loading with an ambiguous-cohort diagnostic when:

- `latest_completed` selects a version set that lacks completed runs for a
  cell present only in excluded version sets (a partial re-run);
- runs of one cell differ in an undeclared hyperparameter other than `seed`;
  the message names the keys to declare as sweep axes or dimensions.

An explicit cohort selector, version selection, or declared axis resolves it.

Nothing is dropped silently. Every planned run matching the cohort produces
either a materialized row or a diagnostic:

| Code | Fatal | Meaning |
| --- | --- | --- |
| `run_failed` | no | only failed executions |
| `run_pending` | no | pending, running, or never executed |
| `malformed_artifact` | yes | completed without a valid artifact record |
| `missing_hyper_config` | yes | the run's hyperparameter config is missing |
| `unresolved_dimension` | yes | a dimension or applicable axis is missing or mistyped |
| `missing_metric_artifact` | yes | no metrics database or no rows for the run |
| `ambiguous_cohort` | yes | see above |
| `invalid_version_selection` | yes | exact versions have no completed runs |
| `excluded_version` | no | a superseded completed version set |
| `non_finite_metric` | no | a metric has NaN or infinite values (a diverged run) |

`partial=True` returns valid rows while retaining every diagnostic.
`ExperimentDataset.accounting` counts, per cell, planned, completed, failed,
pending, and excluded runs (with reasons) and diverged runs per metric; runs
with no resolvable cell are counted as unassigned.

Diverged runs stay in the dataset: NaN is stored as NULL in the metric store
and restored as NaN, and `runs["non_finite_metrics"]` lists the affected
metrics. Statistical consumers raise `NonFiniteValuesError` naming the runs;
`research_analysis.non_finite.substitute_non_finite` applies a declared
worst-case value instead. Consumers also reject two runs sharing one cell and
seed, naming the likely pooling cause.

Step-grid alignment is not a dataset-loader responsibility; curve analysis
continues to validate alignment for the operation that needs it.

## 6. Reporting contract

The following functions accept an `ExperimentDataset` and never accept an
experiments database path:

- `bakeoff_report`
- `best_hypers`
- `slice_hypers`

They operate only on the already-selected runs and materialized metrics in the
dataset. They reduce each run's curve to one score with a declared
`PerformanceMeasure` through `run_scores`. `BakeoffSpec` governs estimands,
pairing, missingness, and resampling. The functions only compute: they never
print or write files.

`research_analysis.plan_report.run_plan` is the one reporting entry point that
takes a database path. It loads the dataset from the declared plan's dimensions,
version selection, and cohort, always with `partial=True`, and embeds the
dataset's diagnostics and accounting in the report document. See the
[reporting API](research-reporting-api.md).

## 7. Artifact loading

Metric databases are loaded only from the recorded
`metadata_json["metrics_db_path"]` value. The loader never searches parent
directories or falls back to `ExecutionArtifacts.root_path`. Metric queries
are batched by database where practical while preserving run and execution
identity in every returned row.
