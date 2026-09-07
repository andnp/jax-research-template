# Technical Specification: Canonical Experiment Dataset API

## 1. Purpose

`research_analysis.dataset` is the only boundary between the relational experiment
database, recorded metric artifacts, and statistical reporting. Reporters consume
an `ExperimentDataset`; they do not open experiment databases or resolve run
artifacts themselves.

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

The canonical dimension column names are `condition_name` and `env_name`.
`env_name` is present only when the experiment declares an environment
dimension. Ablation is retained separately from condition and is part of run
identity even when two runs have identical hyperparameter values.

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
If the requested dimensions identify more than one cohort, the default loader
fails with an ambiguous-cohort diagnostic. An explicit cohort selector is
required to disambiguate it.

Rows that cannot satisfy the requested dimension projection are omitted without
diagnostics. The loader still reports missing recorded metric artifacts,
ambiguous cohorts, and invalid version selections; `partial=True` returns valid
rows while retaining those diagnostics.

Step-grid alignment is not a dataset-loader responsibility; curve analysis
continues to validate alignment for the operation that needs it.

## 6. Reporting contract

The following functions accept an `ExperimentDataset` and never accept an
experiments database path:

- `bakeoff_report`
- `best_hypers`
- `slice_hypers`

They operate only on the already-selected runs and materialized metrics in the
dataset. Statistical behavior, estimands, pairing, missingness, and fixed
reference normalization remain governed by `BakeoffSpec` and the existing
reporting contracts. Low-level array comparisons expose pairing directly.

## 7. Artifact loading

Metric databases are loaded only from the recorded
`metadata_json["metrics_db_path"]` value. The loader never searches parent
directories or falls back to `ExecutionArtifacts.root_path`. Metric queries
are batched by database where practical while preserving run and execution
identity in every returned row.
