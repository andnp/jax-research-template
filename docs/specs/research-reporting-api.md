# Technical Specification: Research Reporting API

## 1. Overview

The reporting API turns an experiment's recorded runs into statistics that an agent can
act on and a person can audit. The rules are fixed by the
[Empirical Design & Analysis Guide](../science_guide.md):

- the resampling unit is the run;
- intervals are percentile-bootstrap intervals with a stated confidence;
- the headline comparison is a seed-paired difference, adjusted for the number of
  comparisons;
- tuned champions carry selection uncertainty;
- variability is reported apart from confidence;
- diverged, failed, and pending runs are never dropped silently.

There are three layers:

1. **Compute**, in `research_analysis.reporting`: `bakeoff_report`, `best_hypers`, and
   `slice_hypers`. They take an `ExperimentDataset` and return frozen dataclasses.
   They never print and never write files.
2. **Document**, in `research_analysis.report`, `plan`, `plan_report`, and
   `plan_snapshots`: a versioned `ReportDocument` that holds provenance, accounting,
   diagnostics, the declared plan and its status, results, failures, and figure
   paths.
3. **Render**, in `research_analysis.report_json`, `report_markdown`, and
   `report_plots`: strict JSON, markdown, and figures, each written only where the
   caller says.

`research experiment report` (see [research-cli](research-cli.md)) connects the three
layers for a spec's declared plan.

## 2. Compute entry points

`bakeoff_report(dataset, metric_name, conditions, *, select_champion_over=None,
block_by=None, spec=None, higher_is_better=True, min_seeds=5, measure=FinalValue(),
pairs=None, shared_across=(), untuned=())` compares two or more conditions:

- It compares block by block, never pooling runs across blocks, and pairs runs by seed.
- `pairs` is the planned comparison family. When it is omitted, every pair is compared.
- Each run is scored by `measure` through `run_scores`. A non-finite score raises
  `NonFiniteValuesError`.
- Under `missingness="error"`, a seed that some conditions observed and others did not
  raises `IncompleteBlocksError`. The error carries every `IncompleteCell` (condition,
  block, missing seeds, and the failed, pending, and excluded counts from the dataset
  accounting).
- Too little data to compute a statistic (fewer blocks, seeds, or tuned configurations
  than the spec's `min_samples` requires) raises `InsufficientDataError`, carrying the
  affected `InsufficientDataCell`s (coordinates, what was required, what was observed).
  `best_hypers`, `slice_hypers`, and the tuning and repeated-blocks internals raise the
  same error for their own data-dependent preconditions. A plain `ValueError` is kept
  only for programmer errors: bad argument types, unknown axis names, out-of-range
  statistical parameters.
- With `select_champion_over`, the differences come from the hierarchical per-sample
  bootstrap over each arm's grid, and no tests are run.
- `shared_across` names `block_by` axes declared with a `SweepAxis.when` rule. A
  condition the axis does not apply to (null on every run) has no block of its own on
  that axis: its runs join every block of it, so one axis-free referent is compared
  against each block's arms. It must be declared; without it, null is its own block.
  The reuse is visible and never double-counted: dataset accounting counts each run
  once, `runs` lists a reused run once per block it joined, `shared_across` in the
  report names the reused conditions, and the markdown states how many distinct runs
  were reused. Reused runs make blocks dependent, so the omnibus test is skipped with
  `omnibus_skipped_reason`; each paired test stays within one block, and Bonferroni
  and Holm hold under that dependence. A referent run whose other coordinates match no
  block raises `InsufficientDataError`; a condition lacking the axis on only some runs
  raises `ValueError`. In a champion comparison the referent is tuned separately
  within each block it joins.
- `untuned` declares single-configuration conditions as fixed referents in a
  champion comparison. They have no selection stage and no champion entry: their
  bootstrap is the plain percentile bootstrap of the mean over their seeds, paired by
  seed with the other arm's hierarchical draws, and each such difference's `method`
  says so. A condition not declared untuned still needs two configurations, one
  declared untuned must have exactly one, and at least one condition must be tuned.
  `tests/large/test_research_analysis_champion_calibration.py` records the measured
  coverage of these differences.

`best_hypers(dataset, metric_name, group_by=None, *, difference_between=None, ...)`
returns a bias-corrected champion per group, or the champion difference between two
values of one axis. `slice_hypers(dataset, metric_name, vary, group_by=None, ...)`
returns the selection-corrected best curve and the champion-pinned slice over one
axis.

`BakeoffSpec` fixes the estimand (`mean` or `median`), alpha, resamples, `min_samples`,
missingness (`error`, `complete_case`, or `blockwise`), RNG seed, and the tolerance
interval's confidence and coverage, and the CI `interval` method: `percentile` (default)
or `expanded_percentile` (Hesterberg 2015), which untuned bakeoffs record in the spec
and in each difference's `method`, and tuned analyses reject. A paired difference takes the estimand of the
per-seed differences (a paired median is median(A − B)); cells take it per condition.

## 3. Result records

- **`BakeoffReport`** contains:
  - `conditions`, `metric_name`, `measure`, `spec`, and `blocks`;
  - `cells[condition][block]`, a `CellSummary`: n, the estimate, the CI with its
    `confidence` (or a `ci_skipped_reason` below `min_samples`), SD, the
    distribution-free tolerance interval (or the runs it needs), and the median
    representative run;
  - `runs`, one `RunScore` per run behind the cells;
  - `champions`, when the arms were tuned;
  - `differences`, one `PairedDifference` per planned (block, pair). Each has an
    estimate, a CI at the Bonferroni-adjusted `comparison_confidence`, `n_paired`, a
    Holm-adjusted p-value, and the method. A skipped pair keeps its `skipped_reason`;
  - `family_size` and `comparison_confidence`;
  - the omnibus test, or `tests_skipped_reason` (no tests at all) or
    `omnibus_skipped_reason` (only the omnibus is absent);
  - `shared_across` (shared axis to the conditions whose runs join its every block)
    and `untuned` (the fixed referents);
  - `warnings`, as `BakeoffWarning` values: boundary champions (numeric axes only),
    insufficient seeds, non-normality, flatlined curves, zero variance, and
    `small_sample_coverage` on every cell or pair CI resting on fewer than 30 runs.
- **`HyperparameterConfigReport`** holds the winning configuration, raw and corrected
  means, the maximization bias, the CI and its confidence, the normalized selection
  entropy, `unpaired_run_ids`, the measure, and `warnings`: `small_sample_coverage`
  when the CI rests on fewer than 30 shared seeds.
- **`HyperparameterDifferenceReport`** follows the same conventions, including
  `warnings`. **`HyperparameterSliceReport`** reports point estimates only, with no
  interval and so no coverage warning.

## 4. Report document, schema version 1

```text
ReportDocument
  schema_version        1 (REPORT_SCHEMA_VERSION)
  status                ok | load_failed | analysis_failed
  plan_status           declared | exploratory
  exploratory_reasons   why the plan could not be shown to predate the results
  plan, plan_fingerprint  the AnalysisPlan and its sha256 over canonical JSON
  provenance            experiment, experiments_db, generated_at,
                        component_versions, executions (commit, dirty, host, times)
  accounting            per-cell planned/completed/failed/pending/diverged/excluded, unassigned
  fatal_diagnostics, non_fatal_diagnostics
  analyses              [{kind: bakeoff|best_hypers|hypers_difference|slice_hypers, name, [group], report}]
  failure               {analysis, error_type, message, non_finite_runs, incomplete_cells,
                        insufficient_data} | null
  plots                 {figure name: path relative to the report directory}
```

- `analyses` is empty unless `status` is `ok`. A document never carries partial
  results.
- In JSON, union members carry a `kind` tag. Measures appear as, for example,
  `{"kind": "tail_mean", "fraction": 0.1}`. Cell keys are objects keyed by axis.
- Serialization is strict: a NaN anywhere raises instead of writing invalid JSON.
- A breaking change to this shape increments `schema_version`.

## 5. Declared plans and the exploratory label

An `AnalysisPlan` fixes:

- the metric and the measure;
- the dimensions, version selection, and cohort;
- the analyses (`BakeoffAnalysis`, `BestHypersAnalysis`, `HypersDifferenceAnalysis`,
  `SliceHypersAnalysis`); a `BakeoffAnalysis` carries its `pairs`, `block_by`,
  `shared_across`, `select_champion_over`, and `untuned` declarations;
- the statistics, as a `BakeoffSpec`;
- `higher_is_better`, and optionally a worst-case score for diverged runs.

`@analysis_plan(spec_factory)` registers a plan beside its spec. The CLI's `run` and
`submit` commands append it to `<results_root>/analysis-plans.jsonl`. The history is
append-only: a plan that is unchanged is not re-recorded, and earlier entries are
never rewritten.

`run_plan(experiments_db, slug, plan, snapshots)` is the only producer of
`plan_status=declared`. It runs the plan verbatim, and requires that, for every
included execution, the latest snapshot at or before that execution's start has
the same fingerprint. Start times are compared in UTC; one without an offset is
never guessed. Otherwise the document is `exploratory`, with a reason per
problem naming the offending execution ids:

- "no plan snapshot before execution";
- "plan changed after results were recorded", with the fingerprint in force;
- the start time was not recorded, or has no UTC offset.

`read_plan_snapshots` validates each history entry's `recorded_at` strictly: an
offset-aware ISO timestamp is converted to UTC, and anything else (naive, or
malformed by a hand edit) is never guessed at. That snapshot comes back with
`recorded_at=None` and its `invalid_reason` instead of raising. `run_plan` then
excludes it from ordering and always labels the report exploratory, with
"plan history has an unusable snapshot: {invalid_reason}" among the reasons,
since an untrustworthy timestamp can hide a plan change at an unknown point in
the history.

`exploratory_report(dataset, experiments_db, results)` wraps results chosen by hand.
It is always `exploratory`.

## 6. Rendering

- `report_to_json(document)` produces strict, indented JSON.
- `render_markdown(document)` gives the audit copy. Status, the declared plan (when
  there is one), provenance, accounting, and diagnostics come first, then each
  analysis, then the figures. Every interval
  states its confidence, and nothing is written as `mean ± x`.
- `render_summary(document)` gives a few console lines.
- `write_report_document(document, output_dir)`, in `research_analysis.report_files`,
  writes the figures, `report.json`, and `report.md` into `output_dir` and returns the
  document naming its figures. `report_directory(results_root, experiment)` gives the
  default `<results_root>/reports/<experiment>/<UTC time>/`. The CLI uses both.
- `write_report_plots(document, output_dir)` draws per-block run distributions and
  sensitivity curves into `output_dir`, and returns the document with relative
  paths. `write_bakeoff_plots` and `write_slice_plot` draw single figures.
- Callers pass a directory under the durable results root.

## 7. Non-Goals

- Interactive rendering or servers.
- Writing to the experiment database. Apart from report output directories, the only
  file written is the append-only plan history, recorded by the CLI launch commands.
- Cross-environment aggregates (IQM, performance profiles, probability of
  improvement, optimality gap) in any default path. They remain opt-in functions in
  `research_analysis.aggregates`.
