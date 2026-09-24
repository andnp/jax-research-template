# Reading Results

There are two ways to read an experiment's results. Both go through the same
dataset boundary and the same statistics:

1. **The declared report.** Declare the primary analysis beside the spec, then run
   `research experiment report <spec_file.py>`. You get one versioned JSON document
   plus a markdown copy for people to audit. This is the default, and the only path
   that can label results *declared*. See [Auto-Research Loop](auto-research.md).
2. **Library calls.** Load an `ExperimentDataset` and call the reporting functions.
   Use this for exploratory questions. Wrap the results in `exploratory_report` so
   they carry the same provenance and accounting, labelled *exploratory*.

For custom SQL that is genuinely outside the dataset contract,
`research_analysis.loader.load_sqlite_query` is an escape hatch. It is not the
normal way to load results.

## Declared report

```python
# experiments/004_ablation.py, beside the @spec factory
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import ExperimentDimensions
from research_analysis.performance import TailMean
from research_analysis.plan import AnalysisPlan, BakeoffAnalysis, against_baseline, analysis_plan


@analysis_plan(ablation)  # the @spec factory this plan belongs to
def ablation_plan() -> AnalysisPlan:
    return AnalysisPlan(
        metric="return",
        measure=TailMean(0.1),
        dimensions=ExperimentDimensions(condition="algorithm", environment="env_name"),
        analyses=(
            BakeoffAnalysis(
                "headline",
                conditions=("rank", "shared"),
                pairs=against_baseline(("rank",), "shared"),
                block_by=("env_name",),
            ),
        ),
        stats=BakeoffSpec(alpha=0.05, n_resamples=10_000, random_seed=0, missingness="error"),
    )
```

`research experiment run` and `submit` append the plan to
`<results_root>/analysis-plans.jsonl` before any execution starts. `research
experiment report` then runs the plan exactly as written. It writes `report.json`,
`report.md`, and figures to `<results_root>/reports/<experiment>/<UTC time>/`.

The report is labelled `exploratory` when:

- no plan snapshot predates the first included execution ("no plan snapshot
  before execution"), or
- the plan in force at that moment differs from the one being run ("plan changed
  after results were recorded").

Exit codes are 0 for a report, 2 for fatal load diagnostics, and 3 for a failed
analysis, such as a diverged run with no declared worst case or an incomplete
block under `missingness="error"`. A failed document still carries the
accounting, the diagnostics, and the runs or cells that caused the failure. It
never carries partial results.

## Library entry points

| You want | Use |
|---|---|
| Typed runs, metrics, accounting, and diagnostics | `research_analysis.dataset.loading.load_experiment_dataset(experiments_db, slug, *, dimensions=..., version_selection=..., cohort=..., metrics=..., partial=...)` |
| A comparison between two or more conditions | `research_analysis.reporting.bakeoff_report(dataset, metric, conditions, spec=..., pairs=..., measure=...)` |
| A bias-corrected champion configuration | `best_hypers(dataset, metric, group_by=...)` |
| Sensitivity to one swept axis | `slice_hypers(dataset, metric, vary, group_by=...)` |
| Per-run scalar scores | `research_analysis.performance.run_scores(frame, metric, measure)` |
| A report document from hand-picked results | `research_analysis.report.exploratory_report(dataset, experiments_db, results)` |
| JSON, markdown, figures | `report_json.report_to_json`, `report_markdown.render_markdown`, `report_plots.write_report_plots(document, output_dir)` |
| A direct array comparison | `run_permutation_test(...)`, `bootstrap_difference_ci(...)` |

`research_analysis.dataset/__init__.py` is empty. Import from the submodules:
`dataset.loading` for the loader, `dataset.types` for `ExperimentDimensions`,
`SweepAxis`, `VersionSelection`, and the rest.

```python
from pathlib import Path

from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDimensions, SweepAxis, SweepValueType, VersionSelection
from research_analysis.report import BakeoffResult, exploratory_report
from research_analysis.report_markdown import render_markdown
from research_analysis.reporting import bakeoff_report

db = Path("projects/my-project/results/study/experiments.sqlite")
dataset = load_experiment_dataset(
    db,
    "study-slug",
    dimensions=ExperimentDimensions(
        condition="algorithm",
        environment="env_name",
        sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT),),
    ),
    version_selection=VersionSelection.latest_completed(),
)
report = bakeoff_report(dataset, "reward", ["baseline", "candidate"], block_by=["env_name"])
print(render_markdown(exploratory_report(dataset, db, [BakeoffResult("look", report)])))
```

The reporting functions only compute. They never print and never write files.
Figures come from `write_report_plots` (or `write_bakeoff_plots` /
`write_slice_plot`) with an explicit output directory. That directory belongs
under the durable results root, never under `/tmp` or the working directory.

## Read accounting and diagnostics before any number

`dataset.accounting` counts each cell's planned, completed, failed, pending,
diverged, and excluded runs, and counts planned runs that fit no cell.
`dataset.diagnostics` names every problem. Failed and pending runs are non-fatal
diagnostics. Malformed artifacts, missing hyperparameter configs, unresolved
dimensions, missing metric artifacts, and ambiguous cohorts are fatal, and
strict loading raises `DatasetLoadError`.

`partial=True` only lets you *see* a dataset with fatal problems. It does not
make them go away. The report path always loads partially, embeds every
diagnostic, and withholds all results when any diagnostic is fatal. Do not use
`partial=True` in library code just to get past a check.

If loading fails with an unresolved condition dimension, check which key labels
your arms. `ExperimentDimensions.condition` defaults to `condition_name`, so an
experiment that labels arms `algorithm` needs `ExperimentDimensions(condition="algorithm")`.

## Version selection is part of the analysis

`VersionSelection.latest_completed()` chooses one coherent component-version set
that completed runs represent. It does not take the maximum algorithm and
environment version independently, and it does not include runs that a later
source change left unexecuted.

**The component-version trap:** editing a declared component mints a new version.
Under `VersionSelection.all()`, runs of the broken and fixed code are pooled
silently into one cell. Use `latest_completed()` or `exact(...)` unless pooling is
the point, and check `EXCLUDED_VERSION` diagnostics and the report's component
versions before you interpret anything. Reproduce a historical cohort with
`VersionSelection.exact(...)` and
`ComponentVersionRef.by_id(...)`/`by_code_snapshot_hash(...)`.

## Diverged runs are outcomes

Runs whose metric went NaN or infinite stay in the dataset. They are listed in
`runs["non_finite_metrics"]` and counted as diverged in the accounting.
`run_scores`, and so every report, raises `NonFiniteValuesError` naming those
runs. You can declare a worst-case score with
`research_analysis.non_finite.substitute_non_finite(dataset, metric, worst_case)`
or with `AnalysisPlan.non_finite_worst_case`. The accounting still shows that the
runs diverged.

## Per-run scores

Every report reduces each run's curve to one number through a declared
`PerformanceMeasure`:

- `FinalValue()`: the last logged value.
- `TailMean(fraction)`: the mean over the final fraction of the logged step span.
- `ReturnRate()`: the mean over every logged point.

The step axis is the logged-curve index recorded by the runner, not an
environment step count. With subsampling, the final logged point may not be the
final training step.

## Tuned champions

Reporting the best raw mean on the seeds that selected it overstates
performance. `best_hypers` returns the raw champion mean, the corrected mean
from core's hierarchical per-sample bootstrap (the selection is re-run inside
every resample), the gap between the two (maximization bias), and a CI. That CI
describes a champion chosen by this procedure. It is not an estimate on held-out
seeds. There is no cross-fitting.

`bakeoff_report(..., select_champion_over=[...])` compares tuned arms by the same
hierarchical bootstrap. It runs no hypothesis tests on the seeds that selected
the champions. See [the science guide](science_guide.md) for why this method is
used rather than the handbook's two-stage bootstrap.

For experiment definition and rerun mechanics, see
[Running Experiments Without Wasting Compute](experiment-efficiency.md) and
[the experiment workflow](workflow.md). The full contracts are in
[the dataset API](specs/research-dataset-api.md) and
[the reporting API](specs/research-reporting-api.md).
