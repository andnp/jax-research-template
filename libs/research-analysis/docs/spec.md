# PRD: `libs/research-analysis`

## 1. Overview

`research-analysis` turns recorded RL experiments into statistics an agent can act on
and a person can audit. Its methodology follows Patterson, Neumann, White & White,
"Empirical Design in Reinforcement Learning" (JMLR 2024). The decisions are recorded in
[ADR 012](../../../docs/adrs/012-percentile-champion-aggregate-decisions.md) and
explained in the [science guide](../../../docs/science_guide.md).

### Goals

- **Nothing is dropped silently.** Every planned run is accounted for per cell, as
  completed, failed, pending, diverged, or excluded with a reason.
- **Honest uncertainty.** Resampling is over runs. Intervals are percentile-bootstrap
  intervals with stated confidence and are multiplicity-adjusted across a comparison
  family. Variability is reported apart from confidence.
- **Selection-aware tuning.** Tuned champions carry selection uncertainty through the
  hierarchical per-sample bootstrap.
- **Self-describing results.** Reports serialize to a versioned JSON document with
  provenance, accounting, diagnostics, the declared plan and its status, and results.
- **Declared before observed.** A report is labelled declared only when a plan recorded
  before the runs was run verbatim.

## 2. Modules

Packages keep empty `__init__.py` files, so import from the submodules.

| Module | Responsibility |
|---|---|
| `dataset.loading`, `dataset.types` | `load_experiment_dataset` and its types: runs, metrics, provenance, execution provenance, diagnostics, `RunAccounting` |
| `diagnostics.types`, `diagnostics.bakeoff` | loader diagnostics (fatal and non-fatal), bakeoff warnings, `IncompleteBlocksError` |
| `performance` | declared per-run measures (`FinalValue`, `TailMean`, `ReturnRate`) and `run_scores`, which raises on non-finite scores |
| `non_finite` | `NonFiniteValuesError`, `require_finite`, `substitute_non_finite` |
| `comparison` | `BakeoffSpec`: estimand, alpha, resamples, `min_samples`, missingness, seed, tolerance settings |
| `reporting` | compute-only `bakeoff_report`, `best_hypers`, `slice_hypers` and their result records |
| `plan`, `plan_snapshots`, `plan_report` | `AnalysisPlan` and `@analysis_plan`, the append-only plan history, `run_plan` |
| `report`, `report_json`, `report_markdown`, `report_plots`, `report_files` | the `ReportDocument` (schema v1), strict JSON and plan fingerprints, markdown and console summary, figures into an explicit directory, and one call writing a whole report directory |
| `bootstrap`, `permutation`, `repeated_blocks`, `multiplicity` | percentile bootstrap CIs, paired or independent permutation tests, the block-stratified omnibus test, Holm and Bonferroni |
| `tuning` | the hierarchical per-sample bootstrap for tuned champions |
| `statistics`, `curves`, `learning_curve`, `trajectory` | tolerance intervals, representative runs, curve summaries, step-weighted returns, trajectory validation |
| `aggregates` | opt-in cross-environment aggregates (IQM, performance profiles, probability of improvement, optimality gap, fixed-reference normalization), never in a default path |
| `loader` | `load_sqlite_query`, an escape hatch for SQL outside the dataset contract |

## 3. Usage

The declared path is `research experiment report <spec_file.py>`. See the
[reporting API](../../../docs/specs/research-reporting-api.md) and
[Reading Results](../../../docs/reading-results.md). In code:

```python
from pathlib import Path

from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDimensions
from research_analysis.report import BakeoffResult, exploratory_report
from research_analysis.report_files import report_directory, write_report_document
from research_analysis.reporting import bakeoff_report

db = Path("projects/my-project/results/study/experiments.sqlite")
dataset = load_experiment_dataset(db, "study-slug", dimensions=ExperimentDimensions(condition="algorithm"))
report = bakeoff_report(dataset, "reward", ["baseline", "candidate"])
difference = report.difference("candidate", "baseline")
document = exploratory_report(dataset, db, [BakeoffResult("look", report)])
write_report_document(document, report_directory(db.parent, document.provenance.experiment))
```

## 4. Technical Constraints

- **Stack:** NumPy and Numba for kernels and Polars for data (ADR 009). No JAX, no Pandas.
- **Database:** reads the relational SQLite schema of ADR 008 through
  `experiment_definition`, and metric databases only from recorded artifact metadata.
- **No side effects in compute:** reporting functions never print and never write.
  Rendering and plotting take an explicit output directory.
- **Multiplicity:** difference CIs are Bonferroni-adjusted across the planned family,
  and p-values use Holm's step-down adjustment. Skipped comparisons still count in
  the family.
- **Determinism:** reports and plans take an explicit random seed (`BakeoffSpec.random_seed`),
  so a rerun of the same plan on the same runs reproduces its numbers.
