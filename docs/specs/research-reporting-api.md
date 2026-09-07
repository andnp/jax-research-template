# Technical Specification: Research Reporting API

## 1. Overview

Reinforcement Learning (RL) researchers—especially graduate students—frequently face challenges when conducting rigorous statistical analysis of experiment data. Common issues include hyperparameter maximization bias, using statistically invalid averages across heterogeneous tasks, and misapplying hypothesis tests (e.g., performing unpaired tests on paired seed runs).

The `research-analysis` reporting API introduces specialized entry points to automate statistical decision-making, enforce monorepo standards defined in the [Empirical Design & Analysis Guide](../science_guide.md), and produce both structured dataclass outputs and human-readable, pedagogical console outputs using `rich`.

The data boundary lives in [dataset.py](../../libs/research-analysis/src/research_analysis/dataset.py); reporting lives in [reporting.py](../../libs/research-analysis/src/research_analysis/reporting.py).

## 2.1 Dataset boundary

Callers first build an `ExperimentDataset` with
`load_experiment_dataset(experiments_db, slug, *, dimensions, version_selection,
cohort, metrics, partial)`. The dataset owns run selection, component
provenance, artifact lookup, metric materialization, and diagnostics. Reporting
does not accept database paths, experiment slugs, `AnalysisSchema`, or execution
roots.

The canonical tables contain `condition_name`, `seed`, `run_id`,
`execution_id`, and `ablation`; `metrics` additionally contains `step`,
`metric`, and `value`. An optional `env_name` and declared typed sweep-axis
columns are included when requested. Curves may impose additional step-grid
requirements for operations that need aligned trajectories.

---

## 2. Goals

- **Pedagogical Value:** Explain the *why* and *how* behind statistical test selections to guide researchers through correct empirical interpretation.
- **Explicit Design:** Require bakeoff callers to declare the estimand, alpha,
  resampling, minimum sample size, and missingness policy.
- **Statistical Rigor:** Directly integrate monorepo primitives such as BCa
  confidence intervals, permutation tests, Holm correction, fixed-reference
  aggregates, and cross-fitted tuning.
- **JAX-Free Execution:** Maintain alignment with [ADR 009: Analysis Stack](../adrs/009-analysis-stack.md) using Polars, NumPy, and Numba instead of JAX.

`BakeoffSpec` is the design contract for `bakeoff_report`. Callers can set the
estimand (`mean` or `median`), alpha, resample count, minimum sample size,
random seed, and missingness policy (`error`, `complete_case`, or `blockwise`).
Bakeoff comparisons always pair observations by block dimensions and seed; the
low-level permutation and bootstrap functions are the separate entry points for
choosing paired or independent designs directly.

---

## 3. Reporting entry points

`bakeoff_report(dataset, metric_name, conditions, ...)` is the canonical
comparison entry point. It handles two or more conditions, optionally grouped
by block dimensions such as environments, and always uses paired block-and-seed
observations. It can also select champions over declared sweep axes before the
comparison. Supply `BakeoffSpec` when the estimand, resampling, or missingness
policy needs to differ from the defaults.

`best_hypers(dataset, metric_name, ...)` finds a bias-corrected champion for a
declared sweep. `slice_hypers(dataset, metric_name, vary, ...)` reports the
sensitivity curve for one axis. These functions own hyperparameter selection;
`bakeoff_report` composes them when `select_champion_over` is requested.

For a direct comparison of two arrays, use the low-level
`run_permutation_test` and `bootstrap_difference_ci` functions. Their `pairing`
argument is the appropriate place to choose paired or independent inference.

---

## 4. Structured Output Data Schemas

The API returns structured dataclasses to support programmatic plotting, paper-writing scripts, or custom dashboards.

```python
@dataclass(frozen=True)
class StatisticalTestDetails:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    is_significant: bool
    assumptions: dict[str, Any]
    justification: str
    intermediate_tests: dict[str, StatisticalTestDetails]

@dataclass(frozen=True)
class BakeoffReport:
    conditions: list[str]
    metric_name: str
    seed_counts: dict[str, dict[str, int]]
    cell_means: dict[str, dict[str, float]]
    cell_variances: dict[str, dict[str, float]]
    champions: dict[str, dict[str, HyperparameterConfigReport]] | None
    comparison: BenchmarkBakeoffReport
    warnings: tuple[BakeoffWarning, ...]

@dataclass(frozen=True)
class BenchmarkBakeoffReport:
    algorithms: list[str]
    environments: list[str]
    normalized_scores: dict[str, dict[str, float]]  # Algo -> Env -> score
    performance_profiles: dict[str, dict[float, float]]
    probability_of_improvement: dict[str, dict[str, float | None]]
    optimality_gaps: dict[str, float]
    dispersion: dict[str, float]
    lower_tail: dict[str, float]
    supplementary_iqm: dict[str, float]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float | None]]  # None means unavailable
    aggregate_plot_path: Path | None
```

Bootstrap primitives report `method="BCa"` for ordinary intervals or
`method="degenerate-point"` for a finite point interval whose declared
statistic is exactly constant. Non-degenerate undefined BCa intervals remain
errors.

---

## 5. Pedagogical Console Output Design

When `verbose=True`, the reporter writes to the standard output using `rich` console panels. Every report must output:
1. **Design Summary Panel:** Outlining identified independent/dependent variables and sample structure (e.g., repeated measures availability).
2. **Methodology Justification Box:** Explaining the declared estimand, pairing,
   permutation design, confidence level, resample count, and missingness policy.
3. **Findings & Interpretation Section:** Presenting statistics, significance, and a plain-English translation of p-values (e.g., "This p-value indicates a 1.2% probability of observing this performance gap purely due to random seed variation under identical algorithms").
4. **Correction Logs (For Hypers):** Explaining maximization bias and showing the contrast between raw averages and corrected estimates.
5. **Artifact Outputs Panel:** Surfacing the file paths of generated diagnostic distribution plots (e.g. violin plots or PDFs) saved in the experiment's analysis directory.

---

## 6. Non-Goals

- **Interactive GUI Rendering:** This API does not launch interactive GUI windows or web servers. Plotting is restricted to generating and saving static diagnostic visual artifacts (including ECDF diagnostic curves) directly to the file system using `research-plot`.
- **Dynamic DB Writing:** This is a read-only API. It does not modify logical runs, hyperparameter configs, or execution metadata tables in the sqlite registry.
