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
- **Explicit Design:** Require callers to declare the estimand, pairing, alpha,
  resampling, minimum sample size, and missingness policy.
- **Statistical Rigor:** Directly integrate monorepo primitives such as BCa
  confidence intervals, permutation tests, Holm correction, fixed-reference
  aggregates, and cross-fitted tuning.
- **JAX-Free Execution:** Maintain alignment with [ADR 009: Analysis Stack](../adrs/009-analysis-stack.md) using Polars, NumPy, and Numba instead of JAX.

`ComparisonSpec` is the canonical design contract. Callers can set the
estimand (`mean` or `median`), pairing (`auto`, `paired`, or `independent`),
alpha, resample count, minimum sample size, pairing keys, random seed, and
missingness policy (`error`, `complete_case`, or `blockwise`). Pairwise
`blockwise` missingness is rejected because it would require silently dropping
unmatched pairs; `blockwise` is reserved for repeated-block bakeoffs.

---

## 3. Specialized Entry Points

To avoid a monolithic API, the reporting framework exposes three distinct functions based on the user's research goal.

### 3.1 Pairwise Comparison: `compare_pairwise`

Used to compare exactly two experimental arms (e.g., a baseline and a proposed method or ablation).

- **Function Signature:**
  ```python
  def compare_pairwise(
      dataset: ExperimentDataset,
      condition_a: str,
      condition_b: str,
      metric_name: str,
      confidence_level: float = 0.95,
      verbose: bool = True,
      spec: ComparisonSpec | None = None,
  ) -> ABComparisonReport: ...
  ```
- **Analysis Steps:**
  1. Select final observations from the already materialized dataset.
  2. Use a paired or independent permutation test for the declared estimand.
  3. Compute a scalar BCa bootstrap interval with explicit confidence and
     resample settings.

### 3.2 Hyperparameter Sweep Analysis: `analyze_hypers`

Used to evaluate hyperparameter tuning sweeps, identify optimal configurations, and report parameter sensitivity.

- **Function Signature:**
  ```python
  def analyze_hypers(
      dataset: ExperimentDataset,
      target_hyperparameter: str,
      metric_name: str,
      group_by: list[str] | None = None,
      verbose: bool = True,
      confidence_level: float = 0.95,
      n_resamples: int = 10000,
      min_samples: int = 3,
      random_seed: int = 0,
  ) -> HyperparameterSensitivityReport: ...
  ```
- **Analysis Steps:**
  1. Read declared typed sweep axes and final observations from the dataset.
  2. Select each fold's winner on training seeds and estimate it on the held-out seed.
  3. Compute the raw winner mean, corrected winner mean, and estimated maximization bias.
  4. Generate sensitivity curve data using both the **Best** and **Slice** strategies described in the science guide.

### 3.3 Multi-Task Bakeoff: `compare_bakeoff`

Used to compare multiple algorithms ($K \ge 3$) across a benchmark suite of several environments.

- **Function Signature:**
  ```python
  def compare_bakeoff(
      dataset: ExperimentDataset,
      algorithms: list[str],
      environments: list[str],
      metric_name: str,
      verbose: bool = True,
      references: dict[str, ReferenceRange] | None = None,
      spec: ComparisonSpec | None = None,
  ) -> BenchmarkBakeoffReport: ...
  ```
- **Analysis Steps:**
  1. Select the requested algorithms and environments from the dataset.
  2. Apply caller-supplied fixed-reference normalization to scale returns across environments with different boundaries.
  3. Detect missing run data (e.g., if certain seeds/tasks failed or are incomplete).
  4. Perform a block-stratified permutation omnibus test equivalent in purpose
     to Skillings–Mack for incomplete blocks without listwise deletion. Require
     explicit minimum data.
  5. Run paired permutation post-hoc tests with true Holm step-down correction.

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
class ABComparisonReport:
    experiment_name: str
    condition_a: str
    condition_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    difference_in_means: float
    difference_ci: tuple[float, float]
    estimand: Literal["mean", "median"]
    pairing: Literal["paired", "independent"]
    alpha: float
    n_resamples: int
    min_samples: int
    missingness: str
    estimate_a: float
    estimate_b: float
    difference_in_estimand: float
    test_details: StatisticalTestDetails
    distribution_plot_path: Path | None  # Path to saved distribution PDF/violin plot

@dataclass(frozen=True)
class HyperparameterSensitivityReport:
    target_hyperparameter: str
    winning_value: Any
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    sensitivity_slice: dict[Any, float]  # Value -> performance
    sensitivity_best: dict[Any, float]   # Value -> performance
    sensitivity_plot_path: Path | None   # Path to saved sensitivity curve PDF
    estimate_method: str
    estimate_ci: tuple[float, float]
    confidence_level: float

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
