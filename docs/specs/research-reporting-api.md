# Technical Specification: Research Reporting API

## 1. Overview

Reinforcement Learning (RL) researchers—especially graduate students—frequently face challenges when conducting rigorous statistical analysis of experiment data. Common issues include hyperparameter maximization bias, using statistically invalid averages across heterogeneous tasks, and misapplying hypothesis tests (e.g., performing unpaired tests on paired seed runs).

The `research-analysis` reporting API introduces specialized entry points to automate statistical decision-making, enforce monorepo standards defined in the [Empirical Design & Analysis Guide (science_guide.md)](file:///home/andy/Projects/research/research-monorepo/core/docs/science_guide.md), and produce both structured dataclass outputs and human-readable, pedagogical console outputs using `rich`.

This API lives in a new module: [reporting.py](file:///home/andy/Projects/research/research-monorepo/core/libs/research-analysis/src/research_analysis/reporting.py).

---

## 2. Goals

- **Pedagogical Value:** Explain the *why* and *how* behind statistical test selections to guide researchers through correct empirical interpretation.
- **Automated Decision-Making:** Automatically inspect database schemas and data properties to pick correct tests (e.g., paired vs. unpaired, parametric vs. non-parametric, handling missing data).
- **Statistical Rigor:** Directly integrate monorepo primitives such as non-parametric tolerance intervals, bootstrapped confidence intervals, and two-stage tuning.
- **JAX-Free Execution:** Maintain alignment with [ADR 009: Analysis Stack](file:///home/andy/Projects/research/research-monorepo/core/docs/adrs/009-analysis-stack.md) using Polars, NumPy, and Numba instead of JAX.

---

## 3. Specialized Entry Points

To avoid a monolithic API, the reporting framework exposes three distinct functions based on the user's research goal.

### 3.1 Pairwise Comparison: `compare_pairwise`

Used to compare exactly two experimental arms (e.g., a baseline and a proposed method or ablation).

- **Function Signature:**
  ```python
  def compare_pairwise(
      db_path: Path | str,
      experiment_slug: str,
      arm_a: str,
      arm_b: str,
      metric_name: str,
      confidence_level: float = 0.95,
      verbose: bool = True,
  ) -> ABComparisonReport: ...
  ```
- **Analysis Steps:**
  1. Detect if random seeds are shared across the two arms.
  2. Perform Shapiro-Wilk normality tests on both arms' distributions.
  3. Select the test:
     - **Paired & Non-Normal:** Wilcoxon Signed-Rank test.
     - **Paired & Normal:** Paired t-test.
     - **Unpaired & Non-Normal:** Mann-Whitney U-test (via [mann_whitney_u_test](file:///home/andy/Projects/research/research-monorepo/core/libs/research-analysis/src/research_analysis/hypothesis.py#L95)).
     - **Unpaired & Normal:** Welch's t-test (via [welch_ttest](file:///home/andy/Projects/research/research-monorepo/core/libs/research-analysis/src/research_analysis/hypothesis.py#L148)).
  4. Compute bootstrapped confidence intervals of the difference in means (via [bootstrap_ci](file:///home/andy/Projects/research/research-monorepo/core/libs/research-analysis/src/research_analysis/bootstrap.py#L45)).

### 3.2 Hyperparameter Sweep Analysis: `analyze_hypers`

Used to evaluate hyperparameter tuning sweeps, identify optimal configurations, and report parameter sensitivity.

- **Function Signature:**
  ```python
  def analyze_hypers(
      db_path: Path | str,
      experiment_slug: str,
      target_hyperparameter: str,
      metric_name: str,
      group_by: list[str] | None = None,
      verbose: bool = True,
  ) -> HyperparameterSensitivityReport: ...
  ```
- **Analysis Steps:**
  1. Retrieve all tuning configurations from `HyperparamConfigs` for the experiment.
  2. Implement **Bootstrapped Two-Stage Tuning** to correct for maximization bias by resampling winner configurations.
  3. Compute the raw winner mean, corrected winner mean, and estimated maximization bias.
  4. Generate sensitivity curve data using both the **Best** and **Slice** strategies described in the science guide.

### 3.3 Multi-Task Bakeoff: `compare_bakeoff`

Used to compare multiple algorithms ($K \ge 3$) across a benchmark suite of several environments.

- **Function Signature:**
  ```python
  def compare_bakeoff(
      db_path: Path | str,
      experiment_slug: str,
      algorithms: list[str],
      environments: list[str],
      metric_name: str,
      verbose: bool = True,
  ) -> BenchmarkBakeoffReport: ...
  ```
- **Analysis Steps:**
  1. Retrieve curves across all tasks and algorithms.
  2. Apply ECDF-based (Empirical Cumulative Distribution Function) normalization to scale returns across environments with different return boundaries.
  3. Detect missing run data (e.g., if certain seeds/tasks failed or are incomplete).
  4. Perform the **Skillings-Mack test** (robust to missing data) or the **Friedman test** (if data is complete) as an omnibus check.
  5. If the omnibus test is significant, run post-hoc pairwise Wilcoxon tests with Bonferroni-Holm family-wise error rate corrections.

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

@dataclass(frozen=True)
class ABComparisonReport:
    experiment_name: str
    arm_a: str
    arm_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    difference_in_means: float
    difference_ci: tuple[float, float]
    test_details: StatisticalTestDetails

@dataclass(frozen=True)
class HyperparameterSensitivityReport:
    target_hyperparameter: str
    winning_value: Any
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    sensitivity_slice: dict[Any, float]  # Value -> performance
    sensitivity_best: dict[Any, float]   # Value -> performance

@dataclass(frozen=True)
class BenchmarkBakeoffReport:
    algorithms: list[str]
    environments: list[str]
    ecdf_scores: dict[str, dict[str, float]]  # Algo -> Env -> ECDF
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float]]  # Algo A -> Algo B -> Adjusted P-value
```

---

## 5. Pedagogical Console Output Design

When `verbose=True`, the reporter writes to the standard output using `rich` console panels. Every report must output:
1. **Design Summary Panel:** Outlining identified independent/dependent variables and sample structure (e.g., repeated measures availability).
2. **Methodology Justification Box:** Explaining *why* a particular test was chosen (e.g., "We selected the Wilcoxon Signed-Rank test instead of a paired t-test because the Shapiro-Wilk test rejected normality (p=0.003), and random seeds are paired").
3. **Findings & Interpretation Section:** Presenting statistics, significance, and a plain-English translation of p-values (e.g., "This p-value indicates a 1.2% probability of observing this performance gap purely due to random seed variation under identical algorithms").
4. **Correction Logs (For Hypers):** Explaining maximization bias and showing the contrast between raw averages and corrected estimates.

---

## 6. Non-Goals

- **Plot Generation:** This API does not write files or render matplotlib figures directly. Visualizations must be handled separately by importing the data into `research-plot`.
- **Dynamic DB Writing:** This is a read-only API. It does not modify logical runs, hyperparameter configs, or execution metadata tables in the sqlite registry.
