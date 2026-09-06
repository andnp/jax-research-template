# PRD: `libs/research-analysis`

## 1. Overview
`research-analysis` is a rigorous statistical library designed for Reinforcement Learning research. It prioritizes scientific validity and statistical evidence over simple performance aggregation. The library is built upon the principles of **Empirical Design in Reinforcement Learning** (Patterson et al., 2023).

### Goals
- **Statistical Rigor:** Provide tools to identify and quantify sources of variation in RL experiments.
- **Robust Comparisons:** Facilitate proper hypothesis testing and policy comparison with sound statistical assumptions.
- **Reproducible Evidence:** Ensure that results reported in papers are backed by strong statistical evidence.
- **Supplementary IQM:** IQM may be reported as a supplementary aggregate, never
  as the sole result. Reports must also include dispersion, lower-tail, and
  relevant constraint or safety metrics.

## 2. Core Features

### 2.1 Variation & Stability Analysis
- Tools to characterize performance variation across seeds, environments, and hyperparameter settings.
- First-class support for reporting full distributions rather than single-point estimates.

### 2.2 Policy Comparison & Hypothesis Testing
- Implementation of explicit mean or median estimands with declared paired or
  independent designs. No normality pretest selects the method.
- Paired or independent permutation tests for pairwise comparisons, plus a
  block-stratified permutation omnibus equivalent for incomplete repeated blocks.
  It retains observed cells within each block instead of listwise deletion.
- BCa confidence intervals with explicit confidence, resample, and minimum-sample
  settings. Insufficient or invalid data must fail clearly.

### 2.3 Experimental Design Primitives
- Support for sophisticated experimental designs in the `define` phase (e.g., Factorial designs, Latin Hypercube Sampling).
- Tools to detect and mitigate **Experimenter Bias** and overfitting to specific seeds/environments.

### 2.4 Baseline Construction
- Utilities for creating and managing statistically sound baselines for comparative studies.

### 2.5 Canonical Experiment Dataset
`research_analysis.dataset.load_experiment_dataset` is the only boundary between
the experiment database, recorded metric artifacts, and statistical reporting.
It returns an `ExperimentDataset` containing typed `runs` and long-form `metrics`
Polars frames, component provenance, selected dimensions, and structured
diagnostics. Reporting functions consume that object; they do not open SQLite
databases or resolve execution paths themselves.

Callers declare source dimensions and sweep axes explicitly:

```python
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
report = compare_pairwise(dataset, "baseline", "candidate", "reward")
```

For a parameter declared only under an experiment condition, declare the same
applicability rule on its axis. Applicable runs must contain a correctly typed
value; other runs receive a null value. Runs that cannot satisfy the requested
dimension projection are omitted:

```python
SweepAxis("eta", SweepValueType.FLOAT, when={"algorithm": "adversarial"})
```

The loader selects one coherent completed component-version set by default.
Historical selection requires qualified component identity plus a version row ID
or recorded code hash. It retains ablation and run/execution identity, reads
metrics only from recorded `metadata_json["metrics_db_path"]`, and fails in
strict mode for ambiguous cohorts or missing artifacts. `partial=True` returns
valid rows with `DatasetDiagnostic` records for the omitted data.

## 3. Technical Constraints
- **Core Stack:** Implementation must use **Numpy** and **Numba** (nopython mode) for high-performance statistical kernels. JAX is NOT required for analysis and should be avoided to minimize dependencies for laptop-based analysis.
- **Data Engine:** **Polars** is the exclusive first-class citizen for data management. All data ingestion from SQLite must go through Polars DataFrames. No Pandas support will be implemented.
- **Database:** Must interface directly with the relational SQLite schema defined in ADR 008.
- **Normalization Policy:** Benchmark normalization must use caller-supplied
  fixed reference anchors or ranges, never a comparison-set-derived pool.
- **Multiplicity Policy:** Pairwise post-hoc p-values use true Holm step-down
  adjustment, not a fixed per-comparison correction.


## 4. Proposed Usage

```python
from research_analysis import analyzer

# Load results from the master DB
results = analyzer.load_experiment("exp_v1_cartpole")

# Statistical Comparison between two algorithms
# This uses robust tests as advocated in Patterson et al.
comparison = results.compare("ppo_v1", "dqn_v2", metric="total_reward")
print(comparison.p_value)
print(comparison.confidence_intervals)

# Plot full distributions to see stability (not just means)
results.plot_distribution("ppo_v1", "total_reward")
```
