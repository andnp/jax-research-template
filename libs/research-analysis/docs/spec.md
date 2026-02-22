# PRD: `libs/research-analysis`

## 1. Overview
`research-analysis` is a rigorous statistical library designed for Reinforcement Learning research. It prioritizes scientific validity and statistical evidence over simple performance aggregation. The library is built upon the principles of **Empirical Design in Reinforcement Learning** (Patterson et al., 2023).

### Goals
- **Statistical Rigor:** Provide tools to identify and quantify sources of variation in RL experiments.
- **Robust Comparisons:** Facilitate proper hypothesis testing and policy comparison with sound statistical assumptions.
- **Reproducible Evidence:** Ensure that results reported in papers are backed by strong statistical evidence.
- **NO IQM:** Explicitly avoid Inter-Quantile Mean (IQM) and other metrics that mask performance variation or stability issues.

## 2. Core Features

### 2.1 Variation & Stability Analysis
- Tools to characterize performance variation across seeds, environments, and hyperparameter settings.
- First-class support for reporting full distributions rather than single-point estimates.

### 2.2 Policy Comparison & Hypothesis Testing
- Implementation of statistical tests with appropriate assumptions for RL data (e.g., handling non-normality and autocorrelation).
- Support for **Skillings-Mack tests** and other robust methods for comparing multiple agents across different tasks.
- Automated confidence interval (CI) generation using appropriate bootstrapping techniques.

### 2.3 Experimental Design Primitives
- Support for sophisticated experimental designs in the `define` phase (e.g., Factorial designs, Latin Hypercube Sampling).
- Tools to detect and mitigate **Experimenter Bias** and overfitting to specific seeds/environments.

### 2.4 Baseline Construction
- Utilities for creating and managing statistically sound baselines for comparative studies.

## 3. Technical Constraints
- Must interface directly with the relational SQLite schema defined in ADR 008.
- Should be optimized for analyzing data from large-scale JAX `vmap` runs.
- **Exclusion Policy:** No implementation or support for Inter-Quantile Mean (IQM) shall be included in this library.

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
