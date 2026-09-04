# Empirical Design & Analysis Guide

This document codifies the standard for Reinforcement Learning experimentation. It serves as the definitive reference for implementing the `research-analysis` and `research-plot` libraries and for designing experiments within the monorepo.

---

## 1. Learning Curve Construction: Step-Weighted Returns

Most RL libraries plot learning curves by averaging the returns of the last $N$ episodes. **This is misleading.** Agents that solve an environment quickly (short episodes) will appear to learn "faster" than agents that take longer, even if their per-step performance is identical.

### The Algorithm: Interpolation via Copy-Forward
We represent performance as a function of **total learning steps**, not episodes. Every step taken by the agent must be assigned a performance value.

1.  **Data Ingress:** From the database, we retrieve pairs of `(cumulative_steps, episodic_return)`.
    - Example: `(50, -100), (120, -80), (180, -75)`.
2.  **Mapping:** For each episode $i$ that ended at step $T_i$ with return $R_i$, we assign the value $R_i$ to all steps in the range $[T_{i-1}, T_i)$.
3.  **Interpolation:** This produces a "step-weighted" curve of length $L$ (total experiment steps).
    - If Episode 1 took 50 steps and got -100, then indices `0-49` of the curve are `-100`.
    - If Episode 2 took 70 steps (ending at step 120) and got -80, then indices `50-119` are `-80`.

### Why this matters
This ensures that an agent taking 10,000 steps to fail once is weighted equally against an agent taking 10,000 steps to fail 200 times. It prevents "Short Episode Bias" and ensures learning curves are comparable across algorithms with different temporal characteristics.

---

## 2. Handling Hyperparameter Maximization Bias

Reporting the "best" performance of an algorithm after a hyperparameter sweep is prone to **Maximization Bias**. We are likely overestimating performance because we selected the hyperparameter configuration that had the "luckiest" random seeds.

### The Solution: Cross-Fitted Tuning
Instead of evaluating a selected setting on the same seeds used to select it, we
report held-out performance after cross-fitted tuning.

1.  **Collect Data:** Run $N$ seeds for every hyperparameter configuration $H$.
2.  **Cross-Fit:** For each seed, select the winner using all other seeds and
    evaluate that selected configuration on the held-out seed. Every tuning fold
    must contain every candidate value and exactly one held-out result.
3.  **Reporting:** Report the mean of held-out outcomes and a BCa interval with
    explicit confidence, resample, and minimum-sample settings.

This procedure captures the uncertainty in hyperparameter selection and provides a realistic estimate of the algorithm's performance in a "new" environment where tuning would be required.

---

## 3. Across-Environment Evaluation: The Small Control Benchmark (SHB)

Comparing algorithms across multiple tasks requires normalization, as returns are not on the same scale.

### Fixed-Reference Normalization
Normalization maps each score using fixed, predeclared reference anchors or
ranges for its environment. The references must not be recalculated when
algorithms or hyperparameter settings are added to a comparison.

---

## 4. Sensitivity Analysis: Slice vs. Best

Visualizing how performance changes with a hyperparameter (e.g., learning rate) requires reducing the other "nuisance" dimensions of the sweep.

### Strategy 1: The "Best" Curve
For each value of the target hyperparameter $\alpha$, show the performance of the **best achievable configuration** across all other hyperparameters.
- *Utility:* Shows the "peak" potential of the algorithm at that $\alpha$.

### Strategy 2: The "Slice" Curve
1.  Identify the single best overall configuration $(\alpha^*, \beta^*, \gamma^*)$.
2.  Plot performance as $\alpha$ varies, while keeping $\beta$ and $\gamma$ fixed to $\beta^*$ and $\gamma^*$.
- *Utility:* Shows the "brittleness" or "smoothness" of the parameter space around the optimum.

---

## 5. Statistical Primitives

### Non-Parametric BCa Bootstrapping
We do not assume performance is Gaussian. Scalar Confidence Intervals (CIs) use
BCa bootstrapping with explicit confidence and resample counts:
1.  Resample indices with replacement.
2.  Compute the statistic (mean, median, etc.).
3.  Repeat 10,000 times.
4.  CI is the bias-corrected and accelerated interval from the bootstrap distribution.

### Repeated-Block Tests in the Reporting API
The reporting API uses paired permutation post-hoc tests and a block-stratified
permutation omnibus equivalent to Skillings–Mack. It permutes observed cells
within each environment-seed block, retaining incomplete blocks rather than
silently listwise-deleting rows. Missingness is reported explicitly; insufficient
data must fail rather than produce a placeholder statistic or p-value.

### Supplementary Robust Aggregates
IQM may be shown as one supplementary center summary. It must appear alongside
dispersion, lower-tail, and constraint or safety metrics, and must never replace
the underlying task-level distributions.

### Non-Parametric Tolerance Intervals
To show performance coverage, use Tolerance Intervals.
- *Statement:* "With 95% confidence, this interval covers 90% of the population performance."
- *Implementation:* Use the binomial distribution PPF to find the rank-order indices in sorted data that satisfy the coverage requirement.

---

## 6. Plotting Philosophy

- **Data Lineage:** Every figure must be traceable to specific `ExecutionIDs`.
- **Intelligent Subsampling:** If plotting more than 30 lines (e.g., all seeds), prune to show only the top 5, bottom 5, and median 5 to maintain visual clarity while showing the full spread.
- **Minimalist Aesthetics:**
    - No top/right spines.
    - No legend frames.
    - Color consistency across all figures in a single project (Agent X is always Blue).
