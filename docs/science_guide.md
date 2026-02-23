# Empirical Design & Analysis Guide

This document codifies the "Patterson Standard" for Reinforcement Learning experimentation. It serves as the definitive reference for implementing the `research-analysis` and `research-plot` libraries and for designing experiments within the monorepo.

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

### The Solution: Bootstrapped Two-Stage Tuning
Instead of a single "best" curve, we report the distribution of "likely winners."

1.  **Collect Data:** Run $N$ seeds for every hyperparameter configuration $H$.
2.  **Bootstrap Resampling (Outer Loop):**
    - For $M$ iterations (e.g., 1000):
        - For each configuration $h \in H$, resample $N$ runs (with replacement) and compute their mean performance.
        - Identify the "Winner": the configuration $h^*$ with the highest resampled mean.
        - Record the **true sample mean** (from the original data) of $h^*$ for this iteration.
3.  **Reporting:** Report the mean and confidence interval of these $M$ recorded "winner" performances.

This procedure captures the uncertainty in hyperparameter selection and provides a realistic estimate of the algorithm's performance in a "new" environment where tuning would be required.

---

## 3. Across-Environment Evaluation: The Small Control Benchmark (SHB)

Comparing algorithms across multiple tasks requires normalization, as returns are not on the same scale.

### CDF-based Normalization
We use an Empirical Cumulative Distribution Function (ECDF) to map any raw score $x$ in environment $e$ to a value in $[0, 1]$.

$$N(x, e) = \frac{1}{|\mathcal{P}_e|} \sum_{g \in \mathcal{P}_e} \mathbb{1}(g < x)$$

Where $\mathcal{P}_e$ is the **pool** of all scores observed in environment $e$ across **all** algorithms and **all** hyperparameter settings tested in the study.
- A value of $0.5$ means the agent outperformed 50% of all other tested configurations on that task.
- This accounts for task difficulty and highlights relative algorithmic improvement.

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

### Non-Parametric Percentile Bootstrapping
We do not assume performance is Gaussian. All Confidence Intervals (CIs) must be generated via bootstrapping:
1.  Resample indices with replacement.
2.  Compute the statistic (mean, median, etc.).
3.  Repeat 10,000 times.
4.  CI is the $(\frac{\alpha}{2}, 1-\frac{\alpha}{2})$ percentiles of the bootstrap distribution.

### Skillings-Mack Test
When comparing $K$ algorithms across $T$ tasks, and especially if some results are missing, use the Skillings-Mack test. It is a non-parametric alternative to Friedman’s test that handles missing data correctly.

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
