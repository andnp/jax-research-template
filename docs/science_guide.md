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

Reporting the best performance after a sweep suffers from **maximization bias**: the
chosen configuration is partly the one whose seeds were luckiest. Comparing tuned
champions by their raw argmax means, on the seeds that selected them, overstates both
the performance and the certainty.

### The method: hierarchical per-sample bootstrap
`research_analysis.tuning` (behind `best_hypers` and `bakeoff_report(...,
select_champion_over=...)`) puts the uncertainty in which configuration wins into the
interval, in two levels:

1.  **Collect data:** run the same seeds under every configuration in the grid.
    Configurations are paired by seed, and runs whose seed is missing from some
    configuration are listed as `unpaired_run_ids` and left out.
2.  **Selection distribution:** repeatedly resample runs with replacement for every
    configuration and record which configuration has the best resampled mean. This
    gives the empirical probability that each configuration would be chosen, and its
    entropy.
3.  **Performance under selection:** each bootstrap mean averages resampled runs,
    drawing the configuration **per sample** from the selection distribution. That
    averages over the whole selection mixture instead of committing each replicate to
    a single winner.
4.  **Report:** the corrected mean and its percentile interval, the raw champion mean,
    their gap (the maximization bias), and the normalized selection entropy (0 when
    one configuration always wins).

This is **not** cross-fitting. No seed is held out, and the estimate is not a
prediction for fresh seeds. It describes the performance of a champion chosen by this
procedure on this seed budget, with its selection uncertainty included.

### Why not the handbook's bootstrapped two-stage method
`projects/hyper-champion-ci` (experiment 001, diary entries of 2026-09-05) simulated
one-stage, bootstrapped two-stage, LOOCV, and two hierarchical variants against known
ground truth, across several run distributions. The two-stage method's coverage
collapsed as the number of configurations grew. LOOCV under-covered everywhere. The
hierarchical per-sample variant was the best-calibrated method, measured as closeness
of coverage to nominal, with narrower intervals than the per-replicate variant. Core
therefore keeps it as the one tuned-champion method.

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

### Unit of resampling
The independent unit is the **run (seed)**. Every interval and test resamples or
permutes whole runs, never episodes or steps within a run.

### Percentile bootstrap
Scalar and paired-difference confidence intervals use the **percentile bootstrap**,
with explicit confidence, resample count, minimum samples, and RNG seed. It does not
assume Gaussian performance. There is no BCa path. A cell with fewer runs than
`BakeoffSpec.min_samples` reports its estimate and n with no interval, and says why.

The percentile bootstrap **under-covers at small n**. Its intervals are narrower than
their stated confidence with 10 runs, and a Bonferroni family reaches its nominal joint
coverage only with many runs. The Monte-Carlo suite in
`tests/large/test_research_analysis_calibration.py` measures this. Reports attach a
`small_sample_coverage` warning to every confidence interval that rests on fewer than
30 runs: bakeoff cells and pairs, tuned champions, and champion differences.
The remedy is more runs (handbook §2.6), not a cleverer interval.

### Headline comparison: paired differences, adjusted for multiplicity
A comparison is a seed-paired difference with its percentile CI: an effect size, not a
verdict. The estimand applies to the per-seed differences, so a paired median is
median(A − B), not median(A) − median(B), and the point estimate, CI, and test all use
it. Only an independent comparison differences each arm's estimand. Across a report's family of planned comparisons (every block × pair, or only
the declared `pairs`), the CIs are Bonferroni-adjusted so that they hold jointly at the
stated confidence, and each one states its own level. Hypothesis tests are secondary.
They are paired permutation tests with Holm-adjusted p-values, where skipped pairs
still count in the family.

### Repeated-block omnibus test
The omnibus test is a block-stratified permutation test equivalent to Skillings–Mack.
It permutes observed cells within each environment-seed block and keeps incomplete
blocks rather than listwise-deleting them. Under the default `missingness="error"`, an
incomplete block fails the report, naming the missing seeds and whether they failed or
are pending. `complete_case` and `blockwise` are declared choices, never fallbacks.

### Variability is reported separately from confidence
Never write `mean ± x`. A CI describes uncertainty about the estimate. Run-to-run
variability is reported separately:

- **SD** across runs.
- **Distribution-free tolerance intervals.** "With 95% confidence, this interval
  covers 90% of runs." The rank indices come from the binomial distribution. A cell
  with too few runs for the requested (confidence, coverage) states how many it needs.
- **The median representative run**: the run whose score is closest to the cell
  median, to inspect as a curve.

### Diverged runs
A run whose metric went NaN or infinite is an outcome (handbook §7, "Discarding or
replacing runs"). It is counted as diverged in the accounting and makes every
statistic raise until a worst-case score is declared. It is never dropped or re-seeded.

### Opt-in aggregates across environments
IQM, performance profiles, probability of improvement, and optimality gap exist for
summarizing a collection of environments. None of them is a default. IQM in particular
trims away catastrophic runs, which are exactly the failures a comparison must show,
and it only makes sense across environments, never within one cell. Any aggregate must
appear beside the per-environment distributions it summarizes.

## 6. Plotting Philosophy

- **Data Lineage:** A report's figures are drawn from its results: run-distribution
  figures from the bakeoff's `runs`, each of which carries its run id, and sensitivity
  figures from the slice result's values. The PNG embeds no ids. The `report.json` in
  the same directory, which names every figure under `plots`, is the lineage record.
- **Show every run:** Distribution figures draw every run; nothing is subsampled.
- **Minimalist Aesthetics:**
    - No top/right spines.
    - No legend frames.
    - Color consistency across all figures in a single project (Agent X is always Blue).
