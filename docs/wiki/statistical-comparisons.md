# Statistical Comparisons

This note explains how `research-analysis` compares conditions, and why. The short
version:

- **Compare paired differences, not two separate summaries.** Runs of different
  conditions that share a seed are paired, and the headline is the distribution of
  their differences.
- **Report the effect with a confidence interval.** The estimate is the mean (or
  median) paired difference, with a percentile-bootstrap CI at a stated confidence.
- **Adjust for the number of comparisons.** Every CI in a report's family is
  Bonferroni-adjusted, so the family holds jointly at the stated confidence.
- **Tests are secondary.** Paired permutation p-values, Holm-adjusted, and a
  block-stratified omnibus test are reported beside the intervals, not instead of them.

## Why paired

Seeds control the sources of randomness a simulation shares across conditions:
initial weights, environment starts, replay sampling. When both conditions see the same
seed, much of the run-to-run noise is common to both. It cancels in the difference, so
a paired comparison needs far fewer runs than two independent samples to resolve the
same effect (handbook §7, "Not controlling seeds"). `bakeoff_report` always pairs by
seed within each block, such as an environment, and never pools runs across blocks.

Unpaired tests such as Welch's t-test and the Mann–Whitney U-test throw that pairing
away. Core no longer ships them as comparison primitives. The low-level
`bootstrap_difference_ci` and `run_permutation_test` still accept
`pairing="independent"` for designs that are genuinely unpaired.

## Why an interval, and which one

A p-value says whether a difference is distinguishable from zero. It does not say how
large the difference is. The headline is therefore the estimate and its CI.

Intervals are **percentile bootstrap** intervals over runs. They assume no
distribution. Their weakness is small samples: with few runs the interval is narrower
than its stated confidence, so reports warn (`small_sample_coverage`) below 30 runs.
Treat that warning as a reason to run more seeds, not as a detail to explain away.

## Mean or median

`BakeoffSpec.estimand` chooses the statistic. The **mean** is the default and answers
"what performance should I expect?". Catastrophic seeds count in full, which is usually
what you want: a method that sometimes collapses should look worse. The **median**
answers "how does a typical run do?" and resists a few extreme seeds, at the price of
hiding them. Choose in the plan, before the runs. Choosing after seeing which one looks
better is metric shopping.

Whichever you choose, report variability separately: the SD across runs, a
distribution-free tolerance interval, and a representative median run. Never write
`mean ± x`.

## Multiple comparisons

With k conditions in b blocks there are b·k(k−1)/2 pairs, or fewer when the plan
declares `pairs` (for example `against_baseline`). Each CI is computed at
1 − α/(family size), so that all of them hold together at 1 − α. Holm's step-down
adjustment does the same for the p-values. A pair skipped for too few paired runs still
counts in the family, and its reason is shown.

## Missing seeds

Under the default `missingness="error"`, a seed observed for some conditions and not
others stops the comparison. The error names the cells, the missing seeds, and whether
they failed or are still pending. Dropping them silently would bias the comparison
toward the seeds that happened to finish. `complete_case`, which uses only seeds every
condition observed, and `blockwise`, which keeps seeds observed by at least two
conditions, are explicit choices to declare in the plan.

## Tuned conditions

When each condition is tuned over a grid, the champion's own runs overstate it. There
the comparison uses the hierarchical per-sample bootstrap
(`select_champion_over=...`), so each difference carries selection uncertainty, and no
tests are run on the seeds that did the selecting. See
[the science guide](../science_guide.md#2-handling-hyperparameter-maximization-bias).

## Related docs

- [Empirical Design & Analysis Guide](../science_guide.md)
- [Reading Results](../reading-results.md)
- [ADR 012: Percentile intervals, hierarchical champions, opt-in aggregates](../adrs/012-percentile-champion-aggregate-decisions.md)
