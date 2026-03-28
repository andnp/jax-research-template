# Statistical Comparisons

This note explains two common ways to compare algorithm performance distributions in `research-analysis`: **Welch's t-test** and the **Mann–Whitney U-test**.

The short version is:

- Use **Welch's t-test** when you care about **differences in means** and the data are reasonably compatible with mean-based inference.
- Use **Mann–Whitney U** when you want a **more robust, non-parametric comparison** and the data may be skewed, heavy-tailed, ordinal, outlier-prone, or clearly non-normal.

For Reinforcement Learning, the second case is common.

## Why this matters in RL

Seed-level returns are often messy:

- one seed collapses,
- one seed gets lucky,
- one algorithm has a long tail,
- another has much lower variance,
- and the resulting distribution is nowhere near Gaussian.

That means the right test depends on the scientific question, not on habit.

## Welch's t-test

### What it is

Welch's t-test is a two-sample test for whether two groups have different **means**.

It is a variant of the classic Student t-test that relaxes the equal-variance assumption. In practice, that makes it safer than the plain pooled-variance t-test when the two algorithms have visibly different variances.

### What question it answers

Welch's t-test is the right tool when your real question is:

> Is the **average** return of algorithm A different from the **average** return of algorithm B?

That is a mean-comparison question.

### Main assumptions

Welch's t-test makes fewer assumptions than the classical t-test, but it is still a **parametric mean-based test**.

The important assumptions are:

1. **Independent observations**
   - Each run/seed should be an independent draw.
2. **A meaningful mean**
   - The mean should be the summary statistic you actually care about.
3. **Reasonably well-behaved sampling distribution**
   - The raw data do not need to be perfectly normal, but the mean should not be dominated by pathological skew or extreme outliers.
4. **Unequal variances are allowed**
   - This is exactly why Welch is preferred over the plain two-sample t-test.

### When someone might use it

Welch's t-test is a good choice when:

- the metric of interest is explicitly the **mean final return**,
- the distributions are not wildly pathological,
- the sample size is moderate or large enough that mean-based inference is tolerable,
- and unequal variance is the main concern.

This can make sense for clean engineering comparisons where the average outcome is the quantity the paper or report wants to claim.

### When it is a bad fit

Welch is a weaker choice when:

- the data are strongly skewed,
- there are major outliers,
- sample sizes are small,
- or the mean is not really the thing you care about.

In RL, a few lucky or catastrophic seeds can move the mean a lot. When that happens, Welch may be answering a mathematically valid question that is still scientifically unhelpful.

## Mann–Whitney U-test

### What it is

The Mann–Whitney U-test is a **non-parametric**, rank-based test for comparing two independent groups.

Instead of working directly with the numeric values and their means, it looks at the **relative ordering** of the observations.

A good intuition is:

> Does algorithm A tend to produce larger values than algorithm B?

### What question it answers

Mann–Whitney is best for questions like:

> Does one method generally outperform the other across runs?

That makes it much more robust when the raw return distributions are ugly.

### Main assumptions

Mann–Whitney makes different assumptions from Welch:

1. **Independent observations**
   - Seeds/runs still need to be independent.
2. **Ordinal comparability**
   - The values must at least be rankable.
3. **Interpretation is rank-based**
   - The safest interpretation is about a **distributional or rank shift**, not automatically a mean difference.

A common simplification is that Mann–Whitney is a "test of medians," but that is only strictly true under extra conditions, especially when the compared distributions have similar shapes.

### When someone might use it

Mann–Whitney is a better choice when:

- the return distribution is **non-normal**,
- there are **outliers**,
- there is visible **skew** or **heavy-tailed behavior**,
- sample sizes are small,
- or the scientific question is closer to **which method usually does better** than **which method has a higher mean**.

This is often the more defensible default in RL.

### Why it can be better for RL

Suppose group A is:

- `[1, 1, 1, 1, 100]`

and group B is:

- `[2, 2, 2, 2, 2]`

The mean of A is heavily affected by the single extreme run. A mean-based test can get pulled toward that outlier.

Mann–Whitney mostly sees that B is larger than A in rank order almost everywhere, except for one wild point.

That often matches the scientific intuition better:

> Most runs of B are better than most runs of A.

## Which one is better?

Neither test is universally better. The right choice depends on the question.

### Use Welch's t-test when:

- you want to compare **means**,
- the mean is the quantity you plan to report,
- and the data are not so pathological that the mean becomes misleading.

### Use Mann–Whitney U when:

- you want a **robust, non-parametric comparison**,
- the distributions are messy,
- or you care more about **general dominance / rank ordering** than exact mean differences.

## Practical rule of thumb

If your actual question is:

> Is the **average** final return different?

then Welch is conceptually aligned.

If your actual question is:

> Which algorithm tends to do better across runs, even when the data are skewed or noisy?

then Mann–Whitney is often the better fit.

## Recommendation for this repo

This repository's science guidance emphasizes:

- non-parametric methods,
- robust uncertainty estimates,
- and caution around fragile assumptions in RL evaluation.

That is why `research-analysis` now treats **Mann–Whitney U** as the primary comparison primitive.

Welch's t-test is still useful to understand and may still be appropriate in some mean-comparison settings, but it should be chosen deliberately rather than by default.

## Related docs

- [Empirical Design & Analysis Guide](../science_guide.md)
- [ADR 009: Analysis Stack Selection (Polars & Numba)](../adrs/009-analysis-stack.md)
