# ADR 010: Robust Statistical Aggregates

## Context

Benchmark results vary across seeds, tasks, and failure modes. A single mean can
hide dispersion, tail failures, and constraint violations, while a pooled rank
normalization changes when the compared algorithms change.

## Decision

`research-analysis` reports task-level distributions and uncertainty first. IQM
is permitted as a supplementary aggregate only; every IQM result must also show
dispersion, lower-tail, and relevant constraint or safety metrics. Benchmark
normalization uses caller-supplied fixed reference anchors or ranges and never a
comparison-set-derived pool.

Incomplete repeated-block comparisons use a validated block-stratified permutation
omnibus equivalent to Skillings–Mack: within-block observed ranks are permuted
without deleting incomplete rows. This preserves each missingness pattern without
claiming the exact Skillings–Mack reference distribution. Insufficient or invalid
data fails explicitly.

## Consequences

- Adding an algorithm cannot change existing normalized scores.
- Robust centers cannot conceal instability or safety failures.
- Missing observations do not silently change the analysis population.
- Reports carry their estimand, pairing, uncertainty, and missingness decisions.

## Update (2026-09-23)

Superseded in part by [ADR 012](012-percentile-champion-aggregate-decisions.md).
IQM and the other aggregates are no longer part of any default report. They are
opt-in for collections of environments only, because IQM trims away exactly the
catastrophic runs a comparison must show. The decision above on the
block-stratified omnibus test and on fixed-reference normalization stands.
