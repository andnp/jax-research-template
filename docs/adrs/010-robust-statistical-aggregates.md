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

Incomplete repeated-block comparisons use Skillings–Mack. Friedman is reserved
for complete blocks. Insufficient or invalid data fails explicitly.

## Consequences

- Adding an algorithm cannot change existing normalized scores.
- Robust centers cannot conceal instability or safety failures.
- Missing observations do not silently change the analysis population.
- Reports carry their estimand, pairing, uncertainty, and missingness decisions.
