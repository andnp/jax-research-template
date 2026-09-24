# ADR 012: Percentile Intervals, Hierarchical Champions, Opt-In Aggregates

## Context

`research-analysis` computed BCa intervals, described cross-fitted tuning that it did
not implement, compared tuned champions by raw argmax on the seeds that selected them,
and folded IQM, performance profiles, and probability of improvement into the default
bakeoff. The methodology reference is Patterson, Neumann, White & White, "Empirical
Design in Reinforcement Learning" (JMLR 2024). Recommendations from Agarwal et al.
("Deep RL at the edge of the statistical precipice") are not adopted as defaults.

## Decision

- **The resampling unit is the run (seed)**, never an episode or step.
- **Percentile bootstrap** is the only default interval. BCa is removed.
- **Confidence and variability are reported separately.** Every CI states its
  confidence. Run-to-run variability is shown as SD, distribution-free tolerance
  intervals, and a median representative run. Nothing is reported as `mean ± x`, and
  there are no standard errors.
- **The headline comparison is a seed-paired difference with its CI**, Bonferroni-
  adjusted across the report's planned family. Permutation p-values, Holm-adjusted,
  and the block-stratified omnibus test are secondary.
- **Tuned champions are compared with the hierarchical per-sample bootstrap**
  (`research_analysis.tuning`), which includes selection uncertainty. No tests are run
  on the selecting seeds. The handbook's bootstrapped two-stage method is not used:
  `projects/hyper-champion-ci` experiment 001 found its coverage collapses as the
  number of configurations grows, while the hierarchical per-sample variant was best
  calibrated.
- **Diverged, failed, and pending runs are never dropped silently.** Diverged runs
  are outcomes: they are counted, and every statistic raises until a worst-case score
  is declared.
- **IQM, performance profiles, probability of improvement, and optimality gap** stay
  as functions, opt-in for collections of environments, never in a default path.
- **Reports come from a declared plan.** A report is labelled declared only when a
  pre-recorded plan was run verbatim. Everything else is exploratory.

### 2026-09-24: plan fingerprints hash declared fields only

A plan's fingerprint now covers only the fields the researcher declared. Fields
at their dataclass defaults are left out, recursively. Before, the whole resolved
plan was hashed, so adding an optional field to any plan dataclass would have
relabelled every existing declared report exploratory without anyone changing
a plan. History entries recorded under the old hash are re-read with the current
schema and compared by their declared fields. An entry the schema cannot read
makes the report exploratory. A changed default is not a plan change. It surfaces
through the report's toolchain provenance (the `research-analysis` version and
commit), not through `plan_status`.

## Consequences

- Small-sample intervals under-cover. The calibration suite
  (`tests/large/test_research_analysis_calibration.py`) measures by how much, and
  reports warn below 30 runs instead of hiding it behind a more complex interval.
- Tuned-arm comparisons are wider than raw argmax comparisons, because they are
  honest about selection.
- ADR 010's permission to show IQM as a supplementary center is narrowed to
  cross-environment use only.
