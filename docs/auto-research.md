# Auto-Research Loop

This guide is for an agent, or a person, running experiments in a loop: form a
hypothesis, run it, read the result, and decide what to do next. The toolchain is
built so that one command gives a report you can trust. The discipline around that
command is what keeps the loop honest.

## The loop

1. **Write the plan first.** Put the question, arms, and protocol in
   `docs/experiments/NNN-slug/NNN-plan.md`. Put the matching spec in
   `experiments/NNN_slug.py`: an `@spec` factory and, beside it, an
   `@analysis_plan(<factory>)` declaring the primary analysis. The plan declares the
   metric, the per-run measure (`FinalValue`, `TailMean`, `ReturnRate`), the dimensions,
   the comparisons (all pairs, `pairs=`, or `against_baseline(...)`), any champion
   selection axes, the statistics (`BakeoffSpec`: alpha, resamples, seed,
   missingness), and a worst-case score for diverged runs if one applies. See
   [Reading Results](reading-results.md#declared-report) for an example.
2. **Run.** `uv run research experiment run experiments/NNN_slug.py --spec <factory>`
   (or `submit`). This snapshots the plan into `<results_root>/analysis-plans.jsonl`
   before any execution starts. The history is append-only, so later edits stay visible.
3. **Report.** `uv run research experiment report experiments/NNN_slug.py --spec <factory> --json`.
   The document is also written, with `report.md` and figures, to
   `<results_root>/reports/<experiment>/<UTC time>/`.
4. **Read the document in this order.**
   1. `status` and the exit code. 0 means results. 2 means `load_failed`: read
      `fatal_diagnostics`. 3 means `analysis_failed`: read `failure`, which names the
      diverged runs or the incomplete cells, their missing seeds, and the failed or
      pending counts behind them. Fix the cause. Do not work around it.
   2. `plan_status`. If it is `exploratory`, the `exploratory_reasons` say why, and
      every result in the document is exploratory.
   3. `accounting`: planned, completed, failed, pending, diverged, and excluded runs per
      cell. A comparison built on 7 of 10 planned seeds is a different claim from one
      built on 10 of 10.
   4. `non_fatal_diagnostics` and each analysis's `warnings`, especially
      `small_sample_coverage`, `hyperparameter_at_boundary`, and `insufficient_seeds`.
   5. Only then the numbers.
5. **Interpret differences, not p-values.** The headline is each paired difference
   with its CI. Family CIs are Bonferroni-adjusted so that all of them hold jointly at
   the stated confidence, and each row states its own level. The Holm p-values and the
   omnibus test are secondary. "The CI excludes 0" is a claim about this family of
   comparisons at this confidence, nothing more. Report variability separately, using
   cell SDs and tolerance intervals. Never report `mean ± something`.
6. **Record.** Write a dated entry in `docs/experiments/NNN-slug/NNN-diary.md` that
   cites the report directory. It should give what was predicted, what the report
   shows (with its plan status), and the decision. When a later result overturns an
   entry, add a new dated entry. Do not edit the old one.

## Rules that keep the loop honest

- **Never change the plan after seeing results and then call the report declared.**
  The tool will not let you: the report becomes exploratory with "plan changed after
  results were recorded". The check compares only what the plan declares. Spelling a
  default out, or a new optional field in the library, does not count as a change,
  but a changed library default does not either: it shows in the report's
  `provenance.toolchain`, so check it when comparing reports. Exploratory follow-ups are fine, but label them as such in
  the diary. Then pre-register a new experiment to confirm them.
- **Do not metric-shop.** Picking the measure, estimand, or comparison that happens
  to look best is the garden of forking paths. The plan fixes all of them.
- **Diverged runs are outcomes.** Never drop or re-seed them (handbook §7,
  "Discarding or replacing runs"). If divergence is bad performance, declare a
  worst-case score in the plan. Otherwise the report fails and names the runs.
- **Crashed or pending seeds are not silently dropped.** Under the default
  `missingness="error"`, an incomplete block fails the report with the missing seeds.
  Finish or rerun them, or declare `complete_case`/`blockwise` in the plan *before*
  running.
- **Tuned champions carry selection uncertainty.** Compare tuned arms with
  `select_champion_over`. It uses the hierarchical per-sample bootstrap, and no
  tests are run on the selecting seeds. Never compare raw argmax champions on the
  same seeds that picked them. A fixed single-configuration referent is declared
  `untuned`, and a referent without a blocking axis is declared `shared_across` that
  axis; neither is ever inferred.
- **Small n under-covers.** The percentile bootstrap's intervals are narrower than
  stated when there are few runs. In the calibration pilots, a single mean CI at 10
  runs covered about 0.88 for normal data. A Bonferroni family of 6 paired CIs
  covered about 0.82–0.85 jointly at 10 runs and 0.91–0.93 at 30, against a nominal
  0.95. `tests/large/test_research_analysis_calibration.py` holds the current bounds.
  Every confidence interval resting on fewer than 30 runs carries a
  `small_sample_coverage` warning in its report's `warnings`: bakeoff cells and
  pairs, tuned champions, and champion differences. Treat a borderline interval as a reason to run more seeds (handbook
  §2.6), not as a finding.
- **IQM, performance profiles, probability of improvement, and optimality gap** are
  opt-in aggregates for collections of environments, never defaults. IQM trims away
  exactly the catastrophic runs you most need to see.
- **Keep results durable.** Reports, plots, and databases live under the project's
  `results/` tree, never in `/tmp` or the working directory.
