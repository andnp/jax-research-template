# PRD: `libs/research-plot`

## 1. Overview

`research-plot` draws static matplotlib figures of experiment results. It does no
statistics and no data loading: it draws the values it is given. The numbers come from
`research-analysis`. In reports, `research_analysis.report_plots` calls it with the
per-run scores and sensitivity curves stored in the report document.

## 2. Functions

| Function | Draws |
|---|---|
| `plot_distributions(samples, save_path, *, title, ylabel)` | each condition's per-run outcomes as a violin plus every run as a point, so a reader sees the whole distribution behind a cell. A condition with fewer than two distinct values gets points only. |
| `plot_sensitivity(param_name, values, slice_performance, best_performance, save_path)` | the selection-corrected best-achievable curve and the champion-pinned slice over one hyperparameter. A slice value that was never run leaves a gap. |
| `plot_trajectory_panels(...)` with `TrajectoryPanel`, `TrajectorySeries`, `EventWindow` | named trajectories sharing y-axes, with optional pointwise bands and highlighted event windows |

Every function returns the saved path.

## 3. Rules

- **Explicit destinations.** Every function writes to the `save_path` it is given and
  creates its parent directory. Nothing is written relative to the working directory
  by default. Callers put figures under the project's durable `results/` tree.
- **Validate, don't hide.** Empty or non-finite inputs raise instead of drawing a
  misleading figure. A diverged run must be scored (or rejected) upstream.
- **Show runs, not just summaries.** Distribution figures draw every run. Bands must
  say what they are (a CI or a tolerance interval), and are never labelled `± x`.
- **Minimal style** (science guide section 6): no top or right spines, no legend
  frames, consistent colors within a project.
- **Deterministic output.** Point jitter uses a fixed seed.
