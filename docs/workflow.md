# Workspace Workflow Guide

This guide details how to develop research projects using the `research` CLI and the "Hub-and-Spoke" monorepo model.

---

## 1. Hub-and-Spoke Workspace Model

A research workspace consists of a single enclosing repository (the "Hub") containing general configuration and a list of experiment-specific repositories under the `projects/` directory (the "Spokes"). 

The **Core** (`core/`) contains shared, stable libraries (`libs/`), command-line utilities (`cli/`), and templates (`templates/`). By isolating experiments in distinct project folders, researchers can iterate on ideas independently without risking compile-time or logic breaks in unrelated projects.

---

## 2. CLI Command Reference

All workspace lifecycle management is handled by the `research` command-line tool. Run commands using `uv run research <command>`.

### Workspace Commands
- `workspace init <name> [--path <path>] [--core-url <url>]`
  Initializes a new workspace directory containing a git repository, a root `pyproject.toml` (configured as a `uv` workspace), a `projects/` directory, a `research.yaml` config file, and a git submodule pointing to the `research-core` repository.
- `workspace repair`
  Repairs the core submodule checkout, aligning it to the specific commit recorded by the workspace, and running clean sweeps (`git clean -ffd`).

### Project Commands
- `project create <name> [--dry-run] [--github-repo <slug>]`
  Creates a new experiment directory under `projects/` using `copier` templates. Automatically runs `git init` inside the new folder. Optional `--github-repo` creates a private remote repository via the GitHub `gh` CLI and registers the project as a shell submodule. The same behavior is automatic when `research.yaml` sets `auto_create_private_project_repos: true` and provides `default_github_org`; the derived repository slug is `<org>/<name>`.

### Lifecycle Commands
- `eject <project> <library>`
  Copies a shared library from `libs/` to the project's local `components/` directory, updates the project's imports, and decouples the project from the shared package.
- `harvest <project> <library>`
  Moves a project-local component from `projects/<project>/components/` into `libs/`, registers the library in the workspace root `pyproject.toml`, and rewrites the import references back to the global package name.

### Diagnostics Command
- `doctor`
  Runs a read-only diagnostic sweep of the workspace to verify package directories, symlink mappings, configuration parameters, and dependency files.

### Experiment Commands
- `experiment plan <spec_file.py> [--spec <name>] [--results-root <path>]`
  Dry-runs an experiment definition: syncs it to `experiments.sqlite` and prints the batches that would run, without executing anything.
- `experiment run <spec_file.py> [--spec <name>] [--results-root <path>] [--max-runs <n>]`
  Executes the unsatisfied runs for an experiment definition, writing provenance to `experiments.sqlite` and metrics to `metrics.sqlite`.
- `experiment status <db_path> [--experiment <name>]`, `experiment list <db_path>`, `experiment executions <db_path>`
  Read-only progress and execution history against an `experiments.sqlite` database.
- `experiment invalidate <db_path> --execution <id>`
  Marks an execution `INVALID` (with a confirmation prompt) so its runs are re-planned by the next `run`.
- `experiment submit <spec_file.py> [--spec <name>] [--slurm-config <file.json>] [...slurm options]`
  Submits the planned batches as a Slurm job array instead of running them locally. Flags override values from `--slurm-config`.
- `experiment report <spec_file.py> [--spec <name>] [--results-root <path>] [--output-dir <path>] [--json]`
  Runs the spec's declared analysis plan and writes `report.json`, `report.md`, and figures under `<results_root>/reports/`. Exits 2 on fatal load diagnostics and 3 on a failed analysis. With `--json`, it prints the document to stdout.

---

## 3. Example Workflows

### Workflow A: Bootstrapping a New Workspace
To create a new workspace for a research phase:

1. **Initialize the workspace container:**
   ```bash
   uv run research workspace init my-thesis-workspace --core-url https://github.com/andnpatterson/research-core
   ```
2. **Synchronize dependencies:**
   Navigate into the newly created directory and synchronize all python packages:
   ```bash
   cd my-thesis-workspace
   uv sync --all-packages
   ```
3. **Verify the environment:**
   Verify the workspace state:
   ```bash
   uv run research doctor
   ```

### Workflow B: Creating a New Experiment Project
To launch a new experiment using a template:

1. **Create the project folder:**
   ```bash
   uv run research project create ppo-cartpole-sweep
   ```
   *Note: This prompts for project details (such as description, algorithm, and environment name) and renders the template inside `projects/ppo-cartpole-sweep/`.*
2. **Execute the local training baseline:**
   Run the generated training script to verify execution:
   ```bash
   cd projects/ppo-cartpole-sweep
   uv run python train.py
   ```

To have project creation create private GitHub repositories and register them as submodules, configure the workspace first:

```yaml
default_github_org: my-org
auto_create_private_project_repos: true
```

The command requires an authenticated GitHub CLI session when this setting is enabled.

### Workflow C: Customizing and Harvesting a Shared Library (The Eject-Harvest Loop)
When a shared library (e.g., `rl-agents`) needs custom modifications that are experimental or breaking for other projects:

1. **Eject the library into the project:**
   ```bash
   uv run research eject ppo-cartpole-sweep rl-agents
   ```
   **What happens under the hood:**
   - The contents of `domains/rl/agents/src/rl_agents/` are copied to `projects/ppo-cartpole-sweep/components/rl_agents/`.
   - The CLI automatically scans Python files in `projects/ppo-cartpole-sweep/` and rewrites imports. For example, `from rl_agents.ppo import make_train` becomes `from components.rl_agents.ppo import make_train`.
   - The project is now fully decoupled from the core library.

2. **Hacking locally:**
   Modify the agent code inside `projects/ppo-cartpole-sweep/components/rl_agents/` to implement your research idea. Test and iterate locally within the project.

3. **Harvesting the component back to Core:**
   Once the modifications are successful, generalized, and ready to share across projects:
   ```bash
   uv run research harvest ppo-cartpole-sweep rl-agents
   ```
   **What happens under the hood:**
   - The component is moved from the project folder back to `domains/rl/agents/src/rl_agents/`.
   - The CLI automatically rewrites import statements back to the global name: `components.rl_agents.ppo` becomes `rl_agents.ppo` in both the project files and the library files.
   - The CLI registers the library in the root workspace `pyproject.toml` dependencies, workspace sources, and `pyrefly` analysis paths.

### Workflow D: Defining and Running an Experiment (Sweeps)
To run a hyperparameter sweep instead of a single training script, define it declaratively rather than hand-rolling a loop over `train.py`:

1. **Define the sweep:**
   In a project file (e.g. `projects/ppo-cartpole-sweep/experiments/001_sweep.py`), build an `experiment_definition.experiment.Experiment` and expose a module-level, zero-argument function returning `ExperimentSpec`, decorated with `@spec` from `research_runner.registry`. The CLI discovers exactly the factories registered by `@spec`.

   ```python
   experiment.add_parameter("env", ENV_NAMES)                     # a named choice
   experiment.add_parameter("agent.learning_rate", [1e-3, 3e-4])  # a dotted config path
   experiment.add_parameter("seed", range(30))
   experiment.add_metric("loop/reward", ema=0.999, every=500)      # smoothed, then one value in 500

   ExperimentSpec(
       experiment,
       config=RLConfig(agent=PPOConfig(), env=GymnaxConfig(ENV_NAMES[0])),
       choices={"env": {name: GymnaxConfig(name) for name in ENV_NAMES}},
       train=rl_train,
       results_root=RESULTS_ROOT,
   )
   ```

   Each parameter either sets a dotted path on the base `config` or names one of `choices`; `seed` is reserved. `train(config, key)` is a JAX function returning `(state, metrics)`, and `rl_agents.experiment.rl_train` is the default; wrap it to emit extra metrics. Only metrics declared with `add_metric` are recorded. Runs that differ only in fields their config marks `traced(...)` share one compiled `jit(vmap(train))`. A misspelt path, an unknown choice or a mistyped value fails at sync, before anything runs.

2. **Preview the run (dry run):**
   ```bash
   uv run research experiment plan sweep.py --spec my_sweep
   ```
   This syncs the definition and prints the batches that would run, without executing anything.

3. **Run it:**
   ```bash
   uv run research experiment run sweep.py --spec my_sweep
   ```
   Results land under the project's `results/` directory: `experiments.sqlite` holds intent and provenance (per-execution git commit, git diff, and component source hashes), while the runner persists the declared metrics to `metrics.sqlite` and each batch's runtime and sampling metadata to its `ExecutionArtifacts` row.

   A single-process run caps XLA's CPU thread pool with the `NPROC` environment variable, e.g. `NPROC=4 uv run research experiment run sweep.py --spec my_sweep`. `XLA_FLAGS` thread-count settings and `OMP_NUM_THREADS`/`MKL_NUM_THREADS` do not cap it. With JAX 0.11.1, a jitted matmul scan used 6.45 cores unset, 1.00 with `NPROC=1`, and 3.83 with `NPROC=4`.

   To use several cores, give the run a thread budget: `uv run research experiment run sweep.py --spec my_sweep --workers 8`. The supervisor starts one worker process per pending group (runs sharing a static config), gives each an `NPROC` so the workers' threads sum to at most the budget, and starts a worker only when free memory fits it. `--max-runs N` splits each group into vmapped batches of N, so a kill loses at most one batch. Cancelling the one command stops every worker, and a worker that dies has its unfinished executions marked FAILED, so the next launch retries them. See [the spec](specs/parallel-experiment-runner.md) for the policy.

4. **Resume after a failure:**
   Re-running the same `run` command re-plans only the unsatisfied runs — those without a linked `COMPLETED` execution — so an interrupted sweep picks back up without repeating finished work. Check progress with:
   ```bash
   uv run research experiment status projects/ppo-cartpole-sweep/results/experiments.sqlite
   ```
   To force specific work to be redone, mark it invalid first: `research experiment invalidate <db> --execution <id>`,
   or select a whole arm by hyperparameter: `research experiment invalidate <db> --experiment <slug> --where algorithm=<arm>`.
   Before re-running anything, read [Running Experiments Without Wasting Compute](experiment-efficiency.md) — editing an
   agent's source invalidates nothing on its own, and re-identifying a study re-runs all of it.

5. **Report the results:**
   Declare the primary analysis beside the spec with `@analysis_plan(my_sweep)` from
   `research_analysis.plan`. `run` and `submit` snapshot it before any execution starts.
   Then:
   ```bash
   uv run research experiment report sweep.py --spec my_sweep
   ```
   This runs the declared plan and writes a versioned JSON report, a markdown copy, and
   figures to `results/reports/<experiment>/<UTC time>/`. The report embeds provenance
   (component versions, execution commits and dirty state), per-cell run accounting,
   every diagnostic, and the results. It is labelled `exploratory` when the plan cannot
   be shown to predate the runs. Read the accounting and diagnostics before the numbers.
   For exploratory questions, load the dataset with
   `research_analysis.dataset.loading.load_experiment_dataset` and call
   `research_analysis.reporting` directly. See [Reading Results](reading-results.md) and
   [Auto-Research Loop](auto-research.md).

For the design rationale behind this database-centric, resumable model, see [ADR 007](adrs/007-declarative-experiment-management.md). For a runnable minimal example wiring definition, training, and checkpointing together, see `core/examples/run_experiment.py`; metrics are persisted by the runner (`research experiment run`).
