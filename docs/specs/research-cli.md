# Technical Specification: `research` CLI Tool

## 1. Overview
The `research` CLI is the orchestration layer for the RL Research Monorepo. It manages the lifecycle of projects, the relationship between the Core submodule and the User Shell, and the "Harvest/Eject" code lifecycle.

- **Backend:** Python 3.12+
- **CLI Framework:** `typer`
- **Template Engine:** `copier`
- **Dependency Management:** `uv`
- **Integrations:** `git`, `gh` (GitHub CLI)

## 2. Core Commands

### 2.1 Workspace Management
- `research workspace init`: 
    - Initializes a new "Shell" monorepo.
    - Adds the `research-core` as a submodule.
    - Sets up the root `pyproject.toml` with `uv` workspace members.
- `research workspace upgrade`:
    - Pulls the latest changes from `research-core` upstream.
    - Warns if there are local modifications in `core/`.
    - Runs `uv sync` to align dependencies.

### 2.2 Project Management
- `research project create <name> [--github-repo <slug>]`:
    - Uses `copier` to spin up a project from `core/templates/standard-project`.
    - Runs `git init` inside the new project.
    - (Optional) Uses `gh repo create` to create a private remote and link it.
- `research project archive <name>`:
    - Moves a project from `projects/active/` to `projects/archive/`.
    - Removes it from the active `uv` workspace to keep resolution fast.

### 2.3 The "Rule of Three" Lifecycle
- `research eject <lib_name>`:
    - Copies `core/libs/<lib_name>` into `projects/<current>/src/components/`.
    - Updates local imports to point to the project-specific version.
- `research harvest <project_path>/<module_name>`:
    - Moves a generalizable component from a project into `core/libs/`.
    - Initializes a new `pyproject.toml` for the library if it's new.
    - Updates project imports to point to the Core library.
- `research propose <lib_name>`:
    - Automates the PR process back to the public `research-core`.
    - Handles forking, branching, and opening the PR via `gh`.

### 2.4 Experiment Orchestration
The CLI manages a multi-stage pipeline for large-scale experiments, using SQLite as the persistent state-of-the-world.

- `research exp define <config.py> [--name <experiment_name>]`:
    - Executes the definition script which uses `rl-sweeper` to generate a Cartesian product of hyperparameters and seeds.
    - populates a `experiments.sqlite` database with rows for every individual "unit of work" (Seed + Hyperparam Set).
    - Returns a unique `experiment_id`.

- `research exp run <experiment_id> [--executor <local|slurm>]`:
    - Reads the database to identify `PENDING` or `FAILED` runs.
    - Groups work into **"vmap-zones"**: sets of runs that can be executed in a single `jax.vmap` call (e.g., all seeds for Algorithm X with Hyperparam Set Y).
    - Dispatches batches to the selected executor.
    - Updates the DB state to `RUNNING` and eventually `COMPLETED` or `FAILED`.

- `research exp status <experiment_id>`:
    - Provides a progress bar and breakdown of status (Pending/Running/Success/Fail) across algorithms and hyperparameters.

- `research exp report <experiment_id> [--best-by <metric_name>]`:
    - Queries the DB to find the optimal hyperparameter combinations.
    - Generates a statistical summary (Mean/Std/Max) per algorithm.

- `research exp analyze <experiment_id> --using <analysis_script.py>`:
    - Injects the experiment data into a project-specific analysis script for publication-ready plotting.

## 3. The "vmap-zone" Batching Logic
The orchestrator must be "JAX-aware." It identifies which parameters are **static** (change the JIT kernel, like hidden layer size) and which are **dynamic** (can be vmapped over, like learning rate or seed).
- Work is batched such that each `vmap-zone` corresponds to exactly one static configuration.
- This ensures maximum hardware utilization by filling the accelerator's memory with dynamic sweeps.


## 3. Configuration
The CLI will look for a `research.yaml` file at the monorepo root to store:
- Path to the `core/` submodule.
- Default GitHub organization/user for new projects.
- Preferred storage backend (Local vs. S3) for the `research-store` integration.

## 4. User Workflows

### Workflow A: The "One Agent, One World" Sweep
1. Researcher writes a single-agent script in `projects/new-idea/train.py`.
2. Runs `research run train.py --vmap 128`.
3. The CLI handles the seed splitting and parallel execution, returning aggregated statistics.

### Workflow B: The Library Contribution
1. Researcher identifies that three projects are using a custom `ReplayBuffer`.
2. Runs `research harvest projects/alpha/src/buffer.py`.
3. CLI moves it to `core/libs/rl-buffers`.
4. Runs `research propose rl-buffers`.
5. PR is opened upstream for the community to benefit.

## 5. Technical Constraints
- The CLI must be installed in "editable" mode (`uv pip install -e core/cli`) within the workspace.
- Commands must be deterministic and provide clear "dry-run" previews before modifying the filesystem or Git state.
