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
    - Sets up the root `pyproject.toml` with `uv` workspace members for `core/cli`, `core/libs/*`, and `projects/*`.
    - Prints truthful manual next steps instead of mutating the environment itself:
        - with `--core-url`, the intended follow-up is `uv sync --all-packages` and then `uv run research doctor`;
        - without `--core-url`, the user must add `core/` first with `git submodule add <url> core` before running that same sync + doctor flow.
- `research workspace repair [--dry-run]`:
    - Mutating remediation command for the configured Core checkout. It is separate from read-only `research doctor` and must never be invoked implicitly by diagnostics.
    - Reads `research.yaml` at the workspace root and resolves the configured `core_path`; all repair actions are scoped from that path.
    - In `--dry-run`, emits a deterministic ordered action plan and exits without mutating the filesystem, dependencies, configuration, or Git state.
    - By default, repairs the Core checkout to the superproject-recorded submodule revision using standard submodule update semantics for the configured `core_path` unless an explicit future option selects a different target.
    - Cleanup scope is intentionally narrow and testable: inside `core_path`, it may discard tracked-file modifications required to match the recorded revision and remove untracked files or directories that block or conflict with that reset; it must not edit `research.yaml`, touch files outside `core_path`, or mutate other workspace members.
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

### 2.5 Diagnostics
- `research doctor`:
    - Runs a read-only diagnostic sweep over workspace configuration, the configured Core checkout, and the local execution environment.
    - Must never mutate the filesystem, install dependencies, rewrite configuration, or modify Git state.
    - Executes all diagnostic groups before exiting and returns a non-zero status code if any check fails.
    - Reports grouped results so the user can see all failures from a single run instead of failing fast on the first problem.
    - Checks that cannot run because an upstream prerequisite is invalid must still be reported in their group as failures caused by missing or invalid inputs.
    - Covers three diagnostic groups:
        - **Config validation** for `research.yaml`:
            - Verify that `research.yaml` exists at the workspace root and is parseable.
            - Validate `core_path` as a required path-like setting used to locate the Core checkout.
            - Validate optional doctor-specific settings under `doctor.expected_accelerators` if present.
        - **Git health** for the configured `core_path`:
            - Verify that `core_path` exists.
            - Verify that `core_path` resolves to a Git working tree.
            - Verify that the working tree can be inspected read-only (for example, `HEAD` resolves and status can be queried).
            - Report dirty or otherwise unhealthy Core state as a failing diagnostic, but do not attempt remediation.
        - **Environment health** around `uv` and JAX:
            - Verify that `uv` is discoverable on `PATH` and responds to a version query.
            - Verify that the current workspace environment can import `jax` without triggering an environment mutation.
            - Record the detected JAX backend/device platforms and compare them against `doctor.expected_accelerators` when configured.
            - Never run mutating environment commands such as `uv sync`, `uv pip install`, cache cleanup, or package upgrades.

## 3. The "vmap-zone" Batching Logic
The orchestrator must be "JAX-aware." It identifies which parameters are **static** (change the JIT kernel, like hidden layer size) and which are **dynamic** (can be vmapped over, like learning rate or seed).
- Work is batched such that each `vmap-zone` corresponds to exactly one static configuration.
- This ensures maximum hardware utilization by filling the accelerator's memory with dynamic sweeps.


## 3. Configuration
The CLI will look for a `research.yaml` file at the monorepo root to store:
- Path to the `core/` submodule.
- Default GitHub organization/user for new projects.
- Preferred storage backend (Local vs. S3) for the `research-store` integration.

The initial schema is intentionally conservative:
- Existing top-level keys remain valid.
- `core_path` remains the canonical setting used by commands that need to locate the Core checkout.
- `research doctor` may additionally read an optional `doctor` section.

Example:

```yaml
core_path: core
github_owner: rlcore
storage_backend: local
doctor:
    expected_accelerators:
        - gpu
```

`doctor.expected_accelerators` is optional. If it is absent, `research doctor` still validates `uv` and JAX availability, but it does not fail solely because no accelerator expectation was configured. The accepted values are conservative platform labels (`cpu`, `gpu`, `tpu`) so the setting can map cleanly onto JAX device discovery without locking the config to a host-specific device string.

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
- Fresh shell workspaces should include `core/cli` as a `uv` workspace member, so the normal bootstrap path is `uv sync --all-packages` once `core/` exists.
- `research workspace init` must not run `uv sync` or `research doctor` implicitly; it only prints the appropriate next-step guidance.
- Commands must be deterministic and provide clear "dry-run" previews before modifying the filesystem or Git state.
