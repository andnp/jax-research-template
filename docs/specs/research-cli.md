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
    - Resolves the enclosing workspace root upward from the current directory, so it can be run from the shell root or from inside an existing child project repo.
    - Uses `copier` to spin up a project from `core/templates/standard-project`.
    - Runs `git init` inside the new project.
    - The shell repo remains the owner of shared workspace files such as `research.yaml` and `uv.lock`; the rendered child project is an independent nested Git repo under `projects/`.
    - (Optional) Uses `gh repo create` to create a private remote and link it.
- `research project archive <name>`:
    - Moves a project from `projects/active/` to `projects/archive/`.
    - Removes it from the active `uv` workspace to keep resolution fast.

### 2.3 The "Rule of Three" Lifecycle
- `research eject <lib_name>`:
    - Resolves the enclosing shell workspace root upward from the current directory using the same contract as `research project create` and `research doctor`.
    - Still requires an explicit project argument; it must not infer the current project from `cwd`.
    - Copies `core/libs/<lib_name>` into `projects/<current>/src/components/`.
    - Updates local imports to point to the project-specific version.
- `research harvest <project_path>/<module_name>`:
    - Resolves the enclosing shell workspace root upward from the current directory using the same contract as `research project create` and `research doctor`.
    - Still requires an explicit project argument; it must not infer the current project from `cwd`.
    - Moves a generalizable component from a project into `core/libs/`.
    - Initializes a new `pyproject.toml` for the library if it's new.
    - Updates project imports to point to the Core library.
- `research propose <lib_name>`:
    - Automates the PR process back to the public `research-core`.
    - Handles forking, branching, and opening the PR via `gh`.

### 2.4 Experiment Orchestration
The CLI manages experiments declared with `experiment-definition`, using SQLite as the persistent state-of-the-world. A *spec file* is a Python module whose factory functions return an `ExperimentSpec`; each names one absolute `results_root`, and the experiment database, metrics database and execution roots are derived from it.

Commands taking a spec file accept `--spec <name>` to select a single factory and `--results-root <path>` to relocate the whole results tree.

- `research experiment plan <spec_file>`:
    - Syncs the definition and reports the batches that would run, without creating executions.
- `research experiment run <spec_file> [--max-runs <n>]`:
    - Plans one batch at a time and executes it before planning the next, so an interrupted sweep leaves no unexecuted `PENDING` executions stranded.
    - Groups work into **"vmap-zones"**: a batch is one static configuration, and the parameters that vary inside it arrive as per-run points the training function can map over.
    - Records each execution as `RUNNING`, then `COMPLETED` or `FAILED`. A `FAILED` execution is retried on the next run.
- `research experiment status <db_path> [--experiment <slug>]`:
    - Reports total, completed and pending logical runs per experiment.
- `research experiment list <db_path>`:
    - Lists the experiments recorded in a database.
- `research experiment executions <db_path> [--experiment <slug>] [--status <status>] [--git-commit <sha>]`:
    - Lists executions with status, hostname, start time and commit.
- `research experiment invalidate <db_path> (--execution <id> | --git-commit <sha>)`:
    - Marks executions `INVALID` so their runs become plannable again. This is also how a run killed mid-execution — left claimed as `RUNNING` — is released.
- `research experiment execute-batch <db_path> --execution-id <id> --spec-file <file> --spec <name>`:
    - Executes a single already-planned execution. This is the entry point a scheduler invokes.
- `research experiment submit <spec_file> [slurm options] [--dry-run]`:
    - Plans every outstanding batch up front, writes a Slurm array script, and submits it. The planned executions stay `PENDING` until Slurm runs each one through `execute-batch`, so a local `run` must never re-plan them.

Analysis is deliberately not a CLI concern: `research-analysis` reads the metrics database produced by a run.

### 2.5 Diagnostics
- `research doctor`:
    - Resolves the enclosing workspace root upward from the current directory, so it can be run from the shell root or from inside child project repos.
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

### Workflow A: The Hyperparameter Sweep
1. Researcher defines an `Experiment` (parameters, static axes, metrics) and a module-level, zero-argument function returning `ExperimentSpec` in a project file, e.g. `projects/new-idea/sweep.py`.
2. Runs `uv run research experiment plan sweep.py --spec my_sweep` to preview the batches that would run, with no side effects.
3. Runs `uv run research experiment run sweep.py --spec my_sweep`. The CLI groups work into vmap-zones sharing static parameters and dispatches them, writing intent/provenance to `experiments.sqlite` and metric series to `metrics.sqlite`.
4. If the run is interrupted, rerunning the same `run` command re-plans and executes only the unsatisfied runs; completed work is never repeated.

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
