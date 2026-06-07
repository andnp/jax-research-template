# Getting Started

This guide provides practical instructions for setting up your environment, running example baselines, and verifying repository integrity.

---

## 1. Prerequisites

Ensure your system has the following installed:
- **Python**: Version 3.13.
- **uv**: Astral's fast Python package manager and resolver.
- **Git**: For version control and submodule management.

---

## 2. Developer Environment Setup

1. **Install dependencies:**
   Sync the virtual environment and workspace packages:
   ```bash
   uv sync
   ```
   This automatically creates a `.venv` directory, installs all required dependencies (including JAX with CPU/GPU support as configured), and links local packages in editable mode.

2. **Install shared git hooks:**
   Install pre-commit hooks that enforce formatting, linting, and typing checks:
   ```bash
   ./scripts/install-git-hooks.sh
   ```
   This sets the git configuration `core.hooksPath` to `.githooks`. On commit, the hooks run:
   - `ruff check .` for linting.
   - `pyright` (or `pyrefly`) for type validation.

---

## 3. Running Example Baselines

The repository includes standalone and pipeline examples under the [examples/](file:///home/andy/Projects/research/research-monorepo/core/examples/) directory.

### Running a Standalone Agent
To train a baseline agent (e.g., PPO on CartPole) and check its learning throughput:
```bash
uv run examples/train_ppo.py
```
This script:
1. Compiles the JAX-native training loop on the first call.
2. Runs a second execution to measure Steps-Per-Second (SPS) without compilation overhead.
3. Generates a training curve saved as `ppo_results.png`.

### Running the End-to-End Experiment Pipeline
To run a complete, instrumented experiment:
```bash
uv run examples/run_experiment.py
```
This script demonstrates the interaction of the core framework libraries:
1. **Definition:** Declares hyperparameters, seeds, and metrics using the `experiment-definition` builder API, syncing them to a local SQLite schema (`experiment.sqlite`).
2. **Telemetry:** Configures `research-instrument` with an active metric whitelist and routes training updates (loss, episode returns) into `metrics.db`.
3. **Execution:** Runs a JIT-compiled DQN agent in JAX.
4. **Checkpointing:** Uses `research-store` to write pytree weights to the local store via atomic temp-and-rename writes.

---

## 4. Running Checks and Tests

Use the testing tiers to verify codebase changes before submitting pull requests.

### Diagnostic Sweep
To run a read-only sweep of your workspace structure, symlinks, and configuration:
```bash
uv run research doctor
```

### Pytest Suites
Tests are structured inside [tests/](file:///home/andy/Projects/research/research-monorepo/core/tests/) and separated by execution duration:

- **Small Tests** (fast math/logic assertions, `< 1ms` execution):
  ```bash
  uv run pytest tests/small
  ```
- **Medium Tests** (JIT compilations and step-based environment interactions, `< 1s` execution):
  ```bash
  uv run pytest tests/medium
  ```
- **Large Tests** (longer learning runs and database migrations):
  ```bash
  uv run pytest tests/large
  ```

To run formatting and typing checks manually:
```bash
uv run ruff check .
uv run pyrefly check
```
