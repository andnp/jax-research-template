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
  Creates a new experiment directory under `projects/` using `copier` templates. Automatically runs `git init` inside the new folder. Optional `--github-repo` creates a private remote repository via the GitHub `gh` CLI.

### Lifecycle Commands
- `eject <project> <library>`
  Copies a shared library from `libs/` to the project's local `components/` directory, updates the project's imports, and decouples the project from the shared package.
- `harvest <project> <library>`
  Moves a project-local component from `projects/<project>/components/` into `libs/`, registers the library in the workspace root `pyproject.toml`, and rewrites the import references back to the global package name.

### Diagnostics Command
- `doctor`
  Runs a read-only diagnostic sweep of the workspace to verify package directories, symlink mappings, configuration parameters, and dependency files.

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

### Workflow C: Customizing and Harvesting a Shared Library (The Eject-Harvest Loop)
When a shared library (e.g., `rl-agents`) needs custom modifications that are experimental or breaking for other projects:

1. **Eject the library into the project:**
   ```bash
   uv run research eject ppo-cartpole-sweep rl-agents
   ```
   **What happens under the hood:**
   - The contents of `libs/rl-agents/src/rl_agents/` are copied to `projects/ppo-cartpole-sweep/components/rl_agents/`.
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
   - The component is moved from the project folder back to `libs/rl-agents/src/rl_agents/`.
   - The CLI automatically rewrites import statements back to the global name: `components.rl_agents.ppo` becomes `rl_agents.ppo` in both the project files and the library files.
   - The CLI registers the library in the root workspace `pyproject.toml` dependencies, workspace sources, and `pyrefly` analysis paths.
