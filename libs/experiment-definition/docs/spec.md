# PRD: `libs/experiment-definition`

## 1. Overview
`experiment-definition` is a JAX-friendly, framework-agnostic library for defining scientific experiments. It decouples the *intent* of an experiment (search spaces, components, seeds) from its *execution* (Slurm, Local, JIT-batching).

### Goals
- **Genericism:** All building blocks (algorithms, simulators, datasets) are treated as `Components`.
- **Researcher-in-the-Loop:** Facilitates versioning by tracking code hashes and prompted intent.
- **Hardware Optimization:** Tags parameters as `static` (recompilation required) or `dynamic` (vmap-ready).
- **Metric-Aware:** Defines expected metrics upfront to optimize the data collection and storage layers.

## 2. Core Concepts

### 2.1 The Component
A logical unit of code. In RL, this might be an `Agent` or an `Environment`. In CV, it might be a `Model` or a `Dataset`. 
- Every component tracks its source path for automatic hashing.

### 2.2 The Parameter Space
- **Standard Parameters:** Fixed values or lists for grid/random sweeps.
- **Conditional Parameters:** Triggered via `exp.when(param=value)`. Essential for complex configurations (e.g., specific optimizer settings).
- **Ablations:** High-level overrides that clone the search space to test the removal of features or the swapping of entire components.

### 2.3 Metric Specifications
Researchers define the metrics they intend to collect (e.g., `reward`, `loss`, `latency`).
- **Standard Metrics:** Simple scalar writes (`collector.write`).
- **Eval Metrics:** Expensive computations executed on-demand (`collector.eval`).
- **Optimization:** Metrics not defined in the `spec` are ignored by the instrumentation layer to save compute and disk space.

## 3. The Definition Script

```python
from pathlib import Path

from experiment_definition import Component, ComponentType, Experiment

exp = Experiment("Policy Gradient Ablations")

# Define Generic Components
ppo = Component(
    name="PPO",
    path=Path("libs/rl-agents/src/rl_agents/ppo.py"),
    type=ComponentType.ALGO,
)
env = Component(
    name="CartPole",
    path=Path("libs/rl-components/src/rl_components/envs.py"),
    type=ComponentType.ENV,
)

# Global Hyperparameters
exp.add_parameter("seed", range(100))
exp.add_parameter("gamma", [0.99])

# Component-Specific Sweeps
with exp.for_component(ppo):
    # A static parameter changes the compiled kernel, so it opens its own
    # vmap zone. Everything else varies along the batch axis.
    exp.add_parameter("arch", ["mlp", "cnn"], is_static=True)
    exp.add_parameter("lr", [1e-3, 3e-4])

    # Conditional logic
    exp.add_parameter("use_gae", [True, False])
    with exp.when(use_gae=True):
        exp.add_parameter("gae_lambda", [0.9, 0.95])

# Ablation: Test without GAE across all PPO runs
exp.add_ablation("no_gae", {"use_gae": False})

# Define Metrics (Informs the collector). `kind` defaults to "float".
exp.add_metric("reward", frequency="per_episode")
exp.add_metric("value_loss", frequency="per_update")
exp.add_metric("eigen_spread", kind="float", frequency="eval_only")

# Sync to the relational database
exp.sync("experiments.sqlite")
```

### Running the definition

`research-runner` turns the definition into executions. An `ExperimentSpec`
names one results root; the database, metrics database and execution roots are
derived from it, and the root must be absolute so it cannot silently follow the
caller's working directory.

```python
from pathlib import Path

from research_runner import ExecutionContext, ExecutionResult, ExperimentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train(ctx: ExecutionContext) -> ExecutionResult:
    # Every run in the batch shares `static_config`, so it is safe to close
    # over when tracing. The parameters that vary arrive per point.
    arch = ctx.static_config["arch"]
    for point in ctx.points:
        print(point.run_id, point.hyperparameters["lr"], point.hyperparameters["seed"])

    # `axes()` stacks the varying parameters in `points` order, ready to be
    # mapped over as a single vmap axis.
    lrs = ctx.axes()["lr"]
    del arch, lrs
    return ExecutionResult(metadata={})


def policy_gradient_ablations() -> ExperimentSpec:
    return ExperimentSpec(
        experiment=exp,
        train_fn=train,
        results_root=PROJECT_ROOT / "results",
    )
```

Run it with `research experiment run <spec_file.py>`, or preview the batches
with `research experiment plan <spec_file.py>`.

## 4. Requirement Gap Analysis (Fulfilled)
- [x] **Relational Schema Support:** Maps directly to ADR 008.
- [x] **JAX Performance:** Support for `is_static` metadata to define vmap-zones.
- [x] **Ablations:** First-class support for both flags and component swaps.
- [x] **Collector Integration:** Metric definitions serve as a whitelist for the instrumentation library.
- [x] **Project Scoping:** Strictly tied to the project-level SQLite database.

## 5. Technical Constraints
- Must be a lightweight Python package.
- Relies on `pydantic` for schema validation.
- Output must be a portable SQLite file that can be shared between researchers.
