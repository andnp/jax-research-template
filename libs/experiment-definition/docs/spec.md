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
Researchers declare the metrics they intend to record (e.g., `reward`, `loss`, `latency`). Only declared metrics are recorded: the experiment's `train` function emits named arrays, and the runner persists the declared ones to `metrics.sqlite` after training.

## 3. The Definition Script

```python
from pathlib import Path

from experiment_definition.component import Component, ComponentType
from experiment_definition.experiment import Experiment

exp = Experiment("Policy Gradient Ablations")

# Define Generic Components
ppo = Component(
    name="PPO",
    path=Path("domains/rl/agents/src/rl_agents/ppo.py"),
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
    # Whether a parameter opens its own vmap zone is decided by the runner's
    # config: a field marked `traced(...)` varies along the batch axis.
    exp.add_parameter("arch", ["mlp", "cnn"])
    exp.add_parameter("lr", [1e-3, 3e-4])

    # Conditional logic
    exp.add_parameter("use_gae", [True, False])
    with exp.when(use_gae=True):
        exp.add_parameter("gae_lambda", [0.9, 0.95])

# Ablation: Test without GAE across all PPO runs
exp.add_ablation("no_gae", {"use_gae": False})

# Define Metrics. `kind` defaults to "float".
exp.add_metric("reward")
exp.add_metric("value_loss")
exp.add_metric("eigen_spread", kind="float")

# Sync to the relational database
exp.sync("experiments.sqlite")
```

### Running the definition

`research-runner` turns the definition into executions. An `ExperimentSpec`
names one results root; the database, metrics database and execution roots are
derived from it, and the root must be absolute so it cannot silently follow the
caller's working directory. The CLI finds a spec file's factories through
registration, not inspection: decorate each zero-argument factory with `@spec`
from `research_runner`, and `research experiment` discovers exactly the
decorated factories in that module.

```python
from dataclasses import dataclass
from pathlib import Path

import jax
from research_runner.config import traced
from research_runner.registry import spec
from research_runner.types import ExperimentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    arch: str = "mlp"                 # read by Python: one compiled kernel per value
    lr: float = traced(1e-3)          # read only by JAX: varies along the batch axis
    gamma: float = traced(0.99)
    use_gae: bool = True
    gae_lambda: float = traced(0.95)


def train(config: Config, key: jax.Array) -> tuple[object, dict[str, jax.Array]]:
    # Runs under jit(vmap(...)); `key` is the run's PRNGKey(seed). It must emit
    # every metric the experiment declares; undeclared ones are dropped.
    ...


@spec
def policy_gradient_ablations() -> ExperimentSpec[Config]:
    return ExperimentSpec(exp, config=Config(), train=train, results_root=PROJECT_ROOT / "results")
```

Run it with `research experiment run <spec_file.py>`, or preview the batches
with `research experiment plan <spec_file.py>`.

## 4. Requirement Gap Analysis (Fulfilled)
- [x] **Relational Schema Support:** Maps directly to ADR 008.
- [x] **JAX Performance:** Each run's vmap zone comes from the static keys its spec computes at sync.
- [x] **Ablations:** First-class support for both flags and component swaps.
- [x] **Project Scoping:** Strictly tied to the project-level SQLite database.

## 5. Technical Constraints
- Must be a lightweight Python package.
- Relies on `pydantic` for schema validation.
- Output must be a portable SQLite file that can be shared between researchers.
