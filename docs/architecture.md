# Architecture & Standards

This document describes the architectural decisions, coding patterns, and enforcement tools used in the RL Research Core.

---

## 1. Execution Philosophy: "One Agent, One World"

To maximize hardware accelerator efficiency, this repository prioritizes seed-parallelism over environment-parallelism.

- **Vmap-Driven Scaling:** Instead of scaling via distributed CPU environment processes communicating with a central GPU agent, we compile the environment-agent interaction loop directly into JAX.
- **Accelerator Bounds:** Running seed sweeps is achieved via JAX's `vmap` (vectorized map) or `pmap` (parallel map) across accelerators. The entire trajectory collection, action selection, and optimization updates execute in a single unified compilation trace.
- **Determinism:** PRNG states (`jax.random.PRNGKey`) are propagated explicitly throughout all components to ensure perfect numerical reproducibility.

---

## 2. Python Module & Import Conventions

To facilitate code search, refactoring, and static analysis, the codebase enforces strict guidelines for imports and module exports:

### No Re-exports in Module Init Files
`__init__.py` files must remain empty (or contain only a module-level docstring). 
- **Conventions:** Always import directly from the defining submodule:
  ```python
  # Correct
  from rl_agents.ppo import make_train

  # Incorrect
  from rl_agents import make_train
  ```
- **Rationale:** Empty init files prevent circular dependency bugs, keep dependency graphs clear, and make it easy for code editors to identify precise function definitions.

### No `__all__`
Do not define `__all__` lists in any module. Every public class, function, or variable (anything not prefixed with an underscore) is treated as part of the public module namespace.

---

## 3. Type Validation with `pyrefly`

We enforce static typing to catch logic errors before running slow execution trials.

- **Typecheck Suite:** `pyrefly` is the static analysis typechecker for code in `libs/`.
- **Zero Type Errors:** Code merged into the core libraries must have 100% type coverage.
- **Strict Any:** Avoid `Any` type annotations in public interfaces. Use generic parameters (using PEP 695 syntax, e.g. `def process[T](data: list[T]) -> T:`) or define explicit protocols for structural typing.
- **Standard Type Annotations:** Use modern Python standards:
  - Union types: `int | str` instead of `typing.Union[int, str]`.
  - Built-in generics: `list[T]` and `dict[K, V]` instead of importing `List` or `Dict`.
  - Optional values: `float | None` instead of `typing.Optional[float]`.

---

## 4. Telemetry and Storage Architecture

Logging metrics and saving model weights must not interfere with performance.

### Post-Training Metric Persistence (`research-instrument`)
Training runs entirely inside JIT and returns its metric curves as arrays; nothing is logged from inside the trace.
- **Runner writes once:** After a batch finishes, `research-runner` persists curves with `seed_batch.write_seed_batch_curve` and scalars with `scalar.write_seed_batch_scalars` / `write_execution_scalars`, each in a single transaction into `metrics.sqlite`.
- **One schema owner:** `sqlite_backend` owns the `metrics` and `scalar_metrics` DDL, the shared insert column lists, schema migration, and the rollback-journal connection that is safe for concurrent Slurm tasks on a network filesystem.
- **Declared metrics:** Which metrics exist is declared on the experiment definition (`add_metric`); the trainer's returned metrics are what get written.

### Atomic Weight Checkpointing (`research-store`)
Checkpoints are serialized using Orbax and managed by `research-store`.
- **Temp-and-Rename writes:** To prevent database or file corruption during training crashes, weights are written to a temporary folder on the same storage volume and then renamed atomically.
- **Storage Abstraction:** The storage layer generates `research://` URIs, allowing runners to seamlessly write to local folders or remote S3 buckets by changing environment variables.

---

## 5. Package Tiers: `libs/` vs `domains/` (ADR-011)

`core` splits shared packages into two dependency tiers instead of one flat `core/libs/`:

- **`libs/`** — domain-agnostic code, or domain-specific primitives with no fixed hyperparameters/environment assumptions (e.g. `jax-nn`, `rl-components`, `ml-components`). `libs` packages may depend on other `libs` packages only.
- **`domains/`** — code tied to one paper's or one environment family's specific implementation, reused across multiple projects (e.g. `domains/rl/{agents,networks,envs}`, `domains/process-control`). `domains` packages may depend on other `domains` packages and any `libs` package.

`libs` must never import from `domains` — this is the one dependency-direction rule the whole split exists to enforce. See ADR-011 for the tier test and full rationale.
