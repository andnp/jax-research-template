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

- **Typecheck Suite:** `pyrefly` is the primary static analysis typechecker for code in `libs/`, with `pyright` as a secondary validator.
- **Zero Type Errors:** Code merged into the core libraries must have 100% type coverage.
- **Strict Any:** Avoid `Any` type annotations in public interfaces. Use generic parameters (using PEP 695 syntax, e.g. `def process[T](data: list[T]) -> T:`) or define explicit protocols for structural typing.
- **Standard Type Annotations:** Use modern Python standards:
  - Union types: `int | str` instead of `typing.Union[int, str]`.
  - Built-in generics: `list[T]` and `dict[K, V]` instead of importing `List` or `Dict`.
  - Optional values: `float | None` instead of `typing.Optional[float]`.

---

## 4. Telemetry and Storage Architecture

Logging metrics and saving model weights must not interfere with performance.

### Double-Buffered Metric Collection (`research-instrument`)
To log metrics from JIT-compiled JAX execution traces, `research-instrument` uses `jax.debug.callback` to safely pass arrays back to Python.
- **Non-blocking Spooling:** To avoid slowing down the GPU, metrics are written to a double-buffered queue in memory. When Buffer A fills up, a background thread flushes it to the SQLite metrics database (`metrics.db`) while JAX continues writing to Buffer B.
- **Declarative Whitelisting:** The instrumentation suite reads the synced experiment configuration database to only capture active metrics, reducing I/O overhead.

### Atomic Weight Checkpointing (`research-store`)
Checkpoints are serialized using Orbax and managed by `research-store`.
- **Temp-and-Rename writes:** To prevent database or file corruption during training crashes, weights are written to a temporary folder on the same storage volume and then renamed atomically.
- **Storage Abstraction:** The storage layer generates `research://` URIs, allowing runners to seamlessly write to local folders or remote S3 buckets by changing environment variables.
