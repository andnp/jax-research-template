# ADR 004: JAX Programming Patterns (E2E JIT)

## Status
Proposed

## Context
Maximum computational performance in RL requires minimizing the boundary between Python and the accelerator (GPU/TPU). Any non-JAX-native code inside the training loop triggers costly CPU-GPU synchronization.

## Decision
All agents and shared components in the Core must be **End-to-End JIT-able**.

### 1. Functional Loops
- Use `jax.lax.scan` for all environment interaction and training loops.
- Use `jax.lax.while_loop` or `jax.lax.fori_loop` only when `scan` is inapplicable.
- Avoid Python `for` loops inside JIT-compiled functions.

### 2. "One Agent, One World" (Parallelism via VMAP)
- Agent logic must be written for a single agent in a single environment (unvectorized).
- Scaling is achieved by `vmap`-ing or `pmap`-ing the entire `train` function over PRNG keys.
- Internal `NUM_ENVS` vectorization is forbidden in shared `libs/`.

### 3. Linearized Control Flow
- Prefer `jax.lax.select` or masking (multiplying by 0/1) over `jax.lax.cond` for small branches to maximize kernel fusion.
- Use `jax.debug.callback` only for non-critical logging and never for training logic.

## Consequences
-   **Pros:** Extreme throughput (~250k+ SPS on CPU for PPO); massive scalability over seeds; deterministic results.
-   **Cons:** Higher cognitive load for researchers; debugging JIT-compiled tracers is more difficult than standard Python.
