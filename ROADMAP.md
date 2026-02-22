# Monorepo Roadmap

This roadmap outlines the path to a fully functional, high-performance RL Research Monorepo.

## Phase 1: Core Orchestration (The Brain)
Focus on the management layer and the primary relational database.
- [ ] **`research-cli`**: Bootstrap Shell monorepos, manage projects, and provide the entry point for runs.
- [ ] **`experiment-definition`**: Implement the fluent builder API for relational experiment planning.
- [ ] **`libs/research-store`**: Implement atomic artifact management with Local and S3 backends.

## Phase 2: Runtime & Data (The Heart)
Develop the JAX-native libraries for execution and data collection.
- [ ] **`research-instrument`**: Build the JAX-native, double-buffered, whitelisted collector.
- [ ] **`jax-utils`**: Implement typed JIT/VMAP wrappers and Pytree math.
- [ ] **`jax-replay`**: Build advanced JIT-native buffers (PER, N-step, Trajectories).

## Phase 3: Science & Analysis (The Journal)
Tools for statistical rigor and publication readiness.
- [ ] **`research-analysis`**: Implement Patterson et al. (2023) statistical metrics and tests.
- [ ] **`research-plot`**: Develop the "Plot-as-Spec" system with data lineage tracking.

## Phase 4: Scaling & Cluster (The Muscle)
Optimizing for high-performance computing.
- [ ] **`research-cluster`**: Implement the Slurm job array mapper and node-local spooling.
- [ ] **`jax-nn`**: Harvest research-grade layers and initializers from successful projects.

## Phase 5: Harvesting (The Cycle)
- [ ] Migrate existing PPO, DQN, and SAC agents from `libs/rl-agents` into the new `experiment-definition` and `research-instrument` ecosystem.
- [ ] Create `copier` templates for standard research projects.

---

## Technical Debt & Maintenance
- [ ] Continuous integration (CI) for all libraries using `uv run pytest`.
- [ ] Enforce 100% type coverage via `ty check` in all `libs/`.
- [ ] Performance regression suite using `pytest-benchmark`.
