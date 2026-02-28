# Monorepo Roadmap

This roadmap outlines the path to a fully functional, high-performance RL Research Monorepo.

## Phase 1: Core Orchestration (The Brain)
Focus on the management layer and the primary relational database.
- [x] **`research-cli`**: Bootstrap Shell monorepos, manage projects, and provide the entry point for runs.
- [x] **`experiment-definition`**: Implement the fluent builder API for relational experiment planning.
- [x] **`libs/research-store`**: Implement atomic artifact management with Local and S3 backends.

## Phase 2: Runtime & Data (The Heart)
Develop the JAX-native libraries for execution and data collection.
- [x] **`research-instrument`**: Build the JAX-native, double-buffered, whitelisted collector.
- [x] **`jax-utils`**: Implement typed JIT/VMAP wrappers and Pytree math.
- [x] **`jax-replay`**: Build advanced JIT-native buffers (PER, N-step, Trajectories).

## Phase 3: Science & Analysis (The Journal)
Tools for statistical rigor and publication readiness.
- [x] **`research-analysis`**: Implement bootstrap CIs and Welch's t-test for algorithm comparison.
- [ ] **`research-plot`**: Develop the "Plot-as-Spec" system with data lineage tracking.

## Phase 4: Scaling & Cluster (The Muscle)
Optimizing for high-performance computing.
- [ ] **`research-cluster`**: Implement the Slurm job array mapper and node-local spooling.
- [x] **`jax-nn`**: Harvest research-grade layers and initializers from successful projects.

## Phase 5: Harvesting (The Cycle)
- [x] Migrate existing PPO, DQN, and SAC agents into the `experiment-definition` and `research-instrument` ecosystem.
- [x] Create `copier` templates for standard research projects.
- [x] Add Double DQN and Dueling DQN agent variants.

---

## Technical Debt & Maintenance
- [x] Continuous integration (CI) for all libraries using `uv run pytest`.
- [ ] Enforce 100% type coverage via `ty check` in all `libs/`.
- [ ] Performance regression suite using `pytest-benchmark`.
- [x] Add `research-instrument` SQLite persistent storage backend.
- [x] Add rl-components test coverage (buffer, networks, types).
- [x] Add rl-agents unit tests (config, loss functions, gradient flow).
