# Product Principles: Research Monorepo

## 1. Research Speed as the North Star
The primary goal is to minimize the latency between "having an idea" and "seeing a result."
- **Low-Friction Context Switching:** Switching between projects must be nearly instantaneous. Shared tooling and consistent structures reduce mental load.
- **CLI-First Management:** Routine tasks (setup, execution, plotting) are offloaded to robust CLI tools, not remembered recipes.

## 2. Low Floor, High Ceiling
We serve both the first-year student and the industrial researcher.
- **Frictionless Onboarding:** A new user should have a running experiment in < 5 minutes with zero infrastructure setup. Defaults are always "local" and "automatic."
- **Industrial Scale:** The same codebase must seamlessly scale to multi-node Slurm clusters with S3 backends and TimescaleDB metrics without code changes.

## 3. Declarative Science
We decouple the scientific intent from the physical execution.
- **Intent via Database:** Experiment configurations, hyperparameters, and seeds are populated into a relational database *before* execution.
- **Stateless Runners:** Runners load work from the DB, allowing for automatic resumptions, gap-filling, and perfect traceability.

## 4. Human-Centric Validity
The researcher is the ultimate arbiter of scientific meaning.
- **Researcher-in-the-loop:** The system uses code-hashing to *recommend* version bumps, but never forces them.
- **Scientific Versioning:** Invalidation of results is handled via semantic versions, not by deleting data. History is immutable.

## 5. Pragmatic Abstraction (The "Rule of Three")
Shared libraries are earned through usage, not predicted through foresight.
- **Harvesting:** Proactively move code from `projects/` to `libs/` only after it has been proven in three distinct projects.
- **Ejecting:** Encourage local "hacking" by allowing users to eject core libraries into their projects for experimentation.

## 6. Transparency of the Machine
We prioritize understanding computational performance as a scientific requirement.
- **Performance Introspection:** Tools for HLO dumping and JAX profiling are first-class citizens.
- **Hardware-Aligned Design:** We adhere to functional JAX patterns (`lax.scan`, `vmap-zones`) to maximize accelerator utilization.

## 7. AI-Native Development
The repository is a collaboration between humans and AI agents.
- **Spec-Driven:** We define "what" before "how," allowing agents to implement against clear interfaces.
- **Context-Aware:** Directory structures are optimized for agent context windows (flat structures, explicit typing).
