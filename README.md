# RL Research Core

A high-performance, E2E JIT-able framework for Reinforcement Learning research.

## Project Philosophy
- **One Agent, One World:** We scale via seed-parallelism, not environment-parallelism.
- **AI-Native:** Designed for agentic programming and automated experimentation.
- **Compound Interest:** Shared libraries grow as research successes are "harvested."

## Repository Structure
- `libs/`: Domain-agnostic shared packages (JAX utilities, RL components, experiment definition, runner, storage, analysis). They may import only other `libs` packages.
- `domains/`: Code tied to one paper or environment family and reused across projects (RL agents, networks, environments, process control). See [ADR 011](docs/adrs/011-libs-domains-split.md).
- `cli/`: The `research` command-line tool for managing the monorepo.
- `templates/`: `copier` templates for spinning up new research projects.
- `examples/`: Reference implementations and baseline results.
- `docs/`: Constitutional documents, ADRs, and technical specs.

## Getting Started
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and [AGENTS.md](AGENTS.md) for agentic workflow instructions.
