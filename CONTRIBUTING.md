# Contributing to Research Core

We prioritize reproducibility, high performance, and code reuse.

## The Contribution Cycle (Hub-and-Spoke)

1.  **Iterate in Projects:** Develop new ideas within a project folder. 
2.  **Eject for Hacking:** If a core library (`libs/`) doesn't quite fit your need, use `research eject <lib>` to fork it into your project.
3.  **Harvest for Reuse:** Once a component has been used successfully in three projects, it should be "harvested" into `libs/`.
4.  **Propose Changes:** Use `research propose <lib>` to submit your harvested or modified code back to the Core via Pull Request.

## Standards

### 1. Development Environment
We use `uv` for dependency management.
```bash
uv sync
uv run ty check .
uv run ruff check .
```

### 2. Testing Expectations
- **Hermetic:** Tests must not depend on external state. Use `tmp_dir` for all I/O.
- **Cleanup:** Tests must clean up all artifacts.
- **Tiers:** Place tests in the correct folder based on runtime (`small/`, `medium/`, `large/`).

### 3. Git Etiquette
- Use **Conventional Commits** (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
- Keep PRs focused. One harvested library per PR.
- Include benchmark results in PR descriptions for performance-sensitive changes.
