# ADR 006: The "Harvesting" Lifecycle

## Status
Proposed

## Context
Code in a research monorepo naturally moves through stages of maturity. Prematurely moving code into shared `libs/` creates "API debt" and slows down exploration.

## Decision
We will follow a strict **Harvesting Lifecycle** based on the "Rule of Three."

### 1. Prototype Stage (in `projects/`)
- New algorithms, buffers, or layers are written directly within a project.
- Duplication between projects is acceptable at this stage to maintain high-churn speed.

### 2. Harvest Stage (into `libs/`)
- Code is "harvested" into the Core only when it is needed by a **third** project.
- During harvesting, the code must be refactored to meet the Core's strict typing, documentation, and testing standards.

### 3. Eject/Fork Stage (out of `libs/`)
- If a project needs to modify a core library in a way that breaks other projects, it must **eject** the library. 
- The CLI will copy the library into the project's source, allowing for local hacking without upstream side-effects.

### 4. Convergence (Propose back to `libs/`)
- Ejected or newly harvested code should be proposed back to the Core via the `research propose` workflow once it has proven its value and performance.

## Consequences
-   **Pros:** Keeps the Core lean and high-quality; prevents "genericism" from killing research speed; provides a clear path for code reuse.
-   **Cons:** Requires manual (or CLI-assisted) management of code movement.
