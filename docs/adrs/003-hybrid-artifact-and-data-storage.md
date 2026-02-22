# ADR 003: Hybrid Artifact & Data Storage

## Status
Proposed

## Context
RL research generates two distinct types of data:
1.  **Lightweight Metadata:** Configs, final plots, metrics (CSV/JSON), and the paper PDF.
2.  **Heavyweight Artifacts:** Neural network checkpoints, massive replay buffers, and high-frequency event logs (TensorBoard).

Git is unsuitable for the latter, even with LFS, due to the sheer volume and churn of research data. However, full reproducibility requires access to these artifacts.

## Decision
We will adopt a hybrid storage strategy managed by a specialized library (`libs/research-store`).

### 1. Git LFS (The "Paper Trail")
- Used for artifacts that are directly referenced in publications or are essential for quick analysis.
- Includes: Final result plots, small processed dataframes, and the compiled paper PDF.

### 2. External Object Storage (The "Research Vault")
- Large binary blobs will be stored in cloud object storage (S3/GCS) or a local NAS.
- Includes: All training checkpoints and full replay buffer dumps.
- **The Abstraction:** The `research-store` library will provide a unified API (e.g., `store.put(name, data)`) that handles local caching and remote syncing.

### 3. Immutable Index
- Every experiment will generate a unique hash (UUID) that serves as the key in the external store. 
- This hash will be recorded in the Git-tracked metadata to link the code version to the binary artifacts.

## Consequences
-   **Pros:** Keeps the monorepo lightweight and fast; ensures long-term reproducibility without repo bloat.
-   **Cons:** Requires external infrastructure (S3 bucket or NAS); slightly more complex data retrieval for collaborators.
