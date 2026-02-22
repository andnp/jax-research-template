# PRD: `libs/research-store`

## 1. Overview
`research-store` is a unified artifact and checkpoint management library. It abstracts the physical location of research data (S3, GCS, Local Disk) behind a simple "Research URI" interface.

### Goals
- **Zero-Config Onboarding:** Default to a simple local directory for students and rapid prototyping.
- **Scale-Ready:** Seamlessly upgrade to Cloud/NAS backends for large-scale PhD research.
- **Atomic Integrity:** Ensure that binary artifacts (like checkpoints) are either fully saved or clearly marked as corrupt/incomplete.
- **Metadata Linked:** Integrate with the relational database to link code versions, hyperparameters, and physical artifacts.

## 2. Storage Tiers

### 2.1 Tier 1: Local Mode (Default)
- **Target:** Students, Local Workstations.
- **Mechanism:** Files are stored in a standard directory structure (e.g., `~/research_results/artifacts/`).
- **Setup:** Zero. The library initializes the folder on the first `put()`.

### 2.2 Tier 2: Cloud/Remote Mode
- **Target:** Cluster users, multi-node experiments.
- **Mechanism:** Files are written to a local cache and asynchronously synced to a remote bucket (S3/GCS).
- **Setup:** Requires credentials via environment variables or a `research.yaml` config.

## 3. Core API

### 3.1 `put(blob: Any, name: str, execution_id: UUID) -> StoreURI`
Persists a binary object. 
- Handles serialization automatically (using `orbax` for JAX Pytrees or `pickle` for generic Python objects).
- Returns a unique `research://` URI.

### 3.2 `get(uri: StoreURI) -> Any`
Retrieves a binary object.
- Handles downloading from remote backends if the local cache is empty.

### 3.3 `sync()`
Forces a synchronization between the local cache and the remote backend. Essential for ensuring Slurm jobs have pushed their final weights before finishing.

## 4. Key Features

### 4.1 "Research URI" Abstraction
Researchers never hardcode paths. They use a standard URI format:
`research://<experiment_id>/<execution_id>/<artifact_name>:<version>`
This allows the same code to run locally or on a massive cluster without changes.

### 4.2 Integrated Checkpointing (Orbax Wrapper)
Since JAX is the primary focus, `research-store` provides a high-level wrapper around **Orbax**.
- Handles standard checkpoint rotations (e.g., "Keep the last 5 checkpoints").
- Automatically generates the metadata required for the "Resume Logic" in `research-instrument`.

### 4.3 Atomic Flushes
- Uses "Hidden Staging": Blobs are written to `.tmp_<name>` and renamed only after successful completion.
- Prevents the "Empty Checkpoint" bug that occurs when a job is killed mid-write.

## 5. Technical Constraints
- Must be a lightweight Python package.
- Relies on `orbax-checkpoint` for JAX-native serialization.
- Backend providers (e.g., `boto3` for S3) should be optional dependencies to keep the "Local Mode" lean.

## 6. Proposed Usage

```python
from research_store import store

# Easy Local Onboarding
# No config required, defaults to ./artifacts
checkpoint_uri = store.put(params, name="policy_weights", execution_id=current_id)

# Later in analysis script
params = store.get(checkpoint_uri)
```
