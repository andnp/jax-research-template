# PRD: `libs/research-instrument`

## 1. Overview
`research-instrument` is a JAX-native, high-performance data collection library. It follows the "One Agent, One World" philosophy, allowing researchers to instrument single-agent code that scales seamlessly via `vmap` or `pmap`.

### Goals
- **JAX-Native:** Works inside `jit`, `vmap`, `scan`, and `cond`.
- **Zero-Contamination:** Instrument code anywhere (agent, env, buffer) without passing through logging objects manually.
- **Whitelist-Driven:** Only collects metrics defined in the `experiment-definition` spec to minimize overhead.
- **Asynchronous & Buffered:** Offloads the heavy lifting of DB writes to background threads to keep the accelerator saturated.

## 2. Core Concepts

### 2.1 The Collector
The primary interface for the researcher. It is globally accessible (via JAX context or a singleton pattern) to avoid "argument drilling."

### 2.2 The Frame
A logical unit of time (usually a timestep or an update). All collected data is associated with a `frame` and an `agent_id` (vmap index).

### 2.3 The Whitelist
At the start of an experiment, the collector loads the spec from the SQLite DB. Any `write()` or `eval()` call for a metric not in the whitelist is converted into a `jax.lax.no_op`, ensuring zero runtime cost for unused instrumentation.

## 3. The API

### 3.1 `write(name: str, value: Array)`
The standard way to log a metric.
- **In JIT:** Wraps the value in `jax.debug.callback` to stream data to the background worker.
- **Vmap-Aware:** Automatically captures the current `vmap` index to associate the value with the correct seed/agent in the DB.

### 3.2 `eval(name: str, fn: Callable[[], Array])`
For computationally expensive metrics (e.g., computing the spectral radius of a weight matrix).
- The `fn` is only executed if the metric is enabled and the current frame satisfies the requested frequency (e.g., every 1000 steps).
- Uses `jax.lax.cond` to ensure the computation is skipped entirely on the accelerator when not needed.

### 3.3 `summary()`
Returns a high-level summary of the current experiment's progress (mean rewards, SPS, etc.) for CLI display.

## 4. Execution Modes

### 4.1 Real-Time Mode (Callback-based)
Uses `jax.debug.callback` to send data points immediately to a background thread.
- **Pros:** Real-time monitoring, lower memory usage on the accelerator.
- **Cons:** Can introduce slight overhead due to Python callbacks.

### 4.2 Batch Mode (Accumulation-based)
Accumulates all metrics into a JAX Pytree during `jax.lax.scan` and returns them at the end of the `train()` call.
- **Pros:** Maximum performance (zero CPU-GPU sync).
- **Cons:** High memory usage (stores the entire history on the GPU).

## 5. Storage Architecture

`research-instrument` uses a pluggable backend system to satisfy different infrastructure constraints.

### 5.1 The Storage Interface
Backends must implement the `StorageBackend` protocol:
- `write_batch(frames: list[DataFrame])`: Writes a chunk of metrics.
- `flush()`: Blocks until all in-flight data is persisted.
- `close()`: Finalizes the connection and ensures data integrity.

### 5.2 Provided Backends

#### A. `SqliteBackend` (The Default)
- **Use Case:** Local dev, small experiments.
- **Mechanism:** Direct writes to a single `.sqlite` file. 
- **Pros:** Zero setup, portable.

#### B. `ParquetBackend` (The HPC / Student Choice)
- **Use Case:** HPC clusters with fragile Distributed File Systems (DFS).
- **Mechanism:** Writes metrics to local node storage (`/tmp` or `/scratch`) as a "Spool." Periodically flushes large, compressed `.parquet` files to a unique directory on the DFS.
- **Pros:** Minimizes metadata operations on Lustre/BeeGFS; extreme read performance for analysis scripts.

#### C. `TimescaleBackend` (The Power-User Choice)
- **Use Case:** Dedicated research servers with Docker/Database support.
- **Mechanism:** Streams data via `psycopg3` to a TimescaleDB instance.
- **Pros:** Relational queries over billions of rows; advanced time-series analysis (e.g., hyperparam importance over time).

### 5.3 Reliability & Consistency
- **Atomic Recovery (Resume Logic):** To prevent gaps or overlaps in curves after a preemption, the collector's state (specifically `last_flushed_frame`) must be stored within the agent's model checkpoint. On resume, the collector fast-forwards to match the exact state of the physics/policy.
- **Data-Driven Checkpointing:** The instrumentation layer can trigger agent checkpoints. `agent.save()` should ideally only occur immediately following a successful `collector.flush()`, ensuring that every saved model has a corresponding and complete log history.
- **Strict Schema Locking:** To prevent "polluted" data that crashes analysis scripts, the schema (names, shapes, types of metrics) is locked per `ExecutionID`. Any attempt to change a metric's type mid-run will trigger an error or force the creation of a new Execution version.
- **Deterministic Sparse Evaluation:** `collector.eval` schedules are strictly tied to `global_step % frequency == 0`. This ensures that sparse metrics are collected at the correct intervals even across multiple resumes.

### 5.4 Performance & Scale
- **Adaptive Throttling:** If the background write queue exceeds a configurable memory limit (e.g., due to DFS latency), the collector automatically shifts from `per_step` to `subsampled` or `aggregated` mode. This "safety valve" prevents a single runaway experiment from saturating the cluster's I/O.
- **Collaborative Live-View:** A dedicated CLI tool (`research monitor <id>`) provides a way to "peak" at the local spool on a worker node, giving researchers real-time feedback even when the persistent backend (like Parquet) has long flush intervals.

## 6. Implementation Strategy: "The Buffer"
To keep JAX running at full speed, the library maintains a **Double Buffer**:
1. JAX pushes data into **Buffer A** (via `jax.debug.callback`).
2. When **Buffer A** is full, it swaps with **Buffer B**.
3. A background thread drains the full buffer into the `StorageBackend`.
4. If the background thread falls behind, the JAX thread can be configured to either **Drop** (for non-critical telemetry) or **Block** (for critical metrics).



## 6. Proposed Usage

```python
# In the experiment definition
# exp.add_metric("reward", frequency="per_step")

# In the agent/environment code
from research_instrument import collector

def step(state, action):
    next_state, reward = env.step(state, action)
    
    # This is a no-op if "reward" isn't in the whitelist
    collector.write("reward", reward)
    
    # Expensive metric run only once per 1000 steps
    collector.eval("weight_norm", lambda: jnp.linalg.norm(params), every=1000)
    
    return next_state, reward
```
