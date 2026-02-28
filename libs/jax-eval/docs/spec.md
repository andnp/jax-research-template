# PRD: `libs/jax-eval` (The RL Microscope)

## 1. Overview
`jax-eval` is a library of high-performance, JAX-native metrics and evaluation primitives. It allows researchers to instrument the hidden internals of RL agents (Representations, Optimization, Buffers) with zero runtime cost when whitelisted.

## 2. Categories of Collectables

### 2.1 Neural Representations
- **`dead_neurons(activations)`:** Returns the fraction of units with zero activation across a batch.
- **`stable_rank(matrix)`:** Computes the ratio of squared Frobenius norm to squared spectral norm.
- **`feature_drift(old_phi, new_phi)`:** Measures the cosine similarity between feature mappings before and after an update.

### 2.2 Optimization & Gradients
- **`gradient_snr(grads_batch)`:** Computes the Signal-to-Noise Ratio of gradients across vectorized seeds.
- **`update_ratio(params, updates)`:** Measures the norm of the update relative to the norm of the parameters.
- **`weight_norm(params)`:** Global and per-layer L2 norms.

### 2.3 Experience & Buffers
- **`priority_utilization(priorities)`:** Computes the entropy of the sampling distribution.
- **`sample_staleness(current_step, sampled_steps)`:** Average age of transitions in the current batch.

## 3. Technical Constraints
- **Trace-Time Whitelisting:** If a metric is not in the `research-instrument` whitelist, the `jax-eval` call must be a pure Python no-op.
- **E2E JIT:** All metrics must be compatible with `jax.jit` and `jax.vmap`.
- **Numerical Stability:** Use `jnp.where` and small epsilons to prevent NaNs in rank/norm calculations.

## 4. Usage Pattern
```python
from jax_eval import stable_rank
from research_instrument import collector

# In the training loop:
phi = model.apply(params, obs)
collector.write("stable_rank_layer1", stable_rank(phi), step)
```
