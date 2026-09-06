# Technical Specification: `libs/jax-eval`

Status: planned; package stub only

Last reconciled: 2026-09-05

## 1. Purpose

`jax-eval` will provide pure, JAX-native diagnostic kernels for inspecting RL
representations, optimization dynamics, and replay behavior. A kernel accepts
arrays or pytrees and returns a scalar or a small fixed-shape JAX pytree. It
must be usable inside `jax.jit`, `jax.vmap`, and `jax.lax.scan` without Python
callbacks, mutable state, storage access, or knowledge of an experiment
database.

The package is a microscope, not the microscope slide or the camera: callers
choose the observations, parameters, gradients, and replay state to inspect;
`research-instrument` decides whether and where the resulting scalar is
recorded.

## 2. Current reality and reconciliation

The original document described an integrated metric library, but the current
workspace contains only `docs/spec.md` and `pyproject.toml`. There is no
`src/jax_eval` package, no test suite for it, and the package is currently
configured with `package = false` and no dependencies.

Several responsibilities already exist elsewhere:

| Concern | Current owner | Consequence for `jax-eval` |
| --- | --- | --- |
| Generic pytree norms and inner products | `jax-utils` | Reuse these operations; do not duplicate tree arithmetic. |
| Feature-covariance stable rank | `jax-utils.linalg.stable_rank` | Treat this as the current canonical implementation. Its contract is feature covariance stable rank, not arbitrary matrix stable rank. |
| Metric whitelist, JIT callbacks, vmap seed IDs, and SQLite frames | `research-instrument` | Remove whitelist behavior from this package. A diagnostic is pure; the collector is the optional effect boundary. |
| Metric declarations and frequencies | `experiment-definition` | Keep metric names and collection cadence in the experiment definition. |
| Fixed-probe construction and project-specific diagnostics | Research projects, currently `reversible-rank-rl` | Keep benchmark-specific probes and scientific aggregates in the project until a second consumer establishes a reusable contract. |
| JAX rollout mechanics | `rl-components.rollout` | Do not build a second environment loop here. |
| Dataset loading and statistical reporting | `research-analysis` | Do not make this package open SQLite files or compute study-level estimands. |

The old usage example is therefore corrected: a caller imports the current
stable-rank implementation from `jax_utils.linalg`, computes a diagnostic, and
passes the scalar to `research_instrument.collector`.

## 3. Goals

- Provide a small set of well-defined diagnostics useful across RL projects.
- Make axis conventions, estimands, tolerances, and zero-denominator behavior
  explicit so results are comparable across experiments.
- Preserve the one-agent/one-world design: a caller writes a single-agent
  function and applies `vmap` over seeds or independent trials.
- Return finite values for ordinary degenerate inputs such as zero parameters,
  zero gradients, uniform priorities, and zero feature vectors.
- Keep diagnostic computation independent of storage, callbacks, experiment
  definitions, and environment implementations.

## 4. Non-goals

The first release will not:

- own a training or evaluation loop, environment reset protocol, or checkpoint
  loader;
- emit callbacks or write SQLite, Parquet, or other artifacts;
- inspect an experiment definition or decide whether a metric is whitelisted;
- provide statistical confidence intervals, curve summaries, or reporting;
- expose arbitrary model internals through reflection;
- reproduce every metric from the research literature without a project-backed
  use case.

If a reusable evaluation loop is later needed, it should compose the canonical
rollout protocol and runner rather than becoming a second orchestration path in
this package.

## 5. API and numerical conventions

The implementation should use direct submodule imports, with an empty
`jax_eval.__init__`. Proposed modules are:

```text
src/jax_eval/
    __init__.py
    representations.py
    optimization.py
    replay.py
```

All public functions should:

- accept JAX arrays or explicitly supported pytrees, not NumPy-only values;
- return a scalar `jax.Array` unless the function explicitly documents a
  fixed-shape result;
- reduce in `float32` for numerical stability, even when inputs are lower
  precision, and return a floating-point scalar;
- use static axis structure and JAX operations only; Python branches may select
  compile-time options but must not inspect tracer values;
- document the sample/batch axis and feature or parameter axes;
- define behavior for zero norms, empty support, and zero denominators;
- avoid silently flattening semantically unrelated axes.

The first implementation should depend on `jax` and the published `jax-utils`
workspace package. It should not depend on `research-instrument`,
`experiment-definition`, or a project package.

## 6. Diagnostic contracts

### 6.1 Representation diagnostics

#### `dead_neurons(activations, *, atol=0.0)`

`activations` has shape `(..., units)` with at least one observation. A one-
dimensional input `(units,)` is treated as one observation. An empty observation
batch returns `0.0`. A unit is dead when every value over the observation axes
satisfies `abs(value) <= atol`. The result is the fraction of dead units, not
the fraction of zero entries. This distinction is important: a unit that
activates on one state is not dead even if most entries in its column are zero.

The default `atol=0.0` measures exact inactivity, which is appropriate for
ReLU-style activations. A caller may choose a positive threshold for noisy or
low-precision activations. The result is `0.0` for an input with no units.

#### `feature_drift(old_features, new_features, *, epsilon=1e-8)`

Both inputs have the same shape `(..., features)`. The kernel computes the mean
row-wise cosine distance over the leading observation axes. For vectors `u` and
`v`, define cosine similarity as `0.0` when either norm is at most `epsilon`;
otherwise compute the cosine ratio and clip it to `[-1.0, 1.0]`.

\[
  D = \operatorname{mean}_{x}\left(1 - \operatorname{sim}(\phi_{old}(x),\phi_{new}(x))\right).
\]

The name now matches the quantity: `0.0` means identical direction and larger
values mean more change. If either vector is effectively zero, its cosine
similarity is defined as `0.0`, so the distance is finite and equals `1.0`.
The explicit similarity clip prevents roundoff from producing negative drift.
Callers should compare features on the same fixed probe states when measuring
training-time representation change.
An empty leading observation batch returns `0.0`.

#### Stable rank

The existing canonical function is `jax_utils.linalg.stable_rank(features)`.
It mean-centers samples through `jnp.cov` and computes
`trace(covariance) / largest_eigenvalue`, with a small eigenvalue floor. This
explains the current behavior that one-hot features of width four have stable
rank three after centering.

The original formula described the stable rank of an arbitrary matrix,
`||M||_F^2 / ||M||_2^2`, while the existing implementation describes the stable
rank of a feature covariance matrix. These are different estimands. The first
`jax-eval` release will not add a second function named `stable_rank` or silently
change the existing behavior. If direct matrix stable rank is required, add a
separately named, tested primitive to the owning linear-algebra package after a
consumer demonstrates that need.

### 6.2 Optimization diagnostics

#### `gradient_snr(grads_batch, *, epsilon=1e-8)`

`grads_batch` is a pytree whose every leaf has at least one dimension and whose
leading batch axis has the same nonzero size `B` across all leaves; an empty
pytree or `B = 0` returns float32 `0.0`. Scalar leaves are rejected. The batch
may represent seeds, observations, or repeated gradient
estimates, but the caller must use one interpretation per metric series. Define
`\bar g = mean_b(g_b)` and compute the global pytree ratio:

\[
  \operatorname{SNR}(g) =
  \frac{\|\bar g\|_2^2}
       {\operatorname{mean}_b\|g_b - \bar g\|_2^2 + \epsilon}.
\]

The numerator and denominator use the unweighted sum of squares over every leaf
element, matching a global pytree L2 norm; leaf size does not cause an
additional weighting factor.
For `B = 1` or identical non-zero gradients, the epsilon-stabilized result is
finite and large; identical zero gradients return zero. Callers should record
the batch size alongside the metric if that distinction matters scientifically.

#### `update_ratio(params, updates, *, epsilon=1e-8)`

`params` and `updates` are identically structured pytrees with matching shapes
for every corresponding leaf. A treedef or leaf-shape mismatch is rejected.
Return

\[
  \frac{\|updates\|_2}{\|params\|_2 + \epsilon}.
\]

The norm is global across leaves. The function must reject mismatched tree
structure through the normal JAX tree operation rather than silently pairing
leaves positionally. A zero parameter tree produces a finite ratio.

Global parameter norms are provided by the existing canonical
`jax_utils.pytree.tree_norm(params)`. Per-layer reporting is achieved by calling
that function on an explicitly selected layer and assigning the layer a distinct
metric name. `jax-eval` will not add a duplicate `weight_norm` wrapper.

### 6.3 Replay diagnostics

#### `priority_utilization(priorities, *, epsilon=1e-8)`

`priorities` is a non-negative, fixed-capacity vector. The kernel forms
`p_i = priority_i / sum(priority)` and returns Shannon entropy normalized by
`log(capacity)`:

\[
  U = \frac{-\sum_i p_i \log(\max(p_i,\epsilon))}{\log(N)}.
\]

The implementation must use a safe total when the priority sum is zero and
must guard the `N <= 1` denominator. Zero-priority entries contribute zero. An
all-zero vector or a capacity-one vector returns `0.0`; a uniform vector over
the full capacity returns `1.0`. This metric describes the provided priority
distribution. It is not guaranteed to equal the replay buffer's actual
sampling distribution when the buffer applies an exponent, mask, or other
sampling rule. A future API should accept sampling probabilities directly if
that distinction is needed.

#### `sample_staleness(current_step, sampled_steps, valid_mask=None)`

Return the mean age of valid sampled transitions:

\[
  A = mean(current\_step - sampled\_step).
\]

`sampled_steps` may have any fixed sample shape. Integer timestamps are
subtracted in a safe integer domain before the signed delta is converted to
float32, preserving small deltas for large counters and avoiding unsigned
underflow. If `valid_mask` is provided, it must have the same shape and only
true entries contribute. Invalid or negative ages are not silently clipped;
callers must ensure that the step counter and sample timestamps use the same
origin. No valid entries returns `0.0`.

## 7. Collection and experiment integration

The diagnostic layer remains pure. Collection is explicit at the call site:

```python
from jax_utils.linalg import stable_rank
from research_instrument.collector import collector
from jax_eval.representations import dead_neurons

collector.eval(
    "diagnostic/representation_stable_rank",
    lambda: stable_rank(model.apply(params, probe_states)),
    step,
    every=1000,
)
collector.eval(
    "diagnostic/dead_neuron_fraction",
    lambda: dead_neurons(model.apply(params, probe_states)),
    step,
    every=1000,
)
```

The experiment definition declares the exact metric names and frequencies. The
instrumentation collector performs the trace-time whitelist check and is the
only component that emits `jax.debug.callback` effects. Expensive diagnostics
on fixed probes should use `collector.eval(..., every=N)` so the expensive probe
forward pass is skipped on non-scheduled steps; the probe itself is prepared
outside the traced training function or passed as a fixed-shape input. A
diagnostic over values already available in the training step may use
`collector.write` directly.

Each scalar series must document its semantic unit in the experiment plan:
which probe or batch was used, whether the batch axis is seeds or samples, the
layer name, the step origin, and any tolerance or normalization setting.

## 8. Implementation roadmap

### Phase 0 — Make the package real

- Change the package metadata from a stub to an installable `src/jax_eval`
  package with the repository's JAX and Python constraints.
- Add the empty package initializer and direct-import module layout.
- Wire the package into `core/pyproject.toml` where needed: workspace source
  mapping, root dependency declaration, and the explicit Pyrefly search path.
- Add public API tests in `core/tests/small/` and JIT smoke tests in
  `core/tests/medium/`.
- Preserve the existing covariance semantics of
  `jax_utils.linalg.stable_rank`; `core/tests/small/test_jax_utils_linalg.py`
  already provides its regression coverage.

### Phase 1 — Representation kernels

Implement and test `dead_neurons` and `feature_drift`. Cover normal inputs,
all-dead inputs, empty observation batches, one-dimensional inputs, identical
small-norm vectors, zero vectors, tolerance thresholds, similarity clipping,
shape mismatch, `jit`, and `vmap` over independent agents or probes.

### Phase 2 — Optimization kernels

Implement `gradient_snr` and `update_ratio`. Reuse `jax-utils` tree reductions;
use `jax_utils.pytree.tree_norm` directly for parameter norms. Cover
single-sample, identical-gradient, zero-gradient, zero-parameter, nested-pytree,
and mixed-leaf-size cases. Verify that the batch axis is preserved under
`vmap`.

### Phase 3 — Replay kernels

Implement `priority_utilization` and `sample_staleness` after confirming with a
real `jax-replay` consumer whether priorities or sampling probabilities are the
appropriate input. Cover uniform, concentrated, all-zero, capacity-one, masked,
and stale-timestamp cases.

### Phase 4 — One end-to-end consumer

Adapt one project diagnostic, preferably a fixed-probe representation report,
to use the published kernels and `research-instrument` collection boundary.
Record the probe, metric names, cadence, and expected output in that project's
experiment plan and diary. Do not generalize project-specific aggregates into
`jax-eval` during this phase.

### Phase 5 — Reassess the harness question

Only after two independent consumers exist should the project decide whether a
shared evaluation harness is missing. The decision should compare the existing
`rl-components` rollout and runner contracts against the concrete duplicated
caller behavior. A harness proposal must specify reset semantics, fixed versus
dynamic evaluation inputs, learning-enabled versus frozen policy behavior,
seed/scenario identity, and artifact ownership before any implementation.

## 9. Verification and acceptance criteria

The first release is complete when:

- the package is installable from the core workspace;
- every published kernel has a written formula and tests for normal, boundary,
  and degenerate inputs;
- all kernels pass `jax.jit` and the relevant `jax.vmap` tests;
- no kernel imports storage, collector, experiment-definition, or project code;
- the collector integration test, owned by `research-instrument`, confirms that
  a non-whitelisted metric emits no frame;
- at least one project records a kernel result with stable run/seed identity;
- the output can be loaded through the existing research-analysis dataset
  boundary without a package-specific database reader.

Run the repository-required `uv run ruff check .` and
`uv run pyrefly check` after adding implementation code. Add a performance
benchmark only for a kernel whose implementation introduces measurable work
beyond its direct JAX expression, using fixed shapes and reporting compilation
and steady-state costs separately. Callback and SQLite overhead belong to the
`research-instrument` benchmarks, not to `jax-eval`.

## 10. Decisions required before implementation

The following scientific choices should be confirmed in the first consumer's
experiment plan rather than hidden in defaults:

1. Whether dead-unit thresholds above exact zero are meaningful for the target
   activation family.
2. Whether gradient SNR batches represent seeds, samples, or repeated updates.
3. Whether replay diagnostics should use raw priorities or post-exponent
   sampling probabilities.
4. Whether direct matrix stable rank is needed in addition to the existing
   feature-covariance stable rank.
5. Which fixed probe states and layer boundaries make representation metrics
   comparable across the first two consumers.
