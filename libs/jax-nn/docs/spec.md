# PRD: `libs/jax-nn`

## 1. Overview
`jax-nn` provides reusable Flax Linen layers, heads, and initialization helpers for RL agents in this repository.

`jax-nn` provides reusable DQN building blocks plus categorical distributional (C51) primitives for RL agents in this repository.

## 2. Goals
- Provide a reusable `NatureCNN` torso for channel-last Atari observations.
- Preserve DQN Zoo preprocessing semantics by casting `uint8` observations to `float32` and scaling by `1 / 255`.
- Expose legacy DQN initialization helpers so future DQN heads can apply the same fan-in-based uniform sampling to both kernels and biases.
- Provide reusable categorical distribution helpers for projection, loss computation, and expectation over fixed atom supports.
- Provide a small categorical value head that emits action-by-atom logits.
- Keep the API JIT-friendly and consistent with existing `jax-nn` modules.

## 3. Scope

### 3.1 `NatureCNN`
- Accept inputs shaped `(..., height, width, channels)`.
- Apply the standard DQN convolution stack:
	- `32` filters, kernel `8x8`, stride `4`
	- `64` filters, kernel `4x4`, stride `2`
	- `64` filters, kernel `3x3`, stride `1`
- Apply `ReLU` after each convolution.
- Flatten the final activation map to `(..., features)`.
- Stop at the torso boundary; do **not** include the DQN head's `512`-unit dense layer.

### 3.2 Legacy DQN initialization
- Provide helpers that implement the original DQN uniform initializer:
	- sample from `[-c, c]`
	- where `c = sqrt(1 / num_input_units)`
- Support kernel initialization by inferring `num_input_units` from shape.
- Support bias initialization by allowing `num_input_units` to be supplied explicitly when shape alone is insufficient.

### 3.3 Categorical distributional helpers
- Provide a reusable projection helper that maps source atom values and probabilities back onto a fixed, uniformly spaced support.
- Provide a cross-entropy helper that reduces only over the atom axis so callers can preserve batch or action dimensions.
- Provide an expected-value helper that converts categorical probabilities and a support into scalar expectations.
- Provide a minimal categorical value head that reshapes a dense projection into `(..., action_dim, num_atoms)` logits.

## 4. Boundaries
- No DQN config or agent wiring yet.
- No full agent-side C51 learner integration.
- No attempt to generalize the torso beyond the canonical Nature-DQN architecture.

## 5. Verification expectations
- Small tests cover initializer bounds / inference and `NatureCNN` output shape plus scaling-sensitive behavior.
- Medium tests cover JIT execution for the new torso.
- Small tests cover categorical projection invariants, cross-entropy, expected-value reduction, and head output shape.
- Medium tests cover JIT execution for the categorical helpers and head.
- Static verification runs through `ruff`, `ty`, and `pyright` on the touched files.
