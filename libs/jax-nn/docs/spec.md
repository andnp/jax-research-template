# PRD: `libs/jax-nn`

## 1. Overview
`jax-nn` provides reusable Flax Linen layers, heads, and initialization helpers for RL agents in this repository.

This first Atari-focused slice adds a reusable `NatureCNN` torso and the legacy DQN uniform initialization needed to match DQN Zoo semantics without yet wiring those pieces into the current DQN agent configuration.

## 2. Goals
- Provide a reusable `NatureCNN` torso for channel-last Atari observations.
- Preserve DQN Zoo preprocessing semantics by casting `uint8` observations to `float32` and scaling by `1 / 255`.
- Expose legacy DQN initialization helpers so future DQN heads can apply the same fan-in-based uniform sampling to both kernels and biases.
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

## 4. Non-goals for this slice
- No DQN config or agent wiring yet.
- No dueling / noisy / distributional head changes.
- No attempt to generalize the torso beyond the canonical Nature-DQN architecture.

## 5. Verification expectations
- Small tests cover initializer bounds / inference and `NatureCNN` output shape plus scaling-sensitive behavior.
- Medium tests cover JIT execution for the new torso.
- Static verification runs through `ruff`, `ty`, and `pyright` on the touched files.
