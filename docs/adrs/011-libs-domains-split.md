# ADR 011: Split `core/libs` into `libs/` and `domains/` Tiers

## Status
Proposed

## Context
`core/libs/` is currently flat: every shared package sits at the same level
regardless of how domain-agnostic it is. This caused real misplacement, found
while investigating duplicated code in `projects/iterated-gtd` — `jax_nn`
mixed framework-generic Flax plumbing with RL-paper-specific network
architectures, and `rl_components` mixed RL-generic primitives with
environment-specific integration code. `generative-replay`'s need for
ELBO/VAE/diffusion-loss machinery had no non-RL home to live in under the
existing flat structure.

## Decision
We will split `core/libs/` into two tiers:

- **`libs/`** — genuinely generic code. A `libs` package may import other
  `libs` packages but may never import from `domains`.
- **`domains/`** — domain-specific code reused across multiple projects (e.g.
  `domains/rl/{agents,networks,envs}`, `domains/process-control`). A `domains`
  package may import other `domains` packages and any `libs` package.

A symbol belongs in `libs` if it is domain-agnostic across all of ML, or if it
is specific to one domain's *concept* but parameterized with no fixed
hyperparameters or environment assumption (reusable across many
implementations within that domain). It belongs in `domains` if it encodes a
specific, named implementation's fixed hyperparameters or conventions tied to
one paper and/or one environment family. `projects/*` may import from either
tier, but never from another project.

## Consequences
- **Pros:** Misplacement like `jax_nn`'s mixed generic/paper-specific content
  becomes structurally visible; new domain-specific machinery (e.g.
  generative-replay's loss functions) gets a natural home without stretching
  an unrelated library's charter.
- **Cons:** Existing packages (`rl-agents`, `rl-components`, `jax-nn`,
  `process-control`) must be split or relocated to match the new boundary —
  tracked as a separate migration, executed incrementally across later
  milestones.
