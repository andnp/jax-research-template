# Technical Specification: Canonical Environment Protocol

## 1. Overview
`rl-components` needs a small internal environment protocol that keeps shared RL code backend-agnostic while remaining fully compatible with end-to-end JAX execution. The canonical abstraction is a single-environment interface with explicit `reset`, `step`, and `spec` entry points.

This protocol defines the shared contract for environment adapters without prescribing backend-specific implementation details.

## 2. Goals
- Keep the shared abstraction smaller than any one backend API.
- Preserve the "One Agent, One World" rule from ADR 004: the protocol models one environment instance, not a vectorized batch wrapper.
- Support both discrete and continuous action spaces.
- Separate termination from truncation so time limits and game-over semantics can be represented without collapsing everything into a single `done` flag.
- Keep the contract compatible with JIT-able JAX code and JAX pytrees.

## 3. Canonical Protocol

### 3.1 Data Structures
- `EnvSpec`
  - Describes the stable shape and dtype contract for observations and actions.
  - Supports discrete environments via `num_actions` and bounded continuous environments via `action_shape`, `action_dtype`, and optional `action_low` / `action_high` arrays.
  - Discrete environments use a scalar integer action index: `num_actions` must be set, `action_shape=()`, and continuous bounds must be omitted.
  - Continuous environments leave `num_actions=None`, use a floating-point `action_dtype`, and may provide elementwise `action_low` / `action_high` bounds whose shapes and dtypes match `action_shape` and `action_dtype`.
  - Malformed combinations should fail fast where practical: partial bounds are invalid, discrete specs cannot also declare continuous bounds, and provided continuous bounds must satisfy `action_low <= action_high` elementwise.
- `EnvReset`
  - Returns the initial observation and backend state.
- `EnvStep`
  - Returns the next observation, next backend state, scalar reward, separate `terminated` / `truncated` flags, and a narrow `info` pytree.
  - `info` is limited to `dict[str, jax.Array]` so shared agents can consume episode statistics without importing backend-specific state or metadata objects.

### 3.2 Interface
Each canonical environment exposes:
- `spec(params=None) -> EnvSpec`
- `reset(key, params=None) -> EnvReset`
- `step(key, state, action, params=None) -> EnvStep`

`params` is optional because Gymnax requires explicit environment parameters, while other backends may carry fixed configuration inside the environment object.

## 4. Backend Positioning

### 4.1 Gymnax is a compatibility layer, not the core abstraction
Gymnax is useful, but its API should not define the shared architecture for the monorepo. Making Gymnax canonical would leak Gymnax-specific choices into every agent and future integration point:
- explicit env-parameter handling on every call,
- backend-specific state/info conventions,
- a narrower view of what an environment backend can look like.

Instead, Gymnax should adapt into the canonical protocol.

### 4.2 JAXAtari fit
JAXAtari adapts into the canonical protocol. The adapter owns ALE-specific state, exposes Atari observation and action metadata through `EnvSpec`, and layers the default DeepMind-style pixel preprocessing contract on top without changing agent-facing interfaces.

The default preprocessing contract for the Atari pixel path is:
- frame stack = 4,
- frame skip = 4,
- grayscale = true,
- max-pooling = true,
- life loss terminal = true,
- resize to `84x84` when the wrapped image pipeline is not already `84x84`.

The adapter also preserves a narrow `info` channel because PPO and DQN consumers read step info for logging and episode accounting.

### 4.3 Brax fit
Brax should adapt into the same protocol with continuous `action_shape` and floating-point `action_dtype`. When backend metadata exposes bounded control ranges, adapters should surface them through `action_low` and `action_high` without changing agent-facing interfaces. No agent code should need a separate abstraction just because the backend is physics-based instead of arcade-style.

## 5. Non-Goals
- No Gymnax adapter refactor.
- No agent code changes.
- No vector-environment framework.

## 6. Dependency Note
The workspace and library configuration pin `jaxatari` directly to the upstream GitHub tag `v0.1` so the adapter dependency remains explicit and reproducible.