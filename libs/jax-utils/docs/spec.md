# PRD: `libs/jax-utils`

## 1. Overview
`jax-utils` is a collection of generic utilities for JAX programming, focused on type safety, Pytree manipulation, and improved developer ergonomics.

### Goals
- **Strong Typing:** Provide wrappers around JAX transformations (`jit`, `vmap`, `grad`) that leverage `jaxtyping` for compile-time and runtime shape/dtype checking.
- **Pytree Ergonomics:** Robust tools for manipulating, math-ing, and checking complex Jytree structures.
- **Functional Cleanliness:** Utilities that assist in maintaining the End-to-End JIT pattern required by the Research Core.

## 2. Core Features

### 2.1 Typed JAX Transformations
- **`typed_jit`**: A wrapper around `jax.jit` that respects and enforces `jaxtyping` annotations on inputs and outputs.
- **`typed_vmap`**: A version of `jax.vmap` that preserves type information and provides clearer error messages for axis mismatches.

### 2.2 Tree-Level Operations
- **Tree Math:** Support for basic arithmetic and statistical operations (Mean, Std, Norm) that operate recursively on all leaves of a Pytree.
- **Shape & Type Checkers:** Validators that ensure two Pytrees have compatible structures and leaf shapes (e.g., comparing agent parameters to buffer storage).
- **Tree Filtering:** Utilities to selectively apply functions to Pytree leaves based on their names or types.

### 2.3 Running Statistics
- JAX-native implementations of **RunningMeanStd** and other stateful statistics that can be easily integrated into `lax.scan` loops.

## 3. Technical Constraints
- Must be zero-overhead at runtime (beyond standard JAX/XLA costs).
- Must adhere to the PEP 695 generics and modern typing standards defined in `AGENTS.md`.
- No dependencies outside of `jax`, `jaxtyping`, and `chex`.

## 4. Proposed Usage

```python
from jax_utils import typed_jit, tree_norm
from jaxtyping import Float, Array

@typed_jit
def my_pure_function(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
    return x.sum(axis=-1)

# Recursive normalization of a complex parameter Pytree
norm = tree_norm(params)
```
