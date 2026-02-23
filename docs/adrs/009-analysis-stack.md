# ADR 009: Analysis Stack Selection (Polars & Numba)

## Context
Reinforcement Learning research generates massive amounts of time-series data. Analyzing this data requires both high-performance computational kernels (for bootstrapping and statistical tests) and efficient data manipulation (for filtering and grouping experiments). 

While the rest of the monorepo is JAX-native, applying JAX to the analysis phase introduces several challenges:
1. **Dependency Heavy:** JAX requires specific accelerator drivers which are often missing on local analysis laptops.
2. **Compile Latency:** JAX JIT compilation overhead is often not worth it for the one-off, variable-length datasets typical of post-hoc analysis.
3. **Data Wrangling:** Post-hoc analysis is primarily a "Data Frame" problem, where JAX is less ergonomic than dedicated libraries.

## Decision
We will standardize the `research-analysis` library on the following stack:
1. **Polars:** As the exclusive first-class citizen for data management. Pandas will not be supported to avoid "API bloat" and maintain high-performance lazy evaluation.
2. **Numba:** For high-performance statistical kernels (bootstrapping, U-tests). Numba provides near-C performance with a much lower overhead than JAX for variable-input post-hoc data.
3. **Numpy:** As the standard array format for Numba kernels.

## Consequences
- **Modularity:** Analysis scripts will be runnable on any machine without needing JAX/CUDA.
- **Speed:** Polars' multithreaded engine and Numba's LLVM compilation ensure that even million-row experiment sweeps can be analyzed in seconds.
- **Consistency:** By mandating Polars, we ensure that all "Science" scripts share a common, modern API for data manipulation.
