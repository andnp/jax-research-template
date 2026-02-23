# Engineering Standards

## 1. Type Checking
We use `ty` (Astral's red-knot) for primary type checking and `pyright` as a secondary validator.

- **Zero Tolerance:** All code in `libs/` must have 100% type coverage.
- **Modern Syntax:**
    - Prefer `list[int]` over `typing.List[int]`.
    - Prefer `int | str` over `typing.Union[int, str]`.
    - Use PEP 695 for generics: `def process[T](data: list[T]) -> T:`.
- **No Casts:** Avoid `typing.cast` unless absolutely necessary for JAX tracers.
- **Strict Any:** `Any` is treated as a failure of design. Use `Mapping[str, Any]` for JSON-like data, but prefer `Pydantic` or `chex.dataclass` for structured data.

## 2. Linting
We use `ruff` for linting and formatting.
- **Rule Set:** We follow `E`, `F`, `B`, and `I` rules.
- **Line Length:** 180 characters (standardized for complex JAX array operations).
- **Format:** `ruff format` is mandatory before commit.

## 3. Testing Tiers
We utilize `pytest` with a directory-based tiering system.

### `tests/small/`
- **Goal:** Fast feedback on logic/math.
- **Constraint:** Must run in `< 1ms` per test.
- **Content:** Unit tests, pure function verification, shape-checking.

### `tests/medium/`
- **Goal:** Integration and interaction.
- **Constraint:** Must run in `< 1s` per test.
- **Content:** Environment-Agent interactions, JIT compilation checks, replay buffer adds/samples.

### `tests/large/`
- **Goal:** Full system verification.
- **Constraint:** None (but should be optimized).
- **Content:** Learning regression tests (hitting reward thresholds), database migrations, Docker-based infra.

### `tests/benchmarks/`
- **Goal:** Performance regression.
- **Content:** `pytest-benchmark` runs measuring SPS and memory throughput.

## 4. Empirical Science Standards
All research code and experiments must adhere to the **[Empirical Design & Analysis Guide](science_guide.md)**.

- **Understanding over Winning:** The goal of an experiment is to understand an algorithm's properties, not just to show it is "good."
- **Scientific Studies vs. Demonstrations:** We prioritize scientific studies (falsifiable hypotheses, controlled confounding effects) over engineering demonstrations.
- **Statistical Rigor:** No results should be reported without appropriate non-parametric confidence intervals and significance tests.
- **Reproducible Data:** Experiments must be declarative and stored in the master database to ensure perfect traceability.

## 5. Documentation Policy
- **Google-Style:** All public functions must have docstrings.
- **Math:** Use LaTeX in docstrings for algorithmic explanations.
- **PRD First:** Major features must have a spec in `docs/specs/` before implementation begins.
- **ADRs:** Record significant architectural decisions in `docs/adrs/`.

