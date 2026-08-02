"""Small, reusable certification checks for process-control modules.

The checks in this module deliberately operate on callables rather than on a
specific benchmark family.  A benchmark can therefore opt in to the checks
that its interface and physics support without pretending that every model
has the same conservation laws.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class CertificationDiagnostic:
    """Result of one named certification check."""

    name: str
    passed: bool
    message: str = ""

    @property
    def ok(self) -> bool:
        """Return the status using the common diagnostic naming convention."""
        return self.passed


@dataclass(frozen=True, slots=True)
class CertificationReport:
    """Named certification results for one module or benchmark."""

    module: str
    diagnostics: tuple[CertificationDiagnostic, ...]

    @property
    def passed(self) -> bool:
        """Return whether every requested check passed."""
        return all(diagnostic.passed for diagnostic in self.diagnostics)

    @property
    def failed(self) -> tuple[CertificationDiagnostic, ...]:
        """Return only failed checks, preserving their diagnostic names."""
        return tuple(diagnostic for diagnostic in self.diagnostics if not diagnostic.passed)

    def raise_for_failure(self) -> None:
        """Raise one concise error when the report contains a failed check."""
        if self.passed:
            return
        details = "; ".join(
            f"{diagnostic.name}: {diagnostic.message}" for diagnostic in self.failed
        )
        raise AssertionError(f"{self.module} certification failed: {details}")


def _tree_equal(left: object, right: object) -> bool:
    if jax.tree.structure(left) != jax.tree.structure(right):
        return False
    return all(
        _leaf_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _leaf_equal(left: object, right: object) -> bool:
    try:
        return bool(jnp.array_equal(jnp.asarray(left), jnp.asarray(right)))
    except (TypeError, ValueError):
        return left == right


def _tree_is_finite(tree: object) -> bool:
    """Return false for any numeric leaf containing NaN or infinity."""
    for leaf in jax.tree.leaves(tree):
        try:
            values = jnp.asarray(leaf)
            if values.dtype.kind not in "biufc":
                continue
            if not bool(jnp.all(jnp.isfinite(values))):
                return False
        except (TypeError, ValueError):
            # Metadata such as event names is not a physical numeric output.
            continue
    return True


def _safe_check(name: str, check: Callable[[], str | None]) -> CertificationDiagnostic:
    try:
        message = check()
    except (
        AssertionError,
        FloatingPointError,
        IndexError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        return CertificationDiagnostic(name, False, f"{type(exc).__name__}: {exc}")
    return CertificationDiagnostic(name, message is None, message or "")


def _run_callable_episode(
    reset: Callable[[jax.Array], Sequence[object]],
    step: Callable[[object, object, jax.Array], Sequence[object]],
    action_for_step: Callable[[int], object],
    *,
    seed: int,
    steps: int,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    root_key = jax.random.PRNGKey(seed)
    reset_output = reset(root_key)
    if not isinstance(reset_output, Sequence) or not reset_output:
        raise TypeError("reset must return a non-empty tuple with state in position zero")
    state = reset_output[0]
    outputs: list[object] = [reset_output]
    actions: list[object] = []
    for step_index in range(1, steps + 1):
        action = action_for_step(step_index)
        actions.append(action)
        output = step(state, action, jax.random.fold_in(root_key, step_index))
        if not isinstance(output, Sequence) or not output:
            raise TypeError("step must return a non-empty tuple with state in position zero")
        outputs.append(output)
        state = output[0]
    return tuple(outputs), tuple(actions)


def certify_reset_step(
    *,
    module: str,
    reset: Callable[[jax.Array], Sequence[object]],
    step: Callable[[object, object, jax.Array], Sequence[object]],
    action_for_step: Callable[[int], object],
    seed: int = 0,
    steps: int = 3,
) -> CertificationReport:
    """Certify a reset/key/step callable pair over bounded actions.

    ``reset`` and ``step`` follow the toolkit's benchmark convention: both
    return tuples whose first element is the next state, and ``step`` receives
    an explicit JAX random key.  Conservation checks are intentionally not
    included because reduced process models often do not expose all flows.
    """

    def finite_reset_and_step() -> str | None:
        outputs, _ = _run_callable_episode(
            reset, step, action_for_step, seed=seed, steps=steps
        )
        return None if all(_tree_is_finite(output) for output in outputs) else "non-finite reset or step output"

    def deterministic_same_seed() -> str | None:
        first, first_actions = _run_callable_episode(
            reset, step, action_for_step, seed=seed, steps=steps
        )
        second, second_actions = _run_callable_episode(
            reset, step, action_for_step, seed=seed, steps=steps
        )
        if not _tree_equal(first_actions, second_actions):
            return "action generator is not deterministic"
        return None if _tree_equal(first, second) else "same seed produced different outputs"

    def bounded_action_no_nan() -> str | None:
        outputs, actions = _run_callable_episode(
            reset, step, action_for_step, seed=seed, steps=steps
        )
        if not all(_tree_is_finite(action) for action in actions):
            return "action generator produced a non-finite action"
        return None if all(_tree_is_finite(output) for output in outputs) else "bounded action produced a non-finite output"

    diagnostics = (
        _safe_check("finite_reset_and_step", finite_reset_and_step),
        _safe_check("deterministic_same_seed", deterministic_same_seed),
        _safe_check("bounded_action_no_nan", bounded_action_no_nan),
    )
    return CertificationReport(module=module, diagnostics=diagnostics)


def check_timestep_refinement(
    *,
    name: str,
    initial_state: object,
    coarse_step: Callable[[object, jax.Array], object],
    fine_step: Callable[[object, jax.Array], object],
    state_value: Callable[[object], jax.Array],
    coarse_dt: float,
    fine_dt: float,
    coarse_steps: int = 1,
    tolerance: float = 1e-6,
) -> CertificationDiagnostic:
    """Check that refining a timestep stays within a declared error envelope."""

    def check() -> str | None:
        if coarse_dt <= 0.0 or fine_dt <= 0.0:
            raise ValueError("coarse_dt and fine_dt must be positive")
        if coarse_steps <= 0:
            raise ValueError("coarse_steps must be positive")
        ratio = coarse_dt / fine_dt
        fine_steps = round(ratio)
        if abs(ratio - fine_steps) > 1e-9:
            raise ValueError("fine_dt must divide coarse_dt")
        coarse_state = initial_state
        fine_state = initial_state
        coarse_delta = jnp.asarray(coarse_dt)
        fine_delta = jnp.asarray(fine_dt)
        for _ in range(coarse_steps):
            coarse_state = coarse_step(coarse_state, coarse_delta)
            for _ in range(fine_steps):
                fine_state = fine_step(fine_state, fine_delta)
        coarse_value = jnp.asarray(state_value(coarse_state))
        fine_value = jnp.asarray(state_value(fine_state))
        if not _tree_is_finite((coarse_value, fine_value)):
            return "refined or coarse trajectory is non-finite"
        error = float(jnp.max(jnp.abs(coarse_value - fine_value)))
        return None if error <= tolerance else f"max refinement error {error:.6g} exceeds {tolerance:.6g}"

    return _safe_check(name, check)


def check_mass_balance(
    *,
    name: str,
    initial_inventory: jax.Array | float,
    final_inventory: jax.Array | float,
    inlet_flow: jax.Array,
    realized_outlet_flow: jax.Array,
    overflow_flow: jax.Array,
    dt: jax.Array | float,
    tolerance: float = 1e-6,
) -> CertificationDiagnostic:
    """Check inventory balance using realized, rather than requested, flows.

    Callers should invoke this only for modules that expose all terms needed by
    the balance.  In particular, an unavailable outlet or overflow must not be
    silently treated as zero for a reduced model.
    """

    def check() -> str | None:
        initial = jnp.asarray(initial_inventory)
        final = jnp.asarray(final_inventory)
        inlet = jnp.asarray(inlet_flow)
        outlet = jnp.asarray(realized_outlet_flow)
        overflow = jnp.asarray(overflow_flow)
        timestep = jnp.asarray(dt)
        residual = initial + jnp.sum(inlet * timestep) - final - jnp.sum(outlet * timestep) - jnp.sum(overflow * timestep)
        if not bool(jnp.all(jnp.isfinite(residual))):
            return "mass-balance residual is non-finite"
        error = float(jnp.max(jnp.abs(residual)))
        return None if error <= tolerance else f"mass-balance residual {error:.6g} exceeds {tolerance:.6g}"

    return _safe_check(name, check)
