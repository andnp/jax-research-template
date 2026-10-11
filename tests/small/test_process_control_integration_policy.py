import jax
import jax.numpy as jnp
import pytest
from process_control.integration_policy import IntegrationPolicy, legacy_policy
from process_control.integration_records import (
    NO_REASON,
    AdvanceChecks,
    AdvanceResult,
    AdvanceStatus,
    CheckResult,
    IntegrationContext,
    check_result,
    nonfinite_check,
)


class TestCheckResult:
    def test_first_failing_flag_decides_the_reason(self) -> None:
        """Reasons follow the declared flag order, so the earliest failing check is reported."""
        result = check_result((jnp.array(False), jnp.array(True), jnp.array(True)))

        assert bool(result.any_failed)
        assert int(result.first_reason) == 2

    def test_no_failure_reports_no_reason(self) -> None:
        """A clean check reports the no-reason id rather than a flag's id."""
        result = check_result((jnp.array(False), jnp.array(False)))

        assert not bool(result.any_failed)
        assert int(result.first_reason) == NO_REASON

    def test_empty_check_never_fails(self) -> None:
        """A policy with no declared checks has an empty, passing result."""
        result = check_result(())

        assert result.failed.shape == (0,)
        assert not bool(result.any_failed)
        assert int(result.first_reason) == NO_REASON

    def test_failure_is_an_array_under_trace(self) -> None:
        """Failing flags survive jit and vmap as data, one verdict per lane, instead of raising."""

        def check(x: jax.Array) -> CheckResult:
            return check_result((x < 0.0, ~jnp.isfinite(x)))

        result = jax.jit(jax.vmap(check))(jnp.array([1.0, -1.0, jnp.nan]))

        assert result.failed.tolist() == [[False, False], [True, False], [False, True]]
        assert result.first_reason.tolist() == [NO_REASON, 1, 2]


class TestNonfiniteCheck:
    def test_flags_only_the_pytrees_holding_a_nonfinite_leaf(self) -> None:
        """Each named tree gets its own flag, in the mapping's order."""
        result = nonfinite_check({"a": {"x": jnp.ones(2)}, "b": [jnp.array([1.0, jnp.inf])], "c": jnp.array(jnp.nan)})

        assert result.failed.tolist() == [False, True, True]

    def test_an_empty_tree_is_finite(self) -> None:
        """A stateless node's None derivative has no leaves and cannot fail."""
        assert nonfinite_check({"a": None}).failed.tolist() == [False]


class TestIntegrationPolicy:
    def test_legacy_policy_declares_nothing(self) -> None:
        """The default policy checks nothing and owns no modes."""
        policy = legacy_policy()

        result = policy.check(None, {}, None, {})

        assert result.failed.shape == (0,)
        assert dict(policy.modes) == {}

    def test_check_must_return_one_flag_per_declared_name(self) -> None:
        """A check returning the wrong number of flags is a declaration error, not a silent mismatch."""
        policy = IntegrationPolicy[None, None](("a", "b"), lambda state, inputs, params, modes: check_result((jnp.array(False),)))

        with pytest.raises(ValueError, match="one flag per declared check"):
            policy.check(None, {}, None, {})

    def test_duplicate_check_names_are_rejected(self) -> None:
        """Reason ids are positional, so repeated names would be ambiguous."""
        with pytest.raises(ValueError, match="check names must be unique"):
            IntegrationPolicy[None, None](("a", "a"), lambda state, inputs, params, modes: check_result(()))

    def test_two_modes_cannot_share_an_input_port(self) -> None:
        """Each mode must own its port exclusively."""
        with pytest.raises(ValueError, match="own input port"):
            IntegrationPolicy[None, None]((), lambda state, inputs, params, modes: check_result(()), modes={"a": "p", "b": "p"})

    def test_continuation_must_name_declared_checks(self) -> None:
        """Only a declared check can bound a domain that location trials may cross."""
        with pytest.raises(ValueError, match="continuation"):
            IntegrationPolicy[None, None](("a",), lambda state, inputs, params, modes: check_result((jnp.array(False),)), continuation=("b",))


class TestAdvanceResult:
    def test_result_is_a_fixed_structure_pytree(self) -> None:
        """An advance result crosses jit with its status and modes intact."""
        zero = jnp.zeros((), dtype=jnp.int32)
        result = AdvanceResult(
            state={"tank": jnp.ones(2)},
            completed_time_h=jnp.array(0.5),
            status=jnp.array(AdvanceStatus.RETRY_EXHAUSTED),
            quadratures={},
            context=IntegrationContext({"n.capacity": jnp.array(1)}),
            checks=AdvanceChecks(zero, zero, zero, zero, zero, zero),
        )

        def identity(value: AdvanceResult) -> AdvanceResult:
            return value

        returned = jax.jit(identity)(result)

        assert int(returned.status) == AdvanceStatus.RETRY_EXHAUSTED
        assert float(returned.completed_time_h) == 0.5
        assert int(returned.context.modes["n.capacity"]) == 1
