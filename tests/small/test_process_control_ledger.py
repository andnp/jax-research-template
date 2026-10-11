from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from process_control.integration_records import IntegrationContext
from process_control.ledger import (
    LedgerSpec,
    commit_if_accepted,
    fast_extent_ledger,
    rk4_weighted_rate,
    segment_integral,
)

EMPTY_CONTEXT = IntegrationContext({})


def _spec(names: tuple[str, ...]) -> LedgerSpec[jax.Array, jax.Array]:
    def rates(state: jax.Array, evaluation: jax.Array, inputs: Mapping[str, jax.Array], context: IntegrationContext) -> Mapping[str, jax.Array]:
        return {"volume": state * inputs["q"], "mass": evaluation}

    return LedgerSpec("plant", dict.fromkeys(names, "m3/h"), rates)


class TestLedgerSpec:
    def test_rates_are_returned_in_declared_order(self) -> None:
        """Extraction yields exactly the declared names, ordered as declared, for a stable pytree."""
        spec = _spec(("mass", "volume"))

        rates = spec.evaluate(jnp.array(2.0), jnp.array(5.0), {"q": jnp.array(3.0)}, EMPTY_CONTEXT)

        assert list(rates) == ["mass", "volume"]
        assert float(rates["volume"]) == 6.0

    def test_undeclared_rates_are_rejected(self) -> None:
        """A rate function that disagrees with its declared names is a declaration error."""
        with pytest.raises(ValueError, match="must return rates"):
            _spec(("volume",)).evaluate(jnp.array(2.0), jnp.array(5.0), {"q": jnp.array(3.0)}, EMPTY_CONTEXT)


class TestSegmentIntegral:
    def test_cubic_rate_integrates_exactly(self) -> None:
        """RK4 stage weights are Simpson's rule in time, exact for a cubic rate sampled at 0, h/2, h/2, h."""
        h = jnp.array(0.5)

        def rate(t: jax.Array) -> jax.Array:
            return 1.0 + 2.0 * t - 3.0 * t**2 + 4.0 * t**3

        mid = h * 0.5
        k = [{"x": rate(t)} for t in (jnp.array(0.0), mid, mid, h)]

        integral = segment_integral(k[0], k[1], k[2], k[3], h)

        exact = h + h**2 - h**3 + h**4
        assert jnp.allclose(integral["x"], exact)

    def test_weights_follow_the_rk4_expression_order(self) -> None:
        """The combination is bit-identical to (a + 2b + 2c + d) / 6 on each leaf, including nested pytrees."""
        a, b, c, d = (jnp.array(v) for v in (0.1, 0.7, 0.3, 0.9))

        weighted = rk4_weighted_rate({"p": [a]}, {"p": [b]}, {"p": [c]}, {"p": [d]})

        assert float(weighted["p"][0]) == float((a + 2.0 * b + 2.0 * c + d) / 6.0)

    def test_constant_rate_gives_rate_times_duration(self) -> None:
        """A constant volume flow integrates to flow times the segment length."""
        flow = {"q": jnp.array(2.0)}

        integral = segment_integral(flow, flow, flow, flow, jnp.array(0.25))

        assert jnp.allclose(integral["q"], 0.5)


class TestCommitIfAccepted:
    def test_rejected_segment_leaves_totals_unchanged(self) -> None:
        """Only accepted segments contribute, so a rejected trial never reaches the ledger."""
        total = {"q": jnp.array(1.0)}
        segment = {"q": jnp.array(0.5)}

        assert float(commit_if_accepted(total, segment, jnp.array(True))["q"]) == 1.5
        assert float(commit_if_accepted(total, segment, jnp.array(False))["q"]) == 1.0

    def test_lanes_commit_independently_under_vmap(self) -> None:
        """Acceptance is per lane, so one rejected lane does not hold back an accepted one."""
        committed = jax.vmap(commit_if_accepted)(jnp.array([1.0, 1.0]), jnp.array([0.5, 0.5]), jnp.array([True, False]))

        assert committed.tolist() == [1.5, 1.0]


class TestFastExtentLedger:
    STOICHIOMETRY = jnp.array([-1.0, 1.0, 0.0])

    def test_extent_and_all_rows_balance(self) -> None:
        """A segment whose inventory change is nonfast flow plus S * extent recovers the extent with zero residual."""
        n_start = jnp.array([4.0, 1.0, 7.0])
        nonfast = jnp.array([0.5, 0.25, -1.0])
        extent = 0.75
        n_end = n_start + nonfast + self.STOICHIOMETRY * extent

        ledger = fast_extent_ledger(n_start, n_end, nonfast, self.STOICHIOMETRY, row=0, rtol=1e-9, atol=1e-12)

        assert jnp.allclose(ledger.extent, extent)
        assert jnp.allclose(ledger.residual, 0.0, atol=1e-12)
        assert bool(ledger.ok)

    def test_an_unexplained_invariant_change_fails_the_row_check(self) -> None:
        """Inventory that moved in an invariant row without a flow to explain it is flagged, not absorbed."""
        n_start = jnp.array([4.0, 1.0, 7.0])
        nonfast = jnp.array([0.5, 0.25, -1.0])
        n_end = n_start + nonfast + self.STOICHIOMETRY * 0.75 + jnp.array([0.0, 0.0, 0.01])

        ledger = fast_extent_ledger(n_start, n_end, nonfast, self.STOICHIOMETRY, row=0, rtol=1e-9, atol=1e-12)

        assert not bool(ledger.rows_balanced)
        assert float(ledger.residual[2]) == pytest.approx(0.01, abs=1e-5)
        assert not bool(ledger.ok)

    def test_negative_progress_fails_even_when_rows_balance(self) -> None:
        """Irreversible progress cannot run backwards."""
        n_start = jnp.array([4.0, 1.0, 7.0])
        nonfast = jnp.zeros(3)
        n_end = n_start + self.STOICHIOMETRY * -0.5

        ledger = fast_extent_ledger(n_start, n_end, nonfast, self.STOICHIOMETRY, row=0, rtol=1e-9, atol=1e-12)

        assert bool(ledger.rows_balanced)
        assert not bool(ledger.progress_nonnegative)
