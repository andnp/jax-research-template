import jax.numpy as jnp
import pytest
from process_control.lag_chain import advance_lag_chain, time_constant_for_t90


class TestLagChain:
    @pytest.mark.parametrize(("t90", "order", "reference"), [(1.0, 2, 0.257), (10.0, 8, 0.849), (20.0, 8, 1.699)])
    def test_time_constants_match_the_bsm1_table(self, t90: float, order: int, reference: float) -> None:
        """The T90 inversion reproduces the BSM1 Table 13 stage time constants (minutes)."""
        assert time_constant_for_t90(t90, order) == pytest.approx(reference, rel=1e-3)

    @pytest.mark.parametrize("order", [1, 2, 8])
    def test_step_response_reaches_ninety_percent_at_t90(self, order: int) -> None:
        """A unit step held for T90 brings the last stage to 0.9."""
        stages = advance_lag_chain(jnp.zeros(order), jnp.array(1.0), time_constant_for_t90(3.0, order), jnp.array(3.0))

        assert jnp.allclose(stages[-1], 0.9, atol=1e-5)

    def test_one_long_step_equals_many_short_steps(self) -> None:
        """The update is exact, so splitting an interval does not change the result."""
        start = jnp.array([4.0, -1.0, 2.0])
        whole = advance_lag_chain(start, jnp.array(1.5), 0.3, jnp.array(2.0))
        pieces = start
        for _ in range(8):
            pieces = advance_lag_chain(pieces, jnp.array(1.5), 0.3, jnp.array(0.25))

        assert jnp.allclose(whole, pieces, atol=1e-5)

    def test_rejects_an_empty_chain(self) -> None:
        """A T90 cannot be realised without any lag stage."""
        with pytest.raises(ValueError, match="positive order"):
            time_constant_for_t90(1.0, 0)
