import jax.numpy as jnp
from process_control.chemistry.demand_consumption import DemandConsumptionParams, compute_consumption


class TestTwoPhaseChlorineDecay:
    def test_demand_limited_regime_never_goes_negative(self) -> None:
        """A dt/tau ratio much larger than 1 must not overshoot to negative chlorine (Euler would)."""
        params = DemandConsumptionParams(tau=0.1, bulk_decay_rate=0.1)
        new_chlorine, new_demand, consumed = compute_consumption(
            chlorine=jnp.array(1.0), demand=jnp.array(5.0), params=params, dt=jnp.array(10.0),
        )
        assert float(new_chlorine) >= 0.0
        assert float(new_demand) >= 0.0

    def test_excess_chlorine_satisfies_demand_and_survives(self) -> None:
        """When chlorine exceeds demand, the demand pool empties and residual chlorine remains."""
        params = DemandConsumptionParams(tau=1.0, bulk_decay_rate=0.0)
        new_chlorine, new_demand, consumed = compute_consumption(
            chlorine=jnp.array(5.0), demand=jnp.array(1.0), params=params, dt=jnp.array(0.1),
        )
        assert float(new_demand) < 1.0
        assert float(new_chlorine) > 0.0
        assert float(consumed) > 0.0

    def test_bulk_decay_continues_after_demand_exhausted(self) -> None:
        """With demand already satisfied, residual chlorine still decays via the slow bulk term."""
        params_no_bulk = DemandConsumptionParams(tau=1.0, bulk_decay_rate=0.0)
        params_bulk = DemandConsumptionParams(tau=1.0, bulk_decay_rate=0.2)

        chlorine_no_bulk, _, _ = compute_consumption(
            chlorine=jnp.array(2.0), demand=jnp.array(0.0), params=params_no_bulk, dt=jnp.array(1.0),
        )
        chlorine_bulk, _, _ = compute_consumption(
            chlorine=jnp.array(2.0), demand=jnp.array(0.0), params=params_bulk, dt=jnp.array(1.0),
        )
        assert float(chlorine_bulk) < float(chlorine_no_bulk)

    def test_demand_consumed_excludes_bulk_decay(self) -> None:
        """demand_consumed reports only the fast-phase draw, for callers attributing it by source."""
        params = DemandConsumptionParams(tau=1.0, bulk_decay_rate=0.5)
        new_chlorine, new_demand, demand_consumed = compute_consumption(
            chlorine=jnp.array(5.0), demand=jnp.array(1.0), params=params, dt=jnp.array(0.1),
        )
        total_consumed = jnp.array(5.0) - new_chlorine
        assert float(demand_consumed) < float(total_consumed)
