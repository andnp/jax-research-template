"""Inventory balances of a well-mixed holdup: transport, volume accounting, division guard."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.inventory import HoldupState, IdentityChart, InventoryBalance, inventory_balance, reconstruct

CHART = IdentityChart()


def _balance(
    volume: float,
    amounts: list[float],
    inlet_flows: list[float],
    inlet_concentrations: list[list[float]],
    draws: list[float],
    overflow: float = 0.0,
    source: list[float] | None = None,
):
    state = HoldupState(jnp.asarray(volume), jnp.asarray(amounts))
    inventory = reconstruct(CHART, state)
    channels = len(amounts)
    return inventory_balance(
        CHART,
        state,
        inventory,
        jnp.asarray(inlet_flows),
        jnp.asarray(inlet_concentrations).reshape(len(inlet_flows), channels),
        jnp.asarray(draws),
        jnp.asarray(overflow),
        jnp.asarray(source if source is not None else [0.0] * channels),
    )


def _concentration_rate(volume: float, amounts: list[float], balance: InventoryBalance) -> np.ndarray:
    """d(n/V)/dt = (n_dot - C V_dot) / V from the balance's own rates."""
    n, v_dot = np.asarray(amounts), float(balance.volume_rate)
    return (np.asarray(balance.coordinate_rate) - n / volume * v_dot) / volume


class TestTransport:
    def test_filling_at_the_tank_concentration_keeps_concentration_constant(self):
        """A feed equal to the contents changes volume and inventory but not concentration, on every channel."""
        amounts = [20.0, 5.0]
        balance = _balance(10.0, amounts, [3.0], [[2.0, 0.5]], [1.0])
        assert float(balance.volume_rate) == pytest.approx(2.0)
        assert np.allclose(balance.coordinate_rate, [4.0, 1.0])
        assert np.allclose(_concentration_rate(10.0, amounts, balance), 0.0)

    def test_draining_without_reaction_keeps_concentration_constant(self):
        """Removing liquid removes inventory in proportion, leaving concentration unchanged."""
        amounts = [20.0, 5.0]
        balance = _balance(10.0, amounts, [0.0], [[0.0, 0.0]], [2.0])
        assert float(balance.volume_rate) == pytest.approx(-2.0)
        assert np.allclose(balance.coordinate_rate, [-4.0, -1.0])
        assert np.allclose(_concentration_rate(10.0, amounts, balance), 0.0)

    def test_opaque_channel_is_transported_like_any_other(self):
        """A channel no chemistry reads still moves with inlets and outlets."""
        balance = _balance(10.0, [0.0, 7.0], [2.0], [[0.0, 3.0]], [0.0], overflow=1.0)
        assert float(balance.coordinate_rate[1]) == pytest.approx(2.0 * 3.0 - 1.0 * 0.7)

    def test_batch_with_no_flows_is_static(self):
        """With no inlets, draws or sources both volume and inventories are constant."""
        balance = _balance(10.0, [4.0], [0.0], [[0.0]], [0.0])
        assert float(balance.volume_rate) == 0.0
        assert float(balance.coordinate_rate[0]) == 0.0


class TestVolumeBalance:
    def test_volume_rate_is_inflow_minus_draws_minus_overflow(self):
        """The volume balance is linear in the flows, with a volume source adding to it."""
        state = HoldupState(jnp.asarray(10.0), jnp.asarray([1.0]))
        inventory = reconstruct(CHART, state)
        balance = inventory_balance(
            CHART,
            state,
            inventory,
            jnp.array([2.0, 3.0]),
            jnp.zeros((2, 1)),
            jnp.array([1.0, 0.5]),
            jnp.asarray(0.25),
            jnp.zeros(1),
            volume_source_flow=0.1,
        )
        assert float(balance.net_flow) == pytest.approx(5.0 - 1.5 + 0.1)
        assert float(balance.volume_rate) == pytest.approx(5.0 - 1.5 + 0.1 - 0.25)
        assert float(balance.outflow) == pytest.approx(1.75)

    def test_draw_exceeding_inflow_drains_volume(self):
        """Stored liquid supplies a draw larger than the feed; nothing caps it."""
        balance = _balance(10.0, [10.0], [1.0], [[1.0]], [4.0])
        assert float(balance.volume_rate) == pytest.approx(-3.0)


class TestSources:
    def test_concentration_source_becomes_an_amount_rate_once(self):
        """A source in concentration per hour is multiplied by the volume exactly once."""
        balance = _balance(10.0, [4.0], [0.0], [[0.0]], [0.0], source=[0.3])
        assert float(balance.coordinate_rate[0]) == pytest.approx(10.0 * 0.3)
        assert float(balance.volume_rate) == 0.0


class TestDivisionGuard:
    @pytest.mark.parametrize("volume", [0.0, -1.0, float("nan"), float("inf")])
    def test_nonpositive_or_nonfinite_volume_is_rejected(self, volume: float):
        """Reconstruction never divides by a volume that is not finite and positive."""
        with pytest.raises(ValueError, match="holdup volume"):
            reconstruct(CHART, HoldupState(jnp.asarray(volume), jnp.asarray([1.0])))

    def test_small_positive_volume_is_still_evaluable(self):
        """The wet operating boundary belongs to the integration policy, so a trial just past it can still be evaluated."""
        assert np.allclose(reconstruct(CHART, HoldupState(jnp.asarray(0.1), jnp.asarray([1.0]))).concentrations, [10.0])

    def test_nonfinite_inventory_is_rejected(self):
        """A NaN inventory is rejected rather than propagated into concentrations."""
        with pytest.raises(ValueError, match="finite"):
            reconstruct(CHART, HoldupState(jnp.asarray(5.0), jnp.asarray([jnp.nan])))

    def test_traced_invalid_volume_poisons_concentrations_with_nan(self):
        """Under jit an invalid stage cannot raise, so it returns NaN instead of a plausible number."""

        @jax.jit
        def concentrations(volume: jax.Array):
            return reconstruct(CHART, HoldupState(volume, jnp.asarray([1.0]))).concentrations

        assert np.isnan(concentrations(jnp.asarray(0.0))).all()
        assert np.allclose(concentrations(jnp.asarray(2.0)), [0.5])
