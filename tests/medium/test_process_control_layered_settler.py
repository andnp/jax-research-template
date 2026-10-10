import dataclasses

import jax
import jax.numpy as jnp
import pytest
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA
from process_control.units.contract import step_unit, unit_outlets
from process_control.units.layered_settler import BSM1_SETTLER, LayeredSettler, SettlerParams

# Alex et al. (2008) BSM1 open-loop steady state: reactor 5 (Table 6) feeds the settler.
REACTOR_5 = jnp.array([30.0, 0.889, 1149.0, 49.3, 2559.0, 150.0, 452.0, 0.491, 10.4, 1.73, 0.688, 3.53, 4.13])
FEED_FLOW = 36892.0 / 24.0
UNDERFLOW = 18831.0 / 24.0
# Table 7, bottom layer first.
LAYER_TSS = jnp.array([6394.0, 356.0, 356.0, 356.0, 356.0, 356.0, 69.0, 29.5, 18.1, 12.5])
# Table 8, particulates X_I, X_S, X_BH, X_BA, X_P, X_ND.
UNDERFLOW_PARTICULATES = jnp.array([2247.0, 96.4, 5005.0, 293.0, 884.0, 6.90])
EFFLUENT_PARTICULATES = jnp.array([4.39, 0.188, 9.78, 0.573, 1.73, 0.0135])
PARTICULATES = jnp.array([2, 3, 4, 5, 6, 11])
SOLUBLES = jnp.array([0, 1, 7, 8, 9, 10, 12])

SETTLER = LayeredSettler(ASM1_SCHEMA)


def _feed(flow: float = FEED_FLOW) -> dict[str, Stream]:
    return {"feed": Stream(flow=jnp.array(flow), components=REACTOR_5, schema=ASM1_SCHEMA)}


def _inputs(underflow: float = UNDERFLOW) -> dict[str, jax.Array]:
    return {"q_underflow": jnp.array(underflow)}


@jax.jit
def _settle(state: jax.Array) -> jax.Array:
    def hour(_: int, current: jax.Array) -> jax.Array:
        return step_unit(SETTLER, current, _feed(), _inputs(), BSM1_SETTLER, jnp.array(1.0), 60)

    return jax.lax.fori_loop(0, 20 * 24, hour, state)


@pytest.fixture(scope="module")
def steady_state() -> jax.Array:
    return _settle(jnp.tile(REACTOR_5, (BSM1_SETTLER.n_layers, 1)))


class TestBSM1SteadyState:
    def test_layer_solids_match_the_reference_profile(self, steady_state: jax.Array) -> None:
        """Twenty days at the open-loop operating point reproduce the BSM1 Table 7 TSS profile."""
        assert steady_state.shape == (10, 13)
        assert jnp.allclose(ASM1_SCHEMA.tss(steady_state), LAYER_TSS, rtol=0.01)

    def test_outlet_particulates_match_the_reference(self, steady_state: jax.Array) -> None:
        """Carrying particulates per layer reaches the feed-fraction outlets of BSM1 Table 8."""
        outlets = unit_outlets(SETTLER, steady_state, _feed(), _inputs(), BSM1_SETTLER)

        assert jnp.allclose(outlets["underflow"].components[PARTICULATES], UNDERFLOW_PARTICULATES, rtol=0.01)
        assert jnp.allclose(outlets["effluent"].components[PARTICULATES], EFFLUENT_PARTICULATES, rtol=0.01)

    def test_solubles_pass_through_unchanged(self, steady_state: jax.Array) -> None:
        """A non-reactive settler leaves every layer at the feed's soluble concentrations."""
        assert jnp.allclose(steady_state[:, SOLUBLES], REACTOR_5[SOLUBLES], rtol=1e-4)

    def test_solids_entering_leave_through_the_outlets(self, steady_state: jax.Array) -> None:
        """At steady state the feed solids load equals the effluent plus underflow loads."""
        outlets = unit_outlets(SETTLER, steady_state, _feed(), _inputs(), BSM1_SETTLER)

        feed_load = FEED_FLOW * ASM1_SCHEMA.tss(REACTOR_5)
        outlet_load = sum(stream.flow * ASM1_SCHEMA.tss(stream.components) for stream in outlets.values())
        assert jnp.allclose(outlet_load, feed_load, rtol=1e-3)


def _balance(params: SettlerParams, underflow: float = UNDERFLOW) -> tuple[jax.Array, jax.Array]:
    """Each component's rate of inventory change and its net inflow, for a non-uniform layer state."""
    layers = jnp.linspace(0.2, 3.0, params.n_layers)[:, None] * REACTOR_5
    layer_volume = params.area * params.height / params.n_layers

    derivatives = SETTLER.derivatives(layers, _feed(), _inputs(underflow), params)
    outlets = unit_outlets(SETTLER, layers, _feed(), _inputs(underflow), params)

    net_inflow = FEED_FLOW * REACTOR_5 - sum(stream.flow * stream.components for stream in outlets.values())
    return layer_volume * derivatives.sum(axis=0), net_inflow


class TestMassBalance:
    def test_inventory_changes_by_inflow_minus_outflow(self) -> None:
        """For any layer state, each component's settler inventory changes by exactly what flows in less what flows out."""
        inventory_change, net_inflow = _balance(BSM1_SETTLER)

        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=1e-5 * FEED_FLOW * REACTOR_5.max())

    def test_outlet_flows_split_the_feed(self) -> None:
        """The underflow is set by its input and the effluent takes the rest of the feed."""
        flows = SETTLER.outlet_flows(jnp.zeros((10, 13)), {"feed": jnp.array(FEED_FLOW)}, _inputs(), BSM1_SETTLER)

        assert jnp.allclose(flows["underflow"], UNDERFLOW)
        assert jnp.allclose(flows["effluent"], FEED_FLOW - UNDERFLOW)

    def test_overdrawn_underflow_is_capped_by_the_feed(self) -> None:
        """An underflow request above the feed takes the whole feed, so flow and mass still balance."""
        flows = SETTLER.outlet_flows(jnp.zeros((10, 13)), {"feed": jnp.array(FEED_FLOW)}, _inputs(2.0 * FEED_FLOW), BSM1_SETTLER)
        inventory_change, net_inflow = _balance(BSM1_SETTLER, underflow=2.0 * FEED_FLOW)

        assert jnp.allclose(flows["underflow"], FEED_FLOW)
        assert jnp.allclose(flows["effluent"], 0.0)
        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=1e-5 * FEED_FLOW * REACTOR_5.max())


class TestConfiguration:
    def test_feed_layer_outside_the_settler_is_rejected(self) -> None:
        """The feed must enter one of the settler's layers."""
        with pytest.raises(ValueError, match="outside the 10 layers"):
            dataclasses.replace(BSM1_SETTLER, feed_layer=10)

    def test_other_layer_counts_conserve_mass(self) -> None:
        """A finer settler is configured by layer count and feed layer alone and still conserves what it receives."""
        inventory_change, net_inflow = _balance(dataclasses.replace(BSM1_SETTLER, n_layers=20, feed_layer=11))

        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=1e-5 * FEED_FLOW * REACTOR_5.max())
