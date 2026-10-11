import dataclasses

import jax
import jax.numpy as jnp
import pytest
from process_control.fields import LocalField, Phase, PrescribedContext, StreamField
from process_control.flowsheet import advance_unit
from process_control.integration_records import AdvanceStatus
from process_control.streams import Stream
from process_control.units.asm1 import ASM1_SCHEMA
from process_control.units.contract import unit_outlets
from process_control.units.layered_settler import BSM1_SETTLER, LayeredSettler, ParticulateOutlets, SettlerParams

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

CONTEXT = PrescribedContext(pressure_pa=101325.0)
SETTLER = LayeredSettler(ASM1_SCHEMA, context=CONTEXT)
TRANSPORTING = LayeredSettler(ASM1_SCHEMA, ParticulateOutlets.LAYER_COMPOSITION, context=CONTEXT)
SETTLERS = pytest.mark.parametrize("settler", [SETTLER, TRANSPORTING], ids=["feed_fractions", "layer_composition"])


def _feed(flow: float = FEED_FLOW, components: jax.Array = REACTOR_5) -> dict[str, Stream]:
    feed = Stream(
        flow=jnp.array(flow),
        components=components,
        temperature_k=jnp.array(288.15),
        pressure_pa=jnp.array(101325.0),
        schema=ASM1_SCHEMA,
        phase=Phase.LIQUID,
    )
    return {"feed": feed}


def _inputs(underflow: float = UNDERFLOW) -> dict[str, jax.Array]:
    return {"q_underflow": jnp.array(underflow), "temperature_k": jnp.array(288.15)}


@jax.jit
def _settle(state: jax.Array) -> tuple[jax.Array, jax.Array]:
    def hour(_: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        result = advance_unit(SETTLER, carry[0], _feed(), _inputs(), BSM1_SETTLER, jnp.array(1.0), 60)
        return result.state, carry[1] & (result.status == AdvanceStatus.SUCCESS)

    return jax.lax.fori_loop(0, 20 * 24, hour, (state, jnp.array(True)))


@pytest.fixture(scope="module")
def steady_state() -> jax.Array:
    state, succeeded = _settle(jnp.tile(REACTOR_5, (BSM1_SETTLER.n_layers, 1)))
    assert bool(succeeded)
    return state


class TestBSM1SteadyState:
    def test_layer_solids_match_the_reference_profile(self, steady_state: jax.Array) -> None:
        """Twenty days at the open-loop operating point reproduce the BSM1 Table 7 TSS profile."""
        assert steady_state.shape == (10, 13)
        assert jnp.allclose(ASM1_SCHEMA.tss(steady_state), LAYER_TSS, rtol=0.01)

    @SETTLERS
    def test_outlet_particulates_match_the_reference(self, settler: LayeredSettler, steady_state: jax.Array) -> None:
        """Both particulate modes reach the outlet concentrations of BSM1 Table 8."""
        outlets = unit_outlets(settler, steady_state, _feed(), _inputs(), BSM1_SETTLER)

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


def _balance(settler: LayeredSettler, params: SettlerParams, underflow: float = UNDERFLOW) -> tuple[jax.Array, jax.Array]:
    """Each component's rate of inventory change and its net inflow, for layers whose solids mix differs from the feed's."""
    depth = jnp.linspace(0.2, 3.0, params.n_layers)[:, None]
    heterotroph_share = jnp.ones((params.n_layers, 13)).at[:, 4].set(jnp.linspace(0.5, 1.5, params.n_layers))
    layers = depth * heterotroph_share * REACTOR_5
    layer_volume = params.area * params.height / params.n_layers

    derivatives = settler.derivatives(layers, _feed(), _inputs(underflow), params)
    outlets = unit_outlets(settler, layers, _feed(), _inputs(underflow), params)

    net_inflow = FEED_FLOW * REACTOR_5 - sum(stream.flow * stream.components for stream in outlets.values())
    return layer_volume * derivatives.sum(axis=0), net_inflow


TOLERANCE = 1e-5 * FEED_FLOW * float(REACTOR_5.max())


class TestMassBalance:
    def test_transported_components_are_each_conserved(self) -> None:
        """Carrying layer compositions, each component's inventory changes by what flows in less what flows out."""
        inventory_change, net_inflow = _balance(TRANSPORTING, BSM1_SETTLER)

        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=TOLERANCE)

    def test_feed_fractions_conserve_solids_and_solubles(self) -> None:
        """The BSM1 rule re-apportions particulates by the feed mix, so it conserves TSS and solubles rather than each particulate."""
        inventory_change, net_inflow = _balance(SETTLER, BSM1_SETTLER)

        assert jnp.allclose(ASM1_SCHEMA.tss(inventory_change), ASM1_SCHEMA.tss(net_inflow), rtol=1e-4, atol=TOLERANCE)
        assert jnp.allclose(inventory_change[SOLUBLES], net_inflow[SOLUBLES], rtol=1e-4, atol=TOLERANCE)

    def test_outlet_flows_split_the_feed(self) -> None:
        """The underflow is set by its input and the effluent takes the rest of the feed."""
        outlets = unit_outlets(SETTLER, jnp.zeros((10, 13)), _feed(), _inputs(), BSM1_SETTLER)

        assert jnp.allclose(outlets["underflow"].flow, UNDERFLOW)
        assert jnp.allclose(outlets["effluent"].flow, FEED_FLOW - UNDERFLOW)

    def test_overdrawn_underflow_is_capped_by_the_feed(self) -> None:
        """An underflow request above the feed takes the whole feed, so flow and mass still balance."""
        outlets = unit_outlets(SETTLER, jnp.zeros((10, 13)), _feed(), _inputs(2.0 * FEED_FLOW), BSM1_SETTLER)
        inventory_change, net_inflow = _balance(TRANSPORTING, BSM1_SETTLER, underflow=2.0 * FEED_FLOW)

        assert jnp.allclose(outlets["underflow"].flow, FEED_FLOW)
        assert jnp.allclose(outlets["effluent"].flow, 0.0)
        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=TOLERANCE)


class TestParticulateOutlets:
    # Moves 500 g COD/m³ from heterotrophs to inerts: the feed TSS is unchanged, its mix is not.
    SHIFTED_FEED = REACTOR_5.at[4].add(-500.0).at[2].add(500.0)

    def test_feed_fraction_step_reaches_the_outlets_immediately(self, steady_state: jax.Array) -> None:
        """Under the BSM1 rule a change in the feed's particulate mix appears at both outlets with no settler delay."""
        outlets = unit_outlets(SETTLER, steady_state, _feed(components=self.SHIFTED_FEED), _inputs(), BSM1_SETTLER)

        for name, layer in (("effluent", steady_state[-1]), ("underflow", steady_state[0])):
            expected = ASM1_SCHEMA.tss(layer) * self.SHIFTED_FEED / ASM1_SCHEMA.tss(self.SHIFTED_FEED)
            assert jnp.allclose(outlets[name].components[PARTICULATES], expected[PARTICULATES], rtol=1e-5)

    def test_layer_composition_ignores_the_current_feed_mix(self, steady_state: jax.Array) -> None:
        """Carrying layer compositions, the outlets show the mix held in the settler until it is displaced."""
        outlets = unit_outlets(TRANSPORTING, steady_state, _feed(components=self.SHIFTED_FEED), _inputs(), BSM1_SETTLER)

        assert jnp.allclose(outlets["underflow"].components, steady_state[0])
        assert jnp.allclose(outlets["effluent"].components, steady_state[-1])

    def test_outlet_dependencies_follow_the_mode(self) -> None:
        """Only the BSM1 rule reads the feed for outlet compositions, which a flowsheet must order before the settler."""
        def composition_reads(settler: LayeredSettler) -> tuple[LocalField, ...]:
            return next(group.reads for group in settler.evaluation_groups if group.name == "composition")

        assert composition_reads(SETTLER) == (LocalField("feed", StreamField.COMPONENTS),)
        assert composition_reads(TRANSPORTING) == ()


class TestConfiguration:
    def test_feed_layer_outside_the_settler_is_rejected(self) -> None:
        """The feed must enter one of the settler's layers."""
        with pytest.raises(ValueError, match="outside the 10 layers"):
            dataclasses.replace(BSM1_SETTLER, feed_layer=10)

    def test_other_layer_counts_conserve_mass(self) -> None:
        """A finer settler is configured by layer count and feed layer alone and still conserves what it receives."""
        inventory_change, net_inflow = _balance(TRANSPORTING, dataclasses.replace(BSM1_SETTLER, n_layers=20, feed_layer=11))

        assert jnp.allclose(inventory_change, net_inflow, rtol=1e-4, atol=TOLERANCE)
