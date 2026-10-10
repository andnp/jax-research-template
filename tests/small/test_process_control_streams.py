import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.streams import Stream, mix, split
from process_control.units.asm1 import ASM1_SCHEMA

TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")

SCHEMA = ComponentSchema(names=("solute", "solids"), particulate=(False, True), tss_factor=(0.0, 0.75), bases=(TRACER,) * 2)


def _stream(flow: float, solute: float, solids: float, schema: ComponentSchema = SCHEMA) -> Stream:
    return Stream(flow=jnp.array(flow), components=jnp.array([solute, solids]), schema=schema)


class TestMix:
    def test_conserves_flow_and_component_mass(self) -> None:
        """The mixture carries the summed flow and the summed mass flux of every component."""
        streams = (_stream(2.0, 10.0, 0.0), _stream(6.0, 2.0, 40.0), _stream(0.5, 0.0, 8.0))

        mixed = mix(*streams)

        assert jnp.allclose(mixed.flow, 8.5)
        expected_mass = sum(stream.flow * stream.components for stream in streams)
        assert jnp.allclose(mixed.flow * mixed.components, expected_mass)

    def test_zero_total_flow_is_finite(self) -> None:
        """With nothing flowing the composition stays finite instead of dividing by zero."""
        mixed = mix(_stream(0.0, 4.0, 2.0), _stream(0.0, 4.0, 2.0))

        assert jnp.allclose(mixed.flow, 0.0)
        assert jnp.allclose(mixed.components, jnp.array([4.0, 2.0]))

    def test_zero_flow_inlet_does_not_dilute(self) -> None:
        """An idle inlet contributes no mass, whatever concentration it reports."""
        mixed = mix(_stream(3.0, 5.0, 1.0), _stream(0.0, 1000.0, 1000.0))

        assert jnp.allclose(mixed.components, jnp.array([5.0, 1.0]))

    def test_rejects_mismatched_schemas(self) -> None:
        """Component vectors ordered by different schemas cannot be averaged elementwise."""
        other = ComponentSchema(names=("a", "b"), particulate=(False, True), tss_factor=(0.0, 0.75), bases=(TRACER,) * 2)

        with pytest.raises(ValueError, match="schemas"):
            mix(_stream(1.0, 1.0, 1.0), _stream(1.0, 1.0, 1.0, schema=other))

    def test_works_under_jit_and_vmap(self) -> None:
        """Mixing a batch of stream pairs under jit matches mixing each pair alone."""
        flows = jnp.array([1.0, 0.0, 3.0])
        first = Stream(flow=flows, components=jnp.ones((3, 13)), schema=ASM1_SCHEMA)
        second = Stream(flow=2.0 * flows, components=jnp.full((3, 13), 4.0), schema=ASM1_SCHEMA)

        mixed = jax.jit(jax.vmap(mix))(first, second)

        assert mixed.components.shape == (3, 13)
        assert jnp.allclose(mixed.flow, 3.0 * flows)
        assert jnp.allclose(mixed.components[0], 3.0)


class TestSplit:
    def test_absolute_take_off_leaves_the_remainder(self) -> None:
        """An absolute draw gets exactly its flow and the remainder outlet the rest."""
        outlets = split(_stream(10.0, 1.0, 2.0), "rest", flows={"waste": 3.0})

        assert jnp.allclose(outlets["waste"].flow, 3.0)
        assert jnp.allclose(outlets["rest"].flow, 7.0)

    def test_take_off_is_limited_to_the_inlet_flow(self) -> None:
        """Asking for more than arrives drains the inlet without a negative remainder."""
        outlets = split(_stream(2.0, 1.0, 2.0), "rest", flows={"waste": 5.0})

        assert jnp.allclose(outlets["waste"].flow, 2.0)
        assert jnp.allclose(outlets["rest"].flow, 0.0)

    def test_fractions_divide_the_inlet_flow(self) -> None:
        """Fractional draws are shares of the inlet flow, drawn after absolute ones."""
        outlets = split(_stream(10.0, 1.0, 2.0), "rest", flows={"pump": 4.0}, fractions={"bleed": 0.25})

        assert jnp.allclose(outlets["pump"].flow, 4.0)
        assert jnp.allclose(outlets["bleed"].flow, 2.5)
        assert jnp.allclose(outlets["rest"].flow, 3.5)

    def test_conserves_flow_and_composition(self) -> None:
        """Every outlet keeps the inlet composition and the outlet flows sum to the inlet flow."""
        inlet = _stream(10.0, 1.0, 2.0)

        outlets = split(inlet, "rest", flows={"a": 4.0, "b": 9.0}, fractions={"c": 0.5})

        assert jnp.allclose(sum(outlet.flow for outlet in outlets.values()), inlet.flow)
        for outlet in outlets.values():
            assert jnp.allclose(outlet.components, inlet.components)

    def test_rejects_duplicate_outlets(self) -> None:
        """Two draws into one outlet name would silently overwrite each other."""
        with pytest.raises(ValueError, match="unique"):
            split(_stream(1.0, 1.0, 1.0), "rest", flows={"rest": 0.5})
