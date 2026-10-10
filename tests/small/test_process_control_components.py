import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.units.asm1 import ASM1_SCHEMA, ASM1State, compute_tss

TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")


class TestASM1Schema:
    def test_tss_matches_bsm1_definition(self) -> None:
        """The schema, in ASM1State field order, reproduces the BSM1 0.75 COD-to-TSS conversion of the particulate COD fractions."""
        state = ASM1State.create()
        components = jnp.stack(jax.tree.leaves(state))

        assert jnp.allclose(ASM1_SCHEMA.tss(components), compute_tss(state))

    def test_particulate_nitrogen_is_particulate_without_tss(self) -> None:
        """X_ND settles with the sludge but carries no TSS of its own."""
        x_nd = ASM1_SCHEMA.index("x_nd")

        assert bool(ASM1_SCHEMA.particulate_mask[x_nd])
        assert ASM1_SCHEMA.tss_factor[x_nd] == 0.0
        assert not bool(ASM1_SCHEMA.soluble_mask[x_nd])


    def test_bases_follow_alex_2008_table_6(self) -> None:
        """Organic channels are COD, oxygen is O2, nitrogen channels are N and alkalinity is mol/m3 of equivalents."""
        bases = dict(zip(ASM1_SCHEMA.names, ASM1_SCHEMA.bases, strict=True))

        assert bases["x_bh"] == ComponentBasis("g/m3", Quantity.COD, "COD")
        assert bases["s_o"] == ComponentBasis("g/m3", Quantity.SPECIES, "O2")
        assert {bases[name] for name in ("s_no", "s_nh", "s_nd", "x_nd")} == {ComponentBasis("g/m3", Quantity.ELEMENT, "N")}
        assert bases["s_alk"] == ComponentBasis("mol/m3", Quantity.EQUIVALENT, "HCO3-")


class TestSchemaValidation:
    def test_rejects_duplicate_names(self) -> None:
        """Index lookup by name is ambiguous with duplicates."""
        with pytest.raises(ValueError, match="unique"):
            ComponentSchema(names=("a", "a"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER,) * 2)

    def test_rejects_mismatched_lengths(self) -> None:
        """Every component needs a particulate flag and a TSS factor."""
        with pytest.raises(ValueError, match="equal lengths"):
            ComponentSchema(names=("a", "b"), particulate=(False,), tss_factor=(0.0, 0.0), bases=(TRACER,) * 2)

    def test_rejects_a_basis_count_that_differs_from_the_names(self) -> None:
        """Every component needs exactly one basis."""
        with pytest.raises(ValueError, match="equal lengths"):
            ComponentSchema(names=("a", "b"), particulate=(False, False), tss_factor=(0.0, 0.0), bases=(TRACER,))

    def test_schemas_differing_only_in_basis_are_unequal(self) -> None:
        """A different unit or reference makes a different carrier, even with identical names."""
        mol = ComponentBasis("mol/m3", Quantity.PROXY, "tracer")

        assert ComponentSchema(("a",), (False,), (0.0,), (TRACER,)) != ComponentSchema(("a",), (False,), (0.0,), (mol,))

    def test_rejects_soluble_tss(self) -> None:
        """A dissolved component cannot contribute suspended solids."""
        with pytest.raises(ValueError, match="cannot contribute"):
            ComponentSchema(names=("a",), particulate=(False,), tss_factor=(0.5,), bases=(TRACER,))

    @pytest.mark.parametrize("factor", [-0.75, float("nan"), float("inf")])
    def test_rejects_invalid_tss_factor(self, factor: float) -> None:
        """A negative or non-finite factor would corrupt every TSS-derived metric."""
        with pytest.raises(ValueError, match="finite and non-negative"):
            ComponentSchema(names=("x",), particulate=(True,), tss_factor=(factor,), bases=(TRACER,))


class TestComponentBasis:
    def test_equal_bases_are_interchangeable_as_keys(self) -> None:
        """A basis is immutable and hashable, so equal bases collapse in sets and dict keys."""
        assert len({ComponentBasis("g/m3", Quantity.COD, "COD"), ComponentBasis("g/m3", Quantity.COD, "COD")}) == 1

    def test_differing_in_any_field_are_distinct(self) -> None:
        """Unit, quantity and reference each distinguish a basis."""
        base = ComponentBasis("g/m3", Quantity.ELEMENT, "N")

        assert base != ComponentBasis("mol/m3", Quantity.ELEMENT, "N")
        assert base != ComponentBasis("g/m3", Quantity.SPECIES, "N")
        assert base != ComponentBasis("g/m3", Quantity.ELEMENT, "P")

    @pytest.mark.parametrize(("unit", "reference"), [("", "N"), ("g/m3", "")])
    def test_rejects_empty_fields(self, unit: str, reference: str) -> None:
        """An unnamed unit or reference cannot support an exact-match check."""
        with pytest.raises(ValueError, match="nonempty"):
            ComponentBasis(unit, Quantity.ELEMENT, reference)
