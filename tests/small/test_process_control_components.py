import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.units.asm1 import ASM1_SCHEMA, ASM1State, compute_tss


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


class TestSchemaValidation:
    def test_rejects_duplicate_names(self) -> None:
        """Index lookup by name is ambiguous with duplicates."""
        with pytest.raises(ValueError, match="unique"):
            ComponentSchema(names=("a", "a"), particulate=(False, False), tss_factor=(0.0, 0.0))

    def test_rejects_mismatched_lengths(self) -> None:
        """Every component needs a particulate flag and a TSS factor."""
        with pytest.raises(ValueError, match="equal lengths"):
            ComponentSchema(names=("a", "b"), particulate=(False,), tss_factor=(0.0, 0.0))

    def test_rejects_soluble_tss(self) -> None:
        """A dissolved component cannot contribute suspended solids."""
        with pytest.raises(ValueError, match="cannot contribute"):
            ComponentSchema(names=("a",), particulate=(False,), tss_factor=(0.5,))

    @pytest.mark.parametrize("factor", [-0.75, float("nan"), float("inf")])
    def test_rejects_invalid_tss_factor(self, factor: float) -> None:
        """A negative or non-finite factor would corrupt every TSS-derived metric."""
        with pytest.raises(ValueError, match="finite and non-negative"):
            ComponentSchema(names=("x",), particulate=(True,), tss_factor=(factor,))


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
