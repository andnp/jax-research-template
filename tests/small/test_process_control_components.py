import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentSchema
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
