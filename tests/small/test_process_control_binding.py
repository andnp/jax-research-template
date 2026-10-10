import jax
import jax.numpy as jnp
import pytest
from process_control.binding import Binding, NamedAccess, Requirement
from process_control.components import ComponentBasis, ComponentSchema, Quantity

OXYGEN = ComponentBasis("g/m3", Quantity.SPECIES, "O2")
NITROGEN = ComponentBasis("g/m3", Quantity.ELEMENT, "N")
TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")


def _schema(*channels: tuple[str, ComponentBasis]) -> ComponentSchema:
    count = len(channels)
    return ComponentSchema(
        names=tuple(name for name, _ in channels),
        particulate=(False,) * count,
        tss_factor=(0.0,) * count,
        bases=tuple(basis for _, basis in channels),
    )


def _oxygen_access() -> NamedAccess:
    oxygen = Requirement("O2", OXYGEN)
    return NamedAccess(reads=(oxygen,), writes=(oxygen,))


class TestChannelResolution:
    def test_reads_follow_the_declared_order_on_a_shuffled_and_extended_carrier(self) -> None:
        """A carrier that reorders the module's channels and adds others still delivers exactly the declared reads, in order."""
        access = NamedAccess(reads=(Requirement("s_no", NITROGEN), Requirement("s_o", OXYGEN)))
        schema = _schema(("extra", TRACER), ("s_o", OXYGEN), ("other", TRACER), ("s_no", NITROGEN))

        binding = Binding.resolve(access, schema)

        assert jnp.array_equal(binding.gather(jnp.array([10.0, 20.0, 30.0, 40.0])), jnp.array([40.0, 20.0]))

    def test_explicit_mapping_binds_a_local_oxygen_name_to_the_carrier_channel(self) -> None:
        """A module declaring O2 reads and writes the carrier's s_o when mapped, and nothing else."""
        schema = _schema(("s_no", NITROGEN), ("s_o", OXYGEN))
        binding = Binding.resolve(_oxygen_access(), schema, {"O2": "s_o"})
        state = jnp.array([1.0, 2.0])

        assert jnp.array_equal(binding.gather(state), jnp.array([2.0]))
        assert jnp.array_equal(binding.scatter_add(state, jnp.array([0.5])), jnp.array([1.0, 2.5]))

    def test_without_a_mapping_local_names_must_match_channel_names(self) -> None:
        """Identity binding does not guess that a reference species identifies a channel."""
        with pytest.raises(ValueError, match="not in the schema"):
            Binding.resolve(_oxygen_access(), _schema(("s_o", OXYGEN)))

    def test_rejects_a_local_name_missing_from_the_mapping(self) -> None:
        """A mapping must cover every declared name."""
        with pytest.raises(ValueError, match="no channel mapped"):
            Binding.resolve(_oxygen_access(), _schema(("s_o", OXYGEN)), {})

    def test_rejects_a_mapped_channel_missing_from_the_carrier(self) -> None:
        """Binding to a nonexistent channel fails at build time, not inside a jitted step."""
        with pytest.raises(ValueError, match="not in the schema"):
            Binding.resolve(_oxygen_access(), _schema(("s_o", OXYGEN)), {"O2": "oxygen"})

    @pytest.mark.parametrize(
        "carrier_basis",
        [ComponentBasis("mol/m3", Quantity.SPECIES, "O2"), ComponentBasis("g/m3", Quantity.COD, "O2"), ComponentBasis("g/m3", Quantity.SPECIES, "N2")],
    )
    def test_rejects_a_channel_whose_basis_differs_in_any_field(self, carrier_basis: ComponentBasis) -> None:
        """Exact basis equality is required; no unit conversion is inferred."""
        with pytest.raises(ValueError, match="requires"):
            Binding.resolve(_oxygen_access(), _schema(("s_o", carrier_basis)), {"O2": "s_o"})

    def test_rejects_two_local_writes_mapped_to_one_channel(self) -> None:
        """Two distinct local writers on one carrier channel would be ambiguous about which owns it."""
        access = NamedAccess(writes=(Requirement("a", TRACER), Requirement("b", TRACER)))

        with pytest.raises(ValueError, match="same channel"):
            Binding.resolve(access, _schema(("c", TRACER)), {"a": "c", "b": "c"})


class TestAccessDeclaration:
    def test_rejects_repeated_names_within_reads_or_writes(self) -> None:
        """A local name identifies one read and one write."""
        twice = (Requirement("a", TRACER), Requirement("a", TRACER))

        with pytest.raises(ValueError, match="reads"):
            NamedAccess(reads=twice)
        with pytest.raises(ValueError, match="writes"):
            NamedAccess(writes=twice)

    def test_rejects_a_name_read_and_written_with_different_bases(self) -> None:
        """One local name denotes one quantity."""
        with pytest.raises(ValueError, match="different bases"):
            NamedAccess(reads=(Requirement("a", TRACER),), writes=(Requirement("a", OXYGEN),))

    def test_a_channel_may_be_both_read_and_written(self) -> None:
        """Read/write overlap is how a reaction consumes what it changes."""
        binding = Binding.resolve(_oxygen_access(), _schema(("c", OXYGEN)), {"O2": "c"})
        state = jnp.array([4.0])

        assert jnp.array_equal(binding.scatter_add(state, -binding.gather(state)), jnp.array([0.0]))


class TestApplication:
    def test_empty_access_gathers_nothing_and_leaves_the_target_unchanged(self) -> None:
        """A module with no reads or writes is a valid no-op."""
        binding = Binding.resolve(NamedAccess(), _schema(("c", TRACER)))
        state = jnp.array([[1.0], [2.0]])

        assert binding.gather(state).shape == (2, 0)
        assert jnp.array_equal(binding.scatter_add(state, jnp.zeros((2, 0))), state)

    def test_scatter_add_returns_a_new_array_and_accumulates(self) -> None:
        """Values add to existing contents without mutating the target or touching other channels."""
        schema = _schema(("a", TRACER), ("b", TRACER), ("c", TRACER))
        access = NamedAccess(writes=(Requirement("c", TRACER), Requirement("a", TRACER)))
        binding = Binding.resolve(access, schema)
        target = jnp.array([1.0, 2.0, 3.0])

        result = binding.scatter_add(target, jnp.array([10.0, 20.0]))

        assert jnp.array_equal(result, jnp.array([21.0, 2.0, 13.0]))
        assert jnp.array_equal(target, jnp.array([1.0, 2.0, 3.0]))

    def test_last_axis_batches_work_under_jit_and_vmap(self) -> None:
        """Leading axes are batch axes, so a batched gather equals the per-row gather and jit agrees with eager."""
        schema = _schema(("a", TRACER), ("b", TRACER), ("c", TRACER))
        access = NamedAccess(reads=(Requirement("c", TRACER), Requirement("a", TRACER)), writes=(Requirement("b", TRACER),))
        binding = Binding.resolve(access, schema)
        batch = jnp.arange(12.0).reshape(4, 3)

        assert jnp.array_equal(jax.jit(binding.gather)(batch), jax.vmap(binding.gather)(batch))
        assert jnp.array_equal(binding.gather(batch), batch[:, jnp.array([2, 0])])
        updated = jax.jit(jax.vmap(binding.scatter_add))(batch, jnp.ones((4, 1)))
        assert jnp.array_equal(updated, batch.at[:, 1].add(1.0))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_scatter_add_preserves_the_target_dtype(self, dtype: jnp.dtype) -> None:
        """Gather and scatter do not promote the carrier's dtype."""
        binding = Binding.resolve(NamedAccess(writes=(Requirement("c", TRACER),)), _schema(("c", TRACER)))

        target = jnp.zeros(1, dtype=dtype)

        assert binding.scatter_add(target, binding.gather(target)).dtype == dtype
