import jax
import jax.numpy as jnp
import pytest
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.field_graph import CompiledGroups, External, GroupId, compile_groups
from process_control.fields import (
    EvaluationGroup,
    FieldRef,
    GroupResult,
    LocalField,
    Phase,
    PortSpec,
    StreamField,
    validate_groups,
)

FLOW = StreamField.FLOW
COMPONENTS = StreamField.COMPONENTS
SCHEMA = ComponentSchema(("tracer",), (False,), (0.0,), (ComponentBasis("g/m3", Quantity.PROXY, "tracer"),))
LIQUID = PortSpec(SCHEMA, Phase.LIQUID)


def field(port: str, kind: StreamField) -> LocalField:
    return LocalField(port, kind)


def group(
    name: str,
    outputs: list[LocalField],
    reads: list[LocalField] | None = None,
    input_names: tuple[str, ...] = (),
    reads_state: bool = False,
    diagnostic_names: tuple[str, ...] = (),
) -> EvaluationGroup:
    return EvaluationGroup(name, tuple(outputs), tuple(reads or ()), input_names, reads_state, diagnostic_names)


def assert_producers_precede_consumers(compiled: CompiledGroups) -> None:
    position = {gid: index for index, gid in enumerate(compiled.order)}
    for gid, sources in compiled.sources.items():
        for source in sources:
            if isinstance(source, FieldRef):
                assert position[compiled.owners[source]] < position[gid], f"{gid} scheduled before owner of {source}"


def passthrough(name: str = "flow") -> EvaluationGroup:
    return group(name, [field("out", FLOW)], [field("in", FLOW)])


class TestScheduling:
    def test_recycle_is_acyclic_when_composition_is_state_only(self) -> None:
        """A recycle of flow and composition resolves when the reactor composition comes from state and the pump flow is prescribed."""
        mixer = [
            group("flow", [field("out", FLOW)], [field("feed", FLOW), field("recycle", FLOW)]),
            group("composition", [field("out", COMPONENTS)], [field("feed", FLOW), field("recycle", FLOW), field("feed", COMPONENTS), field("recycle", COMPONENTS)]),
        ]
        reactor = [
            group("flow", [field("out", FLOW)], [field("in", FLOW)]),
            group("composition", [field("out", COMPONENTS)], reads_state=True),
        ]
        splitter = [
            group("draw", [field("recycle", FLOW)], input_names=("recycle",)),
            group("remainder", [field("out", FLOW)], [field("in", FLOW)]),
            group("composition", [field("recycle", COMPONENTS), field("out", COMPONENTS)], [field("in", COMPONENTS)]),
        ]
        connections = {
            ("mixer", "recycle"): ("splitter", "recycle"),
            ("reactor", "in"): ("mixer", "out"),
            ("splitter", "in"): ("reactor", "out"),
        }
        external = {FieldRef("mixer", "feed", FLOW), FieldRef("mixer", "feed", COMPONENTS)}

        compiled = compile_groups({"mixer": mixer, "reactor": reactor, "splitter": splitter}, connections, external)

        assert set(compiled.order) == {GroupId(node, g.name) for node, gs in [("mixer", mixer), ("reactor", reactor), ("splitter", splitter)] for g in gs}
        assert_producers_precede_consumers(compiled)
        assert compiled.owners[FieldRef("reactor", "out", COMPONENTS)] == GroupId("reactor", "composition")

    def test_independent_groups_keep_declaration_order(self) -> None:
        """Groups with no dependencies are scheduled in the order they were declared, and recompiling gives the same plan."""
        groups = {
            "b": [group("g", [field("out", FLOW)]), group("h", [field("other", FLOW)])],
            "a": [group("g", [field("out", FLOW)])],
        }

        first = compile_groups(groups, {})
        second = compile_groups(groups, {})

        assert first.order == (GroupId("b", "g"), GroupId("b", "h"), GroupId("a", "g"))
        assert first == second

    def test_ready_groups_do_not_jump_ahead_of_earlier_declared_ones(self) -> None:
        """A consumer declared before its producer still runs after it, and unrelated groups keep their relative order."""
        groups = {"n": [passthrough("late"), group("free", [field("x", FLOW)]), group("early", [field("src", FLOW)])]}
        connections = {("n", "in"): ("n", "src")}

        compiled = compile_groups(groups, connections)

        assert compiled.order == (GroupId("n", "free"), GroupId("n", "early"), GroupId("n", "late"))

    def test_groups_with_identical_reads_are_both_scheduled(self) -> None:
        """Two groups reading the same field are distinct computations, each owning its own outputs."""
        groups = {
            "src": [group("flow", [field("out", FLOW)])],
            "n": [
                group("first", [field("a", FLOW)], [field("in", FLOW)]),
                group("second", [field("b", FLOW)], [field("in", FLOW)]),
            ],
        }

        compiled = compile_groups(groups, {("n", "in"): ("src", "out")})

        assert set(compiled.order) == {GroupId("src", "flow"), GroupId("n", "first"), GroupId("n", "second")}
        assert compiled.owners[FieldRef("n", "a", FLOW)] == GroupId("n", "first")
        assert compiled.owners[FieldRef("n", "b", FLOW)] == GroupId("n", "second")
        assert_producers_precede_consumers(compiled)


class TestAtomicGroups:
    def test_group_waits_for_all_of_its_reads(self) -> None:
        """A group owning two outputs runs only after the producers of both of its reads, however early its first read is ready."""
        groups = {
            "n": [
                group("solve", [field("out", FLOW), field("out", COMPONENTS)], [field("a", FLOW), field("b", COMPONENTS)]),
            ],
            "fast": [group("flow", [field("out", FLOW)])],
            "slow_input": [group("flow", [field("out", FLOW)])],
            "slow": [group("composition", [field("out", COMPONENTS)], [field("in", FLOW)])],
        }
        connections = {("n", "a"): ("fast", "out"), ("n", "b"): ("slow", "out"), ("slow", "in"): ("slow_input", "out")}

        compiled = compile_groups(groups, connections)

        order = list(compiled.order)
        solve = GroupId("n", "solve")
        assert order.index(solve) > order.index(GroupId("fast", "flow"))
        assert order.index(solve) > order.index(GroupId("slow", "composition"))
        assert_producers_precede_consumers(compiled)

    def test_coupled_group_closes_a_cycle_that_separate_groups_avoid(self) -> None:
        """Joint ownership of flow and composition makes a loop through the composition read a genuine cycle."""
        downstream = [group("flow", [field("out", FLOW)], [field("in", FLOW)])]
        connections = {("coupled", "in_c"): ("downstream", "out"), ("downstream", "in"): ("coupled", "out")}
        external = {FieldRef("coupled", "in_f", FLOW)}
        separate = [
            group("flow", [field("out", FLOW)], [field("in_f", FLOW)]),
            group("composition", [field("out", COMPONENTS)], [field("in_c", FLOW)]),
        ]
        coupled = [group("solve", [field("out", FLOW), field("out", COMPONENTS)], [field("in_f", FLOW), field("in_c", FLOW)])]

        assert compile_groups({"coupled": separate, "downstream": downstream}, connections, external)
        with pytest.raises(ValueError, match="cycle"):
            compile_groups({"coupled": coupled, "downstream": downstream}, connections, external)


class TestCycles:
    def test_same_field_cycle_is_rejected_with_a_witness(self) -> None:
        """A flow loop between two nodes is rejected, naming each participating group and the field it reads."""
        groups = {"a": [passthrough()], "b": [passthrough()], "bystander": [group("x", [field("out", FLOW)])]}
        connections = {("a", "in"): ("b", "out"), ("b", "in"): ("a", "out")}

        with pytest.raises(ValueError) as excinfo:
            compile_groups(groups, connections)

        message = str(excinfo.value)
        assert "a.flow reads b.out.flow" in message
        assert "b.flow reads a.out.flow" in message
        assert "bystander" not in message

    def test_self_dependency_is_rejected(self) -> None:
        """A group that reads a field it owns, through a connection back to itself, is a cycle."""
        with pytest.raises(ValueError, match=r"n\.flow reads n\.out\.flow"):
            compile_groups({"n": [passthrough()]}, {("n", "in"): ("n", "out")})


class TestOwnershipAndRoots:
    def test_duplicate_output_ownership_is_rejected(self) -> None:
        """Two groups of one node cannot own the same outlet field."""
        groups = {"n": [group("one", [field("out", FLOW)]), group("two", [field("out", FLOW)])]}

        with pytest.raises(ValueError, match="owned by both"):
            compile_groups(groups, {})

    def test_read_of_an_unowned_producer_field_is_rejected(self) -> None:
        """A connection to an outlet field that no group owns cannot be satisfied."""
        groups = {"src": [group("flow", [field("out", FLOW)])], "n": [group("g", [field("out", FLOW)], [field("in", COMPONENTS)])]}

        with pytest.raises(ValueError, match="no group owns"):
            compile_groups(groups, {("n", "in"): ("src", "out")})

    def test_external_fields_are_roots(self) -> None:
        """Reads of an unconnected inlet resolve to the declared external field and impose no ordering."""
        groups = {"n": [passthrough()]}
        external = {FieldRef("n", "in", FLOW)}

        compiled = compile_groups(groups, {}, external)

        assert compiled.order == (GroupId("n", "flow"),)
        assert compiled.sources[GroupId("n", "flow")] == (External(FieldRef("n", "in", FLOW)),)

    def test_unconnected_read_that_is_not_external_is_rejected(self) -> None:
        """An inlet that is neither connected nor supplied externally is a build error."""
        with pytest.raises(ValueError, match="neither connected nor external"):
            compile_groups({"n": [passthrough()]}, {})

    def test_sources_follow_declared_read_order(self) -> None:
        """Resolved sources line up with the group's reads, each pointing at the connected producer field."""
        groups = {
            "a": [group("flow", [field("out", FLOW)])],
            "b": [group("flow", [field("out", FLOW)])],
            "m": [group("flow", [field("out", FLOW)], [field("second", FLOW), field("first", FLOW)])],
        }
        connections = {("m", "first"): ("a", "out"), ("m", "second"): ("b", "out")}

        compiled = compile_groups(groups, connections)

        assert compiled.sources[GroupId("m", "flow")] == (FieldRef("b", "out", FLOW), FieldRef("a", "out", FLOW))


class TestDeclarationValidation:
    def test_complete_declaration_is_accepted(self) -> None:
        """Groups owning every field of every outlet and reading declared inlets validate."""
        groups = [
            group("flow", [field("out", FLOW)], [field("in", FLOW)]),
            group("rest", [field("out", COMPONENTS), field("out", StreamField.TEMPERATURE), field("out", StreamField.PRESSURE)], reads_state=True),
        ]

        validate_groups(groups, {"in": LIQUID}, {"out": LIQUID})

    @pytest.mark.parametrize(
        ("groups", "message"),
        [
            ([group("a", [field("out", FLOW)]), group("a", [field("x", FLOW)])], "group names"),
            ([group("a", [field("out", FLOW)]), group("b", [field("out", FLOW)])], "outputs across groups"),
            ([group("a", [field("out", FLOW), field("out", FLOW)])], "outputs across groups"),
            ([group("a", [])], "owns no outputs"),
            ([group("a", [field("out", FLOW)], [field("ghost", FLOW)])], "undeclared inlet ports"),
            ([group("a", [field("out", FLOW)], diagnostic_names=("d",)), group("b", [field("x", FLOW)], diagnostic_names=("d",))], "diagnostic names"),
            ([group("a", [field("out", FLOW)], input_names=("q", "q"))], "input names"),
        ],
    )
    def test_inconsistent_declarations_are_rejected(self, groups: list[EvaluationGroup], message: str) -> None:
        """Duplicate names, duplicate ownership, empty groups, undeclared reads and duplicate diagnostics fail at construction."""
        with pytest.raises(ValueError, match=message):
            validate_groups(groups, {"in": LIQUID})

    def test_missing_outlet_coverage_is_rejected(self) -> None:
        """Every field of every declared outlet must be owned by some group."""
        groups = [group("flow", [field("out", FLOW)])]

        with pytest.raises(ValueError, match="owned by no group"):
            validate_groups(groups, {}, {"out": LIQUID})

    def test_ownership_of_undeclared_outlet_is_rejected(self) -> None:
        """Groups cannot own fields of a port the unit does not declare as an outlet."""
        groups = [group("flow", [field("ghost", FLOW)])]

        with pytest.raises(ValueError, match="undeclared outlet ports"):
            validate_groups(groups, {}, {"out": LIQUID})


class TestGroupResult:
    def test_values_and_diagnostics_are_pytree_data(self) -> None:
        """Both tuples are traced leaves, so a group result can cross a jit boundary and keep its structure."""
        result = GroupResult(values=(jnp.ones(2), jnp.zeros(())), diagnostics=(jnp.full((3,), 2.0),))

        @jax.jit
        def increment(r: GroupResult) -> GroupResult:
            return GroupResult(tuple(v + 1 for v in r.values), tuple(d + 1 for d in r.diagnostics))

        out = increment(result)

        assert isinstance(out, GroupResult)
        assert [leaf.shape for leaf in out.values] == [(2,), ()]
        assert jnp.allclose(out.diagnostics[0], jnp.full((3,), 3.0))
