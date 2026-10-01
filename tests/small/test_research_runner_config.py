import dataclasses

from research_runner.config import _traced_paths, traced


@dataclasses.dataclass(frozen=True)
class Opt:
    LR: float = traced(1e-3)
    STEPS: int = 10


@dataclasses.dataclass(frozen=True)
class Env:
    NAME: str = "a"
    NOISE: float = traced(0.0)


@dataclasses.dataclass(frozen=True)
class Cfg:
    agent: Opt = Opt()
    env: Env = Env()
    gamma: float = traced(0.99)


class TestTraced:
    def test_field_keeps_its_declared_default(self):
        """A traced field behaves as an ordinary defaulted dataclass field."""
        assert Opt() == Opt(LR=1e-3, STEPS=10)

    def test_field_can_be_overridden_at_construction(self):
        """Marking a field traced does not change how it is set."""
        assert Opt(LR=0.5).LR == 0.5


class TestTracedPaths:
    def test_lists_nested_paths_in_declaration_order(self):
        """Paths are dotted from the root, depth-first, in field order."""
        assert _traced_paths(Cfg()) == ("agent.LR", "env.NOISE", "gamma")

    def test_config_without_traced_fields_lists_none(self):
        """A config with nothing marked traced yields no paths."""

        @dataclasses.dataclass(frozen=True)
        class Plain:
            SIZE: int = 3

        assert _traced_paths(Plain()) == ()

    def test_walks_the_runtime_value_not_the_annotation(self):
        """A field annotated with a base type is walked through the object it holds."""

        @dataclasses.dataclass(frozen=True)
        class Holder:
            inner: object = Opt()

        assert _traced_paths(Holder()) == ("inner.LR",)

    def test_traced_field_is_not_entered(self):
        """A traced value is a number the run reads, never a nested config."""

        @dataclasses.dataclass(frozen=True)
        class Holder:
            inner: object = traced(Opt())

        assert _traced_paths(Holder()) == ("inner",)
