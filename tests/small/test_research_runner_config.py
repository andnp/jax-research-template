import dataclasses

import pytest
from research_runner.config import _merge_traced, _set_path, _split_traced, _traced_paths, traced


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


class TestSplitAndMerge:
    def test_split_resets_traced_fields_and_returns_their_values(self):
        """The static config holds defaults; the values hold what the config had."""
        static, values = _split_traced(Cfg(agent=Opt(LR=0.5, STEPS=7), gamma=0.9))

        assert static == Cfg(agent=Opt(LR=1e-3, STEPS=7))
        assert values == {"agent.LR": 0.5, "env.NOISE": 0.0, "gamma": 0.9}

    def test_configs_differing_only_in_traced_values_share_a_static(self):
        """This is what lets such runs be batched together."""
        a, _ = _split_traced(Cfg(gamma=0.9))
        b, _ = _split_traced(Cfg(gamma=0.5, agent=Opt(LR=2.0)))

        assert a == b

    def test_configs_differing_in_a_static_field_do_not_share_a_static(self):
        """A static field decides which batch a run belongs to."""
        a, _ = _split_traced(Cfg(env=Env(NAME="a")))
        b, _ = _split_traced(Cfg(env=Env(NAME="b")))

        assert a != b

    def test_merge_restores_the_original_config(self):
        """Splitting then merging is the identity."""
        config = Cfg(agent=Opt(LR=0.5, STEPS=7), env=Env(NOISE=0.1), gamma=0.9)

        assert _merge_traced(*_split_traced(config)) == config

    def test_merge_accepts_opaque_values(self):
        """Tracers are never inspected, so any object may be merged in."""
        token = object()

        assert _merge_traced(Cfg(), {"agent.LR": token}).agent.LR is token


class TestUnknownPaths:
    def test_unknown_field_names_the_type_and_valid_fields(self):
        """The error says which config lacked the field and what it has."""
        with pytest.raises(ValueError, match=r"Opt has no field LEARNING_RTE; fields are LR, STEPS"):
            _set_path(Cfg(), "agent.LEARNING_RTE", 1.0)

    def test_path_through_a_non_config_value_is_rejected(self):
        """A dotted path cannot descend into a plain number."""
        with pytest.raises(ValueError, match=r"float has no field x; fields are none"):
            _set_path(Cfg(), "gamma.x", 1.0)
