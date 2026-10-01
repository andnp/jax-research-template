import dataclasses
from typing import Literal

import pytest
from research_runner.config import _check_value, _merge_traced, _set_path, _split_traced, _traced_paths, traced


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


class TestNumericAndBoolChecks:
    @pytest.mark.parametrize(
        ("annotation", "value", "expected"),
        [(int, 3, 3), (int, 3.0, 3), (float, 0.5, 0.5), (float, 2, 2.0), (bool, True, True)],
    )
    def test_accepts_values_that_fit_the_field(self, annotation: type, value: object, expected: object):
        """Integral floats fit int fields and ints fit float fields."""
        checked = _check_value(annotation, "agent.X", value)

        assert checked == expected
        assert type(checked) is annotation

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [(int, 2.5), (int, True), (int, "3"), (float, True), (float, "0.5"), (bool, 1), (bool, "yes")],
    )
    def test_rejects_values_that_do_not_fit_the_field(self, annotation: type, value: object):
        """A bool is not a number, and a non-integral float is not an int."""
        with pytest.raises(ValueError, match=rf"agent.X expects {annotation.__name__}, got {value!r}"):
            _check_value(annotation, "agent.X", value)


class TestStrAndLiteralChecks:
    def test_accepts_str_and_matching_literal(self):
        """A string fits a str field, and a listed option fits a Literal."""
        assert _check_value(str, "env.NAME", "a") == "a"
        assert _check_value(Literal["mlp", "cnn"], "agent.NET", "cnn") == "cnn"

    def test_rejects_a_non_str_for_a_str_field(self):
        """A number is not a string."""
        with pytest.raises(ValueError, match="env.NAME expects str, got 3"):
            _check_value(str, "env.NAME", 3)

    def test_rejects_an_unlisted_literal_naming_the_options(self):
        """The error lists the valid options."""
        with pytest.raises(ValueError, match=r"agent.NET expects one of \('mlp', 'cnn'\), got 'rnn'"):
            _check_value(Literal["mlp", "cnn"], "agent.NET", "rnn")

    def test_a_bool_does_not_match_a_numeric_literal(self):
        """True equals 1 in Python but is not the option 1."""
        with pytest.raises(ValueError):
            _check_value(Literal[1, 2], "agent.K", True)


class TestUnionChecks:
    def test_accepts_none_for_an_optional_field(self):
        """An explicit None is a valid value of X | None."""
        assert _check_value(float | None, "agent.TARGET", None) is None

    def test_converts_through_the_matching_member(self):
        """An int satisfies float | None by converting to float."""
        assert _check_value(float | None, "agent.TARGET", 2) == 2.0

    def test_rejects_a_value_no_member_accepts(self):
        """The error names the whole annotation."""
        with pytest.raises(ValueError, match=r"agent.TARGET expects float \| None, got 'x'"):
            _check_value(float | None, "agent.TARGET", "x")

    def test_rejects_none_for_a_required_field(self):
        """None is only valid where the annotation allows it."""
        with pytest.raises(ValueError, match="agent.LR expects float, got None"):
            _check_value(float, "agent.LR", None)

    def test_passes_through_annotations_it_does_not_know(self):
        """Unrecognised annotations are the field's own business."""
        assert _check_value(tuple[int, ...], "agent.SIZES", (1, 2)) == (1, 2)
