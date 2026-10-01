import dataclasses
from typing import Literal

import pytest
from research_runner.config import _apply_choices, _build_config, _check_value, _merge_traced, _set_path, _split_traced, _static_names, _traced_paths, traced


@dataclasses.dataclass(frozen=True)
class Opt:
    lr: float = traced(1e-3)
    steps: int = 10


@dataclasses.dataclass(frozen=True)
class Env:
    name: str = "a"
    noise: float = traced(0.0)


@dataclasses.dataclass(frozen=True)
class Cfg:
    agent: Opt = Opt()
    env: Env = Env()
    gamma: float = traced(0.99)


class TestTraced:
    def test_field_keeps_its_declared_default(self):
        """A traced field behaves as an ordinary defaulted dataclass field."""
        assert Opt() == Opt(lr=1e-3, steps=10)

    def test_field_can_be_overridden_at_construction(self):
        """Marking a field traced does not change how it is set."""
        assert Opt(lr=0.5).lr == 0.5


class TestTracedPaths:
    def test_lists_nested_paths_in_declaration_order(self):
        """Paths are dotted from the root, depth-first, in field order."""
        assert _traced_paths(Cfg()) == ("agent.lr", "env.noise", "gamma")

    def test_config_without_traced_fields_lists_none(self):
        """A config with nothing marked traced yields no paths."""

        @dataclasses.dataclass(frozen=True)
        class Plain:
            size: int = 3

        assert _traced_paths(Plain()) == ()

    def test_walks_the_runtime_value_not_the_annotation(self):
        """A field annotated with a base type is walked through the object it holds."""

        @dataclasses.dataclass(frozen=True)
        class Holder:
            inner: object = Opt()

        assert _traced_paths(Holder()) == ("inner.lr",)

    def test_traced_field_is_not_entered(self):
        """A traced value is a number the run reads, never a nested config."""

        @dataclasses.dataclass(frozen=True)
        class Holder:
            inner: object = traced(Opt())

        assert _traced_paths(Holder()) == ("inner",)


class TestSplitAndMerge:
    def test_split_resets_traced_fields_and_returns_their_values(self):
        """The static config holds defaults; the values hold what the config had."""
        static, values = _split_traced(Cfg(agent=Opt(lr=0.5, steps=7), gamma=0.9))

        assert static == Cfg(agent=Opt(lr=1e-3, steps=7))
        assert values == {"agent.lr": 0.5, "env.noise": 0.0, "gamma": 0.9}

    def test_configs_differing_only_in_traced_values_share_a_static(self):
        """This is what lets such runs be batched together."""
        a, _ = _split_traced(Cfg(gamma=0.9))
        b, _ = _split_traced(Cfg(gamma=0.5, agent=Opt(lr=2.0)))

        assert a == b

    def test_configs_differing_in_a_static_field_do_not_share_a_static(self):
        """A static field decides which batch a run belongs to."""
        a, _ = _split_traced(Cfg(env=Env(name="a")))
        b, _ = _split_traced(Cfg(env=Env(name="b")))

        assert a != b

    def test_merge_restores_the_original_config(self):
        """Splitting then merging is the identity."""
        config = Cfg(agent=Opt(lr=0.5, steps=7), env=Env(noise=0.1), gamma=0.9)

        assert _merge_traced(*_split_traced(config)) == config

    def test_merge_accepts_opaque_values(self):
        """Tracers are never inspected, so any object may be merged in."""
        token = object()

        assert _merge_traced(Cfg(), {"agent.lr": token}).agent.lr is token


class TestUnknownPaths:
    def test_unknown_field_names_the_type_and_valid_fields(self):
        """The error says which config lacked the field and what it has."""
        with pytest.raises(ValueError, match=r"Opt has no field learning_rte; fields are lr, steps"):
            _set_path(Cfg(), "agent.learning_rte", 1.0)

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
        checked = _check_value(annotation, "agent.x", value)

        assert checked == expected
        assert type(checked) is annotation

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [(int, 2.5), (int, True), (int, "3"), (float, True), (float, "0.5"), (bool, 1), (bool, "yes")],
    )
    def test_rejects_values_that_do_not_fit_the_field(self, annotation: type, value: object):
        """A bool is not a number, and a non-integral float is not an int."""
        with pytest.raises(ValueError, match=rf"agent.x expects {annotation.__name__}, got {value!r}"):
            _check_value(annotation, "agent.x", value)


class TestStrAndLiteralChecks:
    def test_accepts_str_and_matching_literal(self):
        """A string fits a str field, and a listed option fits a Literal."""
        assert _check_value(str, "env.name", "a") == "a"
        assert _check_value(Literal["mlp", "cnn"], "agent.net", "cnn") == "cnn"

    def test_rejects_a_non_str_for_a_str_field(self):
        """A number is not a string."""
        with pytest.raises(ValueError, match="env.name expects str, got 3"):
            _check_value(str, "env.name", 3)

    def test_rejects_an_unlisted_literal_naming_the_options(self):
        """The error lists the valid options."""
        with pytest.raises(ValueError, match=r"agent.net expects one of \('mlp', 'cnn'\), got 'rnn'"):
            _check_value(Literal["mlp", "cnn"], "agent.net", "rnn")

    def test_a_bool_does_not_match_a_numeric_literal(self):
        """True equals 1 in Python but is not the option 1."""
        with pytest.raises(ValueError):
            _check_value(Literal[1, 2], "agent.k", True)


class TestUnionChecks:
    def test_accepts_none_for_an_optional_field(self):
        """An explicit None is a valid value of x | None."""
        assert _check_value(float | None, "agent.target", None) is None

    def test_converts_through_the_matching_member(self):
        """An int satisfies float | None by converting to float."""
        assert _check_value(float | None, "agent.target", 2) == 2.0

    def test_rejects_a_value_no_member_accepts(self):
        """The error names the whole annotation."""
        with pytest.raises(ValueError, match=r"agent.target expects float \| None, got 'x'"):
            _check_value(float | None, "agent.target", "x")

    def test_rejects_none_for_a_required_field(self):
        """None is only valid where the annotation allows it."""
        with pytest.raises(ValueError, match="agent.lr expects float, got None"):
            _check_value(float, "agent.lr", None)

    def test_passes_through_annotations_it_does_not_know(self):
        """Unrecognised annotations are the field's own business."""
        assert _check_value(tuple[int, ...], "agent.sizes", (1, 2)) == (1, 2)


class TestApplyChoices:
    CHOICES = {"env": {"a": Env(name="a"), "b": Env(name="b", noise=0.2)}}

    def test_selected_choice_replaces_the_sub_config(self):
        """The named object is placed at the choice's path."""
        config = _apply_choices(Cfg(), self.CHOICES, {"env": "b"})

        assert config.env == Env(name="b", noise=0.2)

    def test_unselected_choice_keeps_the_base_object(self):
        """A run that does not name the choice keeps the base config's value."""
        assert _apply_choices(Cfg(), self.CHOICES, {"gamma": 0.5}) == Cfg()

    def test_unknown_name_lists_the_choices(self):
        """The error names the bad value, the path and the valid names."""
        with pytest.raises(ValueError, match=r"'c' is not a choice for env; choices are a, b"):
            _apply_choices(Cfg(), self.CHOICES, {"env": "c"})

    def test_non_string_name_is_not_a_choice(self):
        """Choice names are strings, so a number never matches."""
        with pytest.raises(ValueError, match="1 is not a choice for env"):
            _apply_choices(Cfg(), self.CHOICES, {"env": 1})


class TestBuildConfig:
    CHOICES = {"env": {"a": Env(name="a"), "b": Env(name="b")}}

    def test_sets_dotted_paths_on_the_base_config(self):
        """Each parameter replaces the field its path names."""
        config = _build_config(Cfg(), self.CHOICES, {"agent.lr": 0.5, "gamma": 0.9})

        assert config == Cfg(agent=Opt(lr=0.5), gamma=0.9)

    def test_choices_apply_before_paths(self):
        """A path sets its field on whichever object the choice picked."""
        config = _build_config(Cfg(), self.CHOICES, {"env.noise": 0.3, "env": "b"})

        assert config.env == Env(name="b", noise=0.3)

    def test_seed_is_not_a_config_field(self):
        """seed selects the run's key and never touches the config."""
        assert _build_config(Cfg(), self.CHOICES, {"seed": 3}) == Cfg()

    def test_converts_values_to_the_field_type(self):
        """An int given for a float field is stored as a float."""
        assert type(_build_config(Cfg(), {}, {"gamma": 1}).gamma) is float

    def test_unknown_field_is_reported(self):
        """A mistyped path names the config and its fields."""
        with pytest.raises(ValueError, match="Opt has no field learning_rte; fields are lr, steps"):
            _build_config(Cfg(), {}, {"agent.learning_rte": 1.0})

    def test_wrongly_typed_value_is_reported(self):
        """A value of the wrong type names the path and the expected type."""
        with pytest.raises(ValueError, match=r"agent.steps expects int, got 2.5"):
            _build_config(Cfg(), {}, {"agent.steps": 2.5})

    def test_unknown_choice_is_reported(self):
        """A choice name that is not offered lists the offered ones."""
        with pytest.raises(ValueError, match="'c' is not a choice for env; choices are a, b"):
            _build_config(Cfg(), self.CHOICES, {"env": "c"})


class TestStaticNames:
    def test_traced_paths_and_seed_are_not_static(self):
        """Parameters setting traced fields vary within a batch, as does the seed."""
        names = ["agent.lr", "agent.steps", "gamma", "seed"]

        assert _static_names(Cfg(), names) == {"agent.steps"}

    def test_a_choice_is_always_static(self):
        """A choice swaps a whole object, so it splits batches."""
        assert _static_names(Cfg(), ["env", "agent.lr"]) == {"env"}
