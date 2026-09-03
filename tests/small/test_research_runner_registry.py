"""Tests for the experiment spec registration decorator."""

from __future__ import annotations

import types

from research_runner.registry import SPECS_ATTRIBUTE, registered_specs

_MODULE_SOURCE = """
from research_runner.registry import spec

@spec
def make_alpha():
    return "alpha-spec"

@spec
def make_beta():
    return "beta-spec"
"""


def _build_module(source: str) -> types.ModuleType:
    module = types.ModuleType("fake_spec_module")
    exec(source, module.__dict__)
    return module


class TestSpecDecorator:
    def test_decorated_factory_is_returned_unchanged_and_callable(self) -> None:
        module = _build_module(_MODULE_SOURCE)

        assert module.make_alpha() == "alpha-spec"

    def test_registers_both_factories_from_the_same_module_by_name(self) -> None:
        module = _build_module(_MODULE_SOURCE)

        specs = registered_specs(module)

        assert set(specs) == {"make_alpha", "make_beta"}
        assert specs["make_alpha"] is module.make_alpha
        assert specs["make_beta"] is module.make_beta

    def test_module_with_no_registrations_returns_empty_dict(self) -> None:
        module = _build_module("x = 1\n")

        assert registered_specs(module) == {}

    def test_registry_is_stored_on_the_defining_module(self) -> None:
        module = _build_module(_MODULE_SOURCE)

        assert set(getattr(module, SPECS_ATTRIBUTE)) == {"make_alpha", "make_beta"}
