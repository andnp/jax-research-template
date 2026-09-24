"""Contract tests for declared analysis plans and their registration beside specs."""

from __future__ import annotations

import math
from types import ModuleType

import pytest
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import ExperimentDimensions
from research_analysis.plan import AnalysisPlan, BakeoffAnalysis, BestHypersAnalysis, against_baseline, registered_plans

_SPEC_MODULE_SOURCE = """
from research_analysis.plan import AnalysisPlan, BestHypersAnalysis, analysis_plan
from research_analysis.dataset.types import ExperimentDimensions


def sweep():
    return None


@analysis_plan(sweep)
def sweep_plan():
    return AnalysisPlan("return", (BestHypersAnalysis("champion"),), ExperimentDimensions())
"""


def _spec_module(source: str) -> ModuleType:
    module = ModuleType("spec_module")
    exec(source, vars(module))
    return module


class TestAnalysisPlan:
    def test_against_baseline_plans_each_arm_minus_the_baseline(self) -> None:
        """The baseline form compares every other arm to the baseline and nothing else."""
        assert against_baseline(["a", "base", "b"], "base") == (("a", "base"), ("b", "base"))

    def test_rejects_a_plan_with_no_analyses(self) -> None:
        """A plan must commit to at least one analysis."""
        with pytest.raises(ValueError, match="at least one analysis"):
            AnalysisPlan("return", (), ExperimentDimensions())

    def test_rejects_repeated_analysis_names(self) -> None:
        """Names identify results in the report, so two analyses cannot share one."""
        analyses = (BestHypersAnalysis("x"), BakeoffAnalysis("x", ("a", "b")))
        with pytest.raises(ValueError, match="unique"):
            AnalysisPlan("return", analyses, ExperimentDimensions())

    @pytest.mark.parametrize("worst_case", [math.nan, math.inf])
    def test_rejects_a_non_finite_worst_case(self, worst_case: float) -> None:
        """A declared divergence score must itself be a usable number."""
        with pytest.raises(ValueError, match="finite"):
            AnalysisPlan("return", (BestHypersAnalysis("x"),), ExperimentDimensions(), non_finite_worst_case=worst_case)

    def test_expanded_intervals_apply_only_to_untuned_bakeoffs(self) -> None:
        """Tuned analyses use the hierarchical percentile bootstrap, so declaring another interval for them fails."""
        expanded = BakeoffSpec(interval="expanded_percentile")
        untuned = AnalysisPlan("return", (BakeoffAnalysis("headline", ("a", "b")),), ExperimentDimensions(), stats=expanded)
        tuned = (BakeoffAnalysis("headline", ("a", "b"), select_champion_over=("lr",)), BestHypersAnalysis("champion"))

        assert untuned.stats.interval == "expanded_percentile"
        with pytest.raises(ValueError, match=r"\['headline', 'champion'\] use the hierarchical percentile bootstrap"):
            AnalysisPlan("return", tuned, ExperimentDimensions(), stats=expanded)


class TestPlanRegistry:
    def test_finds_the_plan_declared_for_a_spec(self) -> None:
        """A plan is registered under its spec factory's name in the spec's own module."""
        plans = registered_plans(_spec_module(_SPEC_MODULE_SOURCE))

        assert list(plans) == ["sweep"]
        assert plans["sweep"]().metric == "return"

    def test_a_module_without_plans_has_none(self) -> None:
        """Modules that declare no plan report an empty registry rather than failing."""
        assert registered_plans(ModuleType("empty")) == {}

    def test_rejects_a_second_plan_for_one_spec(self) -> None:
        """One spec has one primary analysis; a second declaration is an error, not an override."""
        source = _SPEC_MODULE_SOURCE + """

@analysis_plan(sweep)
def other_plan():
    return AnalysisPlan("loss", (BestHypersAnalysis("champion"),), ExperimentDimensions())
"""
        with pytest.raises(ValueError, match="already has a declared analysis plan"):
            _spec_module(source)
