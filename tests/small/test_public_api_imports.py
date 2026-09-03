"""Small tests guarding the documented package-root import paths."""

from __future__ import annotations

import subprocess
import sys


class TestExperimentDefinitionPublicApi:
    def test_documented_names_resolve_to_canonical_symbols(self) -> None:
        import experiment_definition
        from experiment_definition import Component, Experiment
        from experiment_definition.component import Component as CanonicalComponent
        from experiment_definition.experiment import Experiment as CanonicalExperiment

        assert Experiment is CanonicalExperiment
        assert Component is CanonicalComponent
        assert set(experiment_definition.__all__) == {
            "Component",
            "ComponentType",
            "Experiment",
            "MetricFrequency",
            "MetricType",
            "ParameterValue",
            "metric_whitelist",
        }


class TestResearchRunnerPublicApi:
    def test_documented_names_resolve_to_canonical_symbols(self) -> None:
        import research_runner
        from research_runner import ExecutionContext, ExperimentSpec, run_experiment
        from research_runner.runner import run_experiment as canonical_run_experiment
        from research_runner.types import ExecutionContext as CanonicalExecutionContext
        from research_runner.types import ExperimentSpec as CanonicalExperimentSpec

        assert run_experiment is canonical_run_experiment
        assert ExecutionContext is CanonicalExecutionContext
        assert ExperimentSpec is CanonicalExperimentSpec
        assert set(research_runner.__all__) == {
            "ExecutionContext",
            "ExecutionResult",
            "ExperimentSpec",
            "RunPoint",
            "execute_batch",
            "run_experiment",
        }


class TestPackageInitInFreshInterpreter:
    def test_both_packages_import_without_a_warm_module_cache(self) -> None:
        """A fresh interpreter is the only place a package-init cycle shows up."""
        completed = subprocess.run(
            [sys.executable, "-c", "import experiment_definition, research_runner"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
