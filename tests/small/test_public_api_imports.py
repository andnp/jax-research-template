"""Small tests guarding the documented package-root import paths."""

from __future__ import annotations

import subprocess
import sys
import types as stdlib_types


class TestExperimentDefinitionPackageBoundary:
    def test_documented_names_live_in_defining_modules(self) -> None:
        import experiment_definition
        from experiment_definition.component import Component
        from experiment_definition.experiment import Experiment

        assert Component.__module__ == "experiment_definition.component"
        assert Experiment.__module__ == "experiment_definition.experiment"
        assert all(
            name.startswith("__") or isinstance(value, stdlib_types.ModuleType)
            for name, value in vars(experiment_definition).items()
        )


class TestResearchRunnerPackageBoundary:
    def test_documented_names_live_in_defining_modules(self) -> None:
        import research_runner
        from research_runner.registry import registered_specs, spec
        from research_runner.runner import run_experiment
        from research_runner.types import ExperimentSpec

        assert run_experiment.__module__ == "research_runner.runner"
        assert ExperimentSpec.__module__ == "research_runner.types"
        assert callable(registered_specs)
        assert callable(spec)
        assert all(
            name.startswith("__") or isinstance(value, stdlib_types.ModuleType)
            for name, value in vars(research_runner).items()
        )


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
