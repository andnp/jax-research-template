"""Contract tests for dataset-backed statistical reporting."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
import pytest
from experiment_definition.db import DatabaseManager
from experiment_definition.parameter import ParameterValue
from research_analysis.aggregates import ReferenceRange
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDataset, ExperimentDimensions, SweepAxis, SweepValueType
from research_analysis.diagnostics.bakeoff import BakeoffWarningCode
from research_analysis.non_finite import NonFiniteValuesError
from research_analysis.performance import ReturnRate, TailMean
from research_analysis.reporting import (
    BenchmarkBakeoffReport,
    HyperparameterConfigReport,
    HyperparameterDifferenceReport,
    HyperparameterSliceReport,
    bakeoff_report,
    best_hypers,
    boundary_warning,
    insufficient_seeds_warning,
    non_normality_warning,
    slice_hypers,
    variance_warning,
)
from research_analysis.statistics import pointwise_tolerance_interval


class ExperimentSetup(TypedDict):
    db_path: Path
    metrics_path: Path
    experiment_id: int
    algo_version_id: int
    env_version_id: int
    root: Path


def _ensure_metrics_table(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )


def _add_observation(
    setup: ExperimentSetup,
    metadata: dict[str, ParameterValue],
    seed: int,
    value: float,
    *,
    ablation: str | None = None,
) -> None:
    with DatabaseManager(setup["db_path"]) as database:
        hyper_id = database.add_hyperparam_config({**metadata, "seed": seed})
        run_id = database.add_run(
            setup["experiment_id"],
            setup["algo_version_id"],
            setup["env_version_id"],
            hyper_id,
            seed,
            ablation,
        )
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id,
            str(setup["root"] / str(execution_id)),
            metadata={"metrics_db_path": str(setup["metrics_path"])},
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(setup["metrics_path"]) as connection:
        connection.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
            (run_id, execution_id, "metric", 0, value),
        )
        connection.commit()


def _add_curve_observation(
    setup: ExperimentSetup,
    metadata: dict[str, ParameterValue],
    seed: int,
    values: list[float],
) -> None:
    """Like _add_observation, but records one value per step instead of a single step-0 value."""
    with DatabaseManager(setup["db_path"]) as database:
        hyper_id = database.add_hyperparam_config({**metadata, "seed": seed})
        run_id = database.add_run(
            setup["experiment_id"], setup["algo_version_id"], setup["env_version_id"], hyper_id, seed, None
        )
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id,
            str(setup["root"] / str(execution_id)),
            metadata={"metrics_db_path": str(setup["metrics_path"])},
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(setup["metrics_path"]) as connection:
        connection.executemany(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
            [(run_id, execution_id, "metric", step, value) for step, value in enumerate(values)],
        )
        connection.commit()


def _load(
    setup: ExperimentSetup,
    dimensions: ExperimentDimensions | None = None,
) -> ExperimentDataset:
    return load_experiment_dataset(setup["db_path"], "test-exp", dimensions=dimensions)


@pytest.fixture
def temp_experiment_setup() -> Generator[ExperimentSetup, None, None]:
    """Create a hermetic experiment database and a shared recorded metrics database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "experiments.sqlite"
        metrics_path = root / "metrics.sqlite"
        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_id = database.add_experiment("test-exp")
            algo_component = database.add_component("ppo", "ALGO")
            algo_version_id = database.add_component_version(algo_component, "algo-hash")
            env_component = database.add_component("cartpole", "ENV")
            env_version_id = database.add_component_version(env_component, "env-hash")
        _ensure_metrics_table(metrics_path)
        yield {
            "db_path": db_path,
            "metrics_path": metrics_path,
            "experiment_id": experiment_id,
            "algo_version_id": algo_version_id,
            "env_version_id": env_version_id,
            "root": root,
        }


def test_bakeoff_report_consumes_selected_dataset(temp_experiment_setup: ExperimentSetup) -> None:
    """Compare paired conditions from materialized metrics without database arguments."""
    setup = temp_experiment_setup
    for condition in ("baseline", "candidate"):
        for seed in range(4):
            _add_observation(setup, {"condition_name": condition}, seed, 100.0 + seed + (50.0 if condition == "candidate" else 0.0))

    report = bakeoff_report(_load(setup), "metric", ["baseline", "candidate"], verbose=False)

    diff = report.comparison.difference_estimate["baseline"]["candidate"]
    assert diff is not None
    assert diff == pytest.approx(-50.0)


def test_bakeoff_report_honors_missingness_policy(temp_experiment_setup: ExperimentSetup) -> None:
    """Reject an incomplete block under missingness='error' unless complete_case is explicit."""
    setup = temp_experiment_setup
    for condition, seeds in (("a", (0, 1, 2, 4)), ("b", (0, 1, 2, 3))):
        for seed in seeds:
            _add_observation(setup, {"condition_name": condition}, seed, float(seed))
    dataset = _load(setup)

    with pytest.raises(ValueError, match="incomplete block"):
        bakeoff_report(dataset, "metric", ["a", "b"], verbose=False)

    # complete_case no longer raises -- it pairs down to the shared seeds {0, 1, 2}
    # for the comparison itself, though cells still report each condition's
    # own raw seed count (4), not the paired subset used by the statistical test.
    report = bakeoff_report(
        dataset,
        "metric",
        ["a", "b"],
        verbose=False,
        spec=BakeoffSpec(missingness="complete_case", n_resamples=64),
    )
    assert report.cells["a"][""].n == 4
    assert report.cells["b"][""].n == 4


def test_best_hypers_consumes_declared_sweep_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """Find the best declared sweep value using final values from the dataset metrics table."""
    setup = temp_experiment_setup
    for learning_rate, score in (("low", 1.0), ("high", 3.0)):
        for seed in range(3):
            _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, score + seed)
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    report = best_hypers(dataset, "metric", verbose=False)

    assert isinstance(report, HyperparameterConfigReport)
    assert report.winning_config == {"learning_rate": "high"}
    assert report.raw_mean == pytest.approx(4.0)


def test_best_hypers_rejects_undeclared_group_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """Require grouping keys to be declared typed sweep axes in the dataset."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "learning_rate": "low", "task": "cartpole"},
            seed,
            float(seed),
        )
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    with pytest.raises(ValueError, match="declared typed sweep axes"):
        best_hypers(dataset, "metric", group_by=["task"], verbose=False)


def test_best_hypers_optimizes_jointly_over_multiple_axes(temp_experiment_setup: ExperimentSetup) -> None:
    """A grid champion is the best joint combination, not a per-axis marginal pick."""
    setup = temp_experiment_setup
    combos = {("low", 1): 4.0, ("low", 2): 2.0, ("high", 1): 3.0, ("high", 2): 5.0}
    for (learning_rate, eta), value in combos.items():
        for seed in range(3):
            _add_observation(
                setup,
                {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta},
                seed,
                value,
            )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    report = best_hypers(dataset, "metric", verbose=False)

    assert report.winning_config == {"learning_rate": "high", "eta": 2}
    assert report.raw_mean == pytest.approx(5.0)


def test_best_hypers_higher_is_better_false_finds_the_minimizer(temp_experiment_setup: ExperimentSetup) -> None:
    """For an error-like metric, the champion is the lowest raw mean, reported in its own units."""
    setup = temp_experiment_setup
    for learning_rate, score in (("low", 1.0), ("high", 3.0)):
        for seed in range(3):
            _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, score + seed)
    dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    report = best_hypers(dataset, "metric", higher_is_better=False, verbose=False)

    assert report.winning_config == {"learning_rate": "low"}
    assert report.raw_mean == pytest.approx(2.0)
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.corrected_mean <= ci_high


def test_slice_hypers_pins_champion_config_while_varying_one_axis(
    temp_experiment_setup: ExperimentSetup,
) -> None:
    """The best-achievable curve re-optimizes lr per eta; the slice curve pins lr at the champion's value."""
    setup = temp_experiment_setup
    combos = {("low", 1): 4.0, ("low", 2): 2.0, ("high", 1): 3.0, ("high", 2): 5.0}
    for (learning_rate, eta), value in combos.items():
        for seed in range(3):
            _add_observation(
                setup,
                {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta},
                seed,
                value,
            )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    report = slice_hypers(dataset, "metric", vary="eta", verbose=False)

    assert isinstance(report, HyperparameterSliceReport)
    assert report.values == [1, 2]
    assert report.best_performance == pytest.approx([4.0, 5.0])
    assert report.slice_performance == pytest.approx([3.0, 5.0])


def test_slice_hypers_plots_selection_corrected_best_values(temp_experiment_setup: ExperimentSetup) -> None:
    """The best-achievable curve carries each value's corrected champion, not the raw maximum over noise."""
    setup = temp_experiment_setup
    noise = np.random.default_rng(0)
    for learning_rate in ("low", "high"):
        for eta in (1, 2):
            for seed in range(5):
                metadata: dict[str, ParameterValue] = {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta}
                _add_observation(setup, metadata, seed, float(noise.normal()))
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING), SweepAxis("eta", SweepValueType.INTEGER))
        ),
    )

    report = slice_hypers(dataset, "metric", vary="eta", verbose=False, n_resamples=256)
    profiles = best_hypers(dataset, "metric", group_by=["eta"], verbose=False, n_resamples=256)

    assert report.best_performance == pytest.approx([profiles[(1,)].corrected_mean, profiles[(2,)].corrected_mean])
    assert report.best_performance != pytest.approx([profiles[(1,)].raw_mean, profiles[(2,)].raw_mean])


def test_slice_hypers_leaves_an_unrun_pinned_value_empty(temp_experiment_setup: ExperimentSetup) -> None:
    """A pinned configuration never run at some value shows no slice value rather than a borrowed best one."""
    setup = temp_experiment_setup
    combos = {("low", 1): 10.0, ("mid", 1): 1.0, ("high", 1): 1.0, ("mid", 2): 2.0, ("high", 2): 3.0}
    for (learning_rate, eta), value in combos.items():
        for seed in range(3):
            _add_observation(
                setup, {"condition_name": "baseline", "learning_rate": learning_rate, "eta": eta}, seed, value
            )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING), SweepAxis("eta", SweepValueType.INTEGER))
        ),
    )

    report = slice_hypers(dataset, "metric", vary="eta", verbose=False, n_resamples=64)

    assert report.values == [1, 2]
    assert report.slice_performance == [pytest.approx(10.0), None]


def test_slice_hypers_rejects_varying_a_grouped_axis(temp_experiment_setup: ExperimentSetup) -> None:
    """vary and group_by must name disjoint axes."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "learning_rate": "low", "eta": 1},
            seed,
            float(seed),
        )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    with pytest.raises(ValueError, match="cannot be both varied and grouped by"):
        slice_hypers(dataset, "metric", vary="eta", group_by=["eta"], verbose=False)


class TestBoundaryWarning:
    """Champion hyperparameters at the edge of their swept grid, still trending toward it."""

    LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

    def test_flags_a_champion_still_climbing_at_the_upper_edge(self) -> None:
        """Monotonically improving all the way to the top of the grid."""
        report = self._slice(self.LRS, [-151.8, -143.3, -141.3, -140.3, -138.9])
        warning = boundary_warning(report, condition="sac")
        assert warning is not None
        assert warning.code == BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY
        assert warning.condition == "sac"

    def test_does_not_flag_an_interior_peak(self) -> None:
        """Peaks in the middle of the grid and declines toward the upper edge."""
        report = self._slice(self.LRS, [-193.1, -144.9, -143.4, -144.8, -160.1])
        assert boundary_warning(report, condition="gi_sac") is None

    def test_does_not_flag_a_flat_champion(self) -> None:
        """A tie/plateau at the max is not a boundary concern -- argmax picks the first."""
        report = self._slice(self.LRS, [-6.1, -0.8, -0.8, -0.8, -0.8])
        assert boundary_warning(report) is None

    def test_flags_a_champion_still_climbing_at_the_lower_edge(self) -> None:
        """Mirrors the upper-edge case in the other direction."""
        report = self._slice(self.LRS, [-138.9, -140.3, -141.3, -143.3, -151.8])
        warning = boundary_warning(report)
        assert warning is not None
        assert "3e-05" in warning.message

    def test_respects_higher_is_better_false(self) -> None:
        """A minimizer whose champion is at the edge and still descending is flagged the same way."""
        report = self._slice(self.LRS, [151.8, 143.3, 141.3, 140.3, 138.9])
        warning = boundary_warning(report, higher_is_better=False)
        assert warning is not None

    def test_no_warning_for_a_single_swept_value(self) -> None:
        report = self._slice([3e-4], [-140.0])
        assert boundary_warning(report) is None

    @staticmethod
    def _slice(values: list[float], performance: list[float]) -> HyperparameterSliceReport:
        return HyperparameterSliceReport(
            vary="learning_rate", values=values, best_performance=performance, slice_performance=performance, plot_path=None
        )


class TestNonNormalityWarning:
    """Informational Shapiro-Wilk check; never gates which test runs (ADR-010)."""

    def test_flags_a_heavily_skewed_sample(self) -> None:
        values = [0.0] * 29 + [1000.0]
        warning = non_normality_warning(values, condition="sac_rc")
        assert warning is not None
        assert warning.code == BakeoffWarningCode.NON_NORMAL_DISTRIBUTION
        assert warning.condition == "sac_rc"

    def test_does_not_flag_a_normal_sample(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=30)
        assert non_normality_warning(values) is None

    def test_no_warning_below_the_minimum_sample_size(self) -> None:
        assert non_normality_warning([1.0, 2.0]) is None


def _config_report(*, normalized_selection_entropy: float = 0.0) -> HyperparameterConfigReport:
    return HyperparameterConfigReport(
        winning_config={"learning_rate": 3e-4},
        raw_mean=1.0,
        corrected_mean=1.0,
        maximization_bias=0.0,
        estimate_ci=(0.5, 1.5),
        normalized_selection_entropy=normalized_selection_entropy,
    )


class TestInsufficientSeedsWarning:
    def test_flags_too_few_raw_seeds(self) -> None:
        warning = insufficient_seeds_warning(_config_report(), 3, condition="gi_sac", min_seeds=5)
        assert warning is not None
        assert warning.code == BakeoffWarningCode.INSUFFICIENT_SEEDS
        assert warning.condition == "gi_sac"

    def test_flags_unstable_selection_despite_enough_seeds(self) -> None:
        warning = insufficient_seeds_warning(_config_report(normalized_selection_entropy=0.9), 10, min_seeds=5)
        assert warning is not None
        assert warning.code == BakeoffWarningCode.INSUFFICIENT_SEEDS

    def test_no_warning_with_enough_stable_seeds(self) -> None:
        warning = insufficient_seeds_warning(_config_report(normalized_selection_entropy=0.1), 10, min_seeds=5)
        assert warning is None


class TestVarianceWarning:
    def test_flags_zero_variance(self) -> None:
        warning = variance_warning([1.0, 1.0, 1.0], condition="sac")
        assert warning is not None
        assert warning.code == BakeoffWarningCode.ZERO_VARIANCE

    def test_flags_extreme_variance_relative_to_the_metric_range(self) -> None:
        warning = variance_warning([-500.0, 500.0, -500.0, 500.0], reference_range=100.0)
        assert warning is not None
        assert warning.code == BakeoffWarningCode.EXTREME_VARIANCE

    def test_no_warning_for_ordinary_variance(self) -> None:
        warning = variance_warning([98.0, 100.0, 102.0], reference_range=1000.0)
        assert warning is None

    def test_no_extreme_check_without_a_reference_range(self) -> None:
        warning = variance_warning([-500.0, 500.0, -500.0, 500.0])
        assert warning is None


def _add_arm_observation(setup: ExperimentSetup, algorithm: str, learning_rate: str, seed: int, value: float) -> None:
    _add_observation(setup, {"condition_name": algorithm, "algorithm": algorithm, "learning_rate": learning_rate}, seed, value)


def test_best_hypers_groups_by_the_condition_dimension(temp_experiment_setup: ExperimentSetup) -> None:
    """group_by can name the dataset's condition dimension, not just declared sweep axes."""
    setup = temp_experiment_setup
    for algorithm, base in (("shared", 1.0), ("adversarial", 3.0)):
        for learning_rate, offset in (("low", 0.0), ("high", 1.0)):
            for seed in range(3):
                _add_observation(setup, {"algorithm": algorithm, "learning_rate": learning_rate}, seed, base + offset + seed)
    dataset = _load(
        setup,
        ExperimentDimensions(condition="algorithm", sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)),
    )

    reports = best_hypers(dataset, "metric", group_by=["algorithm"], verbose=False)

    assert reports[("shared",)].winning_config == {"learning_rate": "high"}
    assert reports[("shared",)].raw_mean == pytest.approx(3.0)
    assert reports[("adversarial",)].winning_config == {"learning_rate": "high"}
    assert reports[("adversarial",)].raw_mean == pytest.approx(5.0)


def test_best_hypers_difference_pairs_by_seed_across_two_groups(temp_experiment_setup: ExperimentSetup) -> None:
    """The paired difference compares each arm's own champion, correlated by shared seed."""
    setup = temp_experiment_setup
    arm_a = {("low",): (1.0, 2.0, 3.0), ("high",): (4.0, 5.0, 6.0)}
    arm_b = {("low",): (1.0, 1.0, 1.0), ("high",): (2.0, 2.0, 2.0)}
    for (learning_rate,), values in arm_a.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "a", learning_rate, seed, value)
    for (learning_rate,), values in arm_b.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "b", learning_rate, seed, value)
    dataset = _load(
        setup,
        ExperimentDimensions(sweep_axes=(SweepAxis("algorithm", SweepValueType.STRING), SweepAxis("learning_rate", SweepValueType.STRING))),
    )

    report = best_hypers(dataset, "metric", group_by=["algorithm"], difference_between=("a", "b"), verbose=False)

    assert isinstance(report, HyperparameterDifferenceReport)
    assert report.minuend_config == {"learning_rate": "high"}
    assert report.subtrahend_config == {"learning_rate": "high"}
    assert report.mean_diff > 0
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.mean_diff <= ci_high


def test_best_hypers_difference_higher_is_better_false_finds_minimizers(
    temp_experiment_setup: ExperimentSetup,
) -> None:
    """For an error-like metric, each side's champion is its own minimizer, in original units."""
    setup = temp_experiment_setup
    arm_a = {("low",): (1.0, 2.0, 3.0), ("high",): (4.0, 5.0, 6.0)}
    arm_b = {("low",): (1.0, 1.0, 1.0), ("high",): (2.0, 2.0, 2.0)}
    for (learning_rate,), values in arm_a.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "a", learning_rate, seed, value)
    for (learning_rate,), values in arm_b.items():
        for seed, value in enumerate(values):
            _add_arm_observation(setup, "b", learning_rate, seed, value)
    dataset = _load(
        setup,
        ExperimentDimensions(sweep_axes=(SweepAxis("algorithm", SweepValueType.STRING), SweepAxis("learning_rate", SweepValueType.STRING))),
    )

    report = best_hypers(
        dataset, "metric", group_by=["algorithm"], difference_between=("a", "b"), higher_is_better=False, verbose=False
    )

    assert isinstance(report, HyperparameterDifferenceReport)
    assert report.minuend_config == {"learning_rate": "low"}
    assert report.subtrahend_config == {"learning_rate": "low"}
    assert report.mean_diff == pytest.approx(1.0, abs=0.5)
    ci_low, ci_high = report.estimate_ci
    assert ci_low <= report.mean_diff <= ci_high


def test_best_hypers_difference_requires_a_single_group_axis(temp_experiment_setup: ExperimentSetup) -> None:
    """difference_between only makes sense against exactly one group_by axis."""
    setup = temp_experiment_setup
    for seed in range(3):
        _add_observation(
            setup,
            {"condition_name": "baseline", "algorithm": "a", "learning_rate": "low", "eta": 1},
            seed,
            float(seed),
        )
    dataset = _load(
        setup,
        ExperimentDimensions(
            sweep_axes=(
                SweepAxis("algorithm", SweepValueType.STRING),
                SweepAxis("learning_rate", SweepValueType.STRING),
                SweepAxis("eta", SweepValueType.INTEGER),
            )
        ),
    )

    with pytest.raises(ValueError, match="difference_between requires group_by to name exactly one axis"):
        best_hypers(dataset, "metric", group_by=["algorithm", "learning_rate"], difference_between=("a", "b"), verbose=False)


def test_bakeoff_report_consumes_canonical_environment_dimension(temp_experiment_setup: ExperimentSetup) -> None:
    """Compare algorithms and environments using canonical dataset dimensions, block_by=[env_source]."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac", "td3"]
    environments = ["cartpole", "mountaincar"]
    for algorithm in algorithms:
        for environment in environments:
            for seed in range(3):
                _add_observation(setup, {"algorithm": algorithm, "env_source": environment}, seed, float(seed))
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(
        dataset,
        "metric",
        algorithms,
        block_by=["env_name"],
        verbose=False,
        references={environment: ReferenceRange(0.0, 10.0) for environment in environments},
    )

    assert isinstance(report.comparison, BenchmarkBakeoffReport)
    assert report.conditions == algorithms
    assert sorted(report.comparison.environments) == sorted(environments)


def test_bakeoff_report_retains_blockwise_missingness(temp_experiment_setup: ExperimentSetup) -> None:
    """Keep incomplete environment-seed blocks for block-stratified inference."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac", "td3"]
    for seed in range(3):
        for index, algorithm in enumerate(algorithms):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, float(seed + index))
    for algorithm, seed in (("ppo", 3), ("sac", 3)):
        _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, float(seed))
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(
        dataset,
        "metric",
        algorithms,
        block_by=["env_name"],
        verbose=False,
        references={"cartpole": ReferenceRange(0.0, 10.0)},
    )

    assert report.comparison.omnibus_details.assumptions["has_missing"] is True


def test_bakeoff_report_normalized_scores_respect_a_median_estimand(temp_experiment_setup: ExperimentSetup) -> None:
    """An outlier seed must not drag the reported score when the spec asks for a median."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac", "td3"]
    values = [1.0, 1.0, 1.0, 1.0, 100.0]
    for algorithm in algorithms:
        for seed, value in enumerate(values):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, value)
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(
        dataset,
        "metric",
        algorithms,
        block_by=["env_name"],
        verbose=False,
        references={"cartpole": ReferenceRange(0.0, 200.0)},
        spec=BakeoffSpec(estimand="median", missingness="blockwise"),
    )

    normalized_median = report.comparison.normalized_scores["ppo"]["cartpole"]
    normalized_mean = (sum(values) / len(values)) / 200.0
    assert normalized_median == pytest.approx(1.0 / 200.0)
    assert normalized_median != pytest.approx(normalized_mean)


def test_bakeoff_report_accepts_two_conditions(temp_experiment_setup: ExperimentSetup) -> None:
    """The omnibus test and Holm correction both degenerate correctly at n=2."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac"]
    for algorithm in algorithms:
        for seed in range(3):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, float(seed))
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(
        dataset,
        "metric",
        algorithms,
        block_by=["env_name"],
        verbose=False,
        references={"cartpole": ReferenceRange(0.0, 10.0)},
    )

    assert report.conditions == algorithms
    assert report.comparison.posthoc_matrix["ppo"]["sac"] == report.comparison.posthoc_matrix["sac"]["ppo"]


def test_bakeoff_report_reports_a_signed_difference_and_ci_per_pair(temp_experiment_setup: ExperimentSetup) -> None:
    """difference_estimate/difference_ci must be antisymmetric and carry a magnitude, not just a p-value."""
    setup = temp_experiment_setup
    for algorithm, base in (("ppo", 5.0), ("sac", 3.0)):
        for seed in range(6):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, base + seed * 0.01)
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(
        dataset,
        "metric",
        ["ppo", "sac"],
        block_by=["env_name"],
        verbose=False,
        references={"cartpole": ReferenceRange(0.0, 10.0)},
    )

    # references=(0, 10) normalizes scores, so the raw +2.0 gap becomes +0.2 here.
    assert report.comparison.difference_estimate["ppo"]["sac"] == pytest.approx(0.2, abs=0.01)
    assert report.comparison.difference_estimate["sac"]["ppo"] == pytest.approx(-0.2, abs=0.01)
    ppo_ci = report.comparison.difference_ci["ppo"]["sac"]
    sac_ci = report.comparison.difference_ci["sac"]["ppo"]
    assert ppo_ci is not None
    assert sac_ci is not None
    assert sac_ci[0] == pytest.approx(-ppo_ci[1])
    assert sac_ci[1] == pytest.approx(-ppo_ci[0])


def test_bakeoff_report_reports_raw_scores_without_references(temp_experiment_setup: ExperimentSetup) -> None:
    """No references means no normalization: scores stay in raw units."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac"]
    for algorithm in algorithms:
        for seed in range(3):
            _add_observation(setup, {"algorithm": algorithm, "env_source": "cartpole"}, seed, 42.0)
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    report = bakeoff_report(dataset, "metric", algorithms, block_by=["env_name"], verbose=False)

    assert report.comparison.normalized_scores["ppo"]["cartpole"] == pytest.approx(42.0)
    assert report.comparison.performance_profiles == {}
    assert report.comparison.optimality_gaps == {}


def test_bakeoff_report_rejects_pooling_multiple_environments_without_references(
    temp_experiment_setup: ExperimentSetup,
) -> None:
    """Raw scores from different environments cannot be pooled without a fixed anchor."""
    setup = temp_experiment_setup
    algorithms = ["ppo", "sac"]
    for algorithm in algorithms:
        for environment in ("cartpole", "mountaincar"):
            for seed in range(3):
                _add_observation(setup, {"algorithm": algorithm, "env_source": environment}, seed, 1.0)
    dataset = _load(setup, ExperimentDimensions(condition="algorithm", environment="env_source"))

    with pytest.raises(ValueError, match="requires references"):
        bakeoff_report(dataset, "metric", algorithms, block_by=["env_name"], verbose=False)


class TestBakeoffReport:
    """The consolidated champion-selection + comparison + warnings entry point."""

    def test_rejects_a_single_condition(self, temp_experiment_setup: ExperimentSetup) -> None:
        setup = temp_experiment_setup
        _add_curve_observation(setup, {"condition_name": "a"}, 0, [1.0, 1.0, 1.0])
        dataset = _load(setup)

        with pytest.raises(ValueError, match="at least 2 conditions"):
            bakeoff_report(dataset, "metric", ["a"], verbose=False)

    def test_no_blocking_no_selection_mirrors_a_plain_pairwise_comparison(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        setup = temp_experiment_setup
        for condition, base in (("a", 5.0), ("b", 3.0)):
            for seed in range(6):
                _add_curve_observation(setup, {"condition_name": condition}, seed, [base, base, base])
        dataset = _load(setup)

        report = bakeoff_report(dataset, "metric", ["a", "b"], verbose=False)

        assert report.champions is None
        assert report.cells["a"][""].n == 6
        assert report.cells["a"][""].estimate == pytest.approx(5.0)
        assert report.cells["b"][""].estimate == pytest.approx(3.0)

    def test_champion_selection_picks_the_right_config_per_condition_and_block(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """a's champion is low; b's champion is high; both must hold in both blocks."""
        setup = temp_experiment_setup
        for condition in ("a", "b"):
            for env in ("e1", "e2"):
                for learning_rate, bonus in (("low", 0.0), ("high", 2.0 if condition == "b" else -10.0)):
                    for seed in range(6):
                        base = {"a": 5.0, "b": 3.0}[condition]
                        _add_curve_observation(
                            setup,
                            {"condition_name": condition, "env_name": env, "learning_rate": learning_rate},
                            seed,
                            [base + bonus] * 3,
                        )
        dataset = _load(
            setup,
            ExperimentDimensions(environment="env_name", sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)),
        )

        report = bakeoff_report(
            dataset,
            "metric",
            ["a", "b"],
            select_champion_over=["learning_rate"],
            block_by=["env_name"],
            references={"e1": ReferenceRange(-20.0, 20.0), "e2": ReferenceRange(-20.0, 20.0)},
            verbose=False,
        )

        assert report.champions is not None
        assert report.champions["e1"]["a"].winning_config == {"learning_rate": "low"}
        assert report.champions["e1"]["b"].winning_config == {"learning_rate": "high"}
        assert report.champions["e2"]["a"].winning_config == {"learning_rate": "low"}
        assert report.champions["e2"]["b"].winning_config == {"learning_rate": "high"}

    def test_select_champion_over_must_match_the_declared_sweep_axes(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        setup = temp_experiment_setup
        for condition in ("a", "b"):
            _add_curve_observation(
                setup, {"condition_name": condition, "learning_rate": "low"}, 0, [1.0, 1.0, 1.0]
            )
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

        with pytest.raises(ValueError, match="select_champion_over must name exactly"):
            bakeoff_report(dataset, "metric", ["a", "b"], select_champion_over=["not_a_real_axis"], verbose=False)

    def test_flags_a_boundary_champion_on_a_coarse_grid(self, temp_experiment_setup: ExperimentSetup) -> None:
        """Only two swept values: whichever wins is necessarily at an edge of the grid."""
        setup = temp_experiment_setup
        for learning_rate, value in (("low", 1.0), ("high", 5.0)):
            for seed in range(6):
                _add_curve_observation(
                    setup, {"condition_name": "a", "learning_rate": learning_rate}, seed, [value] * 3
                )
        for learning_rate in ("low", "high"):
            for seed in range(6):
                _add_curve_observation(setup, {"condition_name": "b", "learning_rate": learning_rate}, seed, [1.0, 1.0, 1.0])
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

        report = bakeoff_report(
            dataset, "metric", ["a", "b"], select_champion_over=["learning_rate"], verbose=False
        )

        boundary_warnings = [w for w in report.warnings if w.code == BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY]
        assert any(w.condition == "a" for w in boundary_warnings)


class TestNonFiniteChampionInputs:
    def test_bakeoff_rejects_a_diverged_run_instead_of_treating_it_as_missing(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """A NaN final score is an outcome; the bakeoff must not read it as a missing cell."""
        setup = temp_experiment_setup
        for condition in ("a", "b"):
            for seed in range(4):
                value = float("nan") if (condition, seed) == ("b", 2) else float(seed)
                _add_observation(setup, {"condition_name": condition}, seed, value)

        with pytest.raises(NonFiniteValuesError) as error:
            bakeoff_report(_load(setup), "metric", ["a", "b"], verbose=False)

        assert error.value.offenders.select(["condition_name", "seed"]).to_dicts() == [{"condition_name": "b", "seed": 2}]

    def test_best_hypers_rejects_a_diverged_run(self, temp_experiment_setup: ExperimentSetup) -> None:
        """Champion selection must not rank configurations on NaN scores."""
        setup = temp_experiment_setup
        for learning_rate in ("low", "high"):
            for seed in range(3):
                value = float("-inf") if (learning_rate, seed) == ("high", 1) else 1.0
                _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, value)
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

        with pytest.raises(NonFiniteValuesError, match="'metric' run scores"):
            best_hypers(dataset, "metric", verbose=False)


class TestDuplicateSeedObservations:
    def test_best_hypers_rejects_runs_pooled_across_ablations(self, temp_experiment_setup: ExperimentSetup) -> None:
        """An ablated and a base run of one configuration and seed must not silently overwrite each other."""
        setup = temp_experiment_setup
        for learning_rate in ("low", "high"):
            for seed in range(3):
                _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, 1.0)
        _add_observation(setup, {"condition_name": "baseline", "learning_rate": "low"}, 0, 9.0, ablation="no_target")
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

        with pytest.raises(ValueError, match=r"'learning_rate': 'low'\} has more than one run for seed 0"):
            best_hypers(dataset, "metric", verbose=False)


    def test_bakeoff_rejects_runs_pooled_across_ablations(self, temp_experiment_setup: ExperimentSetup) -> None:
        """A base run and its ablation must not overwrite each other as one paired observation."""
        setup = temp_experiment_setup
        for condition in ("a", "b"):
            for seed in range(3):
                _add_observation(setup, {"condition_name": condition}, seed, float(seed))
        _add_observation(setup, {"condition_name": "a"}, 1, 9.0, ablation="no_target")

        with pytest.raises(ValueError, match="condition 'a' in block '' has more than one run for seed 1"):
            bakeoff_report(_load(setup), "metric", ["a", "b"], verbose=False)


class TestUnpairedRunReporting:
    def test_best_hypers_lists_runs_whose_seed_is_not_shared_by_every_configuration(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """A run with a seed missing from another configuration is excluded from the matrix and reported."""
        setup = temp_experiment_setup
        for learning_rate in ("low", "high"):
            for seed in range(3):
                _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, float(seed))
        _add_observation(setup, {"condition_name": "baseline", "learning_rate": "high"}, 7, 5.0)
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))
        seed_seven_run = dataset.runs.filter(pl.col("seed") == 7)["run_id"].to_list()

        report = best_hypers(dataset, "metric", verbose=False, n_resamples=64)

        assert isinstance(report, HyperparameterConfigReport)
        assert list(report.unpaired_run_ids) == seed_seven_run

    def test_best_hypers_difference_lists_runs_outside_the_shared_seeds(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """Seeds seen by only one compared group are reported rather than silently narrowed away."""
        setup = temp_experiment_setup
        for algorithm, seeds in (("a", range(4)), ("b", range(3))):
            for learning_rate in ("low", "high"):
                for seed in seeds:
                    _add_arm_observation(setup, algorithm, learning_rate, seed, float(seed))
        dataset = _load(
            setup,
            ExperimentDimensions(
                condition="algorithm", sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)
            ),
        )
        seed_three_runs = sorted(dataset.runs.filter(pl.col("seed") == 3)["run_id"].to_list())

        report = best_hypers(
            dataset, "metric", group_by=["algorithm"], difference_between=("a", "b"), verbose=False, n_resamples=64
        )

        assert list(report.unpaired_run_ids) == seed_three_runs

    def test_raw_champion_is_measured_on_the_same_paired_seeds_as_the_correction(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """An unpaired lucky run must not inflate the raw mean, or maximization bias compares two samples."""
        setup = temp_experiment_setup
        for learning_rate, value in (("low", 1.0), ("high", 2.0)):
            for seed in range(3):
                _add_observation(setup, {"condition_name": "baseline", "learning_rate": learning_rate}, seed, value)
        _add_observation(setup, {"condition_name": "baseline", "learning_rate": "low"}, 7, 100.0)
        dataset = _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

        report = best_hypers(dataset, "metric", verbose=False, n_resamples=64)

        assert report.winning_config == {"learning_rate": "high"}
        assert report.raw_mean == pytest.approx(2.0)
        assert report.maximization_bias == pytest.approx(0.0)


class TestDeclaredMeasure:
    @staticmethod
    def _early_and_late_learners(setup: ExperimentSetup) -> ExperimentDataset:
        for seed in range(3):
            _add_curve_observation(setup, {"condition_name": "a", "learning_rate": "early"}, seed, [9.0, 1.0])
            _add_curve_observation(setup, {"condition_name": "a", "learning_rate": "late"}, seed, [0.0, 4.0])
        return _load(setup, ExperimentDimensions(sweep_axes=(SweepAxis("learning_rate", SweepValueType.STRING),)))

    def test_best_hypers_selects_on_the_declared_measure(self, temp_experiment_setup: ExperimentSetup) -> None:
        """The champion depends on the estimand: final value favors the late learner, return rate the early one."""
        dataset = self._early_and_late_learners(temp_experiment_setup)

        final = best_hypers(dataset, "metric", verbose=False, n_resamples=200)
        rate = best_hypers(dataset, "metric", verbose=False, n_resamples=200, measure=ReturnRate())

        assert final.winning_config == {"learning_rate": "late"}
        assert rate.winning_config == {"learning_rate": "early"}
        assert rate.raw_mean == pytest.approx(5.0)
        assert rate.measure == ReturnRate()

    def test_bakeoff_scores_runs_with_the_declared_measure(self, temp_experiment_setup: ExperimentSetup) -> None:
        """Cell estimates summarize each run's tail window, and the report records that measure."""
        setup = temp_experiment_setup
        for seed in range(3):
            _add_curve_observation(setup, {"condition_name": "a"}, seed, [0.0, 2.0, 4.0, 6.0])
            _add_curve_observation(setup, {"condition_name": "b"}, seed, [1.0, 1.0, 1.0, 1.0])

        report = bakeoff_report(_load(setup), "metric", ["a", "b"], verbose=False, measure=TailMean(0.5))

        assert report.measure == TailMean(0.5)
        assert report.cells["a"][""].estimate == pytest.approx(5.0)


class TestBakeoffCells:
    def test_cells_report_estimate_interval_and_spread_but_no_variance(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """Each cell carries its estimand with a labelled percentile CI and SD for variability."""
        setup = temp_experiment_setup
        for seed, (value_a, value_b) in enumerate([(1.0, 0.0), (2.0, 0.5), (4.0, 1.0), (8.0, 1.5)]):
            _add_observation(setup, {"condition_name": "a"}, seed, value_a)
            _add_observation(setup, {"condition_name": "b"}, seed, value_b)

        report = bakeoff_report(
            _load(setup), "metric", ["a", "b"], spec=BakeoffSpec(estimand="median", n_resamples=200), verbose=False
        )

        cell = report.cells["a"][""]
        assert cell.n == 4
        assert cell.estimate == pytest.approx(3.0)
        assert cell.sd == pytest.approx(np.std([1.0, 2.0, 4.0, 8.0], ddof=1))
        assert cell.confidence == pytest.approx(0.95)
        assert cell.ci is not None and 1.0 <= cell.ci[0] <= cell.ci[1] <= 8.0

    def test_cells_report_a_tolerance_interval_only_with_enough_runs(
        self, temp_experiment_setup: ExperimentSetup
    ) -> None:
        """With enough runs the interval is the distribution-free one; below that it states the runs needed."""
        setup = temp_experiment_setup
        for seed in range(14):
            _add_observation(setup, {"condition_name": "a"}, seed, float(seed))
            _add_observation(setup, {"condition_name": "b"}, seed, float(seed % 3))
        spec = BakeoffSpec(n_resamples=50, tolerance_coverage=0.7)

        enough = bakeoff_report(_load(setup), "metric", ["a", "b"], spec=spec, verbose=False).cells["a"][""]
        strict = bakeoff_report(
            _load(setup), "metric", ["a", "b"], spec=replace(spec, tolerance_coverage=0.9), verbose=False
        ).cells["a"][""]

        expected = pointwise_tolerance_interval(np.arange(14.0), confidence=0.95, coverage=0.7)
        assert enough.tolerance == (pytest.approx(expected.ci_low), pytest.approx(expected.ci_high))
        assert enough.tolerance_min_runs == 14
        assert strict.tolerance is None
        assert strict.tolerance_min_runs == 46

    def test_cells_name_the_run_closest_to_the_median(self, temp_experiment_setup: ExperimentSetup) -> None:
        """The representative run is the one whose score sits at the cell median."""
        setup = temp_experiment_setup
        for seed, value in enumerate([5.0, 100.0, 1.0]):
            _add_observation(setup, {"condition_name": "a"}, seed, value)
            _add_observation(setup, {"condition_name": "b"}, seed, 0.0)
        dataset = _load(setup)
        median_run = dataset.runs.filter((pl.col("condition_name") == "a") & (pl.col("seed") == 0))["run_id"].item()

        report = bakeoff_report(dataset, "metric", ["a", "b"], spec=BakeoffSpec(n_resamples=50), verbose=False)

        assert report.cells["a"][""].median_run_id == median_run
