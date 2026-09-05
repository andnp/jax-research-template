"""Small unit tests for research_analysis.tuning."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.tuning import (
    HierarchicalBootstrapCI,
    HierarchicalBootstrapDifferenceCI,
    hierarchical_bootstrap_ci,
    hierarchical_bootstrap_ci_from_experiment,
    hierarchical_bootstrap_difference_ci,
)


def _populate_metrics_db(db_path: Path, run_id: int, execution_id: int, metric_name: str, value: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metrics (run_id INTEGER, execution_id INTEGER, "
        "metric_name TEXT, global_step INTEGER, value REAL)"
    )
    conn.execute(
        "INSERT INTO metrics (run_id, execution_id, metric_name, global_step, value) VALUES (?, ?, ?, ?, ?)",
        (run_id, execution_id, metric_name, 0, value),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def tuning_experiment(tmp_path: Path) -> Generator[tuple[Path, str], None, None]:
    """A tiny experiment with 3 hyper values x 5 seeds, hyper 2 clearly best."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_db_path = tmp_path / "metrics.sqlite"
    slug = "tuning-test-exp"

    with DatabaseManager(db_path) as db:
        db.initialize()
        exp_id = db.add_experiment(slug, "Tuning test experiment")
        algo_ver_id = db.add_component_version(db.add_component("algo", "ALGO"), "hash_algo")
        env_ver_id = db.add_component_version(db.add_component("env", "ENV"), "hash_env")

        for learning_rate in (0.001, 0.01, 0.1):
            for seed in range(5):
                base = 10.0 if learning_rate == 0.1 else 0.0
                hyper_id = db.add_hyperparam_config({"learning_rate": learning_rate, "seed": seed})
                run_id = db.add_run(
                    experiment_id=exp_id,
                    algo_version_id=algo_ver_id,
                    env_version_id=env_ver_id,
                    hyper_id=hyper_id,
                    seed=seed,
                )
                exec_id = db.add_execution()
                db.update_execution_status(exec_id, "COMPLETED")
                db.link_execution_run(exec_id, run_id)
                db.record_execution_artifacts(
                    exec_id, str(tmp_path), metadata={"metrics_db_path": str(metrics_db_path)}
                )
                _populate_metrics_db(metrics_db_path, run_id, exec_id, "reward", base + seed)

    yield db_path, slug


class TestHierarchicalBootstrapCI:
    def test_basic_contract_and_determinism(self) -> None:
        """Verify return type, probability normalization, and deterministic seeding."""
        rng = np.random.default_rng(42)
        # 3 hypers, 20 runs each
        data = rng.normal(loc=0.0, scale=1.0, size=(3, 20))
        data[2] += 2.0  # Hyper 2 is distinctly better

        res1 = hierarchical_bootstrap_ci(data, h_samples=5, e_samples=10, seed=123)
        res2 = hierarchical_bootstrap_ci(data, h_samples=5, e_samples=10, seed=123)

        assert isinstance(res1, HierarchicalBootstrapCI)
        assert np.isclose(res1.mean, res2.mean)
        assert np.isclose(res1.ci_low, res2.ci_low)
        assert np.isclose(res1.ci_high, res2.ci_high)
        assert np.isclose(res1.selection_entropy, res2.selection_entropy)
        assert np.allclose(res1.hyper_probs, res2.hyper_probs)
        assert np.isclose(np.sum(res1.hyper_probs), 1.0)
        assert res1.ci_low <= res1.mean <= res1.ci_high

    def test_selection_uncertainty_expands_ci(self) -> None:
        """Verify that high selection ambiguity yields higher entropy than a dominant hyper."""
        rng = np.random.default_rng(42)
        # Distinct hyper
        dominant = np.zeros((3, 30))
        dominant[0] = rng.normal(loc=10.0, scale=0.5, size=30)
        dominant[1] = rng.normal(loc=0.0, scale=0.5, size=30)
        dominant[2] = rng.normal(loc=0.0, scale=0.5, size=30)

        # Ambiguous hypers with high overlap
        ambiguous = rng.normal(loc=5.0, scale=2.0, size=(3, 30))

        res_dom = hierarchical_bootstrap_ci(dominant, h_samples=5, e_samples=10, seed=42)
        res_amb = hierarchical_bootstrap_ci(ambiguous, h_samples=5, e_samples=10, seed=42)

        # Dominant hyper has near-zero selection entropy and concentrated probability
        assert res_dom.selection_entropy < res_amb.selection_entropy
        assert res_dom.hyper_probs[0] > 0.95

    def test_ci_width_does_not_collapse_as_e_samples_grows(self) -> None:
        """Verify selection-uncertainty variance survives large evaluation budgets.

        A hyper drawn once per bootstrap replicate (not once per sample) must
        leave a non-vanishing floor on CI width as e_samples grows, reflecting
        the genuine 9.5%/90.5% selection split between two well-separated
        hypers. A regression to per-sample redraws would shrink this toward 0.
        """
        rng = np.random.default_rng(0)
        data = np.zeros((2, 10_000))
        data[0] = rng.normal(loc=0.0, scale=1.0, size=10_000)
        data[1] = rng.normal(loc=1.0, scale=1.0, size=10_000)

        # Both e_samples are already large enough that within-hyper noise is
        # negligible; a correct implementation's width should be near-stable
        # between them, while a per-sample-redraw regression keeps shrinking
        # (0.183 -> 0.057, roughly halving each 10x) because it averages away
        # the between-hyper selection variance instead of preserving it.
        medium = hierarchical_bootstrap_ci(data, h_samples=3, e_samples=500, seed=1, n_resamples=20_000)
        large = hierarchical_bootstrap_ci(data, h_samples=3, e_samples=5_000, seed=1, n_resamples=20_000)

        medium_width = medium.ci_high - medium.ci_low
        large_width = large.ci_high - large.ci_low
        assert large_width > 0.8 * medium_width

    def test_validation_rejects_invalid_inputs(self) -> None:
        """Reject 1D inputs, single-hyper inputs, and insufficient runs."""
        with pytest.raises(ValueError, match="must be 2D array"):
            hierarchical_bootstrap_ci(np.ones(10), h_samples=3, e_samples=5)

        with pytest.raises(ValueError, match="need at least 2 hyperparameters"):
            hierarchical_bootstrap_ci(np.ones((1, 10)), h_samples=3, e_samples=5)

        with pytest.raises(ValueError, match="must be >= h_samples"):
            hierarchical_bootstrap_ci(np.ones((3, 5)), h_samples=10, e_samples=5)

    def test_e_samples_may_exceed_n_runs(self) -> None:
        """Evaluation resampling is with replacement, so e_samples > n_runs is valid."""
        rng = np.random.default_rng(7)
        data = rng.normal(size=(2, 8))
        res = hierarchical_bootstrap_ci(data, h_samples=3, e_samples=50, seed=1)
        assert res.ci_low <= res.mean <= res.ci_high

    def test_h_samples_selection_is_without_replacement(self) -> None:
        """A budget equal to n_runs must use every run exactly once, not duplicate any."""
        from research_analysis._tuning_kernels import get_hyper_distribution

        data = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        probs, _entropy = get_hyper_distribution(data, h_samples=4, selection_trials=50, seed=0)
        # Hyper 1 dominates unambiguously regardless of which 4-of-4 draw order is used.
        assert probs[1] == 1.0


class TestHierarchicalBootstrapCIFromExperiment:
    def test_loads_and_evaluates_the_stored_grid(self, tuning_experiment: tuple[Path, str]) -> None:
        """Verify the db-driven wrapper reproduces the array-based function's contract."""
        db_path, slug = tuning_experiment
        res = hierarchical_bootstrap_ci_from_experiment(
            db_path, slug, ["learning_rate"], "reward", h_samples=3, e_samples=5, seed=1
        )
        assert isinstance(res, HierarchicalBootstrapCI)
        # learning_rate=0.1 has base reward 10.0 vs 0.0 for the others: unambiguous winner.
        assert res.hyper_probs[-1] > 0.95
        assert res.ci_low <= res.mean <= res.ci_high

    def test_rejects_ungrouped_hyperparameter(self, tuning_experiment: tuple[Path, str]) -> None:
        """A key absent from every run's config raises, matching _load_hyperparameter_records."""
        db_path, slug = tuning_experiment
        with pytest.raises(ValueError, match="No completed runs found"):
            hierarchical_bootstrap_ci_from_experiment(
                db_path, slug, ["nonexistent_key"], "reward", h_samples=3, e_samples=5
            )


class TestHierarchicalBootstrapDifferenceCI:
    def test_clear_winner_excludes_zero(self) -> None:
        """A large, unambiguous gap between arms' champions must exclude zero."""
        rng = np.random.default_rng(1)
        data_a = np.zeros((2, 200))
        data_a[0] = rng.normal(loc=5.0, scale=0.5, size=200)  # arm A's champion
        data_a[1] = rng.normal(loc=0.0, scale=0.5, size=200)

        data_b = np.zeros((2, 200))
        data_b[0] = rng.normal(loc=0.0, scale=0.5, size=200)
        data_b[1] = rng.normal(loc=0.0, scale=0.5, size=200)  # arm B's champion, much worse

        res = hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=5, e_samples=20, seed=1)
        assert isinstance(res, HierarchicalBootstrapDifferenceCI)
        assert res.ci_low > 0.0
        assert res.mean_diff == pytest.approx(5.0, abs=1.0)

    def test_no_real_difference_includes_zero(self) -> None:
        """Two arms drawn from the same distribution must not exclude zero."""
        rng = np.random.default_rng(2)
        data_a = rng.normal(loc=0.0, scale=1.0, size=(2, 200))
        data_b = rng.normal(loc=0.0, scale=1.0, size=(2, 200))

        res = hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=5, e_samples=20, seed=1)
        assert res.ci_low < 0.0 < res.ci_high

    def test_rejects_mismatched_run_counts(self) -> None:
        """data_a and data_b must be paired by run index, so column counts must match."""
        data_a = np.ones((2, 10))
        data_b = np.ones((2, 8))
        with pytest.raises(ValueError, match="same number of paired runs"):
            hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=3, e_samples=5)
