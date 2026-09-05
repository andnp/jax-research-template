"""Small unit tests for research_analysis.tuning."""

import numpy as np
import pytest
from research_analysis.tuning import HierarchicalBootstrapCI, hierarchical_bootstrap_ci


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
