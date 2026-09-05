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

    def test_validation_rejects_invalid_inputs(self) -> None:
        """Reject 1D inputs, single-hyper inputs, and insufficient runs."""
        with pytest.raises(ValueError, match="must be 2D array"):
            hierarchical_bootstrap_ci(np.ones(10), h_samples=3, e_samples=5)

        with pytest.raises(ValueError, match="need at least 2 hyperparameters"):
            hierarchical_bootstrap_ci(np.ones((1, 10)), h_samples=3, e_samples=5)

        with pytest.raises(ValueError, match="must be >= max"):
            hierarchical_bootstrap_ci(np.ones((3, 5)), h_samples=10, e_samples=5)
