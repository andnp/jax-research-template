"""Small unit tests for research_analysis.tuning."""

import numpy as np
import pytest
from research_analysis.tuning import (
    HierarchicalBootstrapCI,
    HierarchicalBootstrapDifferenceCI,
    hierarchical_bootstrap_ci,
    hierarchical_bootstrap_difference_ci,
)


class TestHierarchicalBootstrapCI:
    def test_basic_contract_and_determinism(self) -> None:
        """Verify return type, probability normalization, and deterministic seeding."""
        rng = np.random.default_rng(42)
        # 3 hypers, 20 runs each
        data = rng.normal(loc=0.0, scale=1.0, size=(3, 20))
        data[2] += 2.0  # Hyper 2 is distinctly better

        res1 = hierarchical_bootstrap_ci(data, h_samples=5, e_samples=10, rng=np.random.default_rng(123))
        res2 = hierarchical_bootstrap_ci(data, h_samples=5, e_samples=10, rng=np.random.default_rng(123))

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

        res_dom = hierarchical_bootstrap_ci(dominant, h_samples=5, e_samples=10, rng=np.random.default_rng(42))
        res_amb = hierarchical_bootstrap_ci(ambiguous, h_samples=5, e_samples=10, rng=np.random.default_rng(42))

        # Dominant hyper has near-zero selection entropy and concentrated probability
        assert res_dom.selection_entropy < res_amb.selection_entropy
        assert res_dom.hyper_probs[0] > 0.95

    def test_ci_width_shrinks_as_the_coupled_budget_grows(self) -> None:
        """Under h_samples == e_samples == n_runs (the only way this is ever actually
        called), CI width should shrink as the shared budget grows, like an ordinary
        bootstrap estimator's -- not the decoupled h_samples-fixed-small/e_samples-swept
        configuration this function is never called under."""
        rng = np.random.default_rng(0)
        widths = []
        for n in (100, 1_000, 10_000):
            data = np.zeros((2, n))
            data[0] = rng.normal(loc=0.0, scale=1.0, size=n)
            data[1] = rng.normal(loc=1.0, scale=1.0, size=n)
            res = hierarchical_bootstrap_ci(data, h_samples=n, e_samples=n, rng=np.random.default_rng(1), n_resamples=5_000)
            widths.append(res.ci_high - res.ci_low)

        assert widths[0] > widths[1] > widths[2]

    def test_selection_distribution_is_not_degenerate_at_full_pool_size(self) -> None:
        """h_samples == n_runs must not collapse to a deterministic point mass: sampling
        with replacement preserves genuine trial-to-trial variation even at full pool
        size, unlike sampling without replacement, where drawing the whole pool is a
        fixed permutation with an invariant mean regardless of true selection ambiguity.
        """
        rng = np.random.default_rng(0)
        n = 30
        data = np.zeros((2, n))
        data[0] = rng.normal(loc=0.0, scale=1.0, size=n)
        data[1] = rng.normal(loc=0.05, scale=1.0, size=n)  # genuinely close true means

        res = hierarchical_bootstrap_ci(data, h_samples=n, e_samples=n, n_resamples=100, rng=np.random.default_rng(1))
        assert res.selection_entropy > 0.0
        assert not np.any(res.hyper_probs == 1.0)

    def test_validation_rejects_invalid_inputs(self) -> None:
        """Reject 1D inputs and single-hyper inputs."""
        with pytest.raises(ValueError, match="must be 2D array"):
            hierarchical_bootstrap_ci(np.ones(10), h_samples=3, e_samples=5, rng=np.random.default_rng(0))

        with pytest.raises(ValueError, match="need at least 2 hyperparameters"):
            hierarchical_bootstrap_ci(np.ones((1, 10)), h_samples=3, e_samples=5, rng=np.random.default_rng(0))

    def test_validation_rejects_non_finite_values(self) -> None:
        """A diverged run must not flow into the resampling, which would propagate NaN silently."""
        data = np.ones((2, 5))
        data[0, 3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            hierarchical_bootstrap_ci(data, h_samples=3, e_samples=5, rng=np.random.default_rng(0))

    def test_h_and_e_samples_may_exceed_n_runs(self) -> None:
        """Both selection and evaluation resampling are with replacement, so neither
        h_samples nor e_samples is bounded by n_runs."""
        rng = np.random.default_rng(7)
        data = rng.normal(size=(2, 8))
        res = hierarchical_bootstrap_ci(data, h_samples=50, e_samples=50, rng=np.random.default_rng(1))
        assert res.ci_low <= res.mean <= res.ci_high

    def test_a_single_run_is_the_minimum_valid_matrix(self) -> None:
        """A matrix with exactly one run per hyperparameter is valid, not an edge failure."""
        data = np.array([[1.0], [2.0]])
        res = hierarchical_bootstrap_ci(data, h_samples=3, e_samples=3, rng=np.random.default_rng(0))
        assert res.ci_low <= res.mean <= res.ci_high

    def test_rejects_a_matrix_with_no_runs(self) -> None:
        with pytest.raises(ValueError, match="need at least 1 run"):
            hierarchical_bootstrap_ci(np.ones((2, 0)), h_samples=3, e_samples=5, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5, True])
    def test_rejects_non_positive_or_non_int_h_samples(self, bad_value: int) -> None:
        data = np.ones((2, 5))
        with pytest.raises(ValueError, match="h_samples must be a positive int"):
            hierarchical_bootstrap_ci(data, h_samples=bad_value, e_samples=3, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5, True])
    def test_rejects_non_positive_or_non_int_e_samples(self, bad_value: int) -> None:
        data = np.ones((2, 5))
        with pytest.raises(ValueError, match="e_samples must be a positive int"):
            hierarchical_bootstrap_ci(data, h_samples=3, e_samples=bad_value, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5, True])
    def test_rejects_non_positive_or_non_int_n_resamples(self, bad_value: int) -> None:
        data = np.ones((2, 5))
        with pytest.raises(ValueError, match="n_resamples must be a positive int"):
            hierarchical_bootstrap_ci(
                data, h_samples=3, e_samples=3, n_resamples=bad_value, rng=np.random.default_rng(0)
            )

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5, True])
    def test_rejects_non_positive_or_non_int_selection_trials(self, bad_value: int) -> None:
        data = np.ones((2, 5))
        with pytest.raises(ValueError, match="selection_trials must be a positive int"):
            hierarchical_bootstrap_ci(
                data, h_samples=3, e_samples=3, selection_trials=bad_value, rng=np.random.default_rng(0)
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

        res = hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=5, e_samples=20, rng=np.random.default_rng(1))
        assert isinstance(res, HierarchicalBootstrapDifferenceCI)
        assert res.ci_low > 0.0
        assert res.mean_diff == pytest.approx(5.0, abs=1.0)

    def test_no_real_difference_includes_zero(self) -> None:
        """Two arms drawn from the same distribution must not exclude zero."""
        rng = np.random.default_rng(2)
        data_a = rng.normal(loc=0.0, scale=1.0, size=(2, 200))
        data_b = rng.normal(loc=0.0, scale=1.0, size=(2, 200))

        res = hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=5, e_samples=20, rng=np.random.default_rng(1))
        assert res.ci_low < 0.0 < res.ci_high

    def test_rejects_mismatched_run_counts(self) -> None:
        """data_a and data_b must be paired by run index, so column counts must match."""
        data_a = np.ones((2, 10))
        data_b = np.ones((2, 8))
        with pytest.raises(ValueError, match="same number of paired runs"):
            hierarchical_bootstrap_difference_ci(data_a, data_b, h_samples=3, e_samples=5, rng=np.random.default_rng(0))

    def test_rejects_a_matrix_with_no_runs(self) -> None:
        with pytest.raises(ValueError, match="need at least 1 run"):
            hierarchical_bootstrap_difference_ci(
                np.ones((2, 0)), np.ones((2, 0)), h_samples=3, e_samples=5, rng=np.random.default_rng(0)
            )

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5, True])
    def test_rejects_non_positive_or_non_int_h_samples(self, bad_value: int) -> None:
        data_a = np.ones((2, 10))
        data_b = np.ones((2, 10))
        with pytest.raises(ValueError, match="h_samples must be a positive int"):
            hierarchical_bootstrap_difference_ci(
                data_a, data_b, h_samples=bad_value, e_samples=5, rng=np.random.default_rng(0)
            )
