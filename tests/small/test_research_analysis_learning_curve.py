"""Small tests for research_analysis.learning_curve."""

import numpy as np
import pytest
from research_analysis.learning_curve import step_weighted_returns


class TestStepWeightedReturns:
    def test_science_guide_worked_example_matches_copy_forward_curve(self):
        cumulative_steps = np.array([50, 120, 180])
        episodic_returns = np.array([-100.0, -80.0, -75.0])

        result = step_weighted_returns(cumulative_steps, episodic_returns)

        expected = np.concatenate(
            [
                np.full(50, -100.0),
                np.full(70, -80.0),
                np.full(60, -75.0),
            ]
        )
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64

    def test_one_episode_input_fills_all_steps(self):
        result = step_weighted_returns(
            np.array([3]),
            np.array([1.5]),
            end_step=3,
        )

        np.testing.assert_array_equal(result, np.array([1.5, 1.5, 1.5]))

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            step_weighted_returns(np.array([], dtype=np.int64), np.array([], dtype=np.float64))

    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            step_weighted_returns(np.array([1, 2]), np.array([1.0]))

    def test_non_monotonic_steps_raise_value_error(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            step_weighted_returns(np.array([2, 5, 4]), np.array([1.0, 2.0, 3.0]))

    def test_duplicate_steps_raise_value_error(self):
        with pytest.raises(ValueError, match="duplicates"):
            step_weighted_returns(np.array([2, 5, 5]), np.array([1.0, 2.0, 3.0]))

    def test_non_positive_step_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly positive"):
            step_weighted_returns(np.array([0, 2]), np.array([1.0, 2.0]))

    def test_end_step_extension_copies_forward_final_return(self):
        result = step_weighted_returns(
            np.array([2, 5]),
            np.array([1.0, 3.0]),
            end_step=8,
        )

        np.testing.assert_array_equal(result, np.array([1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]))

    def test_end_step_smaller_than_last_cumulative_step_raises(self):
        with pytest.raises(ValueError, match="at least the last cumulative step"):
            step_weighted_returns(np.array([2, 5]), np.array([1.0, 3.0]), end_step=4)
