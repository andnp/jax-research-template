"""Behavior tests for multiple-comparison correction."""

import pytest
from research_analysis.multiplicity import bonferroni_confidence, holm_step_down


def test_holm_adjustment_preserves_input_order() -> None:
    """Holm correction returns adjusted values aligned with original hypotheses."""
    adjusted = holm_step_down([0.04, 0.001, 0.02])

    assert adjusted == pytest.approx([0.04, 0.003, 0.04])


def test_holm_adjustment_is_step_down_monotone_in_sorted_order() -> None:
    """Adjusted p-values are monotone after sorting by raw significance."""
    adjusted = holm_step_down([0.03, 0.01, 0.02])
    ordered = [adjusted[index] for index in (1, 2, 0)]

    assert ordered == sorted(ordered)


def test_holm_adjustment_handles_empty_input() -> None:
    """No hypotheses produce an empty adjustment without a placeholder value."""
    assert holm_step_down([]) == []


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_holm_adjustment_rejects_invalid_p_values(value: float) -> None:
    """Invalid p-values fail instead of becoming misleading adjusted results."""
    with pytest.raises(ValueError, match="finite numbers"):
        holm_step_down([value])


def test_bonferroni_confidence_splits_the_family_error_rate() -> None:
    """Five intervals at a joint 95% each use 99%; a single comparison is unadjusted."""
    assert bonferroni_confidence(0.95, 5) == pytest.approx(0.99)
    assert bonferroni_confidence(0.95, 1) == pytest.approx(0.95)


def test_bonferroni_confidence_rejects_an_empty_family() -> None:
    """A family needs at least one comparison and a confidence strictly inside (0, 1)."""
    with pytest.raises(ValueError, match="family_size"):
        bonferroni_confidence(0.95, 0)
    with pytest.raises(ValueError, match="family_confidence"):
        bonferroni_confidence(1.0, 3)
